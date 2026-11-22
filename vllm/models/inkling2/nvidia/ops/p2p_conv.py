# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-rank PCIe P2P all-reduce fused into Inkling2's hidden-stream short conv.

Without NVLink multicast (`ops/allreduce.py`), a small batch's all-reduce and
the short conv after it are two launches. Here the conv kernel itself pushes
each program's rows of this rank's bf16 partial into the peer's CUDA IPC
receive buffer, then reads every row it convolves as its own partial plus the
peer's (polled until no element holds the -0.0 sentinel), summed once in fp32
and rounded once to bf16 (a two-term sum is identical on both ranks). A -0.0
partial is sent and summed as +0.0.

The receive buffer has three generations. The generation is a device counter
the last program of each launch advances, so CUDA graph replays stay in step
on both ranks. A launch re-arms the generation used two launches before: its
local readers are done (stream order) and the peer cannot write it again
before reading this rank's pushes of the next launch.
"""

import os

import torch

from vllm.triton_utils import tl, triton

MAX_TOKENS = 8
_NUM_GENERATIONS = 3
_SENTINEL = -32768  # bf16 -0.0 as int16

_workspaces: dict[int, "P2PConvWorkspace | None"] = {}


def p2p_conv_enabled() -> bool:
    return os.environ.get("VLLM_INKLING2_P2P_AR_CONV", "0") == "1"


@triton.jit
def _fill_kernel(addr, numel, value, BLOCK: tl.constexpr):
    ptr = addr.to(tl.pointer_type(tl.int16))
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + offs, tl.full((BLOCK,), value, tl.int16), mask=offs < numel)


class P2PConvWorkspace:
    """Receive buffers (this rank's and the peer's mapping) and the
    `[generation, arrivals]` counter of the fused kernel."""

    def __init__(self, hidden_size: int) -> None:
        from vllm.distributed import get_tp_group
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        group = get_tp_group()
        assert group.world_size == 2
        self.rank = group.rank_in_group
        self.hidden_size = hidden_size
        self.max_tokens = MAX_TOKENS
        self.generation_numel = MAX_TOKENS * hidden_size
        numel = _NUM_GENERATIONS * self.generation_numel
        self.pointers = CustomAllreduce.create_shared_buffer(
            numel * 2, group=group.cpu_group
        )
        self.recv_addr = self.pointers[self.rank]
        self.peer_addr = self.pointers[1 - self.rank]
        _fill_kernel[(triton.cdiv(numel, 4096),)](
            self.recv_addr, numel, _SENTINEL, BLOCK=4096
        )
        self.counter = torch.zeros(2, dtype=torch.int32, device="cuda")
        torch.accelerator.synchronize()
        torch.distributed.barrier(group=group.cpu_group)

    def usable(self, num_tokens: int, hidden_size: int) -> bool:
        return 0 < num_tokens <= MAX_TOKENS and hidden_size == self.hidden_size


def p2p_conv_workspace(hidden_size: int) -> P2PConvWorkspace | None:
    """The process's workspace for `hidden_size`, created on first use (an
    eager forward on every rank, before graph capture), or None when the
    fused path is off or unsupported."""
    if hidden_size in _workspaces:
        return _workspaces[hidden_size]
    workspace = None
    if p2p_conv_enabled():
        from vllm.distributed import get_tensor_model_parallel_world_size

        if get_tensor_model_parallel_world_size() == 2:
            workspace = P2PConvWorkspace(hidden_size)
    _workspaces[hidden_size] = workspace
    return workspace
