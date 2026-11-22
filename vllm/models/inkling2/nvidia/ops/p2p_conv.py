# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-rank PCIe P2P all-reduce fused into Inkling2's hidden-stream short conv.

Without NVLink multicast (`ops/allreduce.py`), a small batch's all-reduce and
the short conv after it are two launches. Here the conv kernel itself pushes
each program's rows of this rank's bf16 partial into the peer's CUDA IPC
receive buffer, then reads every row it convolves as its own partial plus the
peer's (polled until no element holds the -0.0 sentinel), summed once in fp32
and rounded once to bf16 (a two-term sum is identical on both ranks). A -0.0
partial is sent and summed as +0.0, so two -0.0 partials give +0.0 where an
all-reduce may keep -0.0 (the same value), and a NaN partial may reach the two
ranks with different NaN payloads.

Every program spins on its peer program after pushing, so a launch completes
only if all its programs can be resident on both GPUs at once: the first launch
of each configuration checks the compiled kernel's occupancy and raises instead
of risking a cross-rank hang. Both ranks take the fused path or neither: the
workspace exists only if they report the same device capability and kernel
path, and every other condition depends on the batch both ranks share.

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

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the Inkling2 P2P conv workspace must be created by an eager "
                "forward before CUDA graph capture"
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
        self._resident: set = set()
        torch.accelerator.synchronize()
        torch.distributed.barrier(group=group.cpu_group)

    def usable(self, num_tokens: int, hidden_size: int) -> bool:
        return 0 < num_tokens <= MAX_TOKENS and hidden_size == self.hidden_size

    def check_residency(self, kernel, grid, args, kwargs, key) -> None:
        """Raise unless every program of `kernel[grid]` fits on the GPU at
        once (compiled once per `key` without launching)."""
        if key in self._resident:
            return
        compiled = kernel.warmup(*args, grid=grid, **kwargs)
        compiled._init_handles()
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        warps = compiled.metadata.num_warps
        regs_per_warp = -(-max(compiled.n_regs, 1) * 32 // 256) * 256
        per_sm = min(
            16,  # resident blocks per SM on SM8x
            props.max_threads_per_multi_processor // (warps * 32),
            props.regs_per_multiprocessor // regs_per_warp // warps,
            props.shared_memory_per_multiprocessor
            // (compiled.metadata.shared + 1024),
        )
        programs = grid[0] * grid[1]
        if programs > per_sm * props.multi_processor_count:
            raise RuntimeError(
                f"P2P conv grid of {programs} programs exceeds the "
                f"{per_sm * props.multi_processor_count} resident ones"
            )
        self._resident.add(key)


def _ranks_agree() -> bool:
    """Whether both TP ranks have the same device capability and kernel path
    (the same answer on both, so neither enters the fused kernel alone)."""
    from vllm.distributed import get_tp_group

    from . import use_portable_kernels

    mine = (torch.cuda.get_device_capability(), use_portable_kernels())
    every = [None, None]
    torch.distributed.all_gather_object(every, mine, group=get_tp_group().cpu_group)
    return every[0] == every[1]


def p2p_conv_workspace(hidden_size: int) -> P2PConvWorkspace | None:
    """The process's workspace for `hidden_size`, created on first use (an
    eager forward on every rank, before graph capture), or None when the
    fused path is off or unsupported."""
    if hidden_size in _workspaces:
        return _workspaces[hidden_size]
    workspace = None
    if p2p_conv_enabled():
        from vllm.distributed import get_tensor_model_parallel_world_size

        if get_tensor_model_parallel_world_size() == 2 and _ranks_agree():
            workspace = P2PConvWorkspace(hidden_size)
    _workspaces[hidden_size] = workspace
    return workspace
