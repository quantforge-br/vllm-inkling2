# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools

from vllm.triton_utils import tl, triton


@functools.cache
def use_sm100_kernels() -> bool:
    """Whether to run the Blackwell kernels (tml_fa4, tcgen05 attention,
    TRT-LLM-gen expert GEMMs, e2m1 conversions). Hopper uses Humming experts;
    SM12x uses Triton experts and does not support tcgen05."""
    from vllm.platforms import current_platform

    return current_platform.is_device_capability_family(100)


@functools.cache
def use_pdl() -> bool:
    """Whether the GPU has programmatic dependent launch (`griddepcontrol`,
    SM90+). Before Hopper the kernels launch without it and skip the
    dependency waits (stream order already serializes them)."""
    from vllm.platforms import current_platform

    return current_platform.has_device_capability(90)


@functools.cache
def use_portable_kernels() -> bool:
    """Whether to take the SM12x paths that need neither TMA, `wgmma` nor
    the native Hopper / SM10x conv-mix kernels (~100 KiB of shared memory
    per block): SM12x, and Ampere / Ada (SM8x)."""
    from vllm.platforms import current_platform

    return current_platform.is_device_capability_family(
        120
    ) or not current_platform.has_device_capability(90)


# Launch flag of the Triton kernels written for PDL.
PDL = use_pdl()
_USE_PDL = tl.constexpr(PDL)


@triton.jit
def gdc_wait():
    """`tl.extra.cuda.gdc_wait` where the GPU has PDL, else nothing."""
    if _USE_PDL:
        tl.extra.cuda.gdc_wait()


@triton.jit
def gdc_launch_dependents():
    """`tl.extra.cuda.gdc_launch_dependents` where the GPU has PDL, else
    nothing."""
    if _USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
