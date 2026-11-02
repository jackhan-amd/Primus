# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU target selection shared by backend configuration and patches."""


def is_rocm_target(target="auto"):
    if target in ("cuda", "nvidia"):
        return False
    if target == "amd":
        return True
    if target != "auto":
        raise ValueError(f"Unknown target_gpu: {target!r}; expected auto, amd, or cuda")
    import torch

    return getattr(torch.version, "hip", None) is not None


def configure_gpu_params(params, target="auto"):
    """Disable AMD-only Turbo features before any backend patches run."""
    target = getattr(params, "target_gpu", target)
    params.target_gpu = target
    if is_rocm_target(target):
        return
    for key in vars(params):
        if key == "enable_primus_turbo" or key.startswith("use_turbo_"):
            setattr(params, key, False)
    turbo = getattr(params, "primus_turbo", None)
    if turbo is not None:
        for key in vars(turbo):
            if key.startswith(("enable_", "use_turbo_")):
                setattr(turbo, key, False)
    model = getattr(params, "model", None)
    if model is not None and hasattr(model, "converters"):
        model.converters = [c for c in model.converters if c != "primus_turbo"]
