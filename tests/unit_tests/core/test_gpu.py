# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only regression tests; run directly without PyTorch or backend imports."""

import importlib.util
import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "gpu_utils", Path(__file__).resolve().parents[3] / "primus/core/utils/gpu.py"
)
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


class GpuTargetTests(unittest.TestCase):
    def test_explicit_targets(self):
        self.assertTrue(gpu.is_rocm_target("amd"))
        for target in ("cuda", "nvidia"):
            self.assertFalse(gpu.is_rocm_target(target))
        with self.assertRaises(ValueError):
            gpu.is_rocm_target("typo")

    def test_auto_uses_torch_build(self):
        for hip, expected in (("7.0", True), (None, False)):
            with patch.dict(sys.modules, {"torch": SimpleNamespace(version=SimpleNamespace(hip=hip))}):
                self.assertEqual(gpu.is_rocm_target(), expected)

    def test_cuda_disables_flat_and_nested_turbo(self):
        params = SimpleNamespace(
            enable_primus_turbo=True,
            use_turbo_grouped_gemm=True,
            bf16=True,
            primus_turbo=SimpleNamespace(enable_attention_float8=True, use_turbo_attention=True),
            model=SimpleNamespace(converters=["float8", "primus_turbo"]),
        )
        gpu.configure_gpu_params(params, "cuda")
        self.assertFalse(params.enable_primus_turbo)
        self.assertFalse(params.use_turbo_grouped_gemm)
        self.assertFalse(params.primus_turbo.enable_attention_float8)
        self.assertFalse(params.primus_turbo.use_turbo_attention)
        self.assertTrue(params.bf16)
        self.assertEqual(params.model.converters, ["float8"])

    def test_module_target_takes_precedence(self):
        params = SimpleNamespace(target_gpu="amd", enable_primus_turbo=True)
        gpu.configure_gpu_params(params, "cuda")
        self.assertTrue(params.enable_primus_turbo)

    def test_amd_configuration_is_preserved(self):
        params = SimpleNamespace(enable_primus_turbo=True, use_turbo_attention=True)
        gpu.configure_gpu_params(params, "amd")
        self.assertTrue(params.enable_primus_turbo)
        self.assertTrue(params.use_turbo_attention)

    def test_validate_wrapper_preserves_upstream_and_gates_rocm(self):
        registrations = {}

        def register(name, **kwargs):
            registrations[name] = kwargs
            return lambda fn: fn

        calls = []
        args_mod = SimpleNamespace(validate_args=lambda args: calls.append("upstream"))
        init_mod = SimpleNamespace()
        modules = {
            "megatron": SimpleNamespace(training=SimpleNamespace(arguments=args_mod, initialize=init_mod)),
            "megatron.training": SimpleNamespace(arguments=args_mod, initialize=init_mod),
            "megatron.training.arguments": args_mod,
            "megatron.training.initialize": init_mod,
            "primus.core.patches": SimpleNamespace(
                PatchContext=object, get_args=lambda ctx: ctx, register_patch=register
            ),
            "primus.core.utils.module_utils": SimpleNamespace(log_rank_0=lambda msg: None),
            "primus.core.utils.gpu": gpu,
            "primus.backends.megatron.patches.args.rocm_arg_validation": SimpleNamespace(
                validate_args_on_rocm=lambda args: calls.append("rocm")
            ),
        }
        source = Path(__file__).resolve().parents[3] / (
            "primus/backends/megatron/patches/args/validate_args_patches.py"
        )
        with patch.dict(sys.modules, modules):
            patches = runpy.run_path(str(source))
            patches["patch_validate_args"](None)
            for target, expected in (("cuda", ["upstream"]), ("amd", ["upstream", "rocm"])):
                calls.clear()
                args = SimpleNamespace(target_gpu=target, fp4=True)
                self.assertIs(args_mod.validate_args(args), args)
                self.assertEqual(calls, expected)
                condition = registrations["megatron.validate_args.fp4_te_version"]["condition"]
                self.assertEqual(condition(args), target == "amd")


if __name__ == "__main__":
    unittest.main()
