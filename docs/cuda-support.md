# NVIDIA CUDA support

This branch ports the NVIDIA runtime changes from `haanjack/Primus` onto
`amd-agi/Primus` main at `db5fe1fdb`. Upstream framework pins, trainer APIs and
model implementations are preserved. GPU training and the CUDA image build
remain to be validated on NVIDIA hardware.

## Setup and smoke tests

Initialize the upstream pinned submodules before building:

```bash
git submodule update --init --recursive
docker build -t primus-cuda:latest -f docker/Dockerfile.cuda .
GPUS_PER_NODE=1 ./primus-cli --config runner/cuda.yaml container -- \
  train pretrain --config examples/megatron/configs/cuda/llama3.2_1B-BF16-pretrain.yaml
GPUS_PER_NODE=1 ./primus-cli --config runner/cuda.yaml container -- \
  train pretrain --config examples/torchtitan/configs/cuda/qwen3_0.6B-pretrain.yaml
```

The recipes run five BF16 iterations with sequence length 512. Hugging Face
tokenizer/model assets must be accessible. CUDA Transformer Engine is required
for the Megatron recipe. The image uses the NGC PyTorch base from the fork;
compatibility with the current upstream dependency pins needs a build test.
These are bring-up recipes, not performance configurations.

Use `target_gpu: cuda` at the experiment top level, or `auto` to detect the
PyTorch build. `nvidia` is accepted as an alias. Module-level `target_gpu` takes
precedence. CUDA disables AMD Primus-Turbo flags before backend argument
conversion and patch execution. ROCm-specific validation and CUDA-compilation
suppression only run for AMD. Native Megatron validation still runs on CUDA.

Container mode automatically selects CUDA defaults when `nvidia-smi -L`
succeeds and `/dev/kfd` is absent. AMD keeps the existing defaults. Set
`PRIMUS_GPU_PLATFORM=amd|cuda|auto` to select the host platform explicitly.
Default `rocm/primus:*` images become `primus-cuda:latest` on CUDA; custom images
are retained, `PRIMUS_CUDA_IMAGE` overrides the CUDA image, and explicit container
CLI options take precedence. `runner/cuda.yaml` is also available for explicit use.
The launcher uses
`--gpus all`, omits `/dev/kfd` and `/dev/dri`, and retains any explicitly supplied
InfiniBand devices. On the host, NVIDIA Container Toolkit is required. Set
`GPUS_PER_NODE` to the allocated GPU count. Existing `CUDA_VISIBLE_DEVICES` and
network interface selections are preserved. For direct mode, provide a CUDA
Python environment and use `./primus-cli direct -- train pretrain --config ...`.
Jobs use one GPU platform at a time; the same repository and experiment YAML can
run on either platform with `target_gpu: auto`. Platform-specific optimized
features are disabled for CUDA. This does not claim that a single distributed
job can mix AMD and NVIDIA ranks.

MaxText, FP8, checkpoint save/resume and multi-node training are outside the
initial smoke coverage. After BF16 passes, test each separately before claiming
support. Do not enable automatic requirements installation on CUDA: the shared
requirements file includes AMD packages; the Dockerfile filters them.

## GLM implementation decision

The fork's `glm4_7.yaml` and `glm5.yaml` use the same model architecture settings
as upstream. Current upstream additionally specifies vocabulary sizes 151552
and 154880 respectively, avoiding incorrect mock-data vocabulary sizes from
the placeholder GLM-4 tokenizer. Keep those upstream values. Real-data training
still needs the correct model tokenizer; the placeholder is for bring-up only.

Upstream also owns `glm5_next` and GLM-5.3-Flash (KDA + DSA + mHC), including HF
loading, context-parallel support and tests. This is a separate architecture,
not a replacement for the GLM-5 preset. The fork does not contain an equivalent
Primus implementation to preserve. Importing its older Megatron submodule pin
or API compatibility wrappers would undo upstream work, so neither is ported.

Recommendation: retain upstream GLM code and recipes unchanged. Evaluate CUDA
kernel portability of `glm5_next` separately after dense BF16 bring-up; no GLM
CUDA compatibility or numerical parity is claimed by this branch. A comparison
should then cover identical tokenizer/configuration, parameter count, checkpoint
load coverage, forward/backward loss parity, and only then performance.

### Fine-tuning path audit

`MegatronSFTTrainer.train()` selects the same `get_model_provider(model_type)`
as pretraining and optionally wraps it with LoRA. Thus the upstream model
provider has an SFT entry path, but this alone is not proof of usable GLM SFT.
For GLM-5.3, `megatron.glm5_next.hf_load` can load HF tensors online through
`glm5_hf_load_path` / `PRIMUS_GLM5_HF_LOAD` after model/optimizer creation. It
refreshes optimizer main parameters, which makes this the relevant upstream
route to assess for fine-tuning. There is no dedicated validated GLM SFT/LoRA
recipe in this tree. Adapter injection targets, loss masking, save/resume and
HF weight coverage still require an end-to-end test.

For ordinary GLM-4.7/GLM-5, the generic native checkpoint conversion hook only
recognizes Qwen3, DeepSeek and Llama. It falls back to Megatron-Bridge for other
families. The adapter mentions GLM-4 MoE bridge classes, but this does not
establish GLM-5 checkpoint compatibility or a ready GLM-5 fine-tuning recipe.
The fork's added posttraining recipe/converter is for Kimi-K2, not GLM; it
provides no GLM-specific fine-tuning implementation worth carrying forward.
Keep upstream and treat GLM-5 SFT checkpoint loading as a validation gap rather
than advertising pretrain support as complete fine-tuning support.

## CPU validation

```bash
python tests/unit_tests/core/test_gpu.py
bash tests/runner/helpers/test_cuda_env.sh
```

These cover target selection, preservation of AMD settings, disabling flat and
nested Turbo settings, retaining upstream argument validation on CUDA, and
preserving scheduler GPU masks and network settings. GPU acceptance should
include finite loss and backward/optimizer completion, then save/resume,
multi-GPU collectives, and an AMD regression run.

Local validation on Windows (without PyTorch/GPU): six isolated CPU regression
tests passed, all 13 existing environment tests passed, CUDA environment and
launcher-default checks passed, and the changed shell/Python files and three
YAML files passed syntax/parsing checks. Container dry runs confirmed the CUDA
image/defaults and explicit image/GPU overrides after normalizing the Windows
checkout's shell/YAML line endings to LF. No GPU training or image build ran.
