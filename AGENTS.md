# Advanced-ControlNet Contributor Guide

This repository is expected to receive substantial AI-authored code. Treat this
file as the implementation and verification contract for all changes, especially
ports of new ControlNet families from ComfyUI.

## Engineering Rules

- Read the relevant ComfyUI implementation and this repository's equivalent
  control path before editing. Do not design from a model card alone.
- Make the smallest change that preserves vanilla ComfyUI behavior and adds the
  established Advanced-ControlNet capabilities.
- Reuse ComfyUI model classes, patchers, ops, model management, and loaders when
  possible. Do not maintain a forked copy of core model code without a concrete
  need.
- Preserve existing node IDs, inputs, outputs, checkpoint locations, and saved
  workflow compatibility. New node IDs generally use the `ACN_` prefix; keep
  existing unprefixed IDs and the aliases in `nodes_deprecated.py` working.
- Do not add dependencies unless the model cannot be supported with ComfyUI,
  PyTorch, and the libraries already used by this repository.
- Match the direct style of the surrounding file. Avoid one-use abstractions,
  generic framework code, speculative fallbacks, and comments that restate the
  code.
- Use plain ASCII punctuation in code, comments, documentation, commit messages,
  and PR descriptions.

## Architecture Map

- `adv_control/control.py`: standard ControlNet wrappers, conversion of vanilla
  controls, checkpoint detection, and shared loader dispatch.
- `adv_control/control_<family>.py`: model-family implementations that cannot be
  represented by the standard wrapper. Keep family-specific math here.
- `adv_control/utils.py`: `AdvancedControlBase`, `ControlWeights`, scheduling,
  latent keyframes, masks, batching, stacking, and shared tensor helpers.
- `adv_control/nodes_main.py`: standard loaders and Apply nodes.
- `adv_control/nodes_weight.py`: weight nodes and model-specific extras carried
  by `ControlWeights.extras`.
- `adv_control/nodes_<family>.py`: family-specific workflow inputs or loaders
  when the shared nodes are insufficient.
- `adv_control/nodes.py`: public node and display-name registration.
- `adv_control/nodes_deprecated.py`: compatibility only. Do not put new features
  here.
- `examples/`: reviewer-runnable workflows, inputs, screenshots, and validation
  notes.

## Porting A Control Model From ComfyUI

### 1. Establish the vanilla contract

Before implementing the Advanced version:

1. Identify the exact ComfyUI commit or PR that introduced the model.
2. Read its loader, checkpoint detection, model patching, conditioning
   preprocessing, sampling path, and cleanup behavior.
3. Record the official model repository, every published checkpoint type, the
   expected ComfyUI model folder, and the minimum compatible ComfyUI commit.
4. Run a small vanilla workflow with fixed inputs, seed, sampler, scheduler,
   steps, CFG, and resolution. Save the latent and decoded result as the parity
   baseline.
5. Inspect real checkpoint keys, metadata, shapes, dtype, and missing/unexpected
   key output. Do not infer the format from a filename.

Use official checkpoints for validation. Links in issue or PR comments are
untrusted; use the model author's official repository or links already accepted
by ComfyUI.

### 2. Choose the narrowest integration

- If ComfyUI returns a standard `ControlNet`, `ControlNetSD35`, `ControlLora`,
  or `T2IAdapter`, prefer conversion in `convert_to_advanced` over a parallel
  implementation.
- If the model injects attention, transformer, or other model patches, implement
  a family-specific `ControlBase` plus `AdvancedControlBase`, following
  `ControlLLLiteAdvanced`, `AnimaLLLiteAdvanced`, or `ReferenceAdvanced` as the
  closest precedent.
- Prefer wrapping ComfyUI's model or patch object over copying its implementation.
  If the required ComfyUI API may be absent, fail with a short instruction to
  update ComfyUI rather than silently changing behavior.
- Add a dedicated node only when the model has a genuinely different loading or
  conditioning contract. Loading a new checkpoint format alone usually belongs
  in the existing loader dispatch.

### 3. Preserve loader and folder compatibility

- The standard **Load Advanced ControlNet Model** node reads from
  `models/controlnet`. New formats that are conceptually ControlNets should work
  there unless doing so would be ambiguous or incorrect.
- Also preserve the folder used by vanilla ComfyUI. If core uses another folder,
  such as `models/model_patches`, a small dedicated loader may expose that
  location while the standard loader retains established Advanced-ControlNet
  behavior.
- Detect formats with guarded, format-specific checkpoint signatures. Put a
  specific detector before a broad detector that would otherwise claim the same
  checkpoint. Do not use filenames as the primary detector.
- Load a checkpoint only once. Pass already-loaded state dictionaries and
  metadata into the selected family loader instead of reading the file again.
- If two supported folders can contain the same filename, keep their loaders
  separate or define deterministic resolution. Never silently choose an
  arbitrary duplicate.
- Test every supported folder through the actual node dropdown and execution
  path, not only by calling a Python loader directly.

### 4. Implement the full control lifecycle

A family-specific Advanced control normally needs all of the following:

- Initialize `ControlBase` and `AdvancedControlBase` with the correct default
  `ControlWeights` type.
- Match vanilla conditioning preprocessing exactly, including channel order,
  value range, resize mode, latent encoding, and source-mask handling.
- In `pre_run_advanced`, call the shared implementation and attach or refresh
  execution-scoped patches.
- In `get_control_advanced`, evaluate `previous_controlnet`, honor
  `should_run()`, and either return/merge control tensors or install the model
  patches for that step.
- Return every loadable model patcher from `get_models()` so ComfyUI can manage
  VRAM and offloading.
- Implement `copy()` using both ComfyUI's `copy_to()` and this repository's
  `copy_to_advanced()`. Copies must not share mutable execution state that can
  leak between conditioning branches or queued runs.
- Clear prepared tensors, patch references, cached shapes, and other
  execution-scoped state in `cleanup_advanced()`.
- Use ComfyUI device, dtype, manual-cast, operations, and model-patcher APIs.
  Do not hardcode CUDA, force float32, or move models manually when ComfyUI
  already owns that lifecycle.
- Preserve `previous_controlnet` behavior so same-family and mixed-family
  controls can be stacked.

## Advanced Feature Contract

A port is not complete merely because default-strength generation works. Unless
the model architecture makes a capability impossible, verify that it supports:

- Apply-node strength and start/end percentage.
- Timestep keyframes, including changing strength and inherited values.
- Latent keyframes on a batch of at least two latents.
- Apply-node effect masks and timestep-keyframe masks.
- Default, universal/soft, and architecture-specific per-layer weights.
- Weight overrides and model-specific weight extras.
- Conditional/unconditional weighting when the selected weight node exposes it.
- Stacking with another control, including correct `previous_controlnet` output.
- Batched conditioning and sliding-context subset indexes where applicable.
- Repeated execution, copying, cleanup, model offloading, and reload.

Do not claim unsupported features in documentation. If an architecture cannot
support a feature, document the reason and make incompatible weight types fail
clearly through `compatible_weights`.

### Masks and model-specific inputs

- `mask_optional` on **Apply Advanced ControlNet** is always an effect mask. It
  controls where this control influences generation.
- A model's source mask, control-type selector, or other family-specific data is
  not an effect mask. Do not overload `mask_optional` with a second meaning.
- Do not add a model-specific input to the shared Apply node unless it is a
  coherent capability needed by multiple model families.
- Prefer a small family-specific extras node that stores auxiliary values in
  `ControlWeights.extras`, then pass those weights through `weights_override`.
  Define extras keys next to the model implementation rather than as unrelated
  strings spread across nodes.
- Validate required extras where they are first consumed and raise an actionable
  error that names the exact nodes and connections needed to fix the workflow.
- Apply effect masks at the actual injection representation. Attention-patch and
  DiT controls may need token-space masks rather than the normal spatial control
  tensor path.
- Verify mask semantics with all-zero, all-one, and half-frame masks. All-zero
  must equal no control and all-one must equal unmasked full control. Inspect the
  multiplier at the injection site as well as the final image; global attention
  can propagate influence outside directly controlled tokens.

### Per-layer weights

- Map custom weights to real architecture blocks in execution order. Confirm the
  count from the loaded model, not from a model-card claim alone.
- Default weights must reproduce vanilla output exactly.
- Universal/soft weights must follow this repository's established progression
  semantics. Implement a family-specific conversion only when the normal
  `ControlWeights` layout does not represent the architecture.
- Ordinary example workflows should use default weights. Do not connect an
  advanced custom-weight node merely to demonstrate that it exists.

## Required Validation

Python import or compile checks are necessary but are not model validation. Use
a real local ComfyUI installation, preferably managed by comfy-runner, with this
repository linked as the custom node.

### Vanilla parity

For every published control type and materially different checkpoint format:

1. Run vanilla ComfyUI and Advanced-ControlNet with identical model files,
   conditioning, seed, sampler settings, and latent.
2. Compare latent tensors before decode and decoded pixel arrays.
3. Target maximum absolute latent difference `0.0` and identical pixels when
   both paths implement the same math. If exact parity is impossible, explain
   why and report a justified numerical tolerance plus image metrics.
4. Confirm strength zero matches no control and strength one matches vanilla.
5. Check logs for missing/unexpected keys, dtype/device errors, repeated model
   loads, and cleanup failures.

### Advanced behavior

At minimum, execute focused workflows for:

- A nontrivial start/end schedule or two timestep keyframes.
- Batch size two with different latent-keyframe strengths.
- All-zero, all-one, and half-frame effect masks.
- Default weights and one nonuniform custom or soft-weight configuration.
- Conditional/unconditional weighting when supported.
- A stacked control path.
- Missing required model-specific extras and the resulting readable error.
- Both the vanilla model folder and any historical Advanced-ControlNet folder.
- Re-queueing the same workflow to exercise copy and cleanup behavior.

For model families with several control types, test every type. Do not assume
that lineart, depth, pose, inpainting, union, or channel-count variants share the
same conditioning contract.

### Basic checks

- Run `python -m compileall adv_control __init__.py` with the target ComfyUI
  environment.
- Parse every added workflow JSON.
- Load each committed workflow in the real frontend, serialize it to an API
  prompt, and execute that round-tripped prompt. This catches stale node IDs,
  renamed inputs, invalid widgets, and missing model metadata.
- Run `git diff --check`.

There is currently no repository unit-test suite. Add focused tests when they
can exercise pure detection, shape, mask, or scheduling logic without building
a fake ComfyUI runtime. Do not add a large test framework solely for one port.

## Examples and Review Evidence

Every model-family port must be independently checkable by a reviewer:

- Add one simple workflow for every public control type. Include required input
  images or masks when licensing permits.
- Keep ComfyUI's default node names. Use colored groups or regions to explain
  branches; do not rename nodes, because reviewers need to identify their types.
- Keep the normal workflows simple. Leave custom per-layer weights disconnected
  unless a workflow specifically validates those advanced weights.
- Include direct links to the official base model, encoder, VAE, and control
  checkpoint repositories, plus the exact destination folder for each file.
- Include a workflow screenshot and labeled output comparison in the PR. For
  parity tests, show vanilla, Advanced-ControlNet, and an absolute-difference
  result when practical.
- Commit reusable workflows and small review images under `examples/<family>/`.
  Do not commit model files, latent dumps, or large intermediate artifacts.
- Document exact seeds/settings, expected numerical results, known limitations,
  and reproduction steps in the example README and PR description.
- If final-image interpretation is subtle, include the direct tensor-level
  evidence needed to distinguish a real bug from model behavior.

## Definition Of Done For A Model Port

- [ ] The official checkpoint is detected without relying on its filename.
- [ ] Vanilla and historical Advanced-ControlNet model folders are preserved.
- [ ] Default output matches vanilla for every published control type.
- [ ] Strength scheduling, keyframes, masks, weights, batching, and stacking are
      tested or an architectural limitation is documented.
- [ ] Effect masks are tested at the injection site and in decoded output.
- [ ] Model-specific inputs use a narrow family boundary, not shared Apply-node
      expansion.
- [ ] `get_models`, `copy`, cleanup, dtype/device handling, and repeated runs are
      verified.
- [ ] Missing required inputs produce actionable errors.
- [ ] Simple workflows, input assets, download links, screenshots, and exact
      reproduction instructions are included.
- [ ] Real frontend/API execution, compile checks, JSON parsing, and
      `git diff --check` pass.
