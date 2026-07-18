# Qwen Image ControlNet inpainting

This reviewer example adapts the active inpainting branch of ComfyUI's official
Qwen Image workflow. It uses **Load Advanced ControlNet Model** and **Apply
Advanced ControlNet Inpainting**, while retaining the official Qwen base
pipeline and the bypassed optional Lightning LoRA. Node titles are left at
their ComfyUI defaults; the workflow stores no node title overrides.

## Inputs and models

Download the official inputs to `ComfyUI/input` with these exact names:

- [`acn_qwen_inpaint_source.png`](https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting/resolve/main/assets/images/image1.png)
- [`acn_qwen_inpaint_mask.png`](https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting/resolve/main/assets/masks/mask1.png)

The model author's repository is
[`InstantX/Qwen-Image-ControlNet-Inpainting`](https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting).
Download every model below to the listed folder under `ComfyUI/models`:

| File and exact download | Folder |
| --- | --- |
| [`qwen_image_fp8_e4m3fn.safetensors`](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors) | `diffusion_models` |
| [`qwen_2.5_vl_7b_fp8_scaled.safetensors`](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors) | `text_encoders` |
| [`qwen_image_vae.safetensors`](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors) | `vae` |
| [`Qwen-Image-InstantX-ControlNet-Inpainting.safetensors`](https://huggingface.co/Comfy-Org/Qwen-Image-InstantX-ControlNets/resolve/main/split_files/controlnet/Qwen-Image-InstantX-ControlNet-Inpainting.safetensors) | `controlnet` |
| [`Qwen-Image-Lightning-4steps-V1.0.safetensors`](https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0.safetensors) | `loras` (optional and bypassed) |

## Run

1. Download the two inputs and five model files to the folders above.
2. Load `qwen_image_inpainting.json` in ComfyUI.
3. Queue the workflow unchanged.

For command-line input reproduction:

```sh
curl -L https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting/resolve/main/assets/images/image1.png -o ComfyUI/input/acn_qwen_inpaint_source.png
curl -L https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting/resolve/main/assets/masks/mask1.png -o ComfyUI/input/acn_qwen_inpaint_mask.png
```

The unchanged example uses seed `134554158057228` (fixed), 20 steps, CFG 2.5,
Euler, the simple scheduler, denoise 1.0, model shift 3.1, control strength 1.0,
and control start/end 0.0/1.0. Its prompt is `The Queen, on a throne,
surrounded by Knights, HD, Realistic, Octane Render, Unreal engine`; the
negative prompt is one space. The source is scaled with area interpolation to a
maximum dimension of 1536. The optional 4-step LoRA remains bypassed; enabling
it requires changing the sampler settings appropriately.

The two native **Load Image** nodes are intentionally separate. **Image To
Mask** reads the red channel of the mask PNG. That source `inpaint_mask` defines
the region supplied to the inpainting ControlNet and the latent noise mask. It
is not the Advanced-ControlNet effect mask. `effect_mask_optional` is left
unconnected and independently limits where control is injected. The Apply node
also exposes unconnected timestep keyframe, latent keyframe, and weights ports
for focused reviewer experiments.

## Measured validation evidence

These results were measured with fixed inputs and settings; they are recorded
here rather than inferred from the example image:

- A fresh isolated vanilla-versus-Advanced run had latent maximum/mean absolute
  differences `0/0`, pixel maximum/mean differences `0/0`, and 0 changed
  pixels.
- An all-one effect mask exactly equaled unmasked Advanced output at latent and
  pixel level. An all-zero effect mask exactly equaled no ControlNet at latent
  and pixel level.
- For right-half token-mask injection, relative to full control the left latent
  mean delta was `0` and the right was `0.1332103`; relative to no control the
  left was `0` and the right was `0.1835042`.
- In a per-latent batch, the sample with strength 0 exactly equaled no control
  at latent and pixel level.
- Soft weights, timestep scheduling, and two-control stacking each executed
  successfully.
- The existing Anima real workflow rerun retained exact before/after latent and
  pixel equality.

Frontend and API validation for a missing source mask names `inpaint_mask`.
Supplying an incompatible model produces this exact error:
`The provided ControlNet does not use an inpaint source mask; use Apply Advanced
ControlNet instead.`

Workflow and result screenshots are linked from the PR instead of stored here
to avoid repository growth.
