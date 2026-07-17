# Anima LLLite validation workflows

This folder contains simple Advanced-ControlNet workflows for the two Anima
LLLite v2 checkpoints: the five conditioning inputs documented for the v2
any-test-like model and the v2 inpainting model. It also includes vanilla parity
and effect-mask validation workflows.

![Simple workflows for all six v2 examples](https://ampcode.com/attachments/3ef84a8ad6947c624cd82b5ac6fb9c664b224678c774a5eba79f4f7e6e8d5c3f.jpeg)

![Control images and tested results for all six v2 examples](https://ampcode.com/attachments/addd1fdb85aa331cc622ef55a9397a996f34c5243e268a1b9bcc42ca39a25120.jpeg)

## Simple v2 workflows

| Type | Workflow | Input image | Checkpoint |
| --- | --- | --- | --- |
| Any - Grayscale A | [`anima_lllite_any_grayscale_a.json`](anima_lllite_any_grayscale_a.json) | [`anima_lllite_any_grayscale_a_control.png`](https://ampcode.com/attachments/911b31c414bcf5c9aeaa254356c89f04e1133ce2181d0414cd8b7e9a7fa7847c.png) | `anima-lllite-any-test-like-v2.safetensors` |
| Any - Grayscale B | [`anima_lllite_any_grayscale_b.json`](anima_lllite_any_grayscale_b.json) | [`anima_lllite_any_grayscale_b_control.png`](https://ampcode.com/attachments/6950b3c8a47ff62311cc8ac147e2c8aa6aad9176abe7046db3a1bd6cf97cf705.png) | `anima-lllite-any-test-like-v2.safetensors` |
| Any - Lineart | [`anima_lllite_any_lineart.json`](anima_lllite_any_lineart.json) | [`anima_lllite_any_lineart_control.png`](https://ampcode.com/attachments/338c2f07267d564ca9fb60bf79c229b5689509202734875f0cb674efb8d691b1.png) | `anima-lllite-any-test-like-v2.safetensors` |
| Any - HED scribble | [`anima_lllite_any_hed_scribble.json`](anima_lllite_any_hed_scribble.json) | [`anima_lllite_any_hed_scribble_control.png`](https://ampcode.com/attachments/a52217f09c5852652667152cb901b98c3472c2e4cbf86a6cae53a36ec7e84192.png) | `anima-lllite-any-test-like-v2.safetensors` |
| Any - PiDiNet scribble | [`anima_lllite_any_pidinet_scribble.json`](anima_lllite_any_pidinet_scribble.json) | [`anima_lllite_any_pidinet_scribble_control.png`](https://ampcode.com/attachments/57bc6b0d8cba0df17f92115917bfc6ccfce83107e56af624f0be1db32c402579.png) | `anima-lllite-any-test-like-v2.safetensors` |
| Inpainting | [`anima_lllite_inpainting.json`](anima_lllite_inpainting.json) | [`anima_lllite_v2_control.png`](https://ampcode.com/attachments/3220b2f533a014619e3c0422eca8a3343e8488eb113795af5bae5f05d38a8ada.png) | `anima-lllite-inpainting-v2.safetensors` |

Download the selected input image from the table and save it under the displayed
filename in `ComfyUI/input`. Load its workflow and queue it unchanged. Results
are saved under `ComfyUI/output/acn_anima_examples`. Binary inputs and
screenshots are linked externally instead of being committed to this repository.

These examples use default node names and do not connect the custom 28-layer
Anima weights node. The inpainting workflow uses Anima LLLite Extras and
Default Weights only because the model's source mask must be carried through
`cn_extras`. Strength, start/end scheduling, effect masks, timestep keyframes,
latent keyframes, and stacking remain available on the standard
Advanced-ControlNet nodes.

The model author trained the v2 any-test-like checkpoint on five conditioning
types: HED scribble, PiDiNet scribble, Grayscale A, Grayscale B, and lineart,
all with heavy augmentation. The examples above exercise each input type using
that one v2 checkpoint instead of the lower-quality Preview3 depth, pose,
lineart, and scribble checkpoints.

The HED and PiDiNet controls were prepared with the
[comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux)
HED and Scribble PiDiNet preprocessors. The HED soft-edge output was thresholded
at 80 to make the documented white-on-black scribble representation. The model
card names two grayscale generation patterns but does not define their
individual construction; Grayscale A is the author's sample and Grayscale B
demonstrates the documented inversion, contrast, and blur augmentation.

## Requirements

Official Hugging Face repositories:

- [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) - base model, text encoder, and VAE
- [kohya-ss/Anima-LLLite](https://huggingface.co/kohya-ss/Anima-LLLite) - Anima LLLite control models

Use ComfyUI commit `0f42ba514631` or later and place these files in the listed
model folders. These are direct downloads from the official model repositories:

| Download | ComfyUI model folder |
| --- | --- |
| [`anima-base-v1.0.safetensors`](https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-base-v1.0.safetensors?download=true) | `models/diffusion_models` |
| [`qwen_3_06b_base.safetensors`](https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors?download=true) | `models/text_encoders` |
| [`qwen_image_vae.safetensors`](https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors?download=true) | `models/vae` |
| [`anima-lllite-inpainting-v2.safetensors`](https://huggingface.co/kohya-ss/Anima-LLLite/resolve/main/anima-lllite-inpainting-v2.safetensors?download=true) | `models/model_patches` |
| [`anima-lllite-any-test-like-v2.safetensors`](https://huggingface.co/kohya-ss/Anima-LLLite/resolve/main/anima-lllite-any-test-like-v2.safetensors?download=true) (3-channel any-type model) | `models/model_patches` |

The included workflows use **Load Anima LLLite Model**, so their LLLite files
belong in `models/model_patches`, matching vanilla ComfyUI. For compatibility
with existing Advanced-ControlNet LLLite workflows, the files may instead be
placed in `models/controlnet` and loaded with **Load Advanced ControlNet
Model**. The standard loader automatically distinguishes Anima checkpoints
from older SDXL LLLite checkpoints.

## Run the inpainting comparison

This validation workflow runs the inpainting model through vanilla ComfyUI and
Advanced-ControlNet with identical inputs and sampling settings. It saves both
decoded results, both latent tensors, and an absolute pixel-difference image.

![Workflow showing the vanilla and Advanced-ControlNet branches](https://ampcode.com/attachments/f056cae89132c45bc133f456a2832a81255fee9c0782968a24adce2095b2fe1b.jpeg)

![Bit-exact result comparison](https://ampcode.com/attachments/fcf073d1253ea721a2d46d34efde0e45ff6ccfb751fbd295a438ab459cc26037.jpeg)

1. Download [`anima_lllite_v2_control.png`](https://ampcode.com/attachments/3220b2f533a014619e3c0422eca8a3343e8488eb113795af5bae5f05d38a8ada.png)
   to `ComfyUI/input`.
2. Load `anima_lllite_v2_inpaint_comparison.json` in ComfyUI.
3. Queue the workflow without changing its settings.
4. Inspect `ComfyUI/output/acn_anima_pr`.

The control PNG contains a transparent edit region. ComfyUI's Load Image node
provides that alpha channel as the source inpainting mask.

The vanilla branch passes the mask directly to Apply Anima LLLite. The
Advanced-ControlNet branch passes it through Anima LLLite Extras, the
`cn_extras` input on Default Weights, and `weights_override` on Apply Advanced
ControlNet.

If an inpainting checkpoint reaches sampling without that source mask,
Advanced-ControlNet raises an error that describes these connections instead
of silently substituting an empty mask.

With the included seed and settings, the expected results are:

- Identical latent tensors with maximum and mean absolute differences of `0.0`.
- Identical decoded PNG pixels.
- A completely black `absolute_difference` image.

## Run the any-type effect-mask comparison

This workflow verifies the official 3-channel any-type checkpoint against
vanilla ComfyUI and applies an Advanced-ControlNet effect mask to only the left
half of the image. Its nodes retain their default names; the colored regions
identify the comparison branches.

![Any-type effect-mask workflow](https://ampcode.com/attachments/3a547b4914a84f0d578b0223149d3ed14e9b22542093e4bc51b6501aeb18f29f.jpeg)

![Any-type parity and effect-mask results](https://ampcode.com/attachments/340c286b28d74d943f03a95d71208f2408c94934f0809fbb060cc774ae8c1b67.jpeg)

1. Download [`anima_lllite_v2_any_control.png`](https://ampcode.com/attachments/db18d41c476324bdf5a5b6117476ed117cc590f8f48e193f6316d08b959bc49c.png)
   and [`anima_lllite_v2_left_half_mask.png`](https://ampcode.com/attachments/e4fe3a88415357f29315c9afb124ee975a48dee8abccad1fe1903dd5d21b91b6.png)
   to `ComfyUI/input`.
2. Load `anima_lllite_v2_any_effect_mask.json` in ComfyUI.
3. Queue the workflow without changing its settings.
4. Inspect `ComfyUI/output/acn_anima_any_mask`.

The expected results are:

- Advanced-ControlNet full control and vanilla full control have identical
  latent tensors and decoded pixels, with maximum absolute difference `0.0`.
- A completely black effect mask produces the no-control latent exactly.
- A completely white effect mask produces the unmasked full-control latent
  exactly.
- At the model token resolution, the included half mask is exactly `1.0` on
  the left and `0.0` on the right. Direct LLLite injection is therefore zero
  for every right-side token.
- In the final image, the masked right side is closer to the no-control
  baseline than full control: PSNR improves from `16.39` to `17.69`, and SSIM
  improves from `0.554` to `0.600`.

The final right half is not pixel-identical to the no-control image. This model
patches self-attention Q, so controlled left-side tokens can influence
right-side tokens through global self-attention and later diffusion steps.
The effect mask guarantees local control injection, not hard image-space
isolation after attention.
