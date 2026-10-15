# Anima (Cosmos Predict 2) VACE ControlNet support.
#
# The VACE control branch is a standalone module trained by the sd-scripts
# trainer (networks/control_net_vace_anima.py). Its per-block hints are added
# to the outputs of the Anima DiT blocks through forward hooks registered in
# pre_run_advanced.

import torch
import torch.nn as nn

import comfy.model_management
import comfy.ops
import comfy.utils
from comfy.latent_formats import Wan21
from comfy.ldm.common_dit import pad_to_patch_size
from comfy.ldm.cosmos.predict2 import Block as AnimaBlock, PatchEmbed

from .control import ControlNetAdvanced
from .utils import broadcast_image_to_extend, extend_to_batch_size


# transformer_options key holding the current step's hints ({base_block_idx: tensor}).
# transformer_options is rebuilt for every model call, so hints never outlive a step.
ANIMA_VACE_HINTS_KEY = "anima_vace_hints"


class VACEControlBlock(nn.Module):
    """A copy of the Anima DiT Block plus zero-conv projections. Block 0 fuses
    the control tokens with the base model's patch-embedded input through
    before_proj; every block emits an after_proj hint for its base block.
    """

    def __init__(self, block: nn.Module, x_dim: int, block_id: int = 0, use_after_proj: bool = True):
        super().__init__()
        self.block = block
        self.block_id = block_id
        self.use_after_proj = use_after_proj

        if block_id == 0:
            self.before_proj = nn.Linear(x_dim, x_dim)
        if use_after_proj:
            self.after_proj = nn.Linear(x_dim, x_dim)

    def forward(self, c: torch.Tensor, x_base: torch.Tensor, **block_kwargs):
        """Returns (c_skip, c): the hint for the base block and the running
        stack of block outputs passed to the next control block."""
        all_c = []
        if self.block_id == 0:
            # the projections are plain Linears in the model dtype; keep the residual dtype around them
            c = self.before_proj(c.to(self.before_proj.weight.dtype)).to(x_base.dtype) + x_base
        elif self.use_after_proj:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        c = self.block(c, **block_kwargs)

        if self.use_after_proj:
            c_skip = self.after_proj(c.to(self.after_proj.weight.dtype)).to(c.dtype)
            all_c = all_c + [c_skip, c]
            c = torch.stack(all_c)
        else:
            c_skip = c
        return c_skip, c


class VACEControlNetAnima(nn.Module):
    """PatchEmbed for the control latent plus the VACE control blocks.
    forward returns the hints keyed by base block index."""

    def __init__(
        self,
        model_channels: int = 2048,
        num_heads: int = 16,
        num_blocks: int = 28,
        crossattn_emb_channels: int = 1024,
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        latent_channels: int = 16,
        vace_block_every_n: int = 7,
        condition_strategy: str = "spaced",
        use_after_proj: bool = True,
        use_adaln_lora: bool = True,
        adaln_lora_dim: int = 256,
        concat_padding_mask: bool = True,
        operations=None,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.concat_padding_mask = concat_padding_mask

        # base block indices that receive a hint, in control block order
        if condition_strategy == "spaced":
            self.control_layers = list(range(0, num_blocks, vace_block_every_n))
        else:  # "first_n"
            self.control_layers = list(range(max(1, num_blocks // vace_block_every_n)))

        # +1 for the padding-mask channel the base x_embedder also receives
        self.control_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=latent_channels + (1 if concat_padding_mask else 0),
            out_channels=model_channels,
            operations=operations,
        )

        self.control_blocks = nn.ModuleList([
            VACEControlBlock(
                block=AnimaBlock(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    operations=operations,
                ),
                x_dim=model_channels,
                block_id=i,
                use_after_proj=use_after_proj,
            )
            for i in range(len(self.control_layers))
        ])

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, control_latent, x_base_tokens, emb_B_T_D, crossattn_emb, rope_emb, adaln_lora,
                extra_per_block_pos_emb, transformer_options):
        # AnimaBlock takes its attention compute dtype from emb_B_T_D, while the manual_cast
        # ops follow their input dtype, so put every input in the control model's dtype to
        # keep q/k/v consistent. rope_emb stays in its own (fp32) dtype.
        cdtype = control_latent.dtype
        x_base_tokens = x_base_tokens.to(cdtype)
        emb_B_T_D = emb_B_T_D.to(cdtype)
        crossattn_emb = crossattn_emb.to(cdtype)
        if adaln_lora is not None:
            adaln_lora = adaln_lora.to(cdtype)
        if extra_per_block_pos_emb is not None:
            extra_per_block_pos_emb = extra_per_block_pos_emb.to(cdtype)

        if self.concat_padding_mask and control_latent.shape[1] == self.latent_channels:
            zeros_mask = torch.zeros(
                control_latent.shape[0], 1, *control_latent.shape[2:],
                device=control_latent.device, dtype=control_latent.dtype,
            )
            control_latent = torch.cat([control_latent, zeros_mask], dim=1)

        c = self.control_embedder(control_latent)

        # Same as the base _forward: keep the residual stream in fp32 for fp16 compute,
        # since its large values overflow in fp16.
        if c.dtype == torch.float16:
            c = c.float()
            x_base_tokens = x_base_tokens.float()

        block_kwargs = dict(
            emb_B_T_D=emb_B_T_D,
            crossattn_emb=crossattn_emb,
            rope_emb_L_1_1_D=rope_emb.unsqueeze(1).unsqueeze(0),  # same layout _forward passes to blocks
            adaln_lora_B_T_3D=adaln_lora,
            extra_per_block_pos_emb=extra_per_block_pos_emb,
            transformer_options=transformer_options,
        )

        hints = {}
        for base_idx, cb in zip(self.control_layers, self.control_blocks):
            c_skip, c = cb(c, x_base_tokens, **block_kwargs)
            hints[base_idx] = c_skip
        return hints


def load_anima_vace(ckpt_path: str, controlnet_data: dict, metadata=None, timestep_keyframe=None):
    # The sd-scripts trainer stores the architecture in the safetensors metadata;
    # fall back to Anima 2B defaults and the checkpoint keys when it is absent.
    meta = metadata or {}

    model_channels = int(meta.get("vace.model_channels", "2048"))
    num_blocks = int(meta.get("vace.num_blocks", "28"))
    latent_channels = int(meta.get("vace.in_channels", "16"))
    vace_block_every_n = int(meta.get("vace.block_every_n", "7"))
    condition_strategy = meta.get("vace.condition_strategy", "spaced")
    use_after_proj = meta.get("vace.use_after_proj", "true").lower() == "true"

    if "vace.model_channels" not in meta and "control_blocks.0.after_proj.weight" in controlnet_data:
        model_channels = controlnet_data["control_blocks.0.after_proj.weight"].shape[0]

    # only derive the spacing from the block count when the metadata does not store it
    n_ctrl = len({k.split(".")[1] for k in controlnet_data if k.startswith("control_blocks.")})
    if n_ctrl > 0 and condition_strategy == "spaced" and "vace.block_every_n" not in meta:
        vace_block_every_n = num_blocks // n_ctrl

    vace_model = VACEControlNetAnima(
        model_channels=model_channels,
        num_heads=16,
        num_blocks=num_blocks,
        crossattn_emb_channels=1024,
        patch_spatial=2,
        patch_temporal=1,
        latent_channels=latent_channels,
        vace_block_every_n=vace_block_every_n,
        condition_strategy=condition_strategy,
        use_after_proj=use_after_proj,
        use_adaln_lora=True,
        adaln_lora_dim=256,
        operations=comfy.ops.manual_cast,
    )

    # missing keys would leave uninitialized weights and produce noise
    missing, unexpected = vace_model.load_state_dict(controlnet_data, strict=False)
    if missing or unexpected:
        raise Exception(
            f"Anima VACE checkpoint '{ckpt_path}' does not match the resolved architecture "
            f"(control layers {vace_model.control_layers}). Missing keys ({len(missing)}): {missing[:5]}; "
            f"unexpected keys ({len(unexpected)}): {unexpected[:5]}."
        )

    return AnimaVACEAdvanced(
        vace_model=vace_model.to(comfy.model_management.unet_dtype()),
        timestep_keyframes=timestep_keyframe,
        load_device=comfy.model_management.get_torch_device(),
    )


class AnimaVACEAdvanced(ControlNetAdvanced):
    """get_control_advanced runs the VACE branch each step and publishes the hints
    on transformer_options under ANIMA_VACE_HINTS_KEY. Forward hooks on the target
    DiT blocks add them to the block outputs.
    """

    def __init__(self, vace_model, timestep_keyframes, load_device=None, manual_cast_dtype=None):
        # the control image is VAE-encoded, so it is upscaled to pixel resolution (8x the latent) first
        super().__init__(
            control_model=vace_model,
            timestep_keyframes=timestep_keyframes,
            compression_ratio=8,
            latent_format=Wan21(),
            load_device=load_device,
            manual_cast_dtype=manual_cast_dtype,
        )
        self.vace = vace_model
        self.require_vae = True
        self._hook_handles = []
        self._dit_ref = None

    def pre_run_advanced(self, model, percent_to_timestep_function):
        super().pre_run_advanced(model, percent_to_timestep_function)
        dit = model.diffusion_model
        required_attrs = ("blocks", "prepare_embedded_sequence", "t_embedder",
                          "t_embedding_norm", "patch_temporal", "patch_spatial")
        missing = [attr for attr in required_attrs if not hasattr(dit, attr)]
        if missing:
            raise Exception(
                f"{type(self).__name__} requires a Cosmos Predict2 / Anima DiT model, but "
                f"the target model's diffusion_model is missing: {missing}. "
                f"Is the ControlNet applied to an Anima (Cosmos Predict2) model?"
            )
        if len(dit.blocks) <= self.vace.control_layers[-1]:
            raise Exception(
                f"{type(self).__name__} expects at least {self.vace.control_layers[-1] + 1} DiT blocks, but "
                f"the target model only has {len(dit.blocks)}. Is the ControlNet applied to "
                f"an Anima (Cosmos Predict2) model?"
            )
        self._dit_ref = dit

        for base_idx in self.vace.control_layers:
            block = dit.blocks[base_idx]

            # ComfyUI skips control cleanup when sampling is interrupted, so a hook from
            # that run can still be attached; remove it or the hints get added twice.
            stale = getattr(block, "_acn_vace_hook", None)
            if stale is not None:
                stale.remove()

            def make_hook(bi):
                def hook(module, args, kwargs, output):
                    hint = kwargs.get("transformer_options", {}).get(ANIMA_VACE_HINTS_KEY, {}).get(bi)
                    if hint is None:
                        return output
                    return output + hint.to(dtype=output.dtype)
                return hook

            handle = block.register_forward_hook(make_hook(base_idx), with_kwargs=True)
            block._acn_vace_hook = handle
            self._hook_handles.append(handle)

    def get_control_advanced(self, x_noisy, t, cond, batched_number, transformer_options):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number, transformer_options)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                return control_prev

        dtype = self.control_model.dtype
        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        # x_noisy is (B, C, T, H, W)
        x_h, x_w = x_noisy.shape[-2], x_noisy.shape[-1]
        if self.sub_idxs is not None or self.cond_hint is None or \
           x_h * self.real_compression_ratio != self.cond_hint.shape[-2] or \
           x_w * self.real_compression_ratio != self.cond_hint.shape[-1]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            self.real_compression_ratio = self.compression_ratio

            self.cond_hint = comfy.utils.common_upscale(
                self.cond_hint_original,
                x_w * self.real_compression_ratio,
                x_h * self.real_compression_ratio,
                self.upscale_algorithm, "center")
            self.cond_hint = self.preprocess_image(self.cond_hint)

            if self.vae is not None:
                loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
                self.cond_hint = self.vae.encode(self.cond_hint.movedim(1, -1))
                comfy.model_management.load_models_gpu(loaded_models)
                self.real_compression_ratio = 1

            if self.latent_format is not None:
                self.cond_hint = self.latent_format.process_in(self.cond_hint)
            self.cond_hint = self.cond_hint.to(device=x_noisy.device, dtype=dtype)

        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to_extend(self.cond_hint, x_noisy.shape[0], batched_number)

        self.prepare_mask_cond_hint(x_noisy=x_noisy, t=t, cond=cond,
                                    batched_number=batched_number, dtype=dtype)

        # Build the control branch inputs with the DiT's own embedders, as _forward does.
        # get_control receives the raw x, while _apply_model passes calculate_input(t, x).
        dit = self._dit_ref
        x_noisy = self.model_sampling_current.calculate_input(t, x_noisy)
        x_5d = x_noisy if x_noisy.ndim == 5 else x_noisy.unsqueeze(2)

        # Pad both latents to the patch grid like _forward, so the control tokens line up
        # with the padded base tokens; _forward crops the padding from the final output.
        patch_size = (dit.patch_temporal, dit.patch_spatial, dit.patch_spatial)
        x_5d = pad_to_patch_size(x_5d, patch_size)
        control_latent = pad_to_patch_size(
            self.cond_hint.to(device=x_noisy.device, dtype=dtype), patch_size
        )

        x_B_T_H_W_D, rope_emb, extra_pos_emb = dit.prepare_embedded_sequence(x_5d)

        timestep = self.model_sampling_current.timestep(t)
        if timestep.ndim == 1:
            timestep = timestep.unsqueeze(1)
        t_sin = dit.t_embedder[0](timestep.to(x_B_T_H_W_D.device)).to(x_B_T_H_W_D.dtype)
        t_emb, adaln_lora = dit.t_embedder[1](t_sin)
        t_emb = dit.t_embedding_norm(t_emb)

        context = cond.get('crossattn_controlnet', cond['c_crossattn'])
        context = comfy.model_management.cast_to_device(context, x_B_T_H_W_D.device, dtype)

        hints = self.vace(
            control_latent=control_latent,
            x_base_tokens=x_B_T_H_W_D,
            emb_B_T_D=t_emb,
            crossattn_emb=context,
            rope_emb=rope_emb,
            adaln_lora=adaln_lora,
            extra_per_block_pos_emb=extra_pos_emb,
            transformer_options=transformer_options,
        )

        hint_mask = self._build_hint_mask(hints)
        # a stacked previous VACE control has already published its hints, so add to them
        published = transformer_options.setdefault(ANIMA_VACE_HINTS_KEY, {})
        for ci, hint in hints.items():
            if hint_mask is not None:
                hint = hint * hint_mask
            hint = self._apply_hint_strengths(hint, batched_number) * self.strength
            published[ci] = published[ci] + hint if ci in published else hint

        return control_prev

    def _apply_hint_strengths(self, hint: torch.Tensor, batched_number: int) -> torch.Tensor:
        """The hints never go through control_merge_inject, so apply the batch-wise parts of
        apply_advanced_strengths_and_masks here: latent keyframes, the uncond multiplier and
        the timestep keyframe strength. Per-layer weights are not supported."""
        if self.latent_keyframes is not None:
            hint = hint * self.calc_latent_keyframe_mults(x=hint, batched_number=batched_number)
        if self.weights is not None and self.weights.has_uncond_multiplier \
                and self.cond_or_uncond is not None:
            actual_length = hint.size(0) // batched_number
            for idx, cond_type in enumerate(self.cond_or_uncond):
                if cond_type == 1:
                    hint[actual_length*idx:actual_length*(idx+1)] *= self.weights.uncond_multiplier
        if self._current_timestep_keyframe is not None \
                and self._current_timestep_keyframe.strength != 1.0:
            hint = hint * self._current_timestep_keyframe.strength
        return hint

    def _build_hint_mask(self, hints: dict):
        """Combine the effect mask and timestep keyframe mask into one (B, 1, H_tok, W_tok, 1)
        token-space mask, or None when neither is set."""
        orig_masks = [m for m in (self.mask_cond_hint_original, self.tk_mask_cond_hint_original) if m is not None]
        if not orig_masks:
            return None
        sample = next(iter(hints.values()))
        H_tok, W_tok = sample.shape[-3], sample.shape[-2]
        B_hint = sample.shape[0]
        combined = None
        for om in orig_masks:
            mm = om.reshape(-1, 1, om.shape[-2], om.shape[-1]).to(torch.float32)
            mm = torch.nn.functional.interpolate(
                mm, size=(H_tok, W_tok), mode="bilinear", align_corners=False)
            mm = extend_to_batch_size(mm, B_hint).to(device=sample.device, dtype=sample.dtype)
            mm = mm.reshape(B_hint, 1, H_tok, W_tok, 1)
            combined = mm if combined is None else combined * mm
        return combined

    def cleanup_advanced(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []
        if self._dit_ref is not None:
            for base_idx in self.vace.control_layers:
                block = self._dit_ref.blocks[base_idx]
                if getattr(block, "_acn_vace_hook", None) is not None:
                    del block._acn_vace_hook
        self._dit_ref = None
        return super().cleanup_advanced()

    def copy(self, subtype=None):
        if subtype is None:
            subtype = AnimaVACEAdvanced
        c = subtype(
            vace_model=self.vace,
            timestep_keyframes=self.timestep_keyframes,
            load_device=self.load_device,
            manual_cast_dtype=self.manual_cast_dtype,
        )
        c.control_model_wrapped = self.control_model_wrapped
        self.copy_to(c)
        self.copy_to_advanced(c)
        return c
