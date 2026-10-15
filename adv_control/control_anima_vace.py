# Anima (Cosmos Predict 2) VACE ControlNet support for Advanced-ControlNet.
#
# Ports the spaced-block-duplication + zero-conv control architecture (VACE,
# Alibaba ICCV 2025; as adopted by NVIDIA Cosmos Transfer 2.5) into ACN. The
# trained weights are produced by the companion sd-scripts trainer
# (networks/control_net_vace_anima.py).
#
# The VACE control branch is a standalone nn.Module. Its per-block hint
# outputs are injected into the Anima DiT's residual stream via forward hooks
# registered at runtime by pre_run_advanced — no ComfyUI core changes required.
# Mask, timestep-keyframes, and CFG-aware strength are inherited from
# AdvancedControlBase.

import torch
import torch.nn as nn

import comfy
import comfy.model_management
import comfy.utils
from comfy.latent_formats import Wan21
from comfy.ldm.common_dit import pad_to_patch_size
from comfy.ldm.cosmos.predict2 import Block as AnimaBlock, PatchEmbed

from .control import ControlNetAdvanced
from .logger import logger
from .utils import (
    AdvancedControlBase,
    ControlWeights,
    ControlWeightType,
    TimestepKeyframeGroup,
    broadcast_image_to_extend,
    extend_to_batch_size,
)


# transformer_options key under which get_control_advanced publishes the
# per-step VACE hints ({base_block_idx: tensor}). Stored on the per-run
# transformer_options dict (rebuilt every step by ComfyUI) so hints cannot
# leak across runs — see AnimaVACEAdvanced.pre_run_advanced.
ANIMA_VACE_HINTS_KEY = "anima_vace_hints"


# --------------------------------------------------------------------------- #
# VACE control block — an Anima DiT Block wrapped with zero-conv projections
# --------------------------------------------------------------------------- #

class VACEControlBlock(nn.Module):
    """One block of the VACE control branch: a copy of the Anima DiT Block
    (same architecture, so base weights copy directly) plus zero-conv
    projections. Block 0 additionally fuses the control latent with the base
    model's patch-embedded input via ``before_proj``.

    Implements the stacked-tensor skip protocol from the VACE paper: each
    block carries a running stack of prior outputs, runs the inner Block on
    only the top element, and emits a zero-conv ``after_proj`` hint for
    injection into the corresponding base DiT block.
    """

    def __init__(self, block: nn.Module, x_dim: int, block_id: int = 0, use_after_proj: bool = True):
        super().__init__()
        self.block = block
        self.block_id = block_id
        self.x_dim = x_dim
        self.use_after_proj = use_after_proj

        # Zero-conv projections (zero-initialised so hints start at exactly 0).
        if block_id == 0:
            self.before_proj = nn.Linear(x_dim, x_dim)
        if use_after_proj:
            self.after_proj = nn.Linear(x_dim, x_dim)

    def init_weights(self):
        if hasattr(self.block, "init_weights"):
            self.block.init_weights()
        if self.block_id == 0:
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        if self.use_after_proj:
            nn.init.zeros_(self.after_proj.weight)
            nn.init.zeros_(self.after_proj.bias)

    def forward(self, c: torch.Tensor, x_base: torch.Tensor, **block_kwargs):
        """Run one control block.

        c: working control tensor (B, T, H, W, D).
        x_base: base DiT patch-embedded input (block 0 only).
        Returns (c_skip, c): the injection hint and the evolved working tensor.
        """
        all_c = []
        if self.block_id == 0:
            # before_proj is a model-dtype (fp16) Linear; run it in that dtype
            # and keep the fused residual in x_base's (fp32) dtype.
            c = self.before_proj(c.to(self.before_proj.weight.dtype)).to(x_base.dtype) + x_base
        elif self.use_after_proj:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        c = self.block(c, **block_kwargs)

        if self.use_after_proj:
            # after_proj is fp16; produce the hint in the residual dtype so the
            # stacked tensor stays uniform (the hook casts to the base block
            # dtype anyway).
            c_skip = self.after_proj(c.to(self.after_proj.weight.dtype)).to(c.dtype)
            all_c = all_c + [c_skip, c]
            c = torch.stack(all_c)
        else:
            c_skip = c
        return c_skip, c


# --------------------------------------------------------------------------- #
# VACE control branch — the standalone control model
# --------------------------------------------------------------------------- #

class VACEControlNetAnima(nn.Module):
    """Standalone VACE control branch: a PatchEmbed for the control latent
    plus a ModuleList of VACEControlBlock. ``forward`` returns per-base-block
    hint tensors keyed by base block index."""

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
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.latent_channels = latent_channels
        self.vace_block_every_n = vace_block_every_n
        self.condition_strategy = condition_strategy
        self.concat_padding_mask = concat_padding_mask

        # Spaced layer mapping: one control block every N base blocks, starting
        # at block 0 (VACE / Cosmos Transfer 2.5 distributed-injection scheme).
        if condition_strategy == "spaced":
            self.control_layers = list(range(0, num_blocks, vace_block_every_n))
            self.control_layers_mapping = {i: n for n, i in enumerate(self.control_layers)}
        else:  # "first_n"
            n_ctrl = max(1, num_blocks // vace_block_every_n)
            self.control_layers = list(range(0, n_ctrl))
            self.control_layers_mapping = {i: i for i in range(n_ctrl)}
        assert 0 in self.control_layers, "first control block must map to base block 0"

        # Anima concatenates a padding-mask channel to the latent
        # (latent_channels + 1), matching what the base DiT's x_embedder sees.
        if operations is None:
            operations = comfy.ops.manual_cast

        embed_in_channels = latent_channels + (1 if concat_padding_mask else 0)
        self.control_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=embed_in_channels,
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

    def init_weights(self):
        # comfy.ldm.cosmos.predict2.PatchEmbed has no init_weights method; only
        # the zero-conv projections on each control block need explicit init.
        for cb in self.control_blocks:
            cb.init_weights()

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(
        self,
        control_latent: torch.Tensor,   # (B, C, T, H, W) — VAE-encoded control image
        x_base_tokens: torch.Tensor,    # (B, T, H, W, D) — base DiT patch-embedded input
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb=None,
        adaln_lora=None,
        extra_per_block_pos_emb=None,
        transformer_options=None,
    ):
        """Run the control branch. Returns {base_block_idx: hint_tensor}."""
        if transformer_options is None:
            transformer_options = {}

        # Pin every input to one compute dtype. AnimaBlock derives its
        # attention compute_dtype from emb_B_T_D, and comfy's manual_cast ops
        # emit in the *input* dtype — so q.dtype follows emb while k/v.dtype
        # follow crossattn_emb. The DiT hands these over in whatever dtype it
        # runs in (emb can arrive fp32 from the timestep embedder, crossattn_emb
        # bf16), which would mismatch inside attention. Casting all inputs to
        # control_latent.dtype keeps q/k/v consistent.
        cdtype = control_latent.dtype
        x_base_tokens = x_base_tokens.to(cdtype)
        emb_B_T_D = emb_B_T_D.to(cdtype)
        crossattn_emb = crossattn_emb.to(cdtype)
        if adaln_lora is not None:
            adaln_lora = adaln_lora.to(cdtype)
        if extra_per_block_pos_emb is not None:
            extra_per_block_pos_emb = extra_per_block_pos_emb.to(cdtype)
        # rope_emb is left in its arrival dtype: self-attention applies it to
        # q/k that are already consistent, and RoPE precision benefits from fp32.

        if self.concat_padding_mask and control_latent.shape[1] == self.latent_channels:
            # Append the padding-mask channel the base x_embedder expects
            # (zeros = valid everywhere for a fully-unpadded control image).
            zeros_mask = torch.zeros(
                control_latent.shape[0], 1, *control_latent.shape[2:],
                device=control_latent.device, dtype=control_latent.dtype,
            )
            control_latent = torch.cat([control_latent, zeros_mask], dim=1)

        c = self.control_embedder(control_latent)

        # Mirror the base _forward: keep the residual stream in fp32. The
        # AnimaBlock residual has large values and fp16 accumulation over the
        # control blocks can overflow; attention/MLP still run in fp16 via the
        # Block's own compute-dtype handling.
        if c.dtype == torch.float16:
            c = c.float()
            x_base_tokens = x_base_tokens.float()

        # AnimaBlock expects rope in the (1, L, 1, 1, 1, D) layout the base
        # DiT's _forward produces. prepare_embedded_sequence returns the raw
        # (L, 1, 1, D) form; replicate the two unsqueezes _forward applies.
        rope_for_blocks = rope_emb
        if rope_emb is not None and rope_emb.ndim == 4:
            rope_for_blocks = rope_emb.unsqueeze(1).unsqueeze(0)

        block_kwargs = dict(
            emb_B_T_D=emb_B_T_D,
            crossattn_emb=crossattn_emb,
            rope_emb_L_1_1_D=rope_for_blocks,
            adaln_lora_B_T_3D=adaln_lora,
            extra_per_block_pos_emb=extra_per_block_pos_emb,
            transformer_options=transformer_options,
        )

        hints = {}
        for i, cb in enumerate(self.control_blocks):
            c_skip, c = cb(c, x_base_tokens, **block_kwargs)
            hints[self.control_layers[i]] = c_skip
        return hints


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #

def load_anima_vace(ckpt_path: str, controlnet_data: dict = None, timestep_keyframe=None, model=None):
    """Load an Anima VACE ControlNet checkpoint and return an AnimaVACEAdvanced."""
    if controlnet_data is None:
        controlnet_data = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

    # Architecture is read from safetensors metadata (written by the sd-scripts
    # trainer); fall back to sensible Anima-2B defaults and weight-shape inference.
    from safetensors import safe_open
    meta = {}
    try:
        with safe_open(ckpt_path, framework="pt") as f:
            meta = dict(f.metadata() or {})
    except Exception:
        pass

    model_channels = int(meta.get("vace.model_channels", "2048"))
    num_blocks = int(meta.get("vace.num_blocks", "28"))
    latent_channels = int(meta.get("vace.in_channels", "16"))
    vace_block_every_n = int(meta.get("vace.block_every_n", "7"))
    condition_strategy = meta.get("vace.condition_strategy", "spaced")
    use_after_proj = meta.get("vace.use_after_proj", "true").lower() == "true"

    if "vace.model_channels" not in meta:
        for k, v in controlnet_data.items():
            if "control_blocks.0.after_proj.weight" in k:
                model_channels = v.shape[0]
                break

    # Count the control blocks present in the checkpoint. Only fall back to
    # re-deriving the spacing from this count when the metadata is absent —
    # forcing `num_blocks // n_ctrl` over a stored spacing can silently shift
    # the injection layers (e.g. 28 blocks with control every 8 -> [0,8,16,24]
    # would be rewritten as [0,7,14,21]).
    max_ctrl_idx = -1
    for k in controlnet_data:
        if k.startswith("control_blocks."):
            max_ctrl_idx = max(max_ctrl_idx, int(k.split(".")[1]))
    n_ctrl = max_ctrl_idx + 1 if max_ctrl_idx >= 0 else 0
    if n_ctrl > 0 and condition_strategy == "spaced" and "vace.block_every_n" not in meta:
        vace_block_every_n = num_blocks // n_ctrl

    # Anima 2B constants.
    num_heads = 16
    crossattn_emb_channels = 1024
    use_adaln_lora = True
    adaln_lora_dim = 256

    load_device = comfy.model_management.get_torch_device()
    unet_dtype = comfy.model_management.unet_dtype()

    vace_model = VACEControlNetAnima(
        model_channels=model_channels,
        num_heads=num_heads,
        num_blocks=num_blocks,
        crossattn_emb_channels=crossattn_emb_channels,
        patch_spatial=2,
        patch_temporal=1,
        latent_channels=latent_channels,
        vace_block_every_n=vace_block_every_n,
        condition_strategy=condition_strategy,
        use_after_proj=use_after_proj,
        use_adaln_lora=use_adaln_lora,
        adaln_lora_dim=adaln_lora_dim,
        operations=comfy.ops.manual_cast,
    )

    if n_ctrl > 0 and len(vace_model.control_layers) != n_ctrl:
        logger.warning(
            f"[AnimaVACE] checkpoint has {n_ctrl} control blocks but the resolved "
            f"architecture expects {len(vace_model.control_layers)} control layers; "
            f"injection may be misaligned."
        )

    info = vace_model.load_state_dict(controlnet_data, strict=False)
    if info.missing_keys:
        logger.info(f"[AnimaVACE] missing keys ({len(info.missing_keys)}): {info.missing_keys[:5]}...")
    if info.unexpected_keys:
        logger.info(f"[AnimaVACE] unexpected keys ({len(info.unexpected_keys)}): {info.unexpected_keys[:5]}...")

    vace_model = vace_model.to(unet_dtype)

    return AnimaVACEAdvanced(
        vace_model=vace_model,
        timestep_keyframes=timestep_keyframe,
        load_device=load_device,
    )


# --------------------------------------------------------------------------- #
# ACN wrapper — runtime hooks + per-run hint publishing
# --------------------------------------------------------------------------- #

class AnimaVACEAdvanced(ControlNetAdvanced):
    """ACN wrapper for the Anima VACE ControlNet. Inherits mask, timestep
    keyframes, and CFG-aware strength from AdvancedControlBase.

    Injection model: get_control_advanced runs the VACE branch each step and
    publishes the per-block hints onto the per-run transformer_options dict
    (key ANIMA_VACE_HINTS_KEY). Forward hooks registered on the target DiT
    blocks read from that dict and add the hint to the block output. Because
    the hints live on transformer_options (rebuilt every step) rather than a
    persistent attribute, a hook left attached by an interrupted/failed run
    can never inject stale control into a later run.
    """

    def __init__(self, vace_model, timestep_keyframes, load_device=None, manual_cast_dtype=None):
        # compression_ratio=8 upscales the control image to pixel resolution
        # before VAE encoding; real_compression_ratio is then set to 1 so the
        # encoded latent matches x_noisy's spatial dims.
        super().__init__(
            control_model=vace_model,
            timestep_keyframes=timestep_keyframes,
            compression_ratio=8,
            latent_format=Wan21(),
            load_device=load_device,
            manual_cast_dtype=manual_cast_dtype,
        )
        # VACE takes pixel-space control images (e.g. depth maps) that must be
        # VAE-encoded first; require_vae makes the Apply node enforce a VAE.
        self.vace = vace_model
        self.vace_model = vace_model
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
        max_base_idx = max(self.vace.control_layers_mapping.keys())
        if len(dit.blocks) <= max_base_idx:
            raise Exception(
                f"{type(self).__name__} expects at least {max_base_idx + 1} DiT blocks, but "
                f"the target model only has {len(dit.blocks)}. Is the ControlNet applied to "
                f"an Anima (Cosmos Predict2) model?"
            )
        self._dit_ref = dit

        for base_idx, _ctrl_idx in self.vace.control_layers_mapping.items():
            block = dit.blocks[base_idx]

            # Remove a stale hook left by a previous run. cleanup() is not
            # guaranteed to run when sampling is interrupted or crashes, so the
            # previous run's hook can stay attached to the base DiT block and
            # re-apply the control on subsequent runs (the image then degrades
            # into noise as the hooks accumulate across runs).
            stale = getattr(block, "_acn_vace_hook", None)
            if stale is not None and getattr(stale, "id", None) is not None:
                stale.remove()

            def make_hook(bi):
                def hook(module, args, kwargs, output):
                    to = kwargs.get("transformer_options") or {}
                    hints = to.get(ANIMA_VACE_HINTS_KEY)
                    if not hints:
                        return output
                    hint = hints.get(bi)
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

        # VAE-encode the control image to the latent grid. x_noisy is 5D
        # (B, C, T, H, W); use the trailing two dims for spatial H/W.
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

        # Run the VACE branch using the DiT's own pre-block machinery so the
        # control tokens see the same patch-embedding / timestep / rope the
        # base block 0 does.
        dit = self._dit_ref
        if dit is not None:
            # Match the base DiT's input: _apply_model feeds it
            # `calculate_input(t, x)` (for Cosmos RFLOW, x * (1 - σ/(σ+1))).
            # get_control hands us the *unscaled* noise; scaling it here keeps
            # the control branch's block-0 fusion consistent with the base model.
            x_noisy = self.model_sampling_current.calculate_input(t, x_noisy)
            x_5d = x_noisy if x_noisy.ndim == 5 else x_noisy.unsqueeze(2)

            # Pad base + control latent to the DiT patch grid the same way
            # _forward does (circular, (T, S, S)). ComfyUI hands controlnets the
            # *unpadded* latent; without this, latents whose spatial dims aren't
            # multiples of patch_spatial (e.g. 1440x1080 -> 135x180 latent, 135
            # odd) hit the PatchEmbed divisibility assert inside both
            # prepare_embedded_sequence and control_embedder. Both must be padded
            # identically so the control token grid matches the padded base
            # block outputs the forward hooks fire on (block 0 adds them
            # element-wise). No crop is needed on the hints: _forward already
            # crops the final unpatchified output back to the original latent
            # shape, discarding the padded-edge contribution.
            patch_size = (dit.patch_temporal, dit.patch_spatial, dit.patch_spatial)
            x_5d = pad_to_patch_size(x_5d, patch_size)
            control_latent = pad_to_patch_size(
                self.cond_hint.to(device=x_noisy.device, dtype=dtype), patch_size
            )

            x_B_T_H_W_D, rope_emb, extra_pos_emb = dit.prepare_embedded_sequence(x_5d)

            timestep = self.model_sampling_current.timestep(t)
            if timestep.ndim == 1:
                timestep = timestep.unsqueeze(1)
            # Mirror the base _forward: cast the sinusoidal embedding to the
            # model dtype before the MLP (dit.t_embedder[0]=Timesteps,
            # [1]=TimestepEmbedding), avoiding an fp32/bf16 mismatch.
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
            # Publish onto the per-run transformer_options (read by the hooks).
            # A stacked previous VACE control has already published its hints
            # above, so add to them per block instead of replacing them.
            # mask follows ACN convention: 1 = apply control, 0 = suppress.
            published = transformer_options.setdefault(ANIMA_VACE_HINTS_KEY, {})
            for ci, hint in hints.items():
                if hint_mask is not None:
                    hint = hint * hint_mask
                hint = self._apply_hint_strengths(hint, batched_number) * self.strength
                published[ci] = published[ci] + hint if ci in published else hint

        return control_prev

    def _apply_hint_strengths(self, hint: torch.Tensor, batched_number: int) -> torch.Tensor:
        """Apply the ACN strengths that control_merge_inject normally applies.

        The hook-based VACE path never builds self.control, so
        apply_advanced_strengths_and_masks is bypassed. Apply its batch-wise
        pieces here: latent keyframes, the CFG uncond multiplier, and the
        current timestep keyframe strength. Base strength is applied by the
        caller; per-index soft weights are not supported for VACE hints.
        """
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
        """Build a single (B, 1, H_tok, W_tok, 1) mask gating all hints.

        ACN normally applies region/timestep-keyframe masks inside
        control_merge_inject on self.control; this hook-based path never
        builds self.control, so we apply them directly. Hints are 5D
        (B, T, H, W, D), so the mask is reshaped to broadcast over T and D.
        Returns None when no mask is connected (full control everywhere).
        """
        orig_masks = [
            m for m in (
                self.mask_cond_hint_original,
                getattr(self, "tk_mask_cond_hint_original", None),
            ) if m is not None
        ]
        if not orig_masks or not hints:
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
            for base_idx in self.vace.control_layers_mapping:
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
        c.control_model = self.control_model
        c.control_model_wrapped = self.control_model_wrapped
        self.copy_to(c)
        self.copy_to_advanced(c)
        return c

    @staticmethod
    def from_vanilla(v, timestep_keyframe=None, subtype=None):
        # VACE checkpoints are never loaded as vanilla ComfyUI objects;
        # provided for API completeness.
        if subtype is None:
            subtype = AnimaVACEAdvanced
        to_return = subtype(
            vace_model=v.control_model,
            timestep_keyframes=timestep_keyframe,
            load_device=v.load_device,
            manual_cast_dtype=v.manual_cast_dtype,
        )
        v.copy_to(to_return)
        to_return.control_model_wrapped = v.control_model_wrapped.clone()
        return to_return
