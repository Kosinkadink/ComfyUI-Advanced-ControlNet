from typing import Callable, Union

import math
import torch
from torch import Tensor

import comfy.sample
import comfy.model_patcher
import comfy.utils
from comfy.controlnet import ControlBase
from comfy.model_patcher import ModelPatcher
from comfy.ldm.modules.attention import BasicTransformerBlock
from comfy.ldm.modules.diffusionmodules import openaimodel

from .logger import logger
from .utils import (AdvancedControlBase, ControlWeights, TimestepKeyframeGroup, AbstractPreprocWrapper,
                    broadcast_image_to_extend)


REF_ATTN_CONTROL_LIST = "ref_attn_control_list"
REF_ADAIN_CONTROL_LIST = "ref_adain_control_list"
REF_CONTROL_LIST_ALL = "ref_control_list_all"
REF_CONTROL_INFO = "ref_control_info"
REF_ATTN_MACHINE_STATE = "ref_attn_machine_state"
REF_ADAIN_MACHINE_STATE = "ref_adain_machine_state"
REF_COND_IDXS = "ref_cond_idxs"
REF_UNCOND_IDXS = "ref_uncond_idxs"


class MachineState:
    WRITE = "write"
    READ = "read"
    STYLEALIGN = "stylealign"
    OFF = "off"


class ReferenceType:
    ATTN = "reference_attn"
    ADAIN = "reference_adain"
    ATTN_ADAIN = "reference_attn+adain"
    STYLE_ALIGN = "StyleAlign"

    _LIST = [ATTN, ADAIN, ATTN_ADAIN]
    _LIST_ATTN = [ATTN, ATTN_ADAIN]
    _LIST_ADAIN = [ADAIN, ATTN_ADAIN]

    @classmethod
    def is_attn(cls, ref_type: str):
        return ref_type in cls._LIST_ATTN
    
    @classmethod
    def is_adain(cls, ref_type: str):
        return ref_type in cls._LIST_ADAIN


class ReferenceOptions:
    def __init__(self, reference_type: str,
                 attn_style_fidelity: float, adain_style_fidelity: float,
                 attn_ref_weight: float, adain_ref_weight: float,
                 attn_strength: float=1.0, adain_strength: float=1.0,
                 ref_with_other_cns: bool=False):
        self.reference_type = reference_type
        # attn
        self.original_attn_style_fidelity = attn_style_fidelity
        self.attn_style_fidelity = attn_style_fidelity
        self.attn_ref_weight = attn_ref_weight
        self.attn_strength = attn_strength
        # adain
        self.original_adain_style_fidelity = adain_style_fidelity
        self.adain_style_fidelity = adain_style_fidelity
        self.adain_ref_weight = adain_ref_weight
        self.adain_strength = adain_strength
        # other
        self.ref_with_other_cns = ref_with_other_cns
    
    def clone(self):
        return ReferenceOptions(reference_type=self.reference_type,
                                attn_style_fidelity=self.original_attn_style_fidelity, adain_style_fidelity=self.original_adain_style_fidelity,
                                attn_ref_weight=self.attn_ref_weight, adain_ref_weight=self.adain_ref_weight,
                                attn_strength=self.attn_strength, adain_strength=self.adain_strength,
                                ref_with_other_cns=self.ref_with_other_cns)

    @staticmethod
    def create_combo(reference_type: str, style_fidelity: float, ref_weight: float, ref_with_other_cns: bool=False):
        return ReferenceOptions(reference_type=reference_type,
                                attn_style_fidelity=style_fidelity, adain_style_fidelity=style_fidelity,
                                attn_ref_weight=ref_weight, adain_ref_weight=ref_weight,
                                ref_with_other_cns=ref_with_other_cns)



class ReferencePreprocWrapper(AbstractPreprocWrapper):
    error_msg = error_msg = "Invalid use of Reference Preprocess output. The output of Reference preprocessor is NOT a usual image, but a latent pretending to be an image - you must connect the output directly to an Apply Advanced ControlNet node. It cannot be used for anything else that accepts IMAGE input."
    def __init__(self, condhint: Tensor):
        super().__init__(condhint)


class ReferenceAdvanced(ControlBase, AdvancedControlBase):
    CHANNEL_TO_MULT = {320: 1, 640: 2, 1280: 4}

    def __init__(self, ref_opts: ReferenceOptions, timestep_keyframes: TimestepKeyframeGroup, device=None):
        super().__init__(device)
        AdvancedControlBase.__init__(self, super(), timestep_keyframes=timestep_keyframes, weights_default=ControlWeights.controllllite(), allow_condhint_latents=True)
        # TODO: allow vae_optional to be used instead of preprocessor
        #require_vae=True
        self.ref_opts = ref_opts
        self.order = 0
        self.model_latent_format = None
        self.model_sampling_current = None
        self.should_apply_attn_effective_strength = False
        self.should_apply_adain_effective_strength = False
        self.should_apply_effective_masks = False
        self.latent_shape = None

    def any_attn_strength_to_apply(self):
        return self.should_apply_attn_effective_strength or self.should_apply_effective_masks
    
    def any_adain_strength_to_apply(self):
        return self.should_apply_adain_effective_strength or self.should_apply_effective_masks

    def get_effective_strength(self):
        effective_strength = self.strength
        if self._current_timestep_keyframe is not None:
            effective_strength = effective_strength * self._current_timestep_keyframe.strength
        return effective_strength

    def get_effective_attn_mask_or_float(self, x: Tensor, channels: int, is_mid: bool):
        if not self.should_apply_effective_masks:
            return self.get_effective_strength() * self.ref_opts.attn_strength
        if is_mid:
            div = 8
        else:
            div = self.CHANNEL_TO_MULT[channels]
        real_mask = torch.ones([self.latent_shape[0], 1, self.latent_shape[2]//div, self.latent_shape[3]//div]).to(dtype=x.dtype, device=x.device) * self.strength * self.ref_opts.attn_strength
        self.apply_advanced_strengths_and_masks(x=real_mask, batched_number=self.batched_number)
        # mask is now shape [b, 1, h ,w]; need to turn into [b, h*w, 1]
        b, c, h, w = real_mask.shape
        real_mask = real_mask.permute(0, 2, 3, 1).reshape(b, h*w, c)
        return real_mask

    def get_effective_adain_mask_or_float(self, x: Tensor):
        if not self.should_apply_effective_masks:
            return self.get_effective_strength() * self.ref_opts.adain_strength
        b, c, h, w = x.shape
        real_mask = torch.ones([b, 1, h, w]).to(dtype=x.dtype, device=x.device) * self.strength * self.ref_opts.adain_strength
        self.apply_advanced_strengths_and_masks(x=real_mask, batched_number=self.batched_number)
        return real_mask

    def should_run(self):
        running = super().should_run()
        if not running:
            return running
        attn_run = False
        adain_run = False
        if ReferenceType.is_attn(self.ref_opts.reference_type):
            # attn will run as long as neither weight or strength is zero
            attn_run = not (math.isclose(self.ref_opts.attn_ref_weight, 0.0) or math.isclose(self.ref_opts.attn_strength, 0.0))
        if ReferenceType.is_adain(self.ref_opts.reference_type):
            # adain will run as long as neither weight or strength is zero
            adain_run = not (math.isclose(self.ref_opts.adain_ref_weight, 0.0) or math.isclose(self.ref_opts.adain_strength, 0.0))
        return attn_run or adain_run

    def pre_run_advanced(self, model, percent_to_timestep_function):
        AdvancedControlBase.pre_run_advanced(self, model, percent_to_timestep_function)
        if isinstance(self.cond_hint_original, AbstractPreprocWrapper):
            self.cond_hint_original = self.cond_hint_original.condhint
        self.model_latent_format = model.latent_format # LatentFormat object, used to process_in latent cond_hint
        self.model_sampling_current = model.model_sampling
        # SDXL is more sensitive to style_fidelity according to sd-webui-controlnet comments
        if type(model).__name__ == "SDXL":
            self.ref_opts.attn_style_fidelity = self.ref_opts.original_attn_style_fidelity ** 3.0
            self.ref_opts.adain_style_fidelity = self.ref_opts.original_adain_style_fidelity ** 3.0
        else:
            self.ref_opts.attn_style_fidelity = self.ref_opts.original_attn_style_fidelity
            self.ref_opts.adain_style_fidelity = self.ref_opts.original_adain_style_fidelity

    def get_control_advanced(self, x_noisy: Tensor, t, cond, batched_number: int):
        # normal ControlNet stuff
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                return control_prev

        dtype = x_noisy.dtype
        # prepare cond_hint - it is a latent, NOT an image
        #if self.sub_idxs is not None or self.cond_hint is None or x_noisy.shape[2] != self.cond_hint.shape[2] or x_noisy.shape[3] != self.cond_hint.shape[3]:
        if self.cond_hint is not None:
            del self.cond_hint
        self.cond_hint = None
        # if self.cond_hint_original length greater or equal to real latent count, subdivide it before scaling
        if self.sub_idxs is not None and self.cond_hint_original.size(0) >= self.full_latent_length:
            self.cond_hint = comfy.utils.common_upscale(
                self.cond_hint_original[self.sub_idxs],
                x_noisy.shape[3], x_noisy.shape[2], 'nearest-exact', "center").to(dtype).to(self.device)
        else:
            self.cond_hint = comfy.utils.common_upscale(
                self.cond_hint_original,
                x_noisy.shape[3], x_noisy.shape[2], 'nearest-exact', "center").to(dtype).to(self.device)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to_extend(self.cond_hint, x_noisy.shape[0], batched_number, except_one=False)
        # noise cond_hint based on sigma (current step)
        self.cond_hint = self.model_latent_format.process_in(self.cond_hint)
        self.cond_hint = ref_noise_latents(self.cond_hint, sigma=t, noise=None)
        timestep = self.model_sampling_current.timestep(t)
        self.should_apply_attn_effective_strength = not (math.isclose(self.strength, 1.0) and math.isclose(self._current_timestep_keyframe.strength, 1.0) and math.isclose(self.ref_opts.attn_strength, 1.0))
        self.should_apply_adain_effective_strength = not (math.isclose(self.strength, 1.0) and math.isclose(self._current_timestep_keyframe.strength, 1.0) and math.isclose(self.ref_opts.adain_strength, 1.0))
        # prepare mask - use direct_attn, so the mask dims will match source latents (and be smaller)
        self.prepare_mask_cond_hint(x_noisy=x_noisy, t=t, cond=cond, batched_number=batched_number, direct_attn=True)
        self.should_apply_effective_masks = self.latent_keyframes is not None or self.mask_cond_hint is not None or self.tk_mask_cond_hint is not None
        self.latent_shape = list(x_noisy.shape)
        # done preparing; model patches will take care of everything now.
        # return normal controlnet stuff
        return control_prev

    def cleanup_advanced(self):
        super().cleanup_advanced()
        del self.model_latent_format
        self.model_latent_format = None
        del self.model_sampling_current
        self.model_sampling_current = None
        self.should_apply_attn_effective_strength = False
        self.should_apply_adain_effective_strength = False
        self.should_apply_effective_masks = False
    
    def copy(self):
        c = ReferenceAdvanced(self.ref_opts, self.timestep_keyframes)
        c.order = self.order
        self.copy_to(c)
        self.copy_to_advanced(c)
        return c

    # avoid deepcopy shenanigans by making deepcopy not do anything to the reference
    # TODO: do the bookkeeping to do this in a proper way for all Adv-ControlNets
    def __deepcopy__(self, memo):
        return self


def ref_noise_latents(latents: Tensor, sigma: Tensor, noise: Tensor=None):
    sigma = sigma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    alpha_cumprod = 1 / ((sigma * sigma) + 1)
    sqrt_alpha_prod = alpha_cumprod ** 0.5
    sqrt_one_minus_alpha_prod = (1. - alpha_cumprod) ** 0.5
    if noise is None:
        # generator = torch.Generator(device="cuda")
        # generator.manual_seed(0)
        # noise = torch.empty_like(latents).normal_(generator=generator)
        # generator = torch.Generator()
        # generator.manual_seed(0)
        # noise = torch.randn(latents.size(), generator=generator).to(latents.device)
        noise = torch.randn_like(latents).to(latents.device)
    return sqrt_alpha_prod * latents + sqrt_one_minus_alpha_prod * noise


def simple_noise_latents(latents: Tensor, sigma: float, noise: Tensor=None):
    if noise is None:
        noise = torch.rand_like(latents)
    return latents + noise * sigma


class BankStylesBasicTransformerBlock:
    def __init__(self):
        self.bank = []
        self.style_cfgs = []
        self.cn_idx: list[int] = []
    
    def get_avg_style_fidelity(self):
        return sum(self.style_cfgs) / float(len(self.style_cfgs))
    
    def clean(self):
        del self.bank
        self.bank = []
        del self.style_cfgs
        self.style_cfgs = []
        del self.cn_idx
        self.cn_idx = []


class BankStylesTimestepEmbedSequential:
    def __init__(self):
        self.var_bank = []
        self.mean_bank = []
        self.style_cfgs = []
        self.cn_idx: list[int] = []

    def get_avg_var_bank(self):
        return sum(self.var_bank) / float(len(self.var_bank))

    def get_avg_mean_bank(self):
        return sum(self.mean_bank) / float(len(self.mean_bank))
    
    def get_avg_style_fidelity(self):
        return sum(self.style_cfgs) / float(len(self.style_cfgs))

    def clean(self):
        del self.mean_bank
        self.mean_bank = []
        del self.var_bank
        self.var_bank = []
        del self.style_cfgs
        self.style_cfgs = []
        del self.cn_idx
        self.cn_idx = []


class InjectionBasicTransformerBlockHolder:
    def __init__(self, block: BasicTransformerBlock, idx=None):
        if hasattr(block, "_forward"): # backward compatibility
            self.original_forward = block._forward
        else:
            self.original_forward = block.forward
        self.idx = idx
        self.attn_weight = 1.0
        self.is_middle = False
        self.bank_styles = BankStylesBasicTransformerBlock()
    
    def restore(self, block: BasicTransformerBlock):
        if hasattr(block, "_forward"): # backward compatibility
            block._forward = self.original_forward
        else:
            block.forward = self.original_forward

    def clean(self):
        self.bank_styles.clean()


class InjectionTimestepEmbedSequentialHolder:
    def __init__(self, block: openaimodel.TimestepEmbedSequential, idx=None, is_middle=False, is_input=False, is_output=False):
        self.original_forward = block.forward
        self.idx = idx
        self.gn_weight = 1.0
        self.is_middle = is_middle
        self.is_input = is_input
        self.is_output = is_output
        self.bank_styles = BankStylesTimestepEmbedSequential()
    
    def restore(self, block: openaimodel.TimestepEmbedSequential):
        block.forward = self.original_forward
    
    def clean(self):
        self.bank_styles.clean()


class ReferenceInjections:
    def __init__(self, attn_modules: list['RefBasicTransformerBlock']=None, gn_modules: list['RefTimestepEmbedSequential']=None):
        self.attn_modules = attn_modules if attn_modules else []
        self.gn_modules = gn_modules if gn_modules else []
        self.diffusion_model_orig_forward: Callable = None
    
    def clean_module_mem(self):
        for attn_module in self.attn_modules:
            try:
                attn_module.injection_holder.clean()
            except Exception:
                pass
        for gn_module in self.gn_modules:
            try:
                gn_module.injection_holder.clean()
            except Exception:
                pass

    def cleanup(self):
        self.clean_module_mem()
        del self.attn_modules
        self.attn_modules = []
        del self.gn_modules
        self.gn_modules = []
        self.diffusion_model_orig_forward = None


def factory_forward_inject_UNetModel(reference_injections: ReferenceInjections):
    def forward_inject_UNetModel(self, x: Tensor, *args, **kwargs):
        # get control and transformer_options from kwargs
        real_args = list(args)
        real_kwargs = list(kwargs.keys())
        control = kwargs.get("control", None)
        transformer_options = kwargs.get("transformer_options", None)
        # look for ReferenceAttnPatch objects to get ReferenceAdvanced objects
        ref_controlnets: list[ReferenceAdvanced] = transformer_options[REF_CONTROL_LIST_ALL]
        # discard any controlnets that should not run
        ref_controlnets = [x for x in ref_controlnets if x.should_run()]
        # if nothing related to reference controlnets, do nothing special
        if len(ref_controlnets) == 0:
            return reference_injections.diffusion_model_orig_forward(x, *args, **kwargs)
        try:
            # assign cond and uncond idxs
            batched_number = len(transformer_options["cond_or_uncond"])
            per_batch = x.shape[0] // batched_number
            indiv_conds = []
            for cond_type in transformer_options["cond_or_uncond"]:
                indiv_conds.extend([cond_type] * per_batch)
            transformer_options[REF_UNCOND_IDXS] = [i for i, x in enumerate(indiv_conds) if x == 1]
            transformer_options[REF_COND_IDXS] = [i for i, x in enumerate(indiv_conds) if x == 0]
            # check which controlnets do which thing
            attn_controlnets = []
            adain_controlnets = []
            for control in ref_controlnets:
                if ReferenceType.is_attn(control.ref_opts.reference_type):
                    attn_controlnets.append(control)
                if ReferenceType.is_adain(control.ref_opts.reference_type):
                    adain_controlnets.append(control)
            if len(adain_controlnets) > 0:
                # ComfyUI uses forward_timestep_embed with the TimestepEmbedSequential passed into it
                orig_forward_timestep_embed = openaimodel.forward_timestep_embed
                openaimodel.forward_timestep_embed = forward_timestep_embed_ref_inject_factory(orig_forward_timestep_embed)
            # handle running diffusion with ref cond hints
            for control in ref_controlnets:
                if ReferenceType.is_attn(control.ref_opts.reference_type):
                    transformer_options[REF_ATTN_MACHINE_STATE] = MachineState.WRITE
                else:
                    transformer_options[REF_ATTN_MACHINE_STATE] = MachineState.OFF
                if ReferenceType.is_adain(control.ref_opts.reference_type):
                    transformer_options[REF_ADAIN_MACHINE_STATE] = MachineState.WRITE
                else:
                    transformer_options[REF_ADAIN_MACHINE_STATE] = MachineState.OFF
                transformer_options[REF_ATTN_CONTROL_LIST] = [control]
                transformer_options[REF_ADAIN_CONTROL_LIST] = [control]

                orig_kwargs = kwargs
                if not control.ref_opts.ref_with_other_cns:
                    kwargs = kwargs.copy()
                    kwargs["control"] = None
                reference_injections.diffusion_model_orig_forward(control.cond_hint.to(dtype=x.dtype).to(device=x.device), *args, **kwargs)
                kwargs = orig_kwargs
            # run diffusion for real now
            transformer_options[REF_ATTN_MACHINE_STATE] = MachineState.READ
            transformer_options[REF_ADAIN_MACHINE_STATE] = MachineState.READ
            transformer_options[REF_ATTN_CONTROL_LIST] = attn_controlnets
            transformer_options[REF_ADAIN_CONTROL_LIST] = adain_controlnets
            return reference_injections.diffusion_model_orig_forward(x, *args, **kwargs)
        finally:
            # make sure banks are cleared no matter what happens - otherwise, RIP VRAM
            reference_injections.clean_module_mem()
            if len(adain_controlnets) > 0:
                openaimodel.forward_timestep_embed = orig_forward_timestep_embed

    return forward_inject_UNetModel


# dummy class just to help IDE keep track of injected variables
class RefBasicTransformerBlock(BasicTransformerBlock):
    injection_holder: InjectionBasicTransformerBlockHolder = None

def _forward_inject_BasicTransformerBlock(self: RefBasicTransformerBlock, x: Tensor, context: Tensor=None, transformer_options: dict[str]={}):
    extra_options = {}
    block = transformer_options.get("block", None)
    block_index = transformer_options.get("block_index", 0)
    transformer_patches = {}
    transformer_patches_replace = {}

    for k in transformer_options:
        if k == "patches":
            transformer_patches = transformer_options[k]
        elif k == "patches_replace":
            transformer_patches_replace = transformer_options[k]
        else:
            extra_options[k] = transformer_options[k]

    extra_options["n_heads"] = self.n_heads
    extra_options["dim_head"] = self.d_head

    if self.ff_in:
        x_skip = x
        x = self.ff_in(self.norm_in(x))
        if self.is_res:
            x += x_skip

    n: Tensor = self.norm1(x)
    if self.disable_self_attn:
        context_attn1 = context
    else:
        context_attn1 = None
    value_attn1 = None

    # Reference CN stuff
    uc_idx_mask = transformer_options.get(REF_UNCOND_IDXS, [])
    c_idx_mask = transformer_options.get(REF_COND_IDXS, [])
    # WRITE mode will only have one ReferenceAdvanced, other modes will have all ReferenceAdvanced
    ref_controlnets: list[ReferenceAdvanced] = transformer_options.get(REF_ATTN_CONTROL_LIST, None)
    ref_machine_state: str = transformer_options.get(REF_ATTN_MACHINE_STATE, None)
    # if in WRITE mode, save n and style_fidelity
    if ref_controlnets and ref_machine_state == MachineState.WRITE:
        if ref_controlnets[0].ref_opts.attn_ref_weight > self.injection_holder.attn_weight:
            self.injection_holder.bank_styles.bank.append(n.detach().clone())
            self.injection_holder.bank_styles.style_cfgs.append(ref_controlnets[0].ref_opts.attn_style_fidelity)
            self.injection_holder.bank_styles.cn_idx.append(ref_controlnets[0].order)

    if "attn1_patch" in transformer_patches:
        patch = transformer_patches["attn1_patch"]
        if context_attn1 is None:
            context_attn1 = n
        value_attn1 = context_attn1
        for p in patch:
            n, context_attn1, value_attn1 = p(n, context_attn1, value_attn1, extra_options)

    if block is not None:
        transformer_block = (block[0], block[1], block_index)
    else:
        transformer_block = None
    attn1_replace_patch = transformer_patches_replace.get("attn1", {})
    block_attn1 = transformer_block
    if block_attn1 not in attn1_replace_patch:
        block_attn1 = block

    if block_attn1 in attn1_replace_patch:
        if context_attn1 is None:
            context_attn1 = n
            value_attn1 = n
        n = self.attn1.to_q(n)
        # Reference CN READ - use attn1_replace_patch appropriately
        if ref_machine_state == MachineState.READ and len(self.injection_holder.bank_styles.bank) > 0:
            bank_styles = self.injection_holder.bank_styles
            style_fidelity = bank_styles.get_avg_style_fidelity()
            real_bank = bank_styles.bank.copy()
            cn_idx = 0
            for idx, order in enumerate(bank_styles.cn_idx):
                # make sure matching ref cn is selected
                for i in range(cn_idx, len(ref_controlnets)):
                    if ref_controlnets[i].order == order:
                        cn_idx = i
                        break
                assert order == ref_controlnets[cn_idx].order
                if ref_controlnets[cn_idx].any_attn_strength_to_apply():
                    effective_strength = ref_controlnets[cn_idx].get_effective_attn_mask_or_float(x=n, channels=n.shape[2], is_mid=self.injection_holder.is_middle)
                    real_bank[idx] = real_bank[idx] * effective_strength + context_attn1 * (1-effective_strength)
            n_uc = self.attn1.to_out(attn1_replace_patch[block_attn1](
                n,
                self.attn1.to_k(torch.cat([context_attn1] + real_bank, dim=1)),
                self.attn1.to_v(torch.cat([value_attn1] + real_bank, dim=1)),
                extra_options))
            n_c = n_uc.clone()
            if len(uc_idx_mask) > 0 and not math.isclose(style_fidelity, 0.0):
                n_c[uc_idx_mask] = self.attn1.to_out(attn1_replace_patch[block_attn1](
                    n[uc_idx_mask],
                    self.attn1.to_k(context_attn1[uc_idx_mask]),
                    self.attn1.to_v(value_attn1[uc_idx_mask]),
                    extra_options))
            n = style_fidelity * n_c + (1.0-style_fidelity) * n_uc
            bank_styles.clean()
        else:
            context_attn1 = self.attn1.to_k(context_attn1)
            value_attn1 = self.attn1.to_v(value_attn1)
            n = attn1_replace_patch[block_attn1](n, context_attn1, value_attn1, extra_options)
            n = self.attn1.to_out(n)
    else:
        # Reference CN READ - no attn1_replace_patch
        if ref_machine_state == MachineState.READ and len(self.injection_holder.bank_styles.bank) > 0:
            if context_attn1 is None:
                context_attn1 = n
            bank_styles = self.injection_holder.bank_styles
            style_fidelity = bank_styles.get_avg_style_fidelity()
            real_bank = bank_styles.bank.copy()
            cn_idx = 0
            for idx, order in enumerate(bank_styles.cn_idx):
                # make sure matching ref cn is selected
                for i in range(cn_idx, len(ref_controlnets)):
                    if ref_controlnets[i].order == order:
                        cn_idx = i
                        break
                assert order == ref_controlnets[cn_idx].order
                if ref_controlnets[cn_idx].any_attn_strength_to_apply():
                    effective_strength = ref_controlnets[cn_idx].get_effective_attn_mask_or_float(x=n, channels=n.shape[2], is_mid=self.injection_holder.is_middle)
                    real_bank[idx] = real_bank[idx] * effective_strength + context_attn1 * (1-effective_strength)
            n_uc: Tensor = self.attn1(
                n,
                context=torch.cat([context_attn1] + real_bank, dim=1),
                value=torch.cat([value_attn1] + real_bank, dim=1) if value_attn1 is not None else value_attn1)
            n_c = n_uc.clone()
            if len(uc_idx_mask) > 0 and not math.isclose(style_fidelity, 0.0):
                n_c[uc_idx_mask] = self.attn1(
                    n[uc_idx_mask],
                    context=context_attn1[uc_idx_mask],
                    value=value_attn1[uc_idx_mask] if value_attn1 is not None else value_attn1)
            n = style_fidelity * n_c + (1.0-style_fidelity) * n_uc
            bank_styles.clean()
        else:
            n = self.attn1(n, context=context_attn1, value=value_attn1)

    if "attn1_output_patch" in transformer_patches:
        patch = transformer_patches["attn1_output_patch"]
        for p in patch:
            n = p(n, extra_options)

    x += n
    if "middle_patch" in transformer_patches:
        patch = transformer_patches["middle_patch"]
        for p in patch:
            x = p(x, extra_options)

    if self.attn2 is not None:
        n = self.norm2(x)
        if self.switch_temporal_ca_to_sa:
            context_attn2 = n
        else:
            context_attn2 = context
        value_attn2 = None
        if "attn2_patch" in transformer_patches:
            patch = transformer_patches["attn2_patch"]
            value_attn2 = context_attn2
            for p in patch:
                n, context_attn2, value_attn2 = p(n, context_attn2, value_attn2, extra_options)

        attn2_replace_patch = transformer_patches_replace.get("attn2", {})
        block_attn2 = transformer_block
        if block_attn2 not in attn2_replace_patch:
            block_attn2 = block

        if block_attn2 in attn2_replace_patch:
            if value_attn2 is None:
                value_attn2 = context_attn2
            n = self.attn2.to_q(n)
            context_attn2 = self.attn2.to_k(context_attn2)
            value_attn2 = self.attn2.to_v(value_attn2)
            n = attn2_replace_patch[block_attn2](n, context_attn2, value_attn2, extra_options)
            n = self.attn2.to_out(n)
        else:
            n = self.attn2(n, context=context_attn2, value=value_attn2)

    if "attn2_output_patch" in transformer_patches:
        patch = transformer_patches["attn2_output_patch"]
        for p in patch:
            n = p(n, extra_options)

    x += n
    if self.is_res:
        x_skip = x
    x = self.ff(self.norm3(x))
    if self.is_res:
        x += x_skip

    return x


class RefTimestepEmbedSequential(openaimodel.TimestepEmbedSequential):
    injection_holder: InjectionTimestepEmbedSequentialHolder = None

def forward_timestep_embed_ref_inject_factory(orig_timestep_embed_inject_factory: Callable):
    def forward_timestep_embed_ref_inject(*args, **kwargs):
        ts: RefTimestepEmbedSequential = args[0]
        if not hasattr(ts, "injection_holder"):
            return orig_timestep_embed_inject_factory(*args, **kwargs)
        eps = 1e-6
        x: Tensor = orig_timestep_embed_inject_factory(*args, **kwargs)
        y: Tensor = None
        transformer_options: dict[str] = args[4]
        # Reference CN stuff
        uc_idx_mask = transformer_options.get(REF_UNCOND_IDXS, [])
        c_idx_mask = transformer_options.get(REF_COND_IDXS, [])
        # WRITE mode will only have one ReferenceAdvanced, other modes will have all ReferenceAdvanced
        ref_controlnets: list[ReferenceAdvanced] = transformer_options.get(REF_ADAIN_CONTROL_LIST, None)
        ref_machine_state: str = transformer_options.get(REF_ADAIN_MACHINE_STATE, None)
        
        # if in WRITE mode, save var, mean, and style_cfg
        if ref_machine_state == MachineState.WRITE:
            if ref_controlnets[0].ref_opts.adain_ref_weight > ts.injection_holder.gn_weight:
                var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                ts.injection_holder.bank_styles.var_bank.append(var)
                ts.injection_holder.bank_styles.mean_bank.append(mean)
                ts.injection_holder.bank_styles.style_cfgs.append(ref_controlnets[0].ref_opts.adain_style_fidelity)
                ts.injection_holder.bank_styles.cn_idx.append(ref_controlnets[0].order)
        # if in READ mode, do math with saved var, mean, and style_cfg
        if ref_machine_state == MachineState.READ:
            if len(ts.injection_holder.bank_styles.var_bank) > 0:
                bank_styles = ts.injection_holder.bank_styles
                var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                y_uc = torch.zeros_like(x)
                cn_idx = 0
                for idx, order in enumerate(bank_styles.cn_idx):
                    # make sure matching ref cn is selected
                    for i in range(cn_idx, len(ref_controlnets)):
                        if ref_controlnets[i].order == order:
                            cn_idx = i
                            break
                    assert order == ref_controlnets[cn_idx].order
                    style_fidelity = bank_styles.style_cfgs[idx]
                    var_acc = bank_styles.var_bank[idx]
                    mean_acc = bank_styles.mean_bank[idx]
                    std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                    sub_y_uc = (((x - mean) / std) * std_acc) + mean_acc
                    if ref_controlnets[cn_idx].any_adain_strength_to_apply():
                        effective_strength = ref_controlnets[cn_idx].get_effective_adain_mask_or_float(x=x)
                        sub_y_uc = sub_y_uc * effective_strength + x * (1-effective_strength)
                    y_uc += sub_y_uc
                # get average, if more than one
                if len(bank_styles.cn_idx) > 1:
                    y_uc /= len(bank_styles.cn_idx)
                y_c = y_uc.clone()
                if len(uc_idx_mask) > 0 and not math.isclose(style_fidelity, 0.0):
                    y_c[uc_idx_mask] = x.to(y_c.dtype)[uc_idx_mask]
                y = style_fidelity * y_c + (1.0 - style_fidelity) * y_uc
            ts.injection_holder.bank_styles.clean()

        if y is None:
            y = x
        return y.to(x.dtype)

    return forward_timestep_embed_ref_inject
