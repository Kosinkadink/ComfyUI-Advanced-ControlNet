from typing import Callable, Union

import math
import torch
from torch import Tensor

import comfy.model_management
import comfy.sample
import comfy.model_patcher
import comfy.utils
from comfy.controlnet import ControlBase
from comfy.model_patcher import ModelPatcher
from comfy.ldm.modules.attention import BasicTransformerBlock
from comfy.ldm.modules.diffusionmodules import openaimodel

from .logger import logger
from .utils import (AdvancedControlBase, ControlWeights, TimestepKeyframeGroup, TimestepKeyframe, AbstractPreprocWrapper,
                    broadcast_image_to_extend, ORIG_PREVIOUS_CONTROLNET, CONTROL_INIT_BY_ACN)


REF_READ_ATTN_CONTROL_LIST = "ref_read_attn_control_list"
REF_WRITE_ATTN_CONTROL_LIST = "ref_write_attn_control_list"
REF_READ_ADAIN_CONTROL_LIST = "ref_read_adain_control_list"
REF_WRITE_ADAIN_CONTROL_LIST = "ref_write_adain_control_list"

REF_ATTN_CONTROL_LIST = "ref_attn_control_list"
REF_ADAIN_CONTROL_LIST = "ref_adain_control_list"
REF_CONTROL_LIST_ALL = "ref_control_list_all"
REF_CONTROL_INFO = "ref_control_info"
REF_ATTN_MACHINE_STATE = "ref_attn_machine_state"
REF_ADAIN_MACHINE_STATE = "ref_adain_machine_state"
REF_COND_IDXS = "ref_cond_idxs"
REF_UNCOND_IDXS = "ref_uncond_idxs"

CONTEXTREF_OPTIONS_CLASS = "contextref_options_class"
CONTEXTREF_CLEAN_FUNC = "contextref_clean_func"
CONTEXTREF_CONTROL_LIST_ALL = "contextref_control_list_all"
CONTEXTREF_MACHINE_STATE = "contextref_machine_state"
CONTEXTREF_TEMP_COND_IDX = "contextref_temp_cond_idx"

HIGHEST_VERSION_SUPPORT = 1
RETURNED_CONTEXTREF_VERSION = 1


class RefConst:
    OPTS = "refcn_opts"
    CREF_MODE = "contextref_mode"


class MachineState:
    WRITE = "write"
    READ = "read"
    READ_WRITE = "read_write"
    STYLEALIGN = "stylealign"
    OFF = "off"

def is_read(state: str):
    return state in [MachineState.READ, MachineState.READ_WRITE]

def is_write(state: str):
    return state in [MachineState.WRITE, MachineState.READ_WRITE]


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
    
    @staticmethod
    def create_from_kwargs(attn_style_fidelity=0.0, adain_style_fidelity=0.0,
                         attn_ref_weight=0.0, adain_ref_weight=0.0,
                         attn_strength=0.0, adain_strength=0.0, **kwargs):
        has_attn = attn_strength > 0.0
        has_adain = adain_strength > 0.0
        if has_attn and has_adain:
            reference_type = ReferenceType.ATTN_ADAIN
        elif has_adain:
            reference_type = ReferenceType.ADAIN
        else:
            reference_type = ReferenceType.ATTN
        return ReferenceOptions(reference_type=reference_type,
                                attn_style_fidelity=float(attn_style_fidelity), adain_style_fidelity=float(adain_style_fidelity),
                                attn_ref_weight=float(attn_ref_weight), adain_ref_weight=float(adain_ref_weight),
                                attn_strength=float(attn_strength), adain_strength=float(adain_strength))


class ReferencePreprocWrapper(AbstractPreprocWrapper):
    error_msg = error_msg = "Invalid use of Reference Preprocess output. The output of Reference preprocessor is NOT a usual image, but a latent pretending to be an image - you must connect the output directly to an Apply Advanced ControlNet node. It cannot be used for anything else that accepts IMAGE input."
    def __init__(self, condhint: Tensor):
        super().__init__(condhint)


class ReferenceAdvanced(ControlBase, AdvancedControlBase):
    CHANNEL_TO_MULT = {320: 1, 640: 2, 1280: 4}

    def __init__(self, ref_opts: ReferenceOptions, timestep_keyframes: TimestepKeyframeGroup):
        super().__init__()
        AdvancedControlBase.__init__(self, super(), timestep_keyframes=timestep_keyframes, weights_default=ControlWeights.controllllite(), allow_condhint_latents=True)
        # TODO: allow vae_optional to be used instead of preprocessor
        #require_vae=True
        self._ref_opts = ref_opts
        self.order = 0
        self.model_latent_format = None
        self.model_sampling_current = None
        self.should_apply_attn_effective_strength = False
        self.should_apply_adain_effective_strength = False
        self.should_apply_effective_masks = False
        self.latent_shape = None
        # ContextRef stuff
        self.is_context_ref = False
        self.contextref_cond_idx = -1
        self.contextref_version = RETURNED_CONTEXTREF_VERSION

    @property
    def ref_opts(self):
        if self._current_timestep_keyframe is not None and self._current_timestep_keyframe.has_control_weights():
            return self._current_timestep_keyframe.control_weights.extras.get(RefConst.OPTS, self._ref_opts)
        return self._ref_opts

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

    def get_contextref_mode_replace(self):
        # used by ADE to get mode_replace for current keyframe
        if self._current_timestep_keyframe.has_control_weights():
            return self._current_timestep_keyframe.control_weights.extras.get(RefConst.CREF_MODE, None)
        return None

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
        # SDXL is more sensitive to style_fidelity according to sd-webui-controlnet comments;
        # prepare all ref_opts accordingly
        all_ref_opts = [self._ref_opts]
        for kf in self.timestep_keyframes.keyframes:
            if kf.has_control_weights() and RefConst.OPTS in kf.control_weights.extras:
                all_ref_opts.append(kf.control_weights.extras[RefConst.OPTS])
        for ropts in all_ref_opts:
            if type(model).__name__ == "SDXL":
                ropts.attn_style_fidelity = ropts.original_attn_style_fidelity ** 3.0
                ropts.adain_style_fidelity = ropts.original_adain_style_fidelity ** 3.0
            else:
                ropts.attn_style_fidelity = ropts.original_attn_style_fidelity
                ropts.adain_style_fidelity = ropts.original_adain_style_fidelity

    def get_control_advanced(self, x_noisy: Tensor, t, cond, batched_number: int):
        # normal ControlNet stuff
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                return control_prev

        dtype = x_noisy.dtype
        # cond_hint_original only matters for RefCN, NOT ContextRef
        if self.cond_hint_original is not None:
            # prepare cond_hint - it is a latent, NOT an image
            #if self.sub_idxs is not None or self.cond_hint is None or x_noisy.shape[2] != self.cond_hint.shape[2] or x_noisy.shape[3] != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            # if self.cond_hint_original length greater or equal to real latent count, subdivide it before scaling
            if self.sub_idxs is not None and self.cond_hint_original.size(0) >= self.full_latent_length:
                self.cond_hint = comfy.utils.common_upscale(
                    self.cond_hint_original[self.sub_idxs],
                    x_noisy.shape[3], x_noisy.shape[2], 'nearest-exact', "center").to(dtype).to(x_noisy.device)
            else:
                self.cond_hint = comfy.utils.common_upscale(
                    self.cond_hint_original,
                    x_noisy.shape[3], x_noisy.shape[2], 'nearest-exact', "center").to(dtype).to(x_noisy.device)
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
        c.is_context_ref = self.is_context_ref
        self.copy_to(c)
        self.copy_to_advanced(c)
        return c

    # avoid deepcopy shenanigans by making deepcopy not do anything to the reference
    # TODO: do the bookkeeping to do this in a proper way for all Adv-ControlNets
    def __deepcopy__(self, memo):
        return self


def handle_context_ref_setup(contextref_obj, transformer_options: dict, positive, negative):
    transformer_options[CONTEXTREF_MACHINE_STATE] = MachineState.OFF
    # verify version is compatible
    if contextref_obj.version > HIGHEST_VERSION_SUPPORT:
        raise Exception(f"AnimateDiff-Evolved's ContextRef v{contextref_obj.version} is not supported in currently-installed Advanced-ControlNet (only supports ContextRef up to v{HIGHEST_VERSION_SUPPORT}); " +
                        f"update your Advanced-ControlNet nodes for ContextRef to work.")
    # init ReferenceOptions
    cref_opt_dict = contextref_obj.tune.create_dict() # ContextRefTune obj from ADE
    opts = ReferenceOptions.create_from_kwargs(**cref_opt_dict)
    # init TimestepKeyframes
    cref_tks_list = contextref_obj.keyframe.create_list_of_dicts() # ContextRefKeyframeGroup obj from ADE
    timestep_keyframes = _create_tks_from_dict_list(cref_tks_list)
    # create ReferenceAdvanced
    cref = ReferenceAdvanced(ref_opts=opts, timestep_keyframes=timestep_keyframes)
    cref.strength = contextref_obj.strength # ContextRef obj from ADE
    cref.set_cond_hint_mask(contextref_obj.mask)
    cref.order = 99
    cref.is_context_ref = True
    context_ref_list = [cref]
    transformer_options[CONTEXTREF_CONTROL_LIST_ALL] = context_ref_list
    transformer_options[CONTEXTREF_OPTIONS_CLASS] = ReferenceOptions
    _add_context_ref_to_conds([positive, negative], cref)
    return context_ref_list


def _create_tks_from_dict_list(dlist: list[dict[str]]) -> TimestepKeyframeGroup:
    tks = TimestepKeyframeGroup()
    if dlist is None or len(dlist) == 0:
        return tks
    for d in dlist:
        # scheduling
        start_percent = d["start_percent"]
        guarantee_steps = d["guarantee_steps"]
        inherit_missing = d["inherit_missing"]
        # values
        strength = d["strength"]
        mask = d["mask"]
        tune = d["tune"]
        mode = d["mode"]
        weights = None
        extras = {}
        if tune is not None:
            cref_opt_dict = tune.create_dict() # ContextRefTune obj from ADE
            opts = ReferenceOptions.create_from_kwargs(**cref_opt_dict)
            extras[RefConst.OPTS] = opts
        if mode is not None:
            extras[RefConst.CREF_MODE] = mode
        weights = ControlWeights.default(extras=extras)
        # create keyframe
        tk = TimestepKeyframe(start_percent=start_percent, guarantee_steps=guarantee_steps, inherit_missing=inherit_missing,
                              strength=strength, mask_hint_orig=mask, control_weights=weights)
        tks.add(tk)
    return tks


def _add_context_ref_to_conds(conds: list[list[dict[str]]], context_ref: ReferenceAdvanced):
    def _add_context_ref_to_existing_control(control: ControlBase, context_ref: ReferenceAdvanced):
        curr_cn = control
        while curr_cn is not None:
            if type(curr_cn) == ReferenceAdvanced and curr_cn.is_context_ref:
                break
            if curr_cn.previous_controlnet is not None:
                curr_cn = curr_cn.previous_controlnet
                continue
            orig_previous_controlnet = curr_cn.previous_controlnet
            # NOTE: code is already in place to restore any ORIG_PREVIOUS_CONTROLNET props
            setattr(curr_cn, ORIG_PREVIOUS_CONTROLNET, orig_previous_controlnet)
            curr_cn.previous_controlnet = context_ref
            curr_cn = orig_previous_controlnet

    def _add_context_ref(actual_cond: dict[str], context_ref: ReferenceAdvanced):
        # if controls already present on cond, add it to the last previous_controlnet
        if "control" in actual_cond:
            return _add_context_ref_to_existing_control(actual_cond["control"], context_ref)
        # otherwise, need to add it to begin with, and should mark that it should be cleaned after
        actual_cond["control"] = context_ref
        actual_cond[CONTROL_INIT_BY_ACN] = True
    
    # either add context_ref to end of existing cnet chain, or init 'control' key on actual cond
    for cond in conds:
        if cond is not None:
            for sub_cond in cond:
                actual_cond = sub_cond[1]
                _add_context_ref(actual_cond, context_ref)


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
        # ref
        self.bank = []
        self.style_cfgs = []
        self.cn_idx: list[int] = []
        # contextref - list of lists as each cond/uncond stored separately
        self.c_bank: list[list] = []
        self.c_style_cfgs: list[list] = []
        self.c_cn_idx: list[list[int]] = []

    def get_bank(self, cref_idx, ignore_contextref, cdevice=None):
        if ignore_contextref or cref_idx >= len(self.c_bank):
            return self.bank
        real_c_bank_list = self.c_bank[cref_idx]
        if cdevice != None:
            real_c_bank_list = real_c_bank_list.copy()
            for i in range(len(real_c_bank_list)):
                real_c_bank_list[i] = real_c_bank_list[i].to(cdevice)
        return self.bank + real_c_bank_list

    def get_avg_style_fidelity(self, cref_idx, ignore_contextref):
        if ignore_contextref or cref_idx >= len(self.c_style_cfgs):
            return sum(self.style_cfgs) / float(len(self.style_cfgs))
        combined = self.style_cfgs + self.c_style_cfgs[cref_idx]
        return sum(combined) / float(len(combined))
    
    def get_cn_idxs(self, cref_idx, ignore_contxtref):
        if ignore_contxtref or cref_idx >= len(self.c_cn_idx):
            return self.cn_idx
        return self.cn_idx + self.c_cn_idx[cref_idx]

    def init_cref_for_idx(self, cref_idx: int):
        # makes sure cref lists can accommodate cref_idx 
        if cref_idx < 0:
            return
        while cref_idx >= len(self.c_bank):
            self.c_bank.append([])
            self.c_style_cfgs.append([])
            self.c_cn_idx.append([])

    def clear_cref_for_idx(self, cref_idx: int):
        if cref_idx < 0 or cref_idx >= len(self.c_bank):
            return
        self.c_bank[cref_idx] = []
        self.c_style_cfgs[cref_idx] = []
        self.c_cn_idx[cref_idx] = []

    def clean_ref(self):
        del self.bank
        del self.style_cfgs
        del self.cn_idx
        self.bank = []
        self.style_cfgs = []
        self.cn_idx = []

    def clean_contextref(self):
        del self.c_bank
        del self.c_style_cfgs
        del self.c_cn_idx
        self.c_bank = []
        self.c_style_cfgs = []
        self.c_cn_idx = []

    def clean_all(self):
        self.clean_ref()
        self.clean_contextref()


class BankStylesTimestepEmbedSequential:
    def __init__(self):
        # ref
        self.var_bank = []
        self.mean_bank = []
        self.style_cfgs = []
        self.cn_idx: list[int] = []
        # cref
        self.c_var_bank: list[list] = []
        self.c_mean_bank: list[list] = []
        self.c_style_cfgs: list[list] = []
        self.c_cn_idx: list[list[int]] = []

    def get_var_bank(self, cref_idx, ignore_contextref):
        if ignore_contextref or cref_idx >= len(self.c_var_bank):
            return self.var_bank
        return self.var_bank + self.c_var_bank[cref_idx]

    def get_mean_bank(self, cref_idx, ignore_contextref):
        if ignore_contextref or cref_idx >= len(self.c_mean_bank):
            return self.mean_bank
        return self.mean_bank + self.c_mean_bank[cref_idx]

    def get_style_cfgs(self, cref_idx, ignore_contextref):
        if ignore_contextref or cref_idx >= len(self.c_style_cfgs):
            return self.style_cfgs
        return self.style_cfgs + self.c_style_cfgs[cref_idx]

    def get_cn_idxs(self, cref_idx, ignore_contextref):
        if ignore_contextref or cref_idx >= len(self.c_cn_idx):
            return self.cn_idx
        return self.cn_idx + self.c_cn_idx[cref_idx]

    def init_cref_for_idx(self, cref_idx: int):
        # makes sure cref lists can accommodate cref_idx 
        if cref_idx < 0:
            return
        while cref_idx >= len(self.c_var_bank):
            self.c_var_bank.append([])
            self.c_mean_bank.append([])
            self.c_style_cfgs.append([])
            self.c_cn_idx.append([])

    def clear_cref_for_idx(self, cref_idx: int):
        if cref_idx < 0 or cref_idx >= len(self.c_var_bank):
            return
        self.c_var_bank[cref_idx] = []
        self.c_mean_bank[cref_idx] = []
        self.c_style_cfgs[cref_idx] = []
        self.c_cn_idx[cref_idx] = []

    def clean_ref(self):
        del self.mean_bank
        del self.var_bank
        del self.style_cfgs
        del self.cn_idx
        self.mean_bank = []
        self.var_bank = []
        self.style_cfgs = []
        self.cn_idx = []

    def clean_contextref(self):
        del self.c_var_bank
        del self.c_mean_bank
        del self.c_style_cfgs
        del self.c_cn_idx
        self.c_var_bank = []
        self.c_mean_bank = []
        self.c_style_cfgs = []
        self.c_cn_idx = []

    def clean_all(self):
        self.clean_ref()
        self.clean_contextref()


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

    def clean_ref(self):
        self.bank_styles.clean_ref()
    
    def clean_contextref(self):
        self.bank_styles.clean_contextref()

    def clean_all(self):
        self.bank_styles.clean_all()


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
    
    def clean_ref(self):
        self.bank_styles.clean_ref()
    
    def clean_contextref(self):
        self.bank_styles.clean_contextref()

    def clean_all(self):
        self.bank_styles.clean_all()


class ReferenceInjections:
    def __init__(self, attn_modules: list['RefBasicTransformerBlock']=None, gn_modules: list['RefTimestepEmbedSequential']=None):
        self.attn_modules = attn_modules if attn_modules else []
        self.gn_modules = gn_modules if gn_modules else []
        self.diffusion_model_orig_forward: Callable = None
    
    def clean_ref_module_mem(self):
        for attn_module in self.attn_modules:
            try:
                attn_module.injection_holder.clean_ref()
            except Exception:
                pass
        for gn_module in self.gn_modules:
            try:
                gn_module.injection_holder.clean_ref()
            except Exception:
                pass

    def clean_contextref_module_mem(self):
        for attn_module in self.attn_modules:
            try:
                attn_module.injection_holder.clean_contextref()
            except Exception:
                pass
        for gn_module in self.gn_modules:
            try:
                gn_module.injection_holder.clean_contextref()
            except Exception:
                pass

    def clean_all_module_mem(self):
        for attn_module in self.attn_modules:
            try:
                attn_module.injection_holder.clean_all()
            except Exception:
                pass
        for gn_module in self.gn_modules:
            try:
                gn_module.injection_holder.clean_all()
            except Exception:
                pass

    def cleanup(self):
        self.clean_all_module_mem()
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
        transformer_options: dict[str] = kwargs.get("transformer_options", {})
        # NOTE: adds support for both ReferenceCN and ContextRef, so need to track them separately
        # get ReferenceAdvanced objects
        ref_controlnets: list[ReferenceAdvanced] = transformer_options.get(REF_CONTROL_LIST_ALL, [])
        context_controlnets: list[ReferenceAdvanced] = transformer_options.get(CONTEXTREF_CONTROL_LIST_ALL, [])
        # clean contextref stuff if OFF
        if len(context_controlnets) > 0 and transformer_options[CONTEXTREF_MACHINE_STATE] == MachineState.OFF:
            reference_injections.clean_contextref_module_mem()
            context_controlnets = []
        # discard any controlnets that should not run
        ref_controlnets = [z for z in ref_controlnets if z.should_run()]
        context_controlnets = [z for z in context_controlnets if z.should_run()]
        # if nothing related to reference controlnets, do nothing special
        if len(ref_controlnets) == 0 and len(context_controlnets) == 0:
            return reference_injections.diffusion_model_orig_forward(x, *args, **kwargs)
        try:
            # assign cond and uncond idxs
            batched_number = len(transformer_options["cond_or_uncond"])
            per_batch = x.shape[0] // batched_number
            indiv_conds = []
            for cond_type in transformer_options["cond_or_uncond"]:
                indiv_conds.extend([cond_type] * per_batch)
            transformer_options[REF_UNCOND_IDXS] = [i for i, z in enumerate(indiv_conds) if z == 1]
            transformer_options[REF_COND_IDXS] = [i for i, z in enumerate(indiv_conds) if z == 0]
            # check which controlnets do which thing
            attn_controlnets = []
            adain_controlnets = []
            for control in ref_controlnets:
                if ReferenceType.is_attn(control.ref_opts.reference_type):
                    attn_controlnets.append(control)
                if ReferenceType.is_adain(control.ref_opts.reference_type):
                    adain_controlnets.append(control)
            context_attn_controlnets = []
            context_adain_controlnets = []
            # for ease of access, store current contextref_cond_idx value
            if len(context_controlnets) == 0:
                transformer_options[CONTEXTREF_TEMP_COND_IDX] = -1
            else:
                transformer_options[CONTEXTREF_TEMP_COND_IDX] = context_controlnets[0].contextref_cond_idx
                # logger.info(f"{transformer_options[CONTEXTREF_MACHINE_STATE]}: {transformer_options[CONTEXTREF_TEMP_COND_IDX]}")
            
            for control in context_controlnets:
                if ReferenceType.is_attn(control.ref_opts.reference_type):
                    context_attn_controlnets.append(control)
                if ReferenceType.is_adain(control.ref_opts.reference_type):
                    context_adain_controlnets.append(control)
            if len(adain_controlnets) > 0 or len(context_adain_controlnets) > 0:
                # ComfyUI uses forward_timestep_embed with the TimestepEmbedSequential passed into it
                orig_forward_timestep_embed = openaimodel.forward_timestep_embed
                openaimodel.forward_timestep_embed = forward_timestep_embed_ref_inject_factory(orig_forward_timestep_embed)
            
            # if RefCN to be used, handle running diffusion with ref cond hints
            if len(ref_controlnets) > 0:
                for control in ref_controlnets:
                    read_attn_list = []
                    write_attn_list = []
                    read_adain_list = []
                    write_adain_list = []

                    if ReferenceType.is_attn(control.ref_opts.reference_type):
                        write_attn_list.append(control)
                    if ReferenceType.is_adain(control.ref_opts.reference_type):
                        write_adain_list.append(control)
                    # apply lists
                    transformer_options[REF_READ_ATTN_CONTROL_LIST] = read_attn_list
                    transformer_options[REF_WRITE_ATTN_CONTROL_LIST] = write_attn_list
                    transformer_options[REF_READ_ADAIN_CONTROL_LIST] = read_adain_list
                    transformer_options[REF_WRITE_ADAIN_CONTROL_LIST] = write_adain_list

                    orig_kwargs = kwargs
                    # disable other controlnets for this run, if specified
                    if not control.ref_opts.ref_with_other_cns:
                        kwargs = kwargs.copy()
                        kwargs["control"] = None
                    reference_injections.diffusion_model_orig_forward(control.cond_hint.to(dtype=x.dtype).to(device=x.device), *args, **kwargs)
                    kwargs = orig_kwargs
            # prepare running diffusion for real now
            read_attn_list = []
            write_attn_list = []
            read_adain_list = []
            write_adain_list = []

            # add RefCNs to read lists
            read_attn_list.extend(attn_controlnets)
            read_adain_list.extend(adain_controlnets)
            
            # do contextref stuff, if needed
            if len(context_controlnets) > 0:
                # clean contextref stuff if first WRITE
                # if context_controlnets[0].contextref_cond_idx == 0 and is_write(transformer_options[CONTEXTREF_MACHINE_STATE]):
                #     reference_injections.clean_contextref_module_mem()
                ### add ContextRef to appropriate lists
                # attn
                if is_read(transformer_options[CONTEXTREF_MACHINE_STATE]):
                    read_attn_list.extend(context_attn_controlnets)
                if is_write(transformer_options[CONTEXTREF_MACHINE_STATE]):
                    write_attn_list.extend(context_attn_controlnets)
                # adain
                if is_read(transformer_options[CONTEXTREF_MACHINE_STATE]):
                    read_adain_list.extend(context_adain_controlnets)
                if is_write(transformer_options[CONTEXTREF_MACHINE_STATE]):
                    write_adain_list.extend(context_adain_controlnets)
            # apply lists, containing both RefCN and ContextRef
            transformer_options[REF_READ_ATTN_CONTROL_LIST] = read_attn_list
            transformer_options[REF_WRITE_ATTN_CONTROL_LIST] = write_attn_list
            transformer_options[REF_READ_ADAIN_CONTROL_LIST] = read_adain_list
            transformer_options[REF_WRITE_ADAIN_CONTROL_LIST] = write_adain_list
            # run diffusion for real
            try:
                return reference_injections.diffusion_model_orig_forward(x, *args, **kwargs)
            finally:
                # increment current cond idx
                if len(context_controlnets) > 0:
                    for cn in context_controlnets:
                        cn.contextref_cond_idx += 1
        finally:
            # make sure ref banks are cleared no matter what happens - otherwise, RIP VRAM
            reference_injections.clean_ref_module_mem()
            if len(adain_controlnets) > 0 or len(context_adain_controlnets) > 0:
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
    #c_idx_mask = transformer_options.get(REF_COND_IDXS, [])
    # WRITE mode may have only 1 ReferenceAdvanced for RefCN at a time, other modes will have all ReferenceAdvanced
    ref_write_cns: list[ReferenceAdvanced] = transformer_options.get(REF_WRITE_ATTN_CONTROL_LIST, [])
    ref_read_cns: list[ReferenceAdvanced] = transformer_options.get(REF_READ_ATTN_CONTROL_LIST, [])
    cref_cond_idx: int = transformer_options.get(CONTEXTREF_TEMP_COND_IDX, -1)
    ignore_contextref_read = cref_cond_idx < 0 # if writing to bank, should NOT be read in the same execution

    cached_n = None
    cref_write_cns: list[ReferenceAdvanced] = []
    # check if any WRITE cns are applicable; Reference CN WRITEs immediately, ContextREF WRITEs after READ completed
    # if any refs to WRITE, save n and style_fidelity
    for refcn in ref_write_cns:
        if refcn.ref_opts.attn_ref_weight > self.injection_holder.attn_weight:
            if cached_n is None:
                cached_n = n.detach().clone()
            # for ContextRef, make sure relevant lists are long enough to cond_idx
            # store RefCN and ContextRef stuff separately
            if refcn.is_context_ref:
                cref_write_cns.append(refcn)
                self.injection_holder.bank_styles.init_cref_for_idx(cref_cond_idx)
            else: # Reference CN WRITE
                self.injection_holder.bank_styles.bank.append(cached_n)
                self.injection_holder.bank_styles.style_cfgs.append(refcn.ref_opts.attn_style_fidelity)
                self.injection_holder.bank_styles.cn_idx.append(refcn.order)
    if len(cref_write_cns) == 0:
        del cached_n

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
        if len(ref_read_cns) > 0 and len(self.injection_holder.bank_styles.get_bank(cref_cond_idx, ignore_contextref_read)) > 0:
            bank_styles = self.injection_holder.bank_styles
            style_fidelity = bank_styles.get_avg_style_fidelity(cref_cond_idx, ignore_contextref_read)
            real_bank = bank_styles.get_bank(cref_cond_idx, ignore_contextref_read, cdevice=n.device).copy()
            real_cn_idxs = bank_styles.get_cn_idxs(cref_cond_idx, ignore_contextref_read)
            cn_idx = 0
            for idx, order in enumerate(real_cn_idxs):
                # make sure matching ref cn is selected
                for i in range(cn_idx, len(ref_read_cns)):
                    if ref_read_cns[i].order == order:
                        cn_idx = i
                        break
                assert order == ref_read_cns[cn_idx].order
                if ref_read_cns[cn_idx].any_attn_strength_to_apply():
                    effective_strength = ref_read_cns[cn_idx].get_effective_attn_mask_or_float(x=n, channels=n.shape[2], is_mid=self.injection_holder.is_middle)
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
            bank_styles.clean_ref()
        else:
            context_attn1 = self.attn1.to_k(context_attn1)
            value_attn1 = self.attn1.to_v(value_attn1)
            n = attn1_replace_patch[block_attn1](n, context_attn1, value_attn1, extra_options)
            n = self.attn1.to_out(n)
    else:
        # Reference CN READ - no attn1_replace_patch
        if len(ref_read_cns) > 0 and len(self.injection_holder.bank_styles.get_bank(cref_cond_idx, ignore_contextref_read)) > 0:
            if context_attn1 is None:
                context_attn1 = n
            bank_styles = self.injection_holder.bank_styles
            style_fidelity = bank_styles.get_avg_style_fidelity(cref_cond_idx, ignore_contextref_read)
            real_bank = bank_styles.get_bank(cref_cond_idx, ignore_contextref_read, cdevice=n.device).copy()
            real_cn_idxs = bank_styles.get_cn_idxs(cref_cond_idx, ignore_contextref_read)
            cn_idx = 0
            for idx, order in enumerate(real_cn_idxs):
                # make sure matching ref cn is selected
                for i in range(cn_idx, len(ref_read_cns)):
                    if ref_read_cns[i].order == order:
                        cn_idx = i
                        break
                assert order == ref_read_cns[cn_idx].order
                if ref_read_cns[cn_idx].any_attn_strength_to_apply():
                    effective_strength = ref_read_cns[cn_idx].get_effective_attn_mask_or_float(x=n, channels=n.shape[2], is_mid=self.injection_holder.is_middle)
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
            bank_styles.clean_ref()
        else:
            n = self.attn1(n, context=context_attn1, value=value_attn1)

    # ContextRef CN WRITE
    if len(cref_write_cns) > 0:
        # clear so that ContextRef CNs can properly 'replace' previous value at cond_idx
        self.injection_holder.bank_styles.clear_cref_for_idx(cref_cond_idx)
        for refcn in cref_write_cns:
            # add a whole list to match expected type when combining
            self.injection_holder.bank_styles.c_bank[cref_cond_idx].append(cached_n.to(comfy.model_management.unet_offload_device()))
            self.injection_holder.bank_styles.c_style_cfgs[cref_cond_idx].append(refcn.ref_opts.attn_style_fidelity)
            self.injection_holder.bank_styles.c_cn_idx[cref_cond_idx].append(refcn.order)
        del cached_n

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
        #c_idx_mask = transformer_options.get(REF_COND_IDXS, [])
        # WRITE mode will only have one ReferenceAdvanced, other modes will have all ReferenceAdvanced
        ref_write_cns: list[ReferenceAdvanced] = transformer_options.get(REF_WRITE_ADAIN_CONTROL_LIST, [])
        ref_read_cns: list[ReferenceAdvanced] = transformer_options.get(REF_READ_ADAIN_CONTROL_LIST, [])
        cref_cond_idx: int = transformer_options.get(CONTEXTREF_TEMP_COND_IDX, -1)
        ignore_contextref_read = cref_cond_idx < 0 # if writing to bank, should NOT be read in the same execution

        cached_var = None
        cached_mean = None
        cref_write_cns: list[ReferenceAdvanced] = []
        # if any refs to WRITE, save var, mean, and style_cfg
        for refcn in ref_write_cns:
            if refcn.ref_opts.adain_ref_weight > ts.injection_holder.gn_weight:
                if cached_var is None:
                    cached_var, cached_mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                if refcn.is_context_ref:
                    cref_write_cns.append(refcn)
                    ts.injection_holder.bank_styles.init_cref_for_idx(cref_cond_idx)
                else:
                    ts.injection_holder.bank_styles.var_bank.append(cached_var)
                    ts.injection_holder.bank_styles.mean_bank.append(cached_mean)
                    ts.injection_holder.bank_styles.style_cfgs.append(refcn.ref_opts.adain_style_fidelity)
                    ts.injection_holder.bank_styles.cn_idx.append(refcn.order)
        if len(cref_write_cns) == 0:
            del cached_var
            del cached_mean

        # if any refs to READ, do math with saved var, mean, and style_cfg
        if len(ref_read_cns) > 0:
            if len(ts.injection_holder.bank_styles.get_var_bank(cref_cond_idx, ignore_contextref_read)) > 0:
                bank_styles = ts.injection_holder.bank_styles
                var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                y_uc = torch.zeros_like(x)
                cn_idx = 0
                real_style_cfgs = bank_styles.get_style_cfgs(cref_cond_idx, ignore_contextref_read)
                real_var_bank = bank_styles.get_var_bank(cref_cond_idx, ignore_contextref_read)
                real_mean_bank = bank_styles.get_mean_bank(cref_cond_idx, ignore_contextref_read)
                real_cn_idxs = bank_styles.get_cn_idxs(cref_cond_idx, ignore_contextref_read)
                for idx, order in enumerate(real_cn_idxs):
                    # make sure matching ref cn is selected
                    for i in range(cn_idx, len(ref_read_cns)):
                        if ref_read_cns[i].order == order:
                            cn_idx = i
                            break
                    assert order == ref_read_cns[cn_idx].order
                    style_fidelity = real_style_cfgs[idx]
                    var_acc = real_var_bank[idx]
                    mean_acc = real_mean_bank[idx]
                    std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                    sub_y_uc = (((x - mean) / std) * std_acc) + mean_acc
                    if ref_read_cns[cn_idx].any_adain_strength_to_apply():
                        effective_strength = ref_read_cns[cn_idx].get_effective_adain_mask_or_float(x=x)
                        sub_y_uc = sub_y_uc * effective_strength + x * (1-effective_strength)
                    y_uc += sub_y_uc
                # get average, if more than one
                if len(real_cn_idxs) > 1:
                    y_uc /= len(real_cn_idxs)
                y_c = y_uc.clone()
                if len(uc_idx_mask) > 0 and not math.isclose(style_fidelity, 0.0):
                    y_c[uc_idx_mask] = x.to(y_c.dtype)[uc_idx_mask]
                y = style_fidelity * y_c + (1.0 - style_fidelity) * y_uc
            ts.injection_holder.bank_styles.clean_ref()

        # ContextRef CN WRITE
        if len(cref_write_cns) > 0:
            # clear so that ContextRef CNs can properly 'replace' previous value at cond_idx
            ts.injection_holder.bank_styles.clear_cref_for_idx(cref_cond_idx)
            for refcn in cref_write_cns:
                # add a whole list to match expected type when combining
                ts.injection_holder.bank_styles.c_var_bank[cref_cond_idx].append(cached_var)
                ts.injection_holder.bank_styles.c_mean_bank[cref_cond_idx].append(cached_mean)
                ts.injection_holder.bank_styles.c_style_cfgs[cref_cond_idx].append(refcn.ref_opts.adain_style_fidelity)
                ts.injection_holder.bank_styles.c_cn_idx[cref_cond_idx].append(refcn.order)
            del cached_var
            del cached_mean

        if y is None:
            y = x
        return y.to(x.dtype)

    return forward_timestep_embed_ref_inject
