from typing import Callable, Union

import math
import torch
from torch import Tensor

import comfy.model_patcher
import comfy.utils
from comfy.controlnet import ControlBase
from comfy.model_patcher import ModelPatcher
from comfy.ldm.modules.attention import BasicTransformerBlock

from .logger import logger
from .utils import (AdvancedControlBase, ControlWeights, TimestepKeyframeGroup, AbstractPreprocWrapper,
                    deepcopy_with_sharing, prepare_mask_batch, broadcast_image_to_full, ddpm_noise_latents, simple_noise_latents)


REF_CONTROL_LIST = "ref_control_list"
REF_CONTROL_INFO = "ref_control_info"
REF_MACHINE_STATE = "ref_machine_state"


class MachineState:
    WRITE = "write"
    READ = "read"
    STYLEALIGN = "stylealign"


class ReferenceType:
    ATTN = "reference_attn"
    ADAIN = "reference_adain"
    ATTN_ADAIN = "reference_attn+adain"
    STYLE_ALIGN = "StyleAlign"

    _LIST = [ATTN, ADAIN, ATTN_ADAIN]


class ReferenceOptions:
    def __init__(self, reference_type: str, style_fidelity: float):
        self.reference_type = reference_type
        self.original_style_fidelity = style_fidelity
        self.style_fidelity = style_fidelity
    
    def clone(self):
        return ReferenceOptions(reference_type=self.reference_type, style_fidelity=self.original_style_fidelity)


class ReferencePreprocWrapper(AbstractPreprocWrapper):
    error_msg = error_msg = "Invalid use of Reference Preprocess output. The output of RGB SparseCtrl preprocessor is NOT a usual image, but a latent pretending to be an image - you must connect the output directly to an Apply Advanced ControlNet node. It cannot be used for anything else that accepts IMAGE input."
    def __init__(self, condhint: Tensor):
        super().__init__(condhint)


class ReferenceAttnPatch:
    def __init__(self, control: 'ReferenceAdvanced'=None):
        self.control = control

    # def __call__(self, q: Tensor, k: Tensor, v: Tensor, extra_options: dict):
    #     # do nothing here - all ReferenceAttnPatch is trying to do is be a
    #     # ComfyUI-compliant way of tracking the corresponding ControlNet obj
    #     return q, k, v
    
    def __call__(self, x: Tensor, extra_options: dict):
        # do nothing here - all ReferenceAttnPatch is trying to do is be a
        # ComfyUI-compliant way of tracking the corresponding ControlNet obj
        return x

    def set_control(self, control: 'ReferenceAdvanced') -> 'ReferenceAttnPatch':
        self.control = control
        return self

    def cleanup(self):
        pass

    # make sure deepcopy does not copy control, and deepcopied patch should be assigned to control
    def __deepcopy__(self, memo):
        self.cleanup()
        to_return: ReferenceAttnPatch = deepcopy_with_sharing(self, shared_attribute_names = ['control'], memo=memo)
        #logger.warn(f"patch {id(self)} turned into {id(to_return)}")
        try:
            to_return.control.patch_attn1 = to_return
        except Exception:
            pass
        return to_return


class ReferenceAdvanced(ControlBase, AdvancedControlBase):
    def __init__(self, patch_attn1: ReferenceAttnPatch, ref_opts: ReferenceOptions, timestep_keyframes: TimestepKeyframeGroup, device=None):
        super().__init__(device)
        AdvancedControlBase.__init__(self, super(), timestep_keyframes=timestep_keyframes, weights_default=ControlWeights.controllllite(), require_model=True)
        self.patch_attn1 = patch_attn1.set_control(self)
        self.ref_opts = ref_opts
        self.order = 0
        self.latent_format = None
    
    def get_effective_strength(self):
        effective_strength = self.strength
        if self.current_timestep_keyframe is not None:
            effective_strength = effective_strength * self.current_timestep_keyframe.strength
        return effective_strength

    def patch_model(self, model: ModelPatcher):
        # need to patch model so that control can be found later from it
        model.set_model_attn1_output_patch(self.patch_attn1)
        #model.set_model_attn1_patch(self.patch_attn1)
        # need to add model_options to make patch/unpatch injection know it has to run
        if not REF_CONTROL_INFO in model.model_options:
            model.model_options[REF_CONTROL_INFO] = 0
        self.order = model.model_options[REF_CONTROL_INFO]
        model.model_options[REF_CONTROL_INFO]

    def pre_run_advanced(self, model, percent_to_timestep_function):
        AdvancedControlBase.pre_run_advanced(self, model, percent_to_timestep_function)
        if type(self.cond_hint_original) == ReferencePreprocWrapper:
            self.cond_hint_original = self.cond_hint_original.condhint
        self.latent_format = model.latent_format # LatentFormat object, used to process_in latent cond_hint
        # SDXL is more sensitive to style_fidelity according to sd-webui-controlnet comments
        if type(model).__name__ == "SDXL":
            self.ref_opts.style_fidelity = self.ref_opts.style_fidelity ** 3.0
        else:
            self.ref_opts.style_fidelity = self.ref_opts.style_fidelity
        # set control on patches
        self.patch_attn1.set_control(self)

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
            self.cond_hint = broadcast_image_to_full(self.cond_hint, x_noisy.shape[0], batched_number, except_one=False)
        # noise cond_hint based on sigma (current step)
        # TODO: how to handle noise? reproducibility is key...
        # mess with the order here?
        self.cond_hint = ddpm_noise_latents(self.cond_hint, sigma=t[0] / (self.latent_format.scale_factor), noise=None)
        self.cond_hint = self.latent_format.process_in(self.cond_hint)
        #self.cond_hint = simple_noise_latents(self.cond_hint, sigma=t[0], noise=None)

        # prepare mask
        self.prepare_mask_cond_hint(x_noisy=x_noisy, t=t, cond=cond, batched_number=batched_number)
        # done preparing; model patches will take care of everything now.
        # return normal controlnet stuff
        return control_prev

    def cleanup_advanced(self):
        super().cleanup_advanced()
        self.patch_attn1.cleanup()
        if self.latent_format is not None:
            del self.latent_format
            self.latent_format = None
    
    def copy(self):
        c = ReferenceAdvanced(self.patch_attn1, self.ref_opts, self.timestep_keyframes)
        c.order = self.order
        self.copy_to(c)
        self.copy_to_advanced(c)
        return c


class BankStylesBasicTransformerBlock:
    def __init__(self):
        self.bank = []
        self.style_cfgs = []
    
    def get_avg_style_fidelity(self):
        return sum(self.style_cfgs) / float(len(self.style_cfgs))
    
    def clean(self):
        del self.bank
        self.bank = []
        del self.style_cfgs
        self.style_cfgs = []


class InjectionBasicTransformerBlockHolder:
    def __init__(self, block: BasicTransformerBlock):
        self.original_forward = block._forward
        self.attn_weight = 1.0
        self.bank_styles: dict[int, BankStylesBasicTransformerBlock] = {}
    
    def restore(self, block: BasicTransformerBlock):
        block._forward = self.original_forward

    def clean(self):
        for bank_style in list(self.bank_styles.values()):
            bank_style.clean()
        self.bank_styles.clear()


# def factory_clone_injected_ModelPatcher(orig_clone_ModelPatcher: Callable):
#     def clone_injected_ref(self, *args, **kwargs):
#         cloned = orig_clone_ModelPatcher(*args, **kwargs)
#         if InjectMP.is_injected(self):
#             InjectMP.mark_injected(cloned)
#             cloned.clone = factory_clone_injected_ModelPatcher(cloned.clone).__get__(cloned, type(cloned))
#         return cloned
#     return clone_injected_ref


# inject ModelPatcher.clone so that necessary injection will happen when needed
# orig_modelpatcher_clone = comfy.model_patcher.ModelPatcher.clone
# def clone_injection_ref(self: ModelPatcher, *args, **kwargs):
#     cloned = orig_modelpatcher_clone(self, *args, **kwargs)
#     if InjectMP.is_injected(self):
#         InjectMP.mark_injected(cloned)
#     return cloned
# comfy.model_patcher.ModelPatcher.clone = clone_injection_ref


# inject ModelPatcher.patch_model to apply 
orig_modelpatcher_patch_model = comfy.model_patcher.ModelPatcher.patch_model
def patch_model_injection_ref(self: ModelPatcher, *args, **kwargs):
    if REF_CONTROL_INFO in self.model_options:
        # storage for all Reference-related injections
        reference_injections = ReferenceInjections()
        # first, handle attn module injection
        all_modules = torch_dfs(self.model)
        attn_modules: list[RefBasicTransformerBlock] = []
        for module in all_modules:
            if isinstance(module, BasicTransformerBlock):
                attn_modules.append(module)
        attn_modules = [module for module in all_modules if isinstance(module, BasicTransformerBlock)]
        attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])
        for i, module in enumerate(attn_modules):
            injection_holder = InjectionBasicTransformerBlockHolder(block=module)
            injection_holder.attn_weight = float(i) / float(len(attn_modules))
            module._forward = _forward_inject_BasicTransformerBlock.__get__(module, type(module))
            module.injection_holder = injection_holder
        reference_injections.attn_modules = attn_modules
        # handle diffusion_model forward injection
        reference_injections.diffusion_model_orig_forward = self.model.diffusion_model.forward
        self.model.diffusion_model.forward = factory_forward_inject_UNetModel(reference_injections).__get__(self.model.diffusion_model, type(self.model.diffusion_model))
        InjectMP.set_injected(self, reference_injections)
    to_return = orig_modelpatcher_patch_model(self, *args, **kwargs)
    return to_return
comfy.model_patcher.ModelPatcher.patch_model = patch_model_injection_ref


orig_modelpatcher_unpatch_model = comfy.model_patcher.ModelPatcher.unpatch_model
def unpatch_model_injection_ref(self: ModelPatcher, *args, **kwargs):
    if REF_CONTROL_INFO in self.model_options:
        reference_injections: ReferenceInjections = InjectMP.get_injected(self)
        # first, restore attn modules
        attn_modules: list[RefBasicTransformerBlock] = reference_injections.attn_modules
        for module in attn_modules:
            module.injection_holder.restore(module)
            module.injection_holder.clean()
            del module.injection_holder
        del attn_modules
        # restore diffusion_model forward function
        self.model.diffusion_model.forward = reference_injections.diffusion_model_orig_forward.__get__(self.model.diffusion_model, type(self.model.diffusion_model))
        # cleanup
        InjectMP.clean_injected(self)
        reference_injections.cleanup()
    to_return = orig_modelpatcher_unpatch_model(self, *args, **kwargs)
    return to_return
comfy.model_patcher.ModelPatcher.unpatch_model = unpatch_model_injection_ref


class ReferenceInjections:
    def __init__(self, attn_modules: list['RefBasicTransformerBlock']=None):
        self.attn_modules = attn_modules if attn_modules else []
        self.diffusion_model_orig_forward: Callable = None
    
    def clean_module_mem(self):
        for attn_module in self.attn_modules:
            attn_module.injection_holder.clean()

    def cleanup(self):
        del self.attn_modules
        self.attn_modules = []
        self.diffusion_model_orig_forward = None


class InjectMP:
    PARAM_INJECTED_REF = "___injected_ref"

    @staticmethod
    def is_injected(model: ModelPatcher):
        return getattr(model, InjectMP.PARAM_INJECTED_REF, False)

    def mark_injected(model: ModelPatcher):
        setattr(model, InjectMP.PARAM_INJECTED_REF, True)

    def set_injected(model: ModelPatcher, value: ReferenceInjections):
        setattr(model, InjectMP.PARAM_INJECTED_REF, value)

    def get_injected(model: ModelPatcher) -> ReferenceInjections:
        return getattr(model, InjectMP.PARAM_INJECTED_REF)

    def clean_injected(model: ModelPatcher):
        delattr(model, InjectMP.PARAM_INJECTED_REF)
        InjectMP.mark_injected(model)


def factory_forward_inject_UNetModel(reference_injections: ReferenceInjections):
    def forward_inject_UNetModel(self, x: Tensor, *args, **kwargs):
        # get control and transformer_options from kwargs
        real_args = list(args)
        real_kwargs = list(kwargs.keys())
        control = kwargs.get("control", None)
        transformer_options = kwargs.get("transformer_options", None)
        # look for ReferenceAttnPatch objects to get ReferenceAdvanced objects
        # and remove the patch from transformer_options so it won't be ran for no reason
        patch_name = "attn1_output_patch"
        #patch_name = "attn1_patch"
        ref_patches: list[ReferenceAttnPatch] = []
        if "patches" in transformer_options:
            if patch_name in transformer_options["patches"]:
                patches: list = transformer_options["patches"][patch_name]
                remove_idxs = []
                for i in range(len(patches)):
                    if isinstance(patches[i], ReferenceAttnPatch):
                        ref_patches.append(patches[i])
                        remove_idxs.append(i)
                # for i in reversed(remove_idxs):
                #     patches.pop(i)
                # if len(transformer_options["patches"]["attn1_patch"]) == 0:
                #      transformer_options["patches"].pop("attn1_patch")
        ref_controlnets: list[ReferenceAdvanced] = [x.control for x in ref_patches]
        # discard any controlnets that should not run
        ref_controlnets = [x for x in ref_controlnets if x.should_run()]
        ref_controlnets = sorted(ref_controlnets, key=lambda x: x.order)
        # if nothing related to reference controlnets, do nothing special
        if len(ref_controlnets) == 0:
            return reference_injections.diffusion_model_orig_forward(x, *args, **kwargs)
        try:
            # otherwise, need to handle ref controlnet stuff
            for control in ref_controlnets:
                transformer_options[REF_MACHINE_STATE] = MachineState.WRITE
                transformer_options[REF_CONTROL_LIST] = [control]
                # TODO: insert control's cond_hint
                reference_injections.diffusion_model_orig_forward(control.cond_hint.to(dtype=x.dtype).to(device=x.device), *args, **kwargs)
            transformer_options[REF_MACHINE_STATE] = MachineState.READ
            transformer_options[REF_CONTROL_LIST] = ref_controlnets
            return reference_injections.diffusion_model_orig_forward(x, *args, **kwargs)
        finally:
            # make sure banks are cleared no matter what happens - otherwise, RIP VRAM
            reference_injections.clean_module_mem()

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

    n = self.norm1(x)
    if self.disable_self_attn:
        context_attn1 = context
    else:
        context_attn1 = None
    value_attn1 = None

    # Reference CN stuff
    # WRITE mode will only have one ReferenceAdvanced, other modes will have all ReferenceAdvanced
    ref_controlnets: list[ReferenceAdvanced] = transformer_options.get(REF_CONTROL_LIST, None)
    ref_machine_state: str = transformer_options.get(REF_MACHINE_STATE, None)
    # if in WRITE mode, save n and style_fidelity
    if ref_controlnets and ref_machine_state == MachineState.WRITE:
        if ref_controlnets[0].get_effective_strength() > self.injection_holder.attn_weight:
            if not self.injection_holder.bank_styles.get(ref_controlnets[0].order, None):
                self.injection_holder.bank_styles[ref_controlnets[0].order] = BankStylesBasicTransformerBlock()
            bank_style = self.injection_holder.bank_styles[ref_controlnets[0].order]
            bank_style.bank.append(n.detach().clone())
            bank_style.style_cfgs.append(ref_controlnets[0].ref_opts.style_fidelity)
    # create uc_idx_mask
    per_batch = x.shape[0] // len(transformer_options["cond_or_uncond"])
    indiv_conds = []
    for cond_type in transformer_options["cond_or_uncond"]:
        indiv_conds.extend([cond_type] * per_batch)
    uc_idx_mask = [i for i, x in enumerate(indiv_conds) if x == 0]

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
        if ref_machine_state == MachineState.READ:
            bank_styles = self.injection_holder.bank_styles[ref_controlnets[0].order]
            style_fidelity = bank_styles.get_avg_style_fidelity()
            n_uc = self.attn1.to_out(attn1_replace_patch[block_attn1](
                n,
                self.attn1.to_k(torch.cat([context_attn1] + bank_styles.bank, dim=1)),
                self.attn1.to_v(torch.cat([value_attn1] + bank_styles.bank, dim=1)),
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
        if ref_machine_state == MachineState.READ:
            if context_attn1 is None:
                context_attn1 = n
            bank_styles = self.injection_holder.bank_styles[ref_controlnets[0].order]
            style_fidelity = bank_styles.get_avg_style_fidelity()
            n_uc: Tensor = self.attn1(
                n,
                context=torch.cat([context_attn1] + bank_styles.bank, dim=1),
                value=torch.cat([value_attn1] + bank_styles.bank, dim=1) if value_attn1 is not None else value_attn1)
            n_c = n_uc.clone()
            if len(uc_idx_mask) > 0 and not math.isclose(style_fidelity, 0.0):
                n_c[uc_idx_mask] = self.attn1(
                    n[uc_idx_mask],
                    context=context_attn1[uc_idx_mask] if context_attn1 is not None else context_attn1,
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


# DFS Search for Torch.nn.Module, Written by Lvmin
def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result
