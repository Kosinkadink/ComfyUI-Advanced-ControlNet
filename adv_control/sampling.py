from typing import Callable, Union

import comfy.sample
from comfy.model_patcher import ModelPatcher
from comfy.controlnet import ControlBase
from comfy.ldm.modules.attention import BasicTransformerBlock


from .control import convert_all_to_advanced, restore_all_controlnet_conns
from .control_reference import (ReferenceAdvanced, ReferenceInjections,
                                RefBasicTransformerBlock, RefTimestepEmbedSequential,
                                InjectionBasicTransformerBlockHolder, InjectionTimestepEmbedSequentialHolder,
                                _forward_inject_BasicTransformerBlock, factory_forward_inject_UNetModel,
                                handle_context_ref_setup,
                                REF_CONTROL_LIST_ALL, CONTEXTREF_CLEAN_FUNC)
from .control_lllite import (ControlLLLiteAdvanced)
from .utils import torch_dfs


def support_sliding_context_windows(model, positive, negative) -> tuple[bool, dict, dict]:
    # convert to advanced, with report if anything was actually modified
    modified, new_conds = convert_all_to_advanced([positive, negative])
    positive, negative = new_conds
    return modified, positive, negative


def has_sliding_context_windows(model):
    motion_injection_params = getattr(model, "motion_injection_params", None)
    if motion_injection_params is None:
        return False
    context_options = getattr(motion_injection_params, "context_options")
    return context_options.context_length is not None


def get_contextref_obj(model):
    motion_injection_params = getattr(model, "motion_injection_params", None)
    if motion_injection_params is None:
        return None
    context_options = getattr(motion_injection_params, "context_options")
    extras = getattr(context_options, "extras", None)
    if extras is None:
        return None
    return getattr(extras, "context_ref", None)


def acn_sample_factory(orig_comfy_sample: Callable, is_custom=False) -> Callable:
    def get_refcn(control: ControlBase, order: int=-1):
        ref_set: set[ReferenceAdvanced] = set()
        if control is None:
            return ref_set
        if type(control) == ReferenceAdvanced and not control.is_context_ref:
            control.order = order
            order -= 1
            ref_set.add(control)
        ref_set.update(get_refcn(control.previous_controlnet, order=order))
        return ref_set

    def get_lllitecn(control: ControlBase):
        cn_dict: dict[ControlLLLiteAdvanced,None] = {}
        if control is None:
            return cn_dict
        if type(control) == ControlLLLiteAdvanced:
            cn_dict[control] = None
        cn_dict.update(get_lllitecn(control.previous_controlnet))
        return cn_dict

    def acn_sample(model: ModelPatcher, *args, **kwargs):
        controlnets_modified = False
        orig_positive = args[-3]
        orig_negative = args[-2]
        try:
            orig_model_options = model.model_options
            # check if positive or negative conds contain ref cn
            positive = args[-3]
            negative = args[-2]
            # if context options present, perform some special actions that may be required
            context_refs = []
            if has_sliding_context_windows(model):
                model.model_options = model.model_options.copy()
                model.model_options["transformer_options"] = model.model_options["transformer_options"].copy()
                # convert all CNs to Advanced if needed
                controlnets_modified, positive, negative = support_sliding_context_windows(model, positive, negative)
                if controlnets_modified:
                    args = list(args)
                    args[-3] = positive
                    args[-2] = negative
                    args = tuple(args)
                # enable ContextRef, if requested
                existing_contextref_obj = get_contextref_obj(model)
                if existing_contextref_obj is not None:
                    context_refs = handle_context_ref_setup(existing_contextref_obj, model.model_options["transformer_options"], positive, negative)
                    controlnets_modified = True
            # look for Advanced ControlNets that will require intervention to work
            ref_set = set()
            lllite_dict: dict[ControlLLLiteAdvanced, None] = {} # dicts preserve insertion order since py3.7
            if positive is not None:
                for cond in positive:
                    if "control" in cond[1]:
                        ref_set.update(get_refcn(cond[1]["control"]))
                        lllite_dict.update(get_lllitecn(cond[1]["control"]))
            if negative is not None:
                for cond in negative:
                    if "control" in cond[1]:
                        ref_set.update(get_refcn(cond[1]["control"]))
                        lllite_dict.update(get_lllitecn(cond[1]["control"]))
            # if lllite found, apply patches to a cloned model_options, and continue
            if len(lllite_dict) > 0:
                lllite_list = list(lllite_dict.keys())
                model.model_options = model.model_options.copy()
                model.model_options["transformer_options"] = model.model_options["transformer_options"].copy()
                lllite_list.reverse() # reverse so that patches will be applied in expected order
                for lll in lllite_list:
                    lll.live_model_patches(model.model_options)
            # if no ref cn found, do original function immediately
            if len(ref_set) == 0 and len(context_refs) == 0:
                return orig_comfy_sample(model, *args, **kwargs)
            # otherwise, injection time
            try:
                # inject
                # storage for all Reference-related injections
                reference_injections = ReferenceInjections()

                # first, handle attn module injection
                all_modules = torch_dfs(model.model)
                attn_modules: list[RefBasicTransformerBlock] = []
                for module in all_modules:
                    if isinstance(module, BasicTransformerBlock):
                        attn_modules.append(module)
                attn_modules = [module for module in all_modules if isinstance(module, BasicTransformerBlock)]
                attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])
                for i, module in enumerate(attn_modules):
                    injection_holder = InjectionBasicTransformerBlockHolder(block=module, idx=i)
                    injection_holder.attn_weight = float(i) / float(len(attn_modules))
                    if hasattr(module, "_forward"): # backward compatibility
                        module._forward = _forward_inject_BasicTransformerBlock.__get__(module, type(module))
                    else:
                        module.forward = _forward_inject_BasicTransformerBlock.__get__(module, type(module))
                    module.injection_holder = injection_holder
                    reference_injections.attn_modules.append(module)
                # figure out which module is middle block
                if hasattr(model.model.diffusion_model, "middle_block"):
                    mid_modules = torch_dfs(model.model.diffusion_model.middle_block)
                    mid_attn_modules: list[RefBasicTransformerBlock] = [module for module in mid_modules if isinstance(module, BasicTransformerBlock)]
                    for module in mid_attn_modules:
                        module.injection_holder.is_middle = True

                # next, handle gn module injection (TimestepEmbedSequential)
                # TODO: figure out the logic behind these hardcoded indexes
                if type(model.model).__name__ == "SDXL":
                    input_block_indices = [4, 5, 7, 8]
                    output_block_indices = [0, 1, 2, 3, 4, 5]
                else:
                    input_block_indices = [4, 5, 7, 8, 10, 11]
                    output_block_indices = [0, 1, 2, 3, 4, 5, 6, 7]
                if hasattr(model.model.diffusion_model, "middle_block"):
                    module = model.model.diffusion_model.middle_block
                    injection_holder = InjectionTimestepEmbedSequentialHolder(block=module, idx=0, is_middle=True)
                    injection_holder.gn_weight = 0.0
                    module.injection_holder = injection_holder
                    reference_injections.gn_modules.append(module)
                for w, i in enumerate(input_block_indices):
                    module = model.model.diffusion_model.input_blocks[i]
                    injection_holder = InjectionTimestepEmbedSequentialHolder(block=module, idx=i, is_input=True)
                    injection_holder.gn_weight = 1.0 - float(w) / float(len(input_block_indices))
                    module.injection_holder = injection_holder
                    reference_injections.gn_modules.append(module)
                for w, i in enumerate(output_block_indices):
                    module = model.model.diffusion_model.output_blocks[i]
                    injection_holder = InjectionTimestepEmbedSequentialHolder(block=module, idx=i, is_output=True)
                    injection_holder.gn_weight = float(w) / float(len(output_block_indices))
                    module.injection_holder = injection_holder
                    reference_injections.gn_modules.append(module)
                # hack gn_module forwards and update weights
                for i, module in enumerate(reference_injections.gn_modules):
                    module.injection_holder.gn_weight *= 2

                # handle diffusion_model forward injection
                reference_injections.diffusion_model_orig_forward = model.model.diffusion_model.forward
                model.model.diffusion_model.forward = factory_forward_inject_UNetModel(reference_injections).__get__(model.model.diffusion_model, type(model.model.diffusion_model))
                # store ordered ref cns in model's transformer options
                new_model_options = model.model_options.copy()
                new_model_options["transformer_options"] = model.model_options["transformer_options"].copy()
                ref_list: list[ReferenceAdvanced] = list(ref_set)
                new_model_options["transformer_options"][REF_CONTROL_LIST_ALL] = sorted(ref_list, key=lambda x: x.order)
                new_model_options["transformer_options"][CONTEXTREF_CLEAN_FUNC] = reference_injections.clean_contextref_module_mem
                model.model_options = new_model_options
                # continue with original function
                return orig_comfy_sample(model, *args, **kwargs)
            finally:
                # cleanup injections
                # restore attn modules
                attn_modules: list[RefBasicTransformerBlock] = reference_injections.attn_modules
                for module in attn_modules:
                    module.injection_holder.restore(module)
                    module.injection_holder.clean_all()
                    del module.injection_holder
                del attn_modules
                # restore gn modules
                gn_modules: list[RefTimestepEmbedSequential] = reference_injections.gn_modules
                for module in gn_modules:
                    module.injection_holder.restore(module)
                    module.injection_holder.clean_all()
                    del module.injection_holder
                del gn_modules
                # restore diffusion_model forward function
                model.model.diffusion_model.forward = reference_injections.diffusion_model_orig_forward.__get__(model.model.diffusion_model, type(model.model.diffusion_model))
                # cleanup
                reference_injections.cleanup()
        finally:
            # restore model_options
            model.model_options = orig_model_options
            # restore controlnets in conds, if needed
            if controlnets_modified:
                restore_all_controlnet_conns([orig_positive, orig_negative])

    return acn_sample
