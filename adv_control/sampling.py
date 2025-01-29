from typing import Callable, Union

import comfy.hooks
import comfy.model_patcher
import comfy.patcher_extension
import comfy.sample
import comfy.samplers
from comfy.model_patcher import ModelPatcher
from comfy.controlnet import ControlBase
from comfy.ldm.modules.attention import BasicTransformerBlock


from .control import convert_all_to_advanced, restore_all_controlnet_conns
from .control_reference import (ReferenceAdvanced, ReferenceInjections,
                                RefBasicTransformerBlock, RefTimestepEmbedSequential,
                                InjectionBasicTransformerBlockHolder, InjectionTimestepEmbedSequentialHolder,
                                _forward_inject_BasicTransformerBlock,
                                handle_context_ref_setup, handle_reference_injection,
                                REF_CONTROL_LIST_ALL, CONTEXTREF_CLEAN_FUNC)
from .dinklink import get_dinklink
from .utils import torch_dfs, WrapperConsts, CURRENT_WRAPPER_VERSION

def prepare_dinklink_acn_wrapper():
    # expose acn_sampler_sample_wrapper
    d = get_dinklink()
    link_acn = d.setdefault(WrapperConsts.ACN, {})
    link_acn[WrapperConsts.VERSION] = CURRENT_WRAPPER_VERSION
    link_acn[WrapperConsts.ACN_CREATE_SAMPLER_SAMPLE_WRAPPER] = (comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                                                 WrapperConsts.ACN_OUTER_SAMPLE_WRAPPER_KEY,
                                                                 acn_outer_sample_wrapper)


def support_sliding_context_windows(conds) -> tuple[bool, list[dict]]:
    # convert to advanced, with report if anything was actually modified
    modified, new_conds = convert_all_to_advanced(conds)
    return modified, new_conds


def has_sliding_context_windows(model: ModelPatcher):
    params = model.get_attachment("ADE_params")
    if params is None:
        # backwards compatibility
        params = getattr(model, "motion_injection_params", None)
        if params is None:
            return False
    context_options = getattr(params, "context_options")
    return context_options.context_length is not None


def get_contextref_obj(model: ModelPatcher):
    params = model.get_attachment("ADE_params")
    if params is None:
        # backwards compatibility
        params = getattr(model, "motion_injection_params", None)
        if params is None:
            return None
    context_options = getattr(params, "context_options")
    extras = getattr(context_options, "extras", None)
    if extras is None:
        return None
    return getattr(extras, "context_ref", None)


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


def should_register_outer_sample_wrapper(hook, model, model_options: dict, target, registered: list):
    wrappers = comfy.patcher_extension.get_wrappers_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                                  WrapperConsts.ACN_OUTER_SAMPLE_WRAPPER_KEY,
                                                  model_options, is_model_options=True)
    return len(wrappers) == 0

def create_wrapper_hooks():
    wrappers = {}
    comfy.patcher_extension.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                                 WrapperConsts.ACN_OUTER_SAMPLE_WRAPPER_KEY,
                                                 acn_outer_sample_wrapper,
                                                 transformer_options=wrappers)
    hooks = comfy.hooks.HookGroup()
    hook = comfy.hooks.WrapperHook(wrappers)
    hook.hook_id = WrapperConsts.ACN_OUTER_SAMPLE_WRAPPER_KEY
    hook.custom_should_register = should_register_outer_sample_wrapper
    hooks.add(hook)
    return hooks

def acn_outer_sample_wrapper(executor, *args, **kwargs):
    controlnets_modified = False
    guider: comfy.samplers.CFGGuider = executor.class_obj
    model = guider.model_patcher
    orig_conds = guider.conds
    orig_model_options = guider.model_options
    try:
        new_model_options = orig_model_options
        # if context options present, perform some special actions that may be required
        context_refs = []
        if has_sliding_context_windows(guider.model_patcher):
            new_model_options = comfy.model_patcher.create_model_options_clone(new_model_options)
            # convert all CNs to Advanced if needed
            controlnets_modified, conds = support_sliding_context_windows(orig_conds)
            if controlnets_modified:
                guider.conds = conds
            # enable ContextRef, if requested
            existing_contextref_obj = get_contextref_obj(guider.model_patcher)
            if existing_contextref_obj is not None:
                context_refs = handle_context_ref_setup(existing_contextref_obj, new_model_options["transformer_options"], guider.conds)
                controlnets_modified = True
        # look for Advanced ControlNets that will require intervention to work
        ref_set = set()
        for outer_cond in guider.conds.values():
            for cond in outer_cond:
                if "control" in cond:
                    ref_set.update(get_refcn(cond["control"]))
        # if no ref cn found, do original function immediately
        if len(ref_set) == 0 and len(context_refs) == 0:
            return executor(*args, **kwargs)
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

            # store ordered ref cns in model's transformer options
            new_model_options = comfy.model_patcher.create_model_options_clone(new_model_options)
            # handle diffusion_model forward injection
            handle_reference_injection(new_model_options, reference_injections)

            ref_list: list[ReferenceAdvanced] = list(ref_set)
            new_model_options["transformer_options"][REF_CONTROL_LIST_ALL] = sorted(ref_list, key=lambda x: x.order)
            new_model_options["transformer_options"][CONTEXTREF_CLEAN_FUNC] = reference_injections.clean_contextref_module_mem
            guider.model_options = new_model_options
            # continue with original function
            return executor(*args, **kwargs)
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
            # cleanup
            reference_injections.cleanup()
    finally:
        # restore model_options
        guider.model_options = orig_model_options
        # restore guider.conds
        guider.conds = orig_conds
        # restore controlnets in conds, if needed
        if controlnets_modified:
            restore_all_controlnet_conns(guider.conds)
        del orig_conds
        del orig_model_options
        del model
        del guider
