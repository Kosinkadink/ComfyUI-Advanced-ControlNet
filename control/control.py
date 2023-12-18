from typing import Callable, Union
from torch import Tensor
import torch
import os

import comfy.utils
import comfy.model_management
import comfy.model_detection
import comfy.controlnet as comfy_cn
from comfy.controlnet import ControlBase, ControlNet, ControlLora, T2IAdapter, broadcast_image_to

from .control_sparsectrl import SparseControlNet, SparseCtrlMotionWrapper
from .utils import (TimestepKeyframeGroup, LatentKeyframeGroup, ControlWeightType, ControlWeights, WeightTypeException,
                    manual_cast_clean_groupnorm, disable_weight_init_clean_groupnorm, prepare_mask_batch, get_properly_arranged_t2i_weights, load_torch_file_with_dict_factory)
from .logger import logger


class AdvancedControlBase:
    def __init__(self, base: ControlBase, timestep_keyframes: TimestepKeyframeGroup, weights_default: ControlWeights):
        self.base = base
        self.compatible_weights = [ControlWeightType.UNIVERSAL]
        self.add_compatible_weight(weights_default.weight_type)
        # mask for which parts of controlnet output to keep
        self.mask_cond_hint_original = None
        self.mask_cond_hint = None
        self.tk_mask_cond_hint_original = None
        self.tk_mask_cond_hint = None
        self.weight_mask_cond_hint = None
        # actual index values
        self.sub_idxs = None
        self.full_latent_length = 0
        self.context_length = 0
        # timesteps
        self.t: Tensor = None
        self.batched_number: int = None
        # weights + override
        self.weights: ControlWeights = None
        self.weights_default: ControlWeights = weights_default
        self.weights_override: ControlWeights = None
        # latent keyframe + override
        self.latent_keyframes: LatentKeyframeGroup = None
        self.latent_keyframe_override: LatentKeyframeGroup = None
        # initialize timestep_keyframes
        self.set_timestep_keyframes(timestep_keyframes)
        # override some functions
        self.get_control = self.get_control_inject
        self.control_merge = self.control_merge_inject#.__get__(self, type(self))
        self.pre_run = self.pre_run_inject
        self.cleanup = self.cleanup_inject

    def add_compatible_weight(self, control_weight_type: str):
        self.compatible_weights.append(control_weight_type)

    def verify_all_weights(self, throw_error=True):
        # first, check if override exists - if so, only need to check the override
        if self.weights_override is not None:
            if self.weights_override.weight_type not in self.compatible_weights:
                msg = f"Weight override is type {self.weights_override.weight_type}, but loaded {type(self).__name__}" + \
                    f"only supports {self.compatible_weights} weights."
                raise WeightTypeException(msg)
        # otherwise, check all timestep keyframe weights
        else:
            for tk in self.timestep_keyframes.keyframes:
                if tk.has_control_weights() and tk.control_weights.weight_type not in self.compatible_weights:
                    msg = f"Weight on Timestep Keyframe with start_percent={tk.start_percent} is type" + \
                        f"{tk.control_weights.weight_type}, but loaded {type(self).__name__} only supports {self.compatible_weights} weights."
                    raise WeightTypeException(msg)

    def set_timestep_keyframes(self, timestep_keyframes: TimestepKeyframeGroup):
        self.timestep_keyframes = timestep_keyframes if timestep_keyframes else TimestepKeyframeGroup()
        # prepare first timestep_keyframe related stuff
        self.current_timestep_keyframe = None
        self.current_timestep_index = -1
        self.next_timestep_keyframe = None
        self.weights = None
        self.latent_keyframes = None

    def prepare_current_timestep(self, t: Tensor, batched_number: int):
        self.t = t
        self.batched_number = batched_number
        # get current step percent
        curr_t: float = t[0]
        prev_index = self.current_timestep_index
        # if has next index, loop through and see if need to switch
        if self.timestep_keyframes.has_index(self.current_timestep_index+1):
            for i in range(self.current_timestep_index+1, len(self.timestep_keyframes)):
                eval_tk = self.timestep_keyframes[i]
                # check if start percent is less or equal to curr_t
                if eval_tk.start_t >= curr_t:
                    self.current_timestep_index = i
                    self.current_timestep_keyframe = eval_tk
                    # keep track of control weights, latent keyframes, and masks,
                    # accounting for inherit_missing
                    if self.current_timestep_keyframe.has_control_weights():
                        self.weights = self.current_timestep_keyframe.control_weights
                    elif not self.current_timestep_keyframe.inherit_missing:
                        self.weights = self.weights_default
                    if self.current_timestep_keyframe.has_latent_keyframes():
                        self.latent_keyframes = self.current_timestep_keyframe.latent_keyframes
                    elif not self.current_timestep_keyframe.inherit_missing:
                        self.latent_keyframes = None
                    if self.current_timestep_keyframe.has_mask_hint():
                        self.tk_mask_cond_hint_original = self.current_timestep_keyframe.mask_hint_orig
                    elif not self.current_timestep_keyframe.inherit_missing:
                        del self.tk_mask_cond_hint_original
                        self.tk_mask_cond_hint_original = None
                    # if guarantee_usage, stop searching for other TKs
                    if self.current_timestep_keyframe.guarantee_usage:
                        break
                # if eval_tk is outside of percent range, stop looking further
                else:
                    break
        
        # if index changed, apply overrides
        if prev_index != self.current_timestep_index:
            if self.weights_override is not None:
                self.weights = self.weights_override
            if self.latent_keyframe_override is not None:
                self.latent_keyframes = self.latent_keyframe_override

        # make sure weights and latent_keyframes are in a workable state
        # Note: each AdvancedControlBase should create their own get_universal_weights class
        self.prepare_weights()
    
    def prepare_weights(self):
        if self.weights is None or self.weights.weight_type == ControlWeightType.DEFAULT:
            self.weights = self.weights_default
        elif self.weights.weight_type == ControlWeightType.UNIVERSAL:
            # if universal and weight_mask present, no need to convert
            if self.weights.weight_mask is not None:
                return
            self.weights = self.get_universal_weights()
    
    def get_universal_weights(self) -> ControlWeights:
        return self.weights

    def set_cond_hint_mask(self, mask_hint):
        self.mask_cond_hint_original = mask_hint
        return self

    def pre_run_inject(self, model, percent_to_timestep_function):
        self.base.pre_run(model, percent_to_timestep_function)
        self.pre_run_advanced(model, percent_to_timestep_function)
    
    def pre_run_advanced(self, model, percent_to_timestep_function):
        # for each timestep keyframe, calculate the start_t
        for tk in self.timestep_keyframes.keyframes:
            tk.start_t = percent_to_timestep_function(tk.start_percent)
        # clear variables
        self.cleanup_advanced()

    def get_control_inject(self, x_noisy, t, cond, batched_number):
        # prepare timestep and everything related
        self.prepare_current_timestep(t=t, batched_number=batched_number)
        # if should not perform any actions for the controlnet, exit without doing any work
        if self.strength == 0.0 or self.current_timestep_keyframe.strength == 0.0:
            control_prev = None
            if self.previous_controlnet is not None:
                control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)
            if control_prev is not None:
                return control_prev
            else:
                return None
        # otherwise, perform normal function
        return self.get_control_advanced(x_noisy, t, cond, batched_number)

    def get_control_advanced(self, x_noisy, t, cond, batched_number):
        pass

    def calc_weight(self, idx: int, x: Tensor, layers: int) -> Union[float, Tensor]:
        if self.weights.weight_mask is not None:
            # prepare weight mask
            self.prepare_weight_mask_cond_hint(x, self.batched_number)
            # adjust mask for current layer and return
            return torch.pow(self.weight_mask_cond_hint, self.get_calc_pow(idx=idx, layers=layers))
        return self.weights.get(idx=idx)
    
    def get_calc_pow(self, idx: int, layers: int) -> int:
        return (layers-1)-idx

    def apply_advanced_strengths_and_masks(self, x: Tensor, batched_number: int):
        # apply strengths, and get batch indeces to null out
        # AKA latents that should not be influenced by ControlNet
        if self.latent_keyframes is not None:
            latent_count = x.size(0)//batched_number
            indeces_to_null = set(range(latent_count))
            mapped_indeces = None
            # if expecting subdivision, will need to translate between subset and actual idx values
            if self.sub_idxs:
                mapped_indeces = {}
                for i, actual in enumerate(self.sub_idxs):
                    mapped_indeces[actual] = i
            for keyframe in self.latent_keyframes:
                real_index = keyframe.batch_index
                # if negative, count from end
                if real_index < 0:
                    real_index += latent_count if self.sub_idxs is None else self.full_latent_length

                # if not mapping indeces, what you see is what you get
                if mapped_indeces is None:
                    if real_index in indeces_to_null:
                        indeces_to_null.remove(real_index)
                # otherwise, see if batch_index is even included in this set of latents
                else:
                    real_index = mapped_indeces.get(real_index, None)
                    if real_index is None:
                        continue
                    indeces_to_null.remove(real_index)

                # if real_index is outside the bounds of latents, don't apply
                if real_index >= latent_count or real_index < 0:
                    continue

                # apply strength for each batched cond/uncond
                for b in range(batched_number):
                    x[(latent_count*b)+real_index] = x[(latent_count*b)+real_index] * keyframe.strength

            # null them out by multiplying by null_latent_kf_strength
            for batch_index in indeces_to_null:
                # apply null for each batched cond/uncond
                for b in range(batched_number):
                    x[(latent_count*b)+batch_index] = x[(latent_count*b)+batch_index] * self.current_timestep_keyframe.null_latent_kf_strength
        # apply masks, resizing mask to required dims
        if self.mask_cond_hint is not None:
            masks = prepare_mask_batch(self.mask_cond_hint, x.shape)
            x[:] = x[:] * masks
        if self.tk_mask_cond_hint is not None:
            masks = prepare_mask_batch(self.tk_mask_cond_hint, x.shape)
            x[:] = x[:] * masks
        # apply timestep keyframe strengths
        if self.current_timestep_keyframe.strength != 1.0:
            x[:] *= self.current_timestep_keyframe.strength
    
    def control_merge_inject(self: 'AdvancedControlBase', control_input, control_output, control_prev, output_dtype):
        out = {'input':[], 'middle':[], 'output': []}

        if control_input is not None:
            for i in range(len(control_input)):
                key = 'input'
                x = control_input[i]
                if x is not None:
                    self.apply_advanced_strengths_and_masks(x, self.batched_number)

                    x *= self.strength * self.calc_weight(i, x, len(control_input))
                    if x.dtype != output_dtype:
                        x = x.to(output_dtype)
                out[key].insert(0, x)

        if control_output is not None:
            for i in range(len(control_output)):
                if i == (len(control_output) - 1):
                    key = 'middle'
                    index = 0
                else:
                    key = 'output'
                    index = i
                x = control_output[i]
                if x is not None:
                    self.apply_advanced_strengths_and_masks(x, self.batched_number)

                    if self.global_average_pooling:
                        x = torch.mean(x, dim=(2, 3), keepdim=True).repeat(1, 1, x.shape[2], x.shape[3])

                    x *= self.strength * self.calc_weight(i, x, len(control_output))
                    if x.dtype != output_dtype:
                        x = x.to(output_dtype)

                out[key].append(x)
        if control_prev is not None:
            for x in ['input', 'middle', 'output']:
                o = out[x]
                for i in range(len(control_prev[x])):
                    prev_val = control_prev[x][i]
                    if i >= len(o):
                        o.append(prev_val)
                    elif prev_val is not None:
                        if o[i] is None:
                            o[i] = prev_val
                        else:
                            o[i] += prev_val
        return out

    def prepare_mask_cond_hint(self, x_noisy: Tensor, t, cond, batched_number, dtype=None):
        self._prepare_mask("mask_cond_hint", self.mask_cond_hint_original, x_noisy, t, cond, batched_number, dtype)
        self.prepare_tk_mask_cond_hint(x_noisy, t, cond, batched_number, dtype)

    def prepare_tk_mask_cond_hint(self, x_noisy: Tensor, t, cond, batched_number, dtype=None):
        return self._prepare_mask("tk_mask_cond_hint", self.current_timestep_keyframe.mask_hint_orig, x_noisy, t, cond, batched_number, dtype)

    def prepare_weight_mask_cond_hint(self, x_noisy: Tensor, batched_number, dtype=None):
        return self._prepare_mask("weight_mask_cond_hint", self.weights.weight_mask, x_noisy, t=None, cond=None, batched_number=batched_number, dtype=dtype, direct_attn=True)

    def _prepare_mask(self, attr_name, orig_mask: Tensor, x_noisy: Tensor, t, cond, batched_number, dtype=None, direct_attn=False):
        # make mask appropriate dimensions, if present
        if orig_mask is not None:
            out_mask = getattr(self, attr_name)
            if self.sub_idxs is not None or out_mask is None or x_noisy.shape[2] * 8 != out_mask.shape[1] or x_noisy.shape[3] * 8 != out_mask.shape[2]:
                self._reset_attr(attr_name)
                del out_mask
                # TODO: perform upscale on only the sub_idxs masks at a time instead of all to conserve RAM
                # resize mask and match batch count
                multiplier = 1 if direct_attn else 8
                out_mask = prepare_mask_batch(orig_mask, x_noisy.shape, multiplier=multiplier)
                actual_latent_length = x_noisy.shape[0] // batched_number
                out_mask = comfy.utils.repeat_to_batch_size(out_mask, actual_latent_length if self.sub_idxs is None else self.full_latent_length)
                if self.sub_idxs is not None:
                    out_mask = out_mask[self.sub_idxs]
            # make cond_hint_mask length match x_noise
            if x_noisy.shape[0] != out_mask.shape[0]:
                out_mask = broadcast_image_to(out_mask, x_noisy.shape[0], batched_number)
            # default dtype to be same as x_noisy
            if dtype is None:
                dtype = x_noisy.dtype
            setattr(self, attr_name, out_mask.to(dtype=dtype).to(self.device))
            del out_mask

    def _reset_attr(self, attr_name, new_value=None):
        if hasattr(self, attr_name):
            delattr(self, attr_name)
        setattr(self, attr_name, new_value)

    def cleanup_inject(self):
        self.base.cleanup()
        self.cleanup_advanced()

    def cleanup_advanced(self):
        self.sub_idxs = None
        self.full_latent_length = 0
        self.context_length = 0
        self.t = None
        self.batched_number = None
        self.weights = None
        self.latent_keyframes = None
        # timestep stuff
        self.current_timestep_keyframe = None
        self.next_timestep_keyframe = None
        self.current_timestep_index = -1
        # clear mask hints
        if self.mask_cond_hint is not None:
            del self.mask_cond_hint
            self.mask_cond_hint = None
        if self.tk_mask_cond_hint_original is not None:
            del self.tk_mask_cond_hint_original
            self.tk_mask_cond_hint_original = None
        if self.tk_mask_cond_hint is not None:
            del self.tk_mask_cond_hint
            self.tk_mask_cond_hint = None
        if self.weight_mask_cond_hint is not None:
            del self.weight_mask_cond_hint
            self.weight_mask_cond_hint = None
    
    def copy_to_advanced(self, copied: 'AdvancedControlBase'):
        copied.mask_cond_hint_original = self.mask_cond_hint_original
        copied.weights_override = self.weights_override
        copied.latent_keyframe_override = self.latent_keyframe_override


class ControlNetAdvanced(ControlNet, AdvancedControlBase):
    def __init__(self, control_model, timestep_keyframes: TimestepKeyframeGroup, global_average_pooling=False, device=None, load_device=None, manual_cast_dtype=None):
        super().__init__(control_model=control_model, global_average_pooling=global_average_pooling, device=device, load_device=load_device, manual_cast_dtype=manual_cast_dtype)
        AdvancedControlBase.__init__(self, super(), timestep_keyframes=timestep_keyframes, weights_default=ControlWeights.controlnet())

    def get_universal_weights(self) -> ControlWeights:
        raw_weights = [(self.weights.base_multiplier ** float(12 - i)) for i in range(13)]
        return ControlWeights.controlnet(raw_weights, self.weights.flip_weights)

    def get_control_advanced(self, x_noisy, t, cond, batched_number):
        # perform special version of get_control that supports sliding context and masks
        return self.sliding_get_control(x_noisy, t, cond, batched_number)

    def sliding_get_control(self, x_noisy: Tensor, t, cond, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return None

        dtype = self.control_model.dtype
        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        output_dtype = x_noisy.dtype
        # make cond_hint appropriate dimensions
        # TODO: change this to not require cond_hint upscaling every step when self.sub_idxs are present
        if self.sub_idxs is not None or self.cond_hint is None or x_noisy.shape[2] * 8 != self.cond_hint.shape[2] or x_noisy.shape[3] * 8 != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            # if self.cond_hint_original length greater or equal to real latent count, subdivide it before scaling
            if self.sub_idxs is not None and self.cond_hint_original.size(0) >= self.full_latent_length:
                self.cond_hint = comfy.utils.common_upscale(self.cond_hint_original[self.sub_idxs], x_noisy.shape[3] * 8, x_noisy.shape[2] * 8, 'nearest-exact', "center").to(dtype).to(self.device)
            else:
                self.cond_hint = comfy.utils.common_upscale(self.cond_hint_original, x_noisy.shape[3] * 8, x_noisy.shape[2] * 8, 'nearest-exact', "center").to(dtype).to(self.device)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)

        # prepare mask_cond_hint
        self.prepare_mask_cond_hint(x_noisy=x_noisy, t=t, cond=cond, batched_number=batched_number, dtype=dtype)

        context = cond['c_crossattn']
        # uses 'y' in new ComfyUI update
        y = cond.get('y', None)
        if y is None: # TODO: remove this in the future since no longer used by newest ComfyUI
            y = cond.get('c_adm', None)
        if y is not None:
            y = y.to(dtype)
        timestep = self.model_sampling_current.timestep(t)
        x_noisy = self.model_sampling_current.calculate_input(t, x_noisy)

        control = self.control_model(x=x_noisy.to(dtype), hint=self.cond_hint, timesteps=timestep.float(), context=context.to(dtype), y=y)
        return self.control_merge(None, control, control_prev, output_dtype)

    def copy(self):
        c = ControlNetAdvanced(self.control_model, self.timestep_keyframes, global_average_pooling=self.global_average_pooling, load_device=self.load_device, manual_cast_dtype=self.manual_cast_dtype)
        self.copy_to(c)
        self.copy_to_advanced(c)
        return c
    
    @staticmethod
    def from_vanilla(v: ControlNet, timestep_keyframe: TimestepKeyframeGroup=None) -> 'ControlNetAdvanced':
        return ControlNetAdvanced(control_model=v.control_model, timestep_keyframes=timestep_keyframe,
                                  global_average_pooling=v.global_average_pooling, device=v.device, load_device=v.load_device, manual_cast_dtype=v.manual_cast_dtype)


class T2IAdapterAdvanced(T2IAdapter, AdvancedControlBase):
    def __init__(self, t2i_model, timestep_keyframes: TimestepKeyframeGroup, channels_in, device=None):
        super().__init__(t2i_model=t2i_model, channels_in=channels_in, device=device)
        AdvancedControlBase.__init__(self, super(), timestep_keyframes=timestep_keyframes, weights_default=ControlWeights.t2iadapter())

    def get_universal_weights(self) -> ControlWeights:
        raw_weights = [(self.weights.base_multiplier ** float(7 - i)) for i in range(8)]
        raw_weights = [raw_weights[-8], raw_weights[-3], raw_weights[-2], raw_weights[-1]]
        raw_weights = get_properly_arranged_t2i_weights(raw_weights)
        return ControlWeights.t2iadapter(raw_weights, self.weights.flip_weights)

    def get_calc_pow(self, idx: int, layers: int) -> int:
        # match how T2IAdapterAdvanced deals with universal weights
        indeces = [7 - i for i in range(8)]
        indeces = [indeces[-8], indeces[-3], indeces[-2], indeces[-1]]
        indeces = get_properly_arranged_t2i_weights(indeces)
        return indeces[idx]

    def get_control_advanced(self, x_noisy, t, cond, batched_number):
        # prepare timestep and everything related
        self.prepare_current_timestep(t=t, batched_number=batched_number)
        try:
            # if sub indexes present, replace original hint with subsection
            if self.sub_idxs is not None:
                # cond hints
                full_cond_hint_original = self.cond_hint_original
                del self.cond_hint
                self.cond_hint = None
                self.cond_hint_original = full_cond_hint_original[self.sub_idxs]
            # mask hints
            self.prepare_mask_cond_hint(x_noisy=x_noisy, t=t, cond=cond, batched_number=batched_number)
            return super().get_control(x_noisy, t, cond, batched_number)
        finally:
            if self.sub_idxs is not None:
                # replace original cond hint
                self.cond_hint_original = full_cond_hint_original
                del full_cond_hint_original

    def copy(self):
        c = T2IAdapterAdvanced(self.t2i_model, self.timestep_keyframes, self.channels_in)
        self.copy_to(c)
        self.copy_to_advanced(c)
        return c
    
    def cleanup(self):
        super().cleanup()
        self.cleanup_advanced()

    @staticmethod
    def from_vanilla(v: T2IAdapter, timestep_keyframe: TimestepKeyframeGroup=None) -> 'T2IAdapterAdvanced':
        return T2IAdapterAdvanced(t2i_model=v.t2i_model, timestep_keyframes=timestep_keyframe, channels_in=v.channels_in, device=v.device)


class ControlLoraAdvanced(ControlLora, AdvancedControlBase):
    def __init__(self, control_weights, timestep_keyframes: TimestepKeyframeGroup, global_average_pooling=False, device=None):
        super().__init__(control_weights=control_weights, global_average_pooling=global_average_pooling, device=device)
        AdvancedControlBase.__init__(self, super(), timestep_keyframes=timestep_keyframes, weights_default=ControlWeights.controllora())
        # use some functions from ControlNetAdvanced
        self.get_control_advanced = ControlNetAdvanced.get_control_advanced.__get__(self, type(self))
        self.sliding_get_control = ControlNetAdvanced.sliding_get_control.__get__(self, type(self))
    
    def get_universal_weights(self) -> ControlWeights:
        raw_weights = [(self.weights.base_multiplier ** float(9 - i)) for i in range(10)]
        return ControlWeights.controllora(raw_weights, self.weights.flip_weights)

    def copy(self):
        c = ControlLoraAdvanced(self.control_weights, self.timestep_keyframes, global_average_pooling=self.global_average_pooling)
        self.copy_to(c)
        self.copy_to_advanced(c)
        return c
    
    def cleanup(self):
        super().cleanup()
        self.cleanup_advanced()

    @staticmethod
    def from_vanilla(v: ControlLora, timestep_keyframe: TimestepKeyframeGroup=None) -> 'ControlLoraAdvanced':
        return ControlLoraAdvanced(control_weights=v.control_weights, timestep_keyframes=timestep_keyframe,
                                   global_average_pooling=v.global_average_pooling, device=v.device)


class ControlLLLiteAdvanced(ControlBase, AdvancedControlBase):
    # This ControlNet is more of an attention patch than a traditional controlnet
    # So, the pre_run will be responsible for a lot of the functionality,
    #     while the usual get_control is mostly used to set some values
    def __init__(self, timestep_keyframes: TimestepKeyframeGroup, device=None):
        super().__init__(device)
        AdvancedControlBase.__init__(self, super(), timestep_keyframes=timestep_keyframes, weights_default=ControlWeights.controllllite())
        self.already_patched = False

    def set_cond_hint(self, *args, **kwargs):
        super().set_cond_hint(*args, **kwargs)
        # cond hint for LLLite needs to be scaled between (-1, 1) instead of (0, 1)
        self.cond_hint_original = self.cond_hint_original * 2.0 - 1.0

    def pre_run_advanced(self, model, percent_to_timestep_function):
        AdvancedControlBase.pre_run_advanced(self, model, percent_to_timestep_function)
        logger.info(f"In ControlLLLiteAdvanced pre_run_advanced! {self.already_patched}")
        # perform patches if not already patches
        if not self.already_patched:
            self.already_patched = True

    def get_control(self, x_noisy: Tensor, t, cond, batched_number):
        logger.info("In ControlLLLiteAdvanced get_control!")
        # prepare timestep and everything related
        self.prepare_current_timestep(t=t, batched_number=batched_number)
        # perform other controlnets
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)
        if control_prev is not None:
            return control_prev
        else:
            return None
    
    def get_models(self):
        logger.info(f"In ControlLLLiteAdvanced get_models!")
        # get_models is called once at the start of every KSampler run - use to reset already_patched status
        self.already_patched = False
        out = super().get_models()
        return out

    def copy(self):
        c = ControlLLLiteAdvanced(self.timestep_keyframes)
        self.copy_to(c)
        self.copy_to_advanced(c)
        return c

    def cleanup(self):
        super().cleanup()
        self.cleanup_advanced()
        self.already_patched = False


class SparseCtrlAdvanced(ControlNetAdvanced):
    def __init__(self, control_model, timestep_keyframes: TimestepKeyframeGroup, global_average_pooling=False, device=None, load_device=None, manual_cast_dtype=None):
        super().__init__(control_model=control_model, timestep_keyframes=timestep_keyframes, global_average_pooling=global_average_pooling, device=device, load_device=load_device, manual_cast_dtype=manual_cast_dtype)
        self.add_compatible_weight(ControlWeightType.SPARSECTRL)
    
    def copy(self):
        c = SparseCtrlAdvanced(self.control_model, self.timestep_keyframes, self.global_average_pooling, self.device, self.load_device, self.manual_cast_dtype)
        self.copy_to(c)
        self.copy_to_advanced(c)
        return c


def load_controlnet(ckpt_path, timestep_keyframe: TimestepKeyframeGroup=None, model=None):
    controlnet_data = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    control = None
    # check if a non-vanilla ControlNet
    controlnet_type = ControlWeightType.DEFAULT
    has_controlnet_key = False
    has_motion_modules_key = False
    for key in controlnet_data:
        # LLLLite check
        if "lllite" in key:
            logger.info("ControlLLLite controlnet!")
            controlnet_type = ControlWeightType.CONTROLLLLITE
            break
        # SparseCtrl check
        elif "motion_modules" in key:
            has_motion_modules_key = True
        elif "controlnet" in key:
            has_controlnet_key = True
    if has_controlnet_key and has_motion_modules_key:
        controlnet_type = ControlWeightType.SPARSECTRL

    if controlnet_type != ControlWeightType.DEFAULT:
        if controlnet_type == ControlWeightType.CONTROLLLLITE:
            raise NotImplementedError("ControlLLLite has not been fully implemented yet!")
            control = ControlLLLiteAdvanced(timestep_keyframes=timestep_keyframe)
            # load Controll
        elif controlnet_type == ControlWeightType.SPARSECTRL:
            #raise NotImplementedError("SparseCtrl has not been fully implemented yet!")
            control = load_sparsectrl(ckpt_path, controlnet_data=controlnet_data, timestep_keyframe=timestep_keyframe, model=model)
    # otherwise, load vanilla ControlNet
    else:
        try:
            # hacky way of getting load_torch_file in load_controlnet to use already-present controlnet_data and not redo loading
            orig_load_torch_file = comfy.utils.load_torch_file
            comfy.utils.load_torch_file = load_torch_file_with_dict_factory(controlnet_data, orig_load_torch_file)
            control = comfy_cn.load_controlnet(ckpt_path, model=model)
        finally:
            comfy.utils.load_torch_file = orig_load_torch_file
    # from pathlib import Path
    # with open(Path(__file__).parent.parent.parent / "controlnet_keys.txt", "w") as cfile:
    #     controlnet_data = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    #     for key in controlnet_data:
    #         cfile.write(f"{key}\n")
    return convert_to_advanced(control, timestep_keyframe=timestep_keyframe)


def convert_to_advanced(control, timestep_keyframe: TimestepKeyframeGroup=None):
    # if already advanced, leave it be
    if is_advanced_controlnet(control):
        return control
    # if exactly ControlNet returned, transform it into ControlNetAdvanced
    if type(control) == ControlNet:
        return ControlNetAdvanced.from_vanilla(v=control, timestep_keyframe=timestep_keyframe)
    # if exactly ControlLora returned, transform it into ControlLoraAdvanced
    elif type(control) == ControlLora:
        return ControlLoraAdvanced.from_vanilla(v=control, timestep_keyframe=timestep_keyframe)
    # if T2IAdapter returned, transform it into T2IAdapterAdvanced
    elif isinstance(control, T2IAdapter):
        return T2IAdapterAdvanced.from_vanilla(v=control, timestep_keyframe=timestep_keyframe)
    # otherwise, leave it be - might be something I am not supporting yet
    return control


def is_advanced_controlnet(input_object):
    return hasattr(input_object, "sub_idxs")


def load_sparsectrl(ckpt_path: str, controlnet_data: dict[str, Tensor]=None, timestep_keyframe: TimestepKeyframeGroup=None, model=None) -> SparseCtrlAdvanced:
    if controlnet_data is None:
        controlnet_data = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    # first, separate out motion part from normal controlnet part and attempt to load that portion
    motion_data = {}
    for key in list(controlnet_data.keys()):
        if "temporal" in key:
            motion_data[key] = controlnet_data.pop(key)
    motion_wrapper: SparseCtrlMotionWrapper = SparseCtrlMotionWrapper(motion_data).to(comfy.model_management.unet_dtype())
    missing, unexpected = motion_wrapper.load_state_dict(motion_data)
    if len(missing) > 0 or len(unexpected) > 0:
        logger.info(f"SparseCtrlMotionWrapper: {missing}, {unexpected}")

    # now, load as if it was a normal controlnet - mostly copied from comfy load_controlnet function
    controlnet_config = None
    is_diffusers = False
    use_simplified_conditioning_embedding = False
    if "controlnet_cond_embedding.conv_in.weight" in controlnet_data:
        is_diffusers = True
    if "controlnet_cond_embedding.weight" in controlnet_data:
        is_diffusers = True
        use_simplified_conditioning_embedding = True
    if is_diffusers: #diffusers format
        unet_dtype = comfy.model_management.unet_dtype()
        controlnet_config = comfy.model_detection.unet_config_from_diffusers_unet(controlnet_data, unet_dtype)
        diffusers_keys = comfy.utils.unet_to_diffusers(controlnet_config)
        diffusers_keys["controlnet_mid_block.weight"] = "middle_block_out.0.weight"
        diffusers_keys["controlnet_mid_block.bias"] = "middle_block_out.0.bias"

        count = 0
        loop = True
        while loop:
            suffix = [".weight", ".bias"]
            for s in suffix:
                k_in = "controlnet_down_blocks.{}{}".format(count, s)
                k_out = "zero_convs.{}.0{}".format(count, s)
                if k_in not in controlnet_data:
                    loop = False
                    break
                diffusers_keys[k_in] = k_out
            count += 1
        # normal conditioning embedding
        if not use_simplified_conditioning_embedding:
            count = 0
            loop = True
            while loop:
                suffix = [".weight", ".bias"]
                for s in suffix:
                    if count == 0:
                        k_in = "controlnet_cond_embedding.conv_in{}".format(s)
                    else:
                        k_in = "controlnet_cond_embedding.blocks.{}{}".format(count - 1, s)
                    k_out = "input_hint_block.{}{}".format(count * 2, s)
                    if k_in not in controlnet_data:
                        k_in = "controlnet_cond_embedding.conv_out{}".format(s)
                        loop = False
                    diffusers_keys[k_in] = k_out
                count += 1
        # simplified conditioning embedding
        else:
            count = 0
            suffix = [".weight", ".bias"]
            for s in suffix:
                k_in = "controlnet_cond_embedding{}".format(s)
                k_out = "input_hint_block.{}{}".format(count, s)
                diffusers_keys[k_in] = k_out

        new_sd = {}
        for k in diffusers_keys:
            if k in controlnet_data:
                new_sd[diffusers_keys[k]] = controlnet_data.pop(k)

        leftover_keys = controlnet_data.keys()
        if len(leftover_keys) > 0:
            logger.info("leftover keys:", leftover_keys)
        controlnet_data = new_sd

    pth_key = 'control_model.zero_convs.0.0.weight'
    pth = False
    key = 'zero_convs.0.0.weight'
    if pth_key in controlnet_data:
        pth = True
        key = pth_key
        prefix = "control_model."
    elif key in controlnet_data:
        prefix = ""
    else:
        raise ValueError("The provided model is not a valid SparseCtrl model! [ErrorCode: HORSERADISH]")

    if controlnet_config is None:
        unet_dtype = comfy.model_management.unet_dtype()
        controlnet_config = comfy.model_detection.model_config_from_unet(controlnet_data, prefix, unet_dtype, True).unet_config
    load_device = comfy.model_management.get_torch_device()
    manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, load_device)
    if manual_cast_dtype is not None:
        controlnet_config["operations"] = manual_cast_clean_groupnorm
    else:
        controlnet_config["operations"] = disable_weight_init_clean_groupnorm
    controlnet_config.pop("out_channels")
    # get proper hint channels
    if use_simplified_conditioning_embedding:
        controlnet_config["hint_channels"] = controlnet_data["{}input_hint_block.0.weight".format(prefix)].shape[1]
        controlnet_config["use_simplified_conditioning_embedding"] = use_simplified_conditioning_embedding
    else:
        controlnet_config["hint_channels"] = controlnet_data["{}input_hint_block.0.weight".format(prefix)].shape[1]
        controlnet_config["use_simplified_conditioning_embedding"] = use_simplified_conditioning_embedding
    control_model = SparseControlNet(**controlnet_config)

    if pth:
        if 'difference' in controlnet_data:
            if model is not None:
                comfy.model_management.load_models_gpu([model])
                model_sd = model.model_state_dict()
                for x in controlnet_data:
                    c_m = "control_model."
                    if x.startswith(c_m):
                        sd_key = "diffusion_model.{}".format(x[len(c_m):])
                        if sd_key in model_sd:
                            cd = controlnet_data[x]
                            cd += model_sd[sd_key].type(cd.dtype).to(cd.device)
            else:
                logger.warning("WARNING: Loaded a diff SparseCtrl without a model. It will very likely not work.")

        class WeightsLoader(torch.nn.Module):
            pass
        w = WeightsLoader()
        w.control_model = control_model
        missing, unexpected = w.load_state_dict(controlnet_data, strict=False)
    else:
        missing, unexpected = control_model.load_state_dict(controlnet_data, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        logger.info(f"SparseCtrl ControlNet: {missing}, {unexpected}")

    global_average_pooling = False
    filename = os.path.splitext(ckpt_path)[0]
    if filename.endswith("_shuffle") or filename.endswith("_shuffle_fp16"): #TODO: smarter way of enabling global_average_pooling
        global_average_pooling = True

    # both motion portion and controlnet portions are loaded; bring them together
    motion_wrapper.inject(control_model)

    control = SparseCtrlAdvanced(control_model, timestep_keyframes=timestep_keyframe, global_average_pooling=global_average_pooling, load_device=load_device, manual_cast_dtype=manual_cast_dtype)
    new_state_dict = control_model.state_dict()
    from pathlib import Path
    with open(Path(__file__).parent.parent.parent / "sparcectrlstatedict.txt", "w") as cfile:
        for key in new_state_dict:
            cfile.write(f"{key}\n")
    return control
