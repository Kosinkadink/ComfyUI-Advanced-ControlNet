from typing import Callable, Union
from torch import Tensor
import torch
import os

import comfy.utils
import comfy.model_management
import comfy.model_detection
import comfy.controlnet as comfy_cn
from comfy.controlnet import ControlBase, ControlNet, ControlLora, T2IAdapter, broadcast_image_to

from .control_sparsectrl import SparseControlNet, SparseCtrlMotionWrapper, SparseMethod, SparseSettings, SparseSpreadMethod, PreprocSparseRGBWrapper
from .utils import (AdvancedControlBase, TimestepKeyframeGroup, LatentKeyframeGroup, ControlWeightType, ControlWeights, WeightTypeException,
                    manual_cast_clean_groupnorm, disable_weight_init_clean_groupnorm, prepare_mask_batch, get_properly_arranged_t2i_weights, load_torch_file_with_dict_factory)
from .logger import logger


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
    def __init__(self, control_model, timestep_keyframes: TimestepKeyframeGroup, sparse_settings: SparseSettings=None, global_average_pooling=False, device=None, load_device=None, manual_cast_dtype=None):
        super().__init__(control_model=control_model, timestep_keyframes=timestep_keyframes, global_average_pooling=global_average_pooling, device=device, load_device=load_device, manual_cast_dtype=manual_cast_dtype)
        self.add_compatible_weight(ControlWeightType.SPARSECTRL)
        self.control_model: SparseControlNet = self.control_model  # does nothing except help with IDE hints
        self.sparse_settings = sparse_settings if sparse_settings is not None else SparseSettings.default()
        self.latent_format = None
        self.preprocessed = False
    
    def get_control_advanced(self, x_noisy: Tensor, t, cond, batched_number: int):
        # normal ControlNet stuff
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
        # set actual input length on motion model
        actual_length = x_noisy.size(0)//batched_number
        full_length = actual_length if self.sub_idxs is None else self.full_latent_length
        self.control_model.set_actual_length(actual_length=actual_length, full_length=full_length)
        # prepare cond_hint, if needed
        dim_mult = 1 if self.control_model.use_simplified_conditioning_embedding else 8
        if self.sub_idxs is not None or self.cond_hint is None or x_noisy.shape[2]*dim_mult != self.cond_hint.shape[2] or x_noisy.shape[3]*dim_mult != self.cond_hint.shape[3]:
            # clear out cond_hint and conditioning_mask
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            # first, figure out which cond idxs are relevant, and where they fit in
            cond_idxs = self.sparse_settings.sparse_method.get_indexes(hint_length=self.cond_hint_original.size(0), full_length=full_length)
            
            range_idxs = list(range(full_length)) if self.sub_idxs is None else self.sub_idxs
            hint_idxs = [] # idxs in cond_idxs
            local_idxs = []  # idx to pun in final cond_hint
            for i,cond_idx in enumerate(cond_idxs):
                if cond_idx in range_idxs:
                    hint_idxs.append(i)
                    local_idxs.append(range_idxs.index(cond_idx))
            # sub_cond_hint now contains the hints relevant to current x_noisy
            sub_cond_hint = self.cond_hint_original[hint_idxs].to(dtype).to(self.device)

            # scale cond_hints to match noisy input
            if self.control_model.use_simplified_conditioning_embedding:
                # RGB SparseCtrl; the inputs are latents - use bilinear to avoid blocky artifacts
                sub_cond_hint = self.latent_format.process_in(sub_cond_hint)  # multiplies by model scale factor
                sub_cond_hint = comfy.utils.common_upscale(sub_cond_hint, x_noisy.shape[3], x_noisy.shape[2], "nearest-exact", "center").to(dtype).to(self.device)
            else:
                # other SparseCtrl; inputs are typical images
                sub_cond_hint = comfy.utils.common_upscale(sub_cond_hint, x_noisy.shape[3] * 8, x_noisy.shape[2] * 8, 'nearest-exact', "center").to(dtype).to(self.device)
            # prepare cond_hint (b, c, h ,w)
            cond_shape = list(sub_cond_hint.shape)
            cond_shape[0] = len(range_idxs)
            self.cond_hint = torch.zeros(cond_shape).to(dtype).to(self.device)
            self.cond_hint[local_idxs] = sub_cond_hint[:]
            # prepare cond_mask (b, 1, h, w)
            cond_shape[1] = 1
            cond_mask = torch.zeros(cond_shape).to(dtype).to(self.device)
            cond_mask[local_idxs] = 1.0
            # combine cond_hint and cond_mask into (b, c+1, h, w)
            if not self.sparse_settings.merged:
                self.cond_hint = torch.cat([self.cond_hint, cond_mask], dim=1)
            del sub_cond_hint
            del cond_mask
        # make cond_hint match x_noisy batch
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)

        # prepare mask_cond_hint
        self.prepare_mask_cond_hint(x_noisy=x_noisy, t=t, cond=cond, batched_number=batched_number, dtype=dtype)

        context = cond['c_crossattn']
        y = cond.get('y', None)
        if y is not None:
            y = y.to(dtype)
        timestep = self.model_sampling_current.timestep(t)
        x_noisy = self.model_sampling_current.calculate_input(t, x_noisy)

        control = self.control_model(x=x_noisy.to(dtype), hint=self.cond_hint, timesteps=timestep.float(), context=context.to(dtype), y=y)
        return self.control_merge(None, control, control_prev, output_dtype)

    def pre_run_advanced(self, model, percent_to_timestep_function):
        super().pre_run_advanced(model, percent_to_timestep_function)
        if type(self.cond_hint_original) == PreprocSparseRGBWrapper:
            if not self.control_model.use_simplified_conditioning_embedding:
                raise ValueError("Any model besides RGB SparseCtrl should NOT have its images go through the RGB SparseCtrl preprocessor.")
            self.cond_hint_original = self.cond_hint_original.condhint
        self.latent_format = model.latent_format  # LatentFormat object, used to process_in latent cond hint
        if self.control_model.motion_holder is not None:
            self.control_model.motion_holder.motion_wrapper.reset()
            self.control_model.motion_holder.motion_wrapper.set_strength(self.sparse_settings.motion_strength)
            self.control_model.motion_holder.motion_wrapper.set_scale_multiplier(self.sparse_settings.motion_scale)

    def cleanup_advanced(self):
        super().cleanup_advanced()
        if self.latent_format is not None:
            del self.latent_format
            self.latent_format = None

    def copy(self):
        c = SparseCtrlAdvanced(self.control_model, self.timestep_keyframes, self.sparse_settings, self.global_average_pooling, self.device, self.load_device, self.manual_cast_dtype)
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


def load_sparsectrl(ckpt_path: str, controlnet_data: dict[str, Tensor]=None, timestep_keyframe: TimestepKeyframeGroup=None, sparse_settings=SparseSettings.default(), model=None) -> SparseCtrlAdvanced:
    if controlnet_data is None:
        controlnet_data = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    # first, separate out motion part from normal controlnet part and attempt to load that portion
    motion_data = {}
    for key in list(controlnet_data.keys()):
        if "temporal" in key:
            motion_data[key] = controlnet_data.pop(key)
    if len(motion_data) == 0:
        raise ValueError(f"No motion-related keys in '{ckpt_path}'; not a valid SparseCtrl model!")
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

    # both motion portion and controlnet portions are loaded; bring them together if using motion model
    if sparse_settings.use_motion:
        motion_wrapper.inject(control_model)

    control = SparseCtrlAdvanced(control_model, timestep_keyframes=timestep_keyframe, sparse_settings=sparse_settings, global_average_pooling=global_average_pooling, load_device=load_device, manual_cast_dtype=manual_cast_dtype)
    return control
