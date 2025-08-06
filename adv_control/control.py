from typing import Callable, Union
from torch import Tensor
import torch
import os

import comfy.model_base
import comfy.ops
import comfy.utils
import comfy.model_management
import comfy.model_detection
import comfy.controlnet as comfy_cn
from comfy.controlnet import ControlBase, ControlNet, ControlNetSD35, ControlLora, T2IAdapter, StrengthType
from comfy.model_patcher import ModelPatcher

from .control_sparsectrl import SparseControlNet, SparseSettings, SparseConst, InterfaceAnimateDiffModel, create_sparse_modelpatcher, load_sparsectrl_motionmodel
from .control_lllite import LLLiteModule, LLLitePatch, load_controllllite
from .control_svd import svd_unet_config_from_diffusers_unet, SVDControlNet, svd_unet_to_diffusers
from .utils import (AdvancedControlBase, TimestepKeyframeGroup, LatentKeyframeGroup, AbstractPreprocWrapper, ControlWeightType, ControlWeights, WeightTypeException, Extras,
                    manual_cast_clean_groupnorm, disable_weight_init_clean_groupnorm, WrapperConsts, prepare_mask_batch, get_properly_arranged_t2i_weights, load_torch_file_with_dict_factory,
                    broadcast_image_to_extend, extend_to_batch_size, ORIG_PREVIOUS_CONTROLNET, CONTROL_INIT_BY_ACN)
from .logger import logger


class ControlNetAdvanced(ControlNet, AdvancedControlBase):
    def __init__(self, control_model, timestep_keyframes: TimestepKeyframeGroup, global_average_pooling=False, compression_ratio=8, latent_format=None, load_device=None, manual_cast_dtype=None,
                 extra_conds=["y"], strength_type=StrengthType.CONSTANT, concat_mask=False, preprocess_image=lambda a: a):
        super().__init__(control_model=control_model, global_average_pooling=global_average_pooling, compression_ratio=compression_ratio, latent_format=latent_format, load_device=load_device, manual_cast_dtype=manual_cast_dtype,
                         extra_conds=extra_conds, strength_type=strength_type, concat_mask=concat_mask, preprocess_image=preprocess_image)
        AdvancedControlBase.__init__(self, super(type(self), self), timestep_keyframes=timestep_keyframes, weights_default=ControlWeights.controlnet())
        self.is_flux = False
        self.x_noisy_shape = None

    def get_universal_weights(self) -> ControlWeights:
        def cn_weights_func(idx: int, control: dict[str, list[Tensor]], key: str):
            if key == "middle":
                return 1.0 * self.weights.extras.get(Extras.MIDDLE_MULT, 1.0)
            c_len = len(control[key])
            raw_weights = [(self.weights.base_multiplier ** float((c_len) - i)) for i in range(c_len+1)]
            raw_weights = raw_weights[:-1]
            if key == "input":
                raw_weights.reverse()
            return raw_weights[idx]
        return self.weights.copy_with_new_weights(new_weight_func=cn_weights_func)

    def get_control_advanced(self, x_noisy, t, cond, batched_number, transformer_options):
        # perform special version of get_control that supports sliding context and masks
        return self.sliding_get_control(x_noisy, t, cond, batched_number, transformer_options)

    def sliding_get_control(self, x_noisy: Tensor, t, cond, batched_number, transformer_options):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number, transformer_options)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return None

        dtype = self.control_model.dtype
        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        # make cond_hint appropriate dimensions
        # TODO: change this to not require cond_hint upscaling every step when self.sub_idxs are present
        if self.sub_idxs is not None or self.cond_hint is None or x_noisy.shape[2] * self.real_compression_ratio != self.cond_hint.shape[2] or x_noisy.shape[3] * self.real_compression_ratio != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            self.real_compression_ratio = self.compression_ratio
            compression_ratio = self.compression_ratio
            if self.vae is not None and self.mult_by_ratio_when_vae:
                compression_ratio *= self.vae.downscale_ratio
            # if self.cond_hint_original length greater or equal to real latent count, subdivide it before scaling
            if self.sub_idxs is not None:
                actual_cond_hint_orig = self.cond_hint_original
                if self.cond_hint_original.size(0) < self.full_latent_length:
                    actual_cond_hint_orig = extend_to_batch_size(tensor=actual_cond_hint_orig, batch_size=self.full_latent_length)
                self.cond_hint = comfy.utils.common_upscale(actual_cond_hint_orig[self.sub_idxs], x_noisy.shape[3] * compression_ratio, x_noisy.shape[2] * compression_ratio, self.upscale_algorithm, "center")
            else:
                self.cond_hint = comfy.utils.common_upscale(self.cond_hint_original, x_noisy.shape[3] * compression_ratio, x_noisy.shape[2] * compression_ratio, self.upscale_algorithm, "center")
            self.cond_hint = self.preprocess_image(self.cond_hint)
            if self.vae is not None:
                loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
                self.cond_hint = self.vae.encode(self.cond_hint.movedim(1, -1))
                comfy.model_management.load_models_gpu(loaded_models)
                if not self.mult_by_ratio_when_vae:
                    self.real_compression_ratio = 1
            if self.latent_format is not None:
                self.cond_hint = self.latent_format.process_in(self.cond_hint)
            if len(self.extra_concat_orig) > 0:
                to_concat = []
                for c in self.extra_concat_orig:
                    c = c.to(self.cond_hint.device)
                    c = comfy.utils.common_upscale(c, self.cond_hint.shape[3], self.cond_hint.shape[2], self.upscale_algorithm, "center")
                    to_concat.append(comfy.utils.repeat_to_batch_size(c, self.cond_hint.shape[0]))
                self.cond_hint = torch.cat([self.cond_hint] + to_concat, dim=1)

            self.cond_hint = self.cond_hint.to(device=x_noisy.device, dtype=dtype)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to_extend(self.cond_hint, x_noisy.shape[0], batched_number)

        # prepare mask_cond_hint
        self.prepare_mask_cond_hint(x_noisy=x_noisy, t=t, cond=cond, batched_number=batched_number, dtype=dtype)

        context = cond.get('crossattn_controlnet', cond['c_crossattn'])
        extra = self.extra_args.copy()
        for c in self.extra_conds:
            temp = cond.get(c, None)
            if temp is not None:
                extra[c] = comfy.model_base.convert_tensor(temp, dtype, x_noisy.device)

        timestep = self.model_sampling_current.timestep(t)
        x_noisy = self.model_sampling_current.calculate_input(t, x_noisy)
        self.x_noisy_shape = x_noisy.shape

        control = self.control_model(x=x_noisy.to(dtype), hint=self.cond_hint, timesteps=timestep.to(dtype), context=comfy.model_management.cast_to_device(context, x_noisy.device, dtype), **extra)
        return self.control_merge(control, control_prev, output_dtype=None)

    def pre_run_advanced(self, *args, **kwargs):
        self.is_flux = "Flux" in str(type(self.control_model).__name__)
        return super().pre_run_advanced(*args, **kwargs)

    def apply_advanced_strengths_and_masks(self, x: Tensor, batched_number: int, flux_shape=None):
        if self.is_flux:
            flux_shape = self.x_noisy_shape
        return super().apply_advanced_strengths_and_masks(x, batched_number, flux_shape)

    def copy(self, subtype=None):
        if subtype is None:
            subtype = ControlNetAdvanced
        c = subtype(self.control_model, self.timestep_keyframes, global_average_pooling=self.global_average_pooling, load_device=self.load_device, manual_cast_dtype=self.manual_cast_dtype)
        c.control_model = self.control_model
        c.control_model_wrapped = self.control_model_wrapped
        self.copy_to(c)
        self.copy_to_advanced(c)
        return c
    
    def cleanup_advanced(self):
        self.x_noisy_shape = None
        return super().cleanup_advanced()

    @staticmethod
    def from_vanilla(v: ControlNet, timestep_keyframe: TimestepKeyframeGroup=None, subtype=None) -> 'ControlNetAdvanced':
        if subtype is None:
            subtype = ControlNetAdvanced
        to_return = subtype(control_model=v.control_model, timestep_keyframes=timestep_keyframe,
                            global_average_pooling=v.global_average_pooling, compression_ratio=v.compression_ratio, latent_format=v.latent_format, load_device=v.load_device,
                            manual_cast_dtype=v.manual_cast_dtype, extra_conds=v.extra_conds, strength_type=v.strength_type, concat_mask=v.concat_mask, preprocess_image=v.preprocess_image)
        v.copy_to(to_return)
        to_return.control_model_wrapped = v.control_model_wrapped.clone() # needed to avoid breaking memory management system (parent tracking)
        return to_return


class ControlNetSD35Advanced(ControlNetSD35, ControlNetAdvanced):
    def __init__(self, *args, **kwargs):
        ControlNetAdvanced.__init__(self, *args, **kwargs)

    def copy(self):
        return ControlNetAdvanced.copy(self, subtype=ControlNetSD35Advanced)
    
    @staticmethod
    def from_vanilla(v: ControlNetSD35, timestep_keyframe=None):
        return ControlNetAdvanced.from_vanilla(v, timestep_keyframe, subtype=ControlNetSD35Advanced)


class T2IAdapterAdvanced(T2IAdapter, AdvancedControlBase):
    def __init__(self, t2i_model, timestep_keyframes: TimestepKeyframeGroup, channels_in, compression_ratio=8, upscale_algorithm="nearest_exact", device=None):
        super().__init__(t2i_model=t2i_model, channels_in=channels_in, compression_ratio=compression_ratio, upscale_algorithm=upscale_algorithm, device=device)
        AdvancedControlBase.__init__(self, super(), timestep_keyframes=timestep_keyframes, weights_default=ControlWeights.t2iadapter())

    def control_merge_inject(self, control: dict[str, list[Tensor]], control_prev, output_dtype):
        # match batch_size
        # TODO: make this more efficient by modifying the cached self.control_input val instead of doing this every step
        for key in control:
            control_current = control[key]
            for i in range(len(control_current)):
                x = control_current[i]
                if x is not None and x.size(0) == 1 and x.size(0) != self.batch_size:
                    control_current[i] = x.repeat(self.batch_size, 1, 1, 1)[:self.batch_size]
        return AdvancedControlBase.control_merge_inject(self, control, control_prev, output_dtype)

    def get_universal_weights(self) -> ControlWeights:
        def t2i_weights_func(idx: int, control: dict[str, list[Tensor]], key: str):
            if key == "middle":
                return 1.0 * self.weights.extras.get(Extras.MIDDLE_MULT, 1.0)
            c_len = 8 #len(control[key])
            raw_weights = [(self.weights.base_multiplier ** float((c_len-1) - i)) for i in range(c_len)]
            raw_weights = [raw_weights[-c_len], raw_weights[-3], raw_weights[-2], raw_weights[-1]]
            raw_weights = get_properly_arranged_t2i_weights(raw_weights)
            if key == "input":
                raw_weights.reverse()
            return raw_weights[idx]
        return self.weights.copy_with_new_weights(new_weight_func=t2i_weights_func)

    def get_calc_pow(self, idx: int, control: dict[str, list[Tensor]], key: str) -> int:
        if key == "middle":
            return 0
        # match how T2IAdapterAdvanced deals with universal weights
        c_len = 8 #len(control[key])
        indeces = [(c_len-1) - i for i in range(c_len)]
        indeces = [indeces[-c_len], indeces[-3], indeces[-2], indeces[-1]]
        indeces = get_properly_arranged_t2i_weights(indeces)
        if key == "input":
            indeces.reverse()  # need to reverse to match recent ComfyUI changes
        return indeces[idx]

    def get_control_advanced(self, x_noisy, t, cond, batched_number, transformer_options):
        try:
            # if sub indexes present, replace original hint with subsection
            if self.sub_idxs is not None:
                # cond hints
                full_cond_hint_original = self.cond_hint_original
                actual_cond_hint_orig = full_cond_hint_original
                del self.cond_hint
                self.cond_hint = None
                if full_cond_hint_original.size(0) < self.full_latent_length:
                    actual_cond_hint_orig = extend_to_batch_size(tensor=full_cond_hint_original, batch_size=full_cond_hint_original.size(0))
                self.cond_hint_original = actual_cond_hint_orig[self.sub_idxs]
            # mask hints
            self.prepare_mask_cond_hint(x_noisy=x_noisy, t=t, cond=cond, batched_number=batched_number)
            return super().get_control(x_noisy, t, cond, batched_number, transformer_options)
        finally:
            if self.sub_idxs is not None:
                # replace original cond hint
                self.cond_hint_original = full_cond_hint_original
                del full_cond_hint_original

    def copy(self):
        c = T2IAdapterAdvanced(self.t2i_model, self.timestep_keyframes, self.channels_in, self.compression_ratio, self.upscale_algorithm)
        self.copy_to(c)
        self.copy_to_advanced(c)
        return c
    
    def cleanup(self):
        super().cleanup()
        self.cleanup_advanced()

    @staticmethod
    def from_vanilla(v: T2IAdapter, timestep_keyframe: TimestepKeyframeGroup=None) -> 'T2IAdapterAdvanced':
        to_return = T2IAdapterAdvanced(t2i_model=v.t2i_model, timestep_keyframes=timestep_keyframe, channels_in=v.channels_in,
                                  compression_ratio=v.compression_ratio, upscale_algorithm=v.upscale_algorithm, device=v.device)
        v.copy_to(to_return)
        return to_return


class ControlLoraAdvanced(ControlLora, AdvancedControlBase):
    def __init__(self, control_weights, timestep_keyframes: TimestepKeyframeGroup, global_average_pooling=False):
        super().__init__(control_weights=control_weights, global_average_pooling=global_average_pooling)
        AdvancedControlBase.__init__(self, super(), timestep_keyframes=timestep_keyframes, weights_default=ControlWeights.controllora())
        # use some functions from ControlNetAdvanced
        self.get_control_advanced = ControlNetAdvanced.get_control_advanced.__get__(self, type(self))
        self.sliding_get_control = ControlNetAdvanced.sliding_get_control.__get__(self, type(self))
    
    def get_universal_weights(self) -> ControlWeights:
        raw_weights = [(self.weights.base_multiplier ** float(9 - i)) for i in range(10)]
        return self.weights.copy_with_new_weights(raw_weights)

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
        to_return = ControlLoraAdvanced(control_weights=v.control_weights, timestep_keyframes=timestep_keyframe,
                                   global_average_pooling=v.global_average_pooling)
        v.copy_to(to_return)
        return to_return


class SVDControlNetAdvanced(ControlNetAdvanced):
    def __init__(self, control_model: SVDControlNet, timestep_keyframes: TimestepKeyframeGroup, global_average_pooling=False, load_device=None, manual_cast_dtype=None):
        super().__init__(control_model=control_model, timestep_keyframes=timestep_keyframes, global_average_pooling=global_average_pooling, load_device=load_device, manual_cast_dtype=manual_cast_dtype)

    def set_cond_hint_inject(self, *args, **kwargs):
        to_return = super().set_cond_hint_inject(*args, **kwargs)
        # cond hint for SVD-ControlNet needs to be scaled between (-1, 1) instead of (0, 1)
        self.cond_hint_original = self.cond_hint_original * 2.0 - 1.0
        return to_return

    def get_control_advanced(self, x_noisy, t, cond, batched_number, transformer_options):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number, transformer_options)

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
            if self.sub_idxs is not None:
                actual_cond_hint_orig = self.cond_hint_original
                if self.cond_hint_original.size(0) < self.full_latent_length:
                    actual_cond_hint_orig = extend_to_batch_size(tensor=actual_cond_hint_orig, batch_size=self.full_latent_length)
                self.cond_hint = comfy.utils.common_upscale(actual_cond_hint_orig[self.sub_idxs], x_noisy.shape[3] * 8, x_noisy.shape[2] * 8, 'nearest-exact', "center").to(dtype).to(x_noisy.device)
            else:
                self.cond_hint = comfy.utils.common_upscale(self.cond_hint_original, x_noisy.shape[3] * 8, x_noisy.shape[2] * 8, 'nearest-exact', "center").to(dtype).to(x_noisy.device)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to_extend(self.cond_hint, x_noisy.shape[0], batched_number)

        # prepare mask_cond_hint
        self.prepare_mask_cond_hint(x_noisy=x_noisy, t=t, cond=cond, batched_number=batched_number, dtype=dtype)

        context = cond.get('crossattn_controlnet', cond['c_crossattn'])
        # uses 'y' in new ComfyUI update
        y = cond.get('y', None)
        if y is not None:
            y = comfy.model_base.convert_tensor(y, dtype, x_noisy.device)
        timestep = self.model_sampling_current.timestep(t)
        x_noisy = self.model_sampling_current.calculate_input(t, x_noisy)
        # concat c_concat if exists (should exist for SVD), doubling channels to 8
        if cond.get('c_concat', None) is not None:
            x_noisy = torch.cat([x_noisy] + [cond['c_concat']], dim=1)

        control = self.control_model(x=x_noisy.to(dtype), hint=self.cond_hint, timesteps=timestep.float(), context=comfy.model_management.cast_to_device(context, x_noisy.device, dtype), y=y, cond=cond)
        return self.control_merge(control, control_prev, output_dtype)

    def copy(self):
        c = SVDControlNetAdvanced(self.control_model, self.timestep_keyframes, global_average_pooling=self.global_average_pooling, load_device=self.load_device, manual_cast_dtype=self.manual_cast_dtype)
        self.copy_to(c)
        self.copy_to_advanced(c)
        return c


class SparseCtrlAdvanced(ControlNetAdvanced):
    def __init__(self, control_model: SparseControlNet, motion_model: InterfaceAnimateDiffModel,
                 timestep_keyframes: TimestepKeyframeGroup, sparse_settings: SparseSettings=None, global_average_pooling=False, load_device=None, manual_cast_dtype=None):
        super().__init__(control_model=None, timestep_keyframes=timestep_keyframes, global_average_pooling=global_average_pooling, load_device=load_device, manual_cast_dtype=manual_cast_dtype)
        self.control_model = control_model
        if control_model is not None:
            self.control_model_wrapped: ModelPatcher = create_sparse_modelpatcher(self.control_model, motion_model, load_device=load_device, offload_device=comfy.model_management.unet_offload_device())
            self.prepare_conditioning_info()
        self.add_compatible_weight(ControlWeightType.SPARSECTRL)
        self.postpone_condhint_latents_check = True
        self.sparse_settings = sparse_settings if sparse_settings is not None else SparseSettings.default()
        self.model_latent_format = None  # latent format for active SD model, NOT controlnet
        self.preprocessed = False
    
    def prepare_conditioning_info(self):
        if self.control_model.use_simplified_conditioning_embedding:
            # TODO: allow vae_optional to be used instead of preprocessor
            #self.require_vae = True
            self.allow_condhint_latents = True

    @property
    def motion_model(self) -> InterfaceAnimateDiffModel:
        motion_models = self.control_model_wrapped.get_additional_models_with_key(WrapperConsts.ACN)
        if len(motion_models) == 0:
            return None
        return motion_models[0].model

    def get_control_advanced(self, x_noisy: Tensor, t, cond, batched_number: int, transformer_options):
        # normal ControlNet stuff
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number, transformer_options)

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
        if self.motion_model is not None:
            self.motion_model.set_video_length(video_length=actual_length, full_length=full_length)
        # prepare cond_hint, if needed
        dim_mult = 1 if self.control_model.use_simplified_conditioning_embedding else 8
        if self.sub_idxs is not None or self.cond_hint is None or x_noisy.shape[2]*dim_mult != self.cond_hint.shape[2] or x_noisy.shape[3]*dim_mult != self.cond_hint.shape[3]:
            # clear out cond_hint and conditioning_mask
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            # first, figure out which cond idxs are relevant, and where they fit in
            cond_idxs, hint_order  = self.sparse_settings.sparse_method.get_indexes(hint_length=self.cond_hint_original.size(0), full_length=full_length,
                                                                                     sub_idxs=self.sub_idxs if self.sparse_settings.is_context_aware() else None)
            range_idxs = list(range(full_length)) if self.sub_idxs is None else self.sub_idxs
            hint_idxs = [] # idxs in cond_idxs
            local_idxs = []  # idx to put in final cond_hint
            for i,cond_idx in enumerate(cond_idxs):
                if cond_idx in range_idxs:
                    hint_idxs.append(i)
                    local_idxs.append(range_idxs.index(cond_idx))
            # log_string = f"cond_idxs: {cond_idxs}, local_idxs: {local_idxs}, hint_idxs: {hint_idxs}, hint_order: {hint_order}"
            # if self.sub_idxs is not None:
            #     log_string += f" sub_idxs: {self.sub_idxs[0]}-{self.sub_idxs[-1]}"
            # logger.warn(log_string)
            # determine cond/uncond indexes that will get masked
            self.local_sparse_idxs = []
            self.local_sparse_idxs_inverse = list(range(x_noisy.size(0)))
            for batch_idx in range(batched_number):
                for i in local_idxs:
                    actual_i = i+(batch_idx*actual_length)
                    self.local_sparse_idxs.append(actual_i)
                    if actual_i in self.local_sparse_idxs_inverse:
                        self.local_sparse_idxs_inverse.remove(actual_i)
            # sub_cond_hint now contains the hints relevant to current x_noisy
            if hint_order is None:
                sub_cond_hint = self.cond_hint_original[hint_idxs].to(dtype).to(x_noisy.device)
            else:
                sub_cond_hint = self.cond_hint_original[hint_order][hint_idxs].to(dtype).to(x_noisy.device)
            # scale cond_hints to match noisy input
            if self.control_model.use_simplified_conditioning_embedding:
                # RGB SparseCtrl; the inputs are latents - use bilinear to avoid blocky artifacts
                sub_cond_hint = self.model_latent_format.process_in(sub_cond_hint)  # multiplies by model scale factor
                sub_cond_hint = comfy.utils.common_upscale(sub_cond_hint, x_noisy.shape[3], x_noisy.shape[2], "nearest-exact", "center").to(dtype).to(x_noisy.device)
            else:
                # other SparseCtrl; inputs are typical images
                sub_cond_hint = comfy.utils.common_upscale(sub_cond_hint, x_noisy.shape[3] * 8, x_noisy.shape[2] * 8, 'nearest-exact', "center").to(dtype).to(x_noisy.device)
            # prepare cond_hint (b, c, h ,w)
            cond_shape = list(sub_cond_hint.shape)
            cond_shape[0] = len(range_idxs)
            self.cond_hint = torch.zeros(cond_shape).to(dtype).to(x_noisy.device)
            self.cond_hint[local_idxs] = sub_cond_hint[:]
            # prepare cond_mask (b, 1, h, w)
            cond_shape[1] = 1
            cond_mask = torch.zeros(cond_shape).to(dtype).to(x_noisy.device)
            cond_mask[local_idxs] = self.sparse_settings.sparse_mask_mult * self.weights.extras.get(SparseConst.MASK_MULT, 1.0)
            # combine cond_hint and cond_mask into (b, c+1, h, w)
            if not self.sparse_settings.merged:
                self.cond_hint = torch.cat([self.cond_hint, cond_mask], dim=1)
            del sub_cond_hint
            del cond_mask
        # make cond_hint match x_noisy batch
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to_extend(self.cond_hint, x_noisy.shape[0], batched_number)

        # prepare mask_cond_hint
        self.prepare_mask_cond_hint(x_noisy=x_noisy, t=t, cond=cond, batched_number=batched_number, dtype=dtype)

        context = cond['c_crossattn']
        y = cond.get('y', None)
        if y is not None:
            y = comfy.model_base.convert_tensor(y, dtype, x_noisy.device)
        timestep = self.model_sampling_current.timestep(t)
        x_noisy = self.model_sampling_current.calculate_input(t, x_noisy)

        control = self.control_model(x=x_noisy.to(dtype), hint=self.cond_hint, timesteps=timestep.float(), context=comfy.model_management.cast_to_device(context, x_noisy.device, dtype), y=y)
        return self.control_merge(control, control_prev, output_dtype)

    def apply_advanced_strengths_and_masks(self, x: Tensor, batched_number: int, *args, **kwargs):
        # apply mults to indexes with and without a direct condhint
        x[self.local_sparse_idxs] *= self.sparse_settings.sparse_hint_mult * self.weights.extras.get(SparseConst.HINT_MULT, 1.0)
        x[self.local_sparse_idxs_inverse] *= self.sparse_settings.sparse_nonhint_mult * self.weights.extras.get(SparseConst.NONHINT_MULT, 1.0)
        return super().apply_advanced_strengths_and_masks(x, batched_number, *args, **kwargs)

    def pre_run_advanced(self, model, percent_to_timestep_function):
        super().pre_run_advanced(model, percent_to_timestep_function)
        if isinstance(self.cond_hint_original, AbstractPreprocWrapper):
            if not self.control_model.use_simplified_conditioning_embedding:
                raise ValueError("Any model besides RGB SparseCtrl should NOT have its images go through the RGB SparseCtrl preprocessor.")
            self.cond_hint_original = self.cond_hint_original.condhint
        self.model_latent_format = model.latent_format  # LatentFormat object, used to process_in latent cond hint
        if self.motion_model is not None:
            self.motion_model.cleanup()
            self.motion_model.set_effect(self.sparse_settings.motion_strength)
            self.motion_model.set_scale(self.sparse_settings.motion_scale)

    def cleanup_advanced(self):
        super().cleanup_advanced()
        if self.model_latent_format is not None:
            del self.model_latent_format
            self.model_latent_format = None
        self.local_sparse_idxs = None
        self.local_sparse_idxs_inverse = None
        if self.motion_model is not None:
            self.motion_model.cleanup()

    def copy(self):
        c = SparseCtrlAdvanced(None, None, self.timestep_keyframes, self.sparse_settings, self.global_average_pooling, self.load_device, self.manual_cast_dtype)
        c.control_model = self.control_model
        c.control_model_wrapped = self.control_model_wrapped
        self.prepare_conditioning_info()
        self.copy_to(c)
        self.copy_to_advanced(c)
        return c

    def get_models(self):
        to_return = super().get_models()
        to_return.extend(self.control_model_wrapped.get_additional_models())
        return to_return


def load_controlnet(ckpt_path, timestep_keyframe: TimestepKeyframeGroup=None, model=None):
    controlnet_data = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    # from pathlib import Path
    # log_name = ckpt_path.split('\\')[-1]
    # with open(Path(__file__).parent.parent.parent / rf"keys_{log_name}.txt", "w") as afile:
    #     for key, value in controlnet_data.items():
    #         afile.write(f"{key}:\t{value.shape}\n")
    control = None
    # check if a non-vanilla ControlNet
    controlnet_type = ControlWeightType.DEFAULT
    has_controlnet_key = False
    has_motion_modules_key = False
    has_temporal_res_block_key = False
    for key in controlnet_data:
        # LLLite check
        if "lllite" in key:
            controlnet_type = ControlWeightType.CONTROLLLLITE
            break
        # SparseCtrl check
        elif "motion_modules" in key:
            has_motion_modules_key = True
        elif "controlnet" in key:
            has_controlnet_key = True
        # SVD-ControlNet check
        elif "temporal_res_block" in key:
            has_temporal_res_block_key = True
        # ControlNet++ check
        elif "task_embedding" in key:
            pass
        # CtrLoRA check
        elif "lora_layer" in key:
            controlnet_type = ControlWeightType.CTRLORA
            break

    if has_controlnet_key and has_motion_modules_key:
        controlnet_type = ControlWeightType.SPARSECTRL
    elif has_controlnet_key and has_temporal_res_block_key:
        controlnet_type = ControlWeightType.SVD_CONTROLNET

    if controlnet_type != ControlWeightType.DEFAULT:
        if controlnet_type == ControlWeightType.CONTROLLLLITE:
            control = load_controllllite(ckpt_path, controlnet_data=controlnet_data, timestep_keyframe=timestep_keyframe)
        elif controlnet_type == ControlWeightType.SPARSECTRL:
            control = load_sparsectrl(ckpt_path, controlnet_data=controlnet_data, timestep_keyframe=timestep_keyframe, model=model)
        elif controlnet_type == ControlWeightType.SVD_CONTROLNET:
            control = load_svdcontrolnet(ckpt_path, controlnet_data=controlnet_data, timestep_keyframe=timestep_keyframe)
        elif controlnet_type == ControlWeightType.CTRLORA:
            raise Exception("This is a CtrLoRA; use the Load CtrLoRA Model node.")
    # otherwise, load vanilla ControlNet
    else:
        try:
            # hacky way of getting load_torch_file in load_controlnet to use already-present controlnet_data and not redo loading
            orig_load_torch_file = comfy.utils.load_torch_file
            comfy.utils.load_torch_file = load_torch_file_with_dict_factory(controlnet_data, orig_load_torch_file)
            control = comfy_cn.load_controlnet(ckpt_path, model=model)
        finally:
            comfy.utils.load_torch_file = orig_load_torch_file
    if control is None:
        raise Exception(f"Something went wrong when loading '{ckpt_path}'; ControlNet is None.")
    return convert_to_advanced(control, timestep_keyframe=timestep_keyframe)


def convert_to_advanced(control, timestep_keyframe: TimestepKeyframeGroup=None):
    # if already advanced, leave it be
    if is_advanced_controlnet(control):
        return control
    # if exactly ControlNet returned, transform it into ControlNetAdvanced
    if type(control) == ControlNet:
        control = ControlNetAdvanced.from_vanilla(v=control, timestep_keyframe=timestep_keyframe)
        if is_sd3_advanced_controlnet(control):
            control.require_vae = True
        return control
    # if exactly ControlNetSD35 returned, transform into ControlNetSD35Advanced
    elif type(control) == ControlNetSD35:
        control = ControlNetSD35Advanced.from_vanilla(v=control, timestep_keyframe=timestep_keyframe)
        if is_sd3_advanced_controlnet(control):
            control.require_vae = True
        return control
    # if exactly ControlLora returned, transform it into ControlLoraAdvanced
    elif type(control) == ControlLora:
        return ControlLoraAdvanced.from_vanilla(v=control, timestep_keyframe=timestep_keyframe)
    # if T2IAdapter returned, transform it into T2IAdapterAdvanced
    elif isinstance(control, T2IAdapter):
        return T2IAdapterAdvanced.from_vanilla(v=control, timestep_keyframe=timestep_keyframe)
    # otherwise, leave it be - might be something I am not supporting yet
    return control


def convert_all_to_advanced(conds: dict[str, list[dict[str]]]) -> tuple[bool, list]:
    cache = {}
    modified = False
    new_conds = {}
    for cond_type in conds:
        converted_cond: list[dict[str]] = None
        cond = conds[cond_type]
        if cond is not None:
            for actual_cond in cond:
                need_to_convert = False
                if "control" in actual_cond:
                    if not are_all_advanced_controlnet(actual_cond["control"]):
                        need_to_convert = True
                        break
            if not need_to_convert:
                converted_cond = cond
            else:
                converted_cond = []
                for actual_cond in cond:
                    if not isinstance(actual_cond, dict):
                        converted_cond.append(actual_cond)
                        continue
                    if "control" not in actual_cond:
                        converted_cond.append(actual_cond)
                    elif are_all_advanced_controlnet(actual_cond["control"]):
                        converted_cond.append(actual_cond)
                    else:
                        actual_cond = actual_cond.copy()
                        actual_cond["control"] = _convert_all_control_to_advanced(actual_cond["control"], cache)
                        converted_cond.append(actual_cond)
                        modified = True
        new_conds[cond_type] = converted_cond
    return modified, new_conds


def _convert_all_control_to_advanced(input_object: ControlBase, cache: dict):
    output_object = input_object
    # iteratively convert to advanced, if needed
    next_cn = None
    curr_cn = input_object
    iter = 0
    while curr_cn is not None:
        if not is_advanced_controlnet(curr_cn):
            # if already in cache, then conversion was done before, so just link it and exit
            if curr_cn in cache:
                new_cn = cache[curr_cn]
                if next_cn is not None:
                    setattr(next_cn, ORIG_PREVIOUS_CONTROLNET, next_cn.previous_controlnet)
                    next_cn.previous_controlnet = new_cn
                if iter == 0: # if was top-level controlnet, that's the new output
                    output_object = new_cn
                break
            try:
                # convert to advanced, and assign previous_controlnet (convert doesn't transfer it)
                new_cn = convert_to_advanced(curr_cn)
            except Exception as e:
                raise Exception("Failed to automatically convert a ControlNet to Advanced to support sliding window context.", e)
            new_cn.previous_controlnet = curr_cn.previous_controlnet
            if iter == 0: # if was top-level controlnet, that's the new output
                output_object = new_cn
            # if next_cn is present, then it needs to be pointed to new_cn
            if next_cn is not None:
                setattr(next_cn, ORIG_PREVIOUS_CONTROLNET, next_cn.previous_controlnet)
                next_cn.previous_controlnet = new_cn
            # add to cache
            cache[curr_cn] = new_cn
            curr_cn = new_cn
        next_cn = curr_cn
        curr_cn = curr_cn.previous_controlnet
        iter += 1
    return output_object


def restore_all_controlnet_conns(conds: dict[str, list[dict[str]]]):
    # if a cn has an _orig_previous_controlnet property, restore it and delete
    for cond_type in conds:
        cond = conds[cond_type]
        if cond is not None:
            for actual_cond in cond:
                if "control" in actual_cond:
                    # if ACN is the one to have initialized it, delete it
                    # TODO: maybe check if someone else did a similar hack, and carefully pluck out our stuff?
                    if CONTROL_INIT_BY_ACN in actual_cond:
                        actual_cond.pop("control")
                        actual_cond.pop(CONTROL_INIT_BY_ACN)
                    else:
                        _restore_all_controlnet_conns(actual_cond["control"])



def _restore_all_controlnet_conns(input_object: ControlBase):
    # restore original previous_controlnet if needed
    curr_cn = input_object
    while curr_cn is not None:
        if hasattr(curr_cn, ORIG_PREVIOUS_CONTROLNET):
            curr_cn.previous_controlnet = getattr(curr_cn, ORIG_PREVIOUS_CONTROLNET)
            delattr(curr_cn, ORIG_PREVIOUS_CONTROLNET)
        curr_cn = curr_cn.previous_controlnet


def are_all_advanced_controlnet(input_object: ControlBase):
    # iteratively check if linked controlnets objects are all advanced
    curr_cn = input_object
    while curr_cn is not None:
        if not is_advanced_controlnet(curr_cn):
            return False
        curr_cn = curr_cn.previous_controlnet
    return True


def is_advanced_controlnet(input_object):
    return hasattr(input_object, "sub_idxs")


def is_sd3_advanced_controlnet(input_object: ControlNetAdvanced):
    return type(input_object) in [ControlNetAdvanced, ControlNetSD35Advanced] and input_object.latent_format is not None


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

    # now, load as if it was a normal controlnet - mostly copied from comfy load_controlnet function
    controlnet_config: dict[str] = None
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

    # actually load motion portion of model now
    motion_model = load_sparsectrl_motionmodel(ckpt_path=ckpt_path, motion_data=motion_data, ops=controlnet_config.get("operations", None)).to(comfy.model_management.unet_dtype())
    # both motion portion and controlnet portions are loaded; ignore motion_model if shouldn't use motion portion
    if not sparse_settings.use_motion:
        motion_model = None

    control = SparseCtrlAdvanced(control_model, motion_model, timestep_keyframes=timestep_keyframe, sparse_settings=sparse_settings, global_average_pooling=global_average_pooling, load_device=load_device, manual_cast_dtype=manual_cast_dtype)
    return control


def load_svdcontrolnet(ckpt_path: str, controlnet_data: dict[str, Tensor]=None, timestep_keyframe: TimestepKeyframeGroup=None, model=None):
    if controlnet_data is None:
        controlnet_data = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

    controlnet_config = None
    if "controlnet_cond_embedding.conv_in.weight" in controlnet_data: #diffusers format
        unet_dtype = comfy.model_management.unet_dtype()
        controlnet_config = svd_unet_config_from_diffusers_unet(controlnet_data, unet_dtype)
        diffusers_keys = svd_unet_to_diffusers(controlnet_config)
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

        new_sd = {}
        for k in diffusers_keys:
            if k in controlnet_data:
                new_sd[diffusers_keys[k]] = controlnet_data.pop(k)

        leftover_keys = controlnet_data.keys()
        if len(leftover_keys) > 0:
            spatial_leftover_keys = []
            temporal_leftover_keys = []
            other_leftover_keys = []
            for key in leftover_keys:
                if "spatial" in key:
                    spatial_leftover_keys.append(key)
                elif "temporal" in key:
                    temporal_leftover_keys.append(key)
                else:
                    other_leftover_keys.append(key)
            logger.warn(f"spatial_leftover_keys ({len(spatial_leftover_keys)}): {spatial_leftover_keys}")
            logger.warn(f"temporal_leftover_keys ({len(temporal_leftover_keys)}): {temporal_leftover_keys}")
            logger.warn(f"other_leftover_keys ({len(other_leftover_keys)}): {other_leftover_keys}")
            #print("leftover keys:", leftover_keys)
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
        raise ValueError("The provided model is not a valid SVD-ControlNet model! [ErrorCode: MUSTARD]")

    if controlnet_config is None:
        unet_dtype = comfy.model_management.unet_dtype()
        controlnet_config = comfy.model_detection.model_config_from_unet(controlnet_data, prefix, unet_dtype, True).unet_config
    load_device = comfy.model_management.get_torch_device()
    manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, load_device)
    if manual_cast_dtype is not None:
        controlnet_config["operations"] = comfy.ops.manual_cast
    controlnet_config.pop("out_channels")
    controlnet_config["hint_channels"] = controlnet_data["{}input_hint_block.0.weight".format(prefix)].shape[1]
    control_model = SVDControlNet(**controlnet_config)

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
                print("WARNING: Loaded a diff controlnet without a model. It will very likely not work.")

        class WeightsLoader(torch.nn.Module):
            pass
        w = WeightsLoader()
        w.control_model = control_model
        missing, unexpected = w.load_state_dict(controlnet_data, strict=False)
    else:
        missing, unexpected = control_model.load_state_dict(controlnet_data, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        logger.info(f"SVD-ControlNet: {missing}, {unexpected}")

    global_average_pooling = False
    filename = os.path.splitext(ckpt_path)[0]
    if filename.endswith("_shuffle") or filename.endswith("_shuffle_fp16"): #TODO: smarter way of enabling global_average_pooling
        global_average_pooling = True

    control = SVDControlNetAdvanced(control_model, timestep_keyframes=timestep_keyframe, global_average_pooling=global_average_pooling, load_device=load_device, manual_cast_dtype=manual_cast_dtype)
    return control

