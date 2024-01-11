from typing import Callable, Union
import torch
from torch import Tensor
import torch.nn.functional as F
import comfy.ops
import comfy.utils
from comfy.controlnet import ControlBase, broadcast_image_to


def load_torch_file_with_dict_factory(controlnet_data: dict[str, Tensor], orig_load_torch_file: Callable):
    def load_torch_file_with_dict(*args, **kwargs):
        # immediately restore load_torch_file to original version
        comfy.utils.load_torch_file = orig_load_torch_file
        return controlnet_data
    return load_torch_file_with_dict


def get_properly_arranged_t2i_weights(initial_weights: list[float]):
    new_weights = []
    new_weights.extend([initial_weights[0]]*3)
    new_weights.extend([initial_weights[1]]*3)
    new_weights.extend([initial_weights[2]]*3)
    new_weights.extend([initial_weights[3]]*3)
    return new_weights


class ControlWeightType:
    DEFAULT = "default"
    UNIVERSAL = "universal"
    T2IADAPTER = "t2iadapter"
    CONTROLNET = "controlnet"
    CONTROLLORA = "controllora"
    CONTROLLLLITE = "controllllite"
    SPARSECTRL = "sparsectrl"


class ControlWeights:
    def __init__(self, weight_type: str, base_multiplier: float=1.0, flip_weights: bool=False, weights: list[float]=None, weight_mask: Tensor=None):
        self.weight_type = weight_type
        self.base_multiplier = base_multiplier
        self.flip_weights = flip_weights
        self.weights = weights
        if self.weights is not None and self.flip_weights:
            self.weights.reverse()
        self.weight_mask = weight_mask

    def get(self, idx: int) -> Union[float, Tensor]:
        # if weights is not none, return index
        if self.weights is not None:
            return self.weights[idx]
        return 1.0

    @classmethod
    def default(cls):
        return cls(ControlWeightType.DEFAULT)

    @classmethod
    def universal(cls, base_multiplier: float, flip_weights: bool=False):
        return cls(ControlWeightType.UNIVERSAL, base_multiplier=base_multiplier, flip_weights=flip_weights)
    
    @classmethod
    def universal_mask(cls, weight_mask: Tensor):
        return cls(ControlWeightType.UNIVERSAL, weight_mask=weight_mask)

    @classmethod
    def t2iadapter(cls, weights: list[float]=None, flip_weights: bool=False):
        if weights is None:
            weights = [1.0]*12
        return cls(ControlWeightType.T2IADAPTER, weights=weights,flip_weights=flip_weights)

    @classmethod
    def controlnet(cls, weights: list[float]=None, flip_weights: bool=False):
        if weights is None:
            weights = [1.0]*13
        return cls(ControlWeightType.CONTROLNET, weights=weights, flip_weights=flip_weights)
    
    @classmethod
    def controllora(cls, weights: list[float]=None, flip_weights: bool=False):
        if weights is None:
            weights = [1.0]*10
        return cls(ControlWeightType.CONTROLLORA, weights=weights, flip_weights=flip_weights)
    
    @classmethod
    def controllllite(cls, weights: list[float]=None, flip_weights: bool=False):
        if weights is None:
            # TODO: make this have a real value
            weights = [1.0]*200
        return cls(ControlWeightType.CONTROLLLLITE, weights=weights, flip_weights=flip_weights)


class StrengthInterpolation:
    LINEAR = "linear"
    EASE_IN = "ease-in"
    EASE_OUT = "ease-out"
    EASE_IN_OUT = "ease-in-out"
    NONE = "none"


class LatentKeyframe:
    def __init__(self, batch_index: int, strength: float) -> None:
        self.batch_index = batch_index
        self.strength = strength


# always maintain sorted state (by batch_index of LatentKeyframe)
class LatentKeyframeGroup:
    def __init__(self) -> None:
        self.keyframes: list[LatentKeyframe] = []

    def add(self, keyframe: LatentKeyframe) -> None:
        added = False
        # replace existing keyframe if same batch_index
        for i in range(len(self.keyframes)):
            if self.keyframes[i].batch_index == keyframe.batch_index:
                self.keyframes[i] = keyframe
                added = True
                break
        if not added:
            self.keyframes.append(keyframe)
        self.keyframes.sort(key=lambda k: k.batch_index)
    
    def get_index(self, index: int) -> Union[LatentKeyframe, None]:
        try:
            return self.keyframes[index]
        except IndexError:
            return None
    
    def __getitem__(self, index) -> LatentKeyframe:
        return self.keyframes[index]
    
    def is_empty(self) -> bool:
        return len(self.keyframes) == 0

    def clone(self) -> 'LatentKeyframeGroup':
        cloned = LatentKeyframeGroup()
        for tk in self.keyframes:
            cloned.add(tk)
        return cloned


class TimestepKeyframe:
    def __init__(self,
                 start_percent: float = 0.0,
                 strength: float = 1.0,
                 interpolation: str = StrengthInterpolation.NONE,
                 control_weights: ControlWeights = None,
                 latent_keyframes: LatentKeyframeGroup = None,
                 null_latent_kf_strength: float = 0.0,
                 inherit_missing: bool = True,
                 guarantee_usage: bool = True,
                 mask_hint_orig: Tensor = None) -> None:
        self.start_percent = start_percent
        self.start_t = 999999999.9
        self.strength = strength
        self.interpolation = interpolation
        self.control_weights = control_weights
        self.latent_keyframes = latent_keyframes
        self.null_latent_kf_strength = null_latent_kf_strength
        self.inherit_missing = inherit_missing
        self.guarantee_usage = guarantee_usage
        self.mask_hint_orig = mask_hint_orig

    def has_control_weights(self):
        return self.control_weights is not None
    
    def has_latent_keyframes(self):
        return self.latent_keyframes is not None
    
    def has_mask_hint(self):
        return self.mask_hint_orig is not None
    
    
    @classmethod
    def default(cls) -> 'TimestepKeyframe':
        return cls(0.0)


# always maintain sorted state (by start_percent of TimestepKeyFrame)
class TimestepKeyframeGroup:
    def __init__(self) -> None:
        self.keyframes: list[TimestepKeyframe] = []
        self.keyframes.append(TimestepKeyframe.default())

    def add(self, keyframe: TimestepKeyframe) -> None:
        added = False
        # replace existing keyframe if same start_percent
        for i in range(len(self.keyframes)):
            if self.keyframes[i].start_percent == keyframe.start_percent:
                self.keyframes[i] = keyframe
                added = True
                break
        if not added:
            self.keyframes.append(keyframe)
        self.keyframes.sort(key=lambda k: k.start_percent)

    def get_index(self, index: int) -> Union[TimestepKeyframe, None]:
        try:
            return self.keyframes[index]
        except IndexError:
            return None
    
    def has_index(self, index: int) -> int:
        return index >=0 and index < len(self.keyframes)

    def __getitem__(self, index) -> TimestepKeyframe:
        return self.keyframes[index]
    
    def __len__(self) -> int:
        return len(self.keyframes)

    def is_empty(self) -> bool:
        return len(self.keyframes) == 0
    
    def clone(self) -> 'TimestepKeyframeGroup':
        cloned = TimestepKeyframeGroup()
        for tk in self.keyframes:
            cloned.add(tk)
        return cloned
    
    @classmethod
    def default(cls, keyframe: TimestepKeyframe) -> 'TimestepKeyframeGroup':
        group = cls()
        group.keyframes[0] = keyframe
        return group


# depending on model, AnimateDiff may inject into GroupNorm, so make sure GroupNorm will be clean
class disable_weight_init_clean_groupnorm(comfy.ops.disable_weight_init):
    class GroupNorm(comfy.ops.disable_weight_init.GroupNorm):
        def forward(self, input: Tensor) -> Tensor:
            return F.group_norm(
                input, self.num_groups, self.weight, self.bias, self.eps)
class manual_cast_clean_groupnorm(comfy.ops.manual_cast):
    class GroupNorm(comfy.ops.manual_cast.GroupNorm):
        def forward(self, input: Tensor) -> Tensor:
            return F.group_norm(
                input, self.num_groups, self.weight, self.bias, self.eps)


# adapted from comfy/sample.py
def prepare_mask_batch(mask: Tensor, shape: Tensor, multiplier: int=1, match_dim1=False):
    mask = mask.clone()
    mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(shape[2]*multiplier, shape[3]*multiplier), mode="bilinear")
    if match_dim1:
        mask = torch.cat([mask] * shape[1], dim=1)
    return mask


# applies min-max normalization, from:
# https://stackoverflow.com/questions/68791508/min-max-normalization-of-a-tensor-in-pytorch
def normalize_min_max(x: Tensor, new_min = 0.0, new_max = 1.0):
    x_min, x_max = x.min(), x.max()
    return (((x - x_min)/(x_max - x_min)) * (new_max - new_min)) + new_min

def linear_conversion(x, x_min=0.0, x_max=1.0, new_min=0.0, new_max=1.0):
    return (((x - x_min)/(x_max - x_min)) * (new_max - new_min)) + new_min


class WeightTypeException(TypeError):
    "Raised when weight not compatible with AdvancedControlBase object"
    pass


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
                            if o[i].shape[0] < prev_val.shape[0]:
                                o[i] = prev_val + o[i]
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
