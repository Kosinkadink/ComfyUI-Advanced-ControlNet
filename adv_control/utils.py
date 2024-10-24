from copy import deepcopy
from typing import Callable, Union
import torch
from torch import Tensor
import torch.nn.functional
from einops import rearrange
import numpy as np
import math

import comfy.ops
import comfy.utils
import comfy.sample
import comfy.samplers
import comfy.model_base

from comfy.controlnet import ControlBase
from comfy.model_patcher import ModelPatcher
from comfy.sd import VAE

from .logger import logger

BIGMIN = -(2**53-1)
BIGMAX = (2**53-1)

ORIG_PREVIOUS_CONTROLNET = "_orig_previous_controlnet"
CONTROL_INIT_BY_ACN = "_control_init_by_ACN"


def load_torch_file_with_dict_factory(controlnet_data: dict[str, Tensor], orig_load_torch_file: Callable):
    def load_torch_file_with_dict(*args, **kwargs):
        # immediately restore load_torch_file to original version
        comfy.utils.load_torch_file = orig_load_torch_file
        return controlnet_data
    return load_torch_file_with_dict

# wrapping len function so that it will save the thing len is trying to get the length of;
# this will be assumed to be the cond_or_uncond variable;
# automatically restores len to original function after running
def wrapper_len_factory(orig_len: Callable) -> Callable:
    def wrapper_len(*args, **kwargs):
        cond_or_uncond = args[0]
        real_length = orig_len(*args, **kwargs)
        if real_length > 0 and type(cond_or_uncond) == list and isinstance(cond_or_uncond[0], int) and (cond_or_uncond[0] in [0, 1]):
            try:
                to_return = IntWithCondOrUncond(real_length)
                setattr(to_return, "cond_or_uncond", cond_or_uncond)
                return to_return
            finally:
                __builtins__["len"] = orig_len
        else:
            return real_length
    return wrapper_len

# wrapping cond_cat function so that it will wrap around len function to get cond_or_uncond variable value
# from comfy.samplers.calc_conds_batch
def wrapper_cond_cat_factory(orig_cond_cat: Callable):
    def wrapper_cond_cat(*args, **kwargs):
        __builtins__["len"] = wrapper_len_factory(__builtins__["len"])
        return orig_cond_cat(*args, **kwargs)
    return wrapper_cond_cat
orig_cond_cat = comfy.samplers.cond_cat
comfy.samplers.cond_cat = wrapper_cond_cat_factory(orig_cond_cat)


# wrapping apply_model so that len function will be cleaned up fairly soon after being injected
def apply_model_uncond_cleanup_factory(orig_apply_model, orig_len):
    def apply_model_uncond_cleanup_wrapper(self, *args, **kwargs):
        __builtins__["len"] = orig_len
        return orig_apply_model(self, *args, **kwargs)
    return apply_model_uncond_cleanup_wrapper
global_orig_len = __builtins__["len"]
orig_apply_model = comfy.model_base.BaseModel.apply_model
comfy.model_base.BaseModel.apply_model = apply_model_uncond_cleanup_factory(orig_apply_model, global_orig_len)


def uncond_multiplier_check_cn_sample_factory(orig_comfy_sample: Callable, is_custom=False) -> Callable:
    def contains_uncond_multiplier(control: Union[ControlBase, 'AdvancedControlBase']):
        if control is None:
            return False
        if not isinstance(control, AdvancedControlBase):
            return contains_uncond_multiplier(control.previous_controlnet)
        # check if weights_override has an uncond_multiplier
        if control.weights_override is not None and control.weights_override.has_uncond_multiplier:
            return True
        # check if any timestep_keyframes have an uncond_multiplier on their weights
        if control.timestep_keyframes is not None:
            for tk in control.timestep_keyframes.keyframes:
                if tk.has_control_weights() and tk.control_weights.has_uncond_multiplier:
                    return True
        return contains_uncond_multiplier(control.previous_controlnet)

    # check if positive or negative conds contain Adv. Cns that use multiply_negative on weights
    def uncond_multiplier_check_cn_sample(model: ModelPatcher, *args, **kwargs):
        positive = args[-3]
        negative = args[-2]
        has_uncond_multiplier = False
        if positive is not None:
            for cond in positive:
                if "control" in cond[1]:
                    has_uncond_multiplier = contains_uncond_multiplier(cond[1]["control"])
                    if has_uncond_multiplier:
                        break
        if negative is not None and not has_uncond_multiplier:
            for cond in negative:
                if "control" in cond[1]:
                    has_uncond_multiplier = contains_uncond_multiplier(cond[1]["control"])
                    if has_uncond_multiplier:
                        break
        try:
            # if uncond_multiplier found, continue to use wrapped version of function
            if has_uncond_multiplier:
                return orig_comfy_sample(model, *args, **kwargs)
            # otherwise, use original version of function to prevent even the smallest of slowdowns (0.XX%)
            try:
                wrapped_cond_cat = comfy.samplers.cond_cat
                comfy.samplers.cond_cat = orig_cond_cat
                return orig_comfy_sample(model, *args, **kwargs)
            finally:
                comfy.samplers.cond_cat = wrapped_cond_cat
        finally:
            # make sure len function is unwrapped by the time sampling is done, just in case
            __builtins__["len"] = global_orig_len
    return uncond_multiplier_check_cn_sample
# inject sample functions
comfy.sample.sample = uncond_multiplier_check_cn_sample_factory(comfy.sample.sample)
comfy.sample.sample_custom = uncond_multiplier_check_cn_sample_factory(comfy.sample.sample_custom, is_custom=True)


class IntWithCondOrUncond(int):
    def __new__(cls, *args, **kwargs):
        return super(IntWithCondOrUncond, cls).__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.cond_or_uncond = None



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
    CONTROLNETPLUSPLUS = "controlnet++"
    CONTROLLORA = "controllora"
    CONTROLLLLITE = "controllllite"
    SVD_CONTROLNET = "svd_controlnet"
    SPARSECTRL = "sparsectrl"


class ControlWeights:
    def __init__(self, weight_type: str, base_multiplier: float=1.0,
                 weights_input: list[float]=None, weights_middle: list[float]=None, weights_output: list[float]=None,
                 weight_func: Callable=None, weight_mask: Tensor=None,
                 uncond_multiplier=1.0, uncond_mask: Tensor=None, extras: dict[str]={},):
        self.weight_type = weight_type
        self.base_multiplier = base_multiplier
        self.weights_input = weights_input
        self.weights_middle = weights_middle
        self.weights_output = weights_output
        self.weight_func = weight_func
        self.weight_mask = weight_mask
        self.uncond_multiplier = float(uncond_multiplier)
        self.has_uncond_multiplier = not math.isclose(self.uncond_multiplier, 1.0)
        self.uncond_mask = uncond_mask if uncond_mask is not None else 1.0
        self.has_uncond_mask = uncond_mask is not None
        self.extras = extras

    def get(self, idx: int, control: dict[str, list[Tensor]], key: str, default=1.0) -> Union[float, Tensor]:
        # if weight_func present, use it
        if self.weight_func is not None:
            return self.weight_func(idx=idx, control=control, key=key)
        # if weights is not none, return index
        relevant_weights = None
        if key == "middle":
            relevant_weights = self.weights_middle
        elif key == "input":
            relevant_weights = self.weights_input
            if relevant_weights is not None:
                relevant_weights = list(reversed(relevant_weights))
        else:
            relevant_weights = self.weights_output
        if relevant_weights is None:
            return default
        elif idx >= len(relevant_weights):
            return default
        return relevant_weights[idx]

    def copy_with_new_weights(self, new_weights_input: list[float]=None, new_weights_middle: list[float]=None, new_weights_output: list[float]=None,
                              new_weight_func: Callable=None):
        return ControlWeights(weight_type=self.weight_type, base_multiplier=self.base_multiplier,
                              weights_input=new_weights_input, weights_middle=new_weights_middle, weights_output=new_weights_output,
                              weight_func=new_weight_func, weight_mask=self.weight_mask,
                              uncond_multiplier=self.uncond_multiplier, extras=self.extras)

    @classmethod
    def default(cls, extras: dict[str]={}):
        return cls(ControlWeightType.DEFAULT, extras=extras)

    @classmethod
    def universal(cls, base_multiplier: float, uncond_multiplier: float=1.0, extras: dict[str]={}):
        return cls(ControlWeightType.UNIVERSAL, base_multiplier=base_multiplier, uncond_multiplier=uncond_multiplier, extras=extras)
    
    @classmethod
    def universal_mask(cls, weight_mask: Tensor, uncond_multiplier: float=1.0, extras: dict[str]={}):
        return cls(ControlWeightType.UNIVERSAL, weight_mask=weight_mask, uncond_multiplier=uncond_multiplier, extras=extras)

    @classmethod
    def t2iadapter(cls, weights_input: list[float]=None, uncond_multiplier: float=1.0, extras: dict[str]={}):
        return cls(ControlWeightType.T2IADAPTER, weights_input=weights_input, uncond_multiplier=uncond_multiplier, extras=extras)

    @classmethod
    def controlnet(cls, weights_output: list[float]=None, weights_middle: list[float]=None, weights_input: list[float]=None, uncond_multiplier: float=1.0, extras: dict[str]={}):
        return cls(ControlWeightType.CONTROLNET, weights_output=weights_output, weights_middle=weights_middle, weights_input=weights_input, uncond_multiplier=uncond_multiplier, extras=extras)
    
    @classmethod
    def controllora(cls, weights_output: list[float]=None, weights_middle: list[float]=None, weights_input: list[float]=None, uncond_multiplier: float=1.0, extras: dict[str]={}):
        return cls(ControlWeightType.CONTROLLORA, weights_output=weights_output, weights_middle=weights_middle, weights_input=weights_input, uncond_multiplier=uncond_multiplier, extras=extras)
    
    @classmethod
    def controllllite(cls, weights_output: list[float]=None, weights_middle: list[float]=None, weights_input: list[float]=None, uncond_multiplier: float=1.0, extras: dict[str]={}):
        return cls(ControlWeightType.CONTROLLLLITE, weights_output=weights_output, weights_middle=weights_middle, weights_input=weights_input, uncond_multiplier=uncond_multiplier, extras=extras)


class StrengthInterpolation:
    LINEAR = "linear"
    EASE_IN = "ease-in"
    EASE_OUT = "ease-out"
    EASE_IN_OUT = "ease-in-out"
    NONE = "none"

    _LIST = [LINEAR, EASE_IN, EASE_OUT, EASE_IN_OUT]
    _LIST_WITH_NONE = [LINEAR, EASE_IN, EASE_OUT, EASE_IN_OUT, NONE]

    @classmethod
    def get_weights(cls, num_from: float, num_to: float, length: int, method: str, reverse=False):
        diff = num_to - num_from
        if method == cls.LINEAR:
            weights = torch.linspace(num_from, num_to, length)
        elif method == cls.EASE_IN:
            index = torch.linspace(0, 1, length)
            weights = diff * np.power(index, 2) + num_from
        elif method == cls.EASE_OUT:
            index = torch.linspace(0, 1, length)
            weights = diff * (1 - np.power(1 - index, 2)) + num_from
        elif method == cls.EASE_IN_OUT:
            index = torch.linspace(0, 1, length)
            weights = diff * ((1 - np.cos(index * np.pi)) / 2) + num_from
        else:
            raise ValueError(f"Unrecognized interpolation method '{method}'.")
        if reverse:
            weights = weights.flip(dims=(0,))
        return weights


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
                 control_weights: ControlWeights = None,
                 latent_keyframes: LatentKeyframeGroup = None,
                 null_latent_kf_strength: float = 0.0,
                 inherit_missing: bool = True,
                 guarantee_steps: int = 1,
                 mask_hint_orig: Tensor = None) -> None:
        self.start_percent = float(start_percent)
        self.start_t = 999999999.9
        self.strength = strength
        self.control_weights = control_weights
        self.latent_keyframes = latent_keyframes
        self.null_latent_kf_strength = null_latent_kf_strength
        self.inherit_missing = inherit_missing
        self.guarantee_steps = guarantee_steps
        self.mask_hint_orig = mask_hint_orig

    def has_control_weights(self):
        return self.control_weights is not None
    
    def has_latent_keyframes(self):
        return self.latent_keyframes is not None
    
    def has_mask_hint(self):
        return self.mask_hint_orig is not None
    
    
    @staticmethod
    def default() -> 'TimestepKeyframe':
        return TimestepKeyframe(start_percent=0.0, guarantee_steps=0)


# always maintain sorted state (by start_percent of TimestepKeyFrame)
class TimestepKeyframeGroup:
    def __init__(self) -> None:
        self.keyframes: list[TimestepKeyframe] = []
        self.keyframes.append(TimestepKeyframe.default())

    def add(self, keyframe: TimestepKeyframe) -> None:
        # add to end of list, then sort
        self.keyframes.append(keyframe)
        self.keyframes = get_sorted_list_via_attr(self.keyframes, attr="start_percent")

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
        # already sorted, so don't use add function to make cloning quicker
        for tk in self.keyframes:
            cloned.keyframes.append(tk)
        return cloned
    
    @classmethod
    def default(cls, keyframe: TimestepKeyframe) -> 'TimestepKeyframeGroup':
        group = cls()
        group.keyframes[0] = keyframe
        return group


class AbstractPreprocWrapper:
    error_msg = "Invalid use of [InsertHere] output. The output of [InsertHere] preprocessor is NOT a usual image, but a latent pretending to be an image - you must connect the output directly to an Apply ControlNet node (advanced or otherwise). It cannot be used for anything else that accepts IMAGE input."
    def __init__(self, condhint):
        self.condhint = condhint
    
    def movedim(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        raise AttributeError(self.error_msg)
    
    def __setattr__(self, name, value):
        if name != "condhint":
            raise AttributeError(self.error_msg)
        super().__setattr__(name, value)
    
    def __iter__(self, *args, **kwargs):
        raise AttributeError(self.error_msg)
    
    def __next__(self, *args, **kwargs):
        raise AttributeError(self.error_msg)

    def __len__(self, *args, **kwargs):
        raise AttributeError(self.error_msg)
    
    def __getitem__(self, *args, **kwargs):
        raise AttributeError(self.error_msg)
    
    def __setitem__(self, *args, **kwargs):
        raise AttributeError(self.error_msg)


# depending on model, AnimateDiff may inject into GroupNorm, so make sure GroupNorm will be clean
class disable_weight_init_clean_groupnorm(comfy.ops.disable_weight_init):
    class GroupNorm(comfy.ops.disable_weight_init.GroupNorm):
        def forward_comfy_cast_weights(self, input):
            weight, bias = comfy.ops.cast_bias_weight(self, input)
            return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

        def forward(self, input):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(input)
            else:
                return torch.nn.functional.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)

class manual_cast_clean_groupnorm(comfy.ops.manual_cast):
    class GroupNorm(disable_weight_init_clean_groupnorm.GroupNorm):
        comfy_cast_weights = True


# adapted from comfy/sample.py
def prepare_mask_batch(mask: Tensor, shape: Tensor, multiplier: int=1, match_dim1=False, match_shape=False, flux_shape=None):
    mask = mask.clone()
    if flux_shape is not None:
        multiplier = multiplier * 0.5
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(round(flux_shape[-2]*multiplier), round(flux_shape[-1]*multiplier)), mode="bilinear")
        mask = rearrange(mask, "b c h w -> b (h w) c")
    else:
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(round(shape[-2]*multiplier), round(shape[-1]*multiplier)), mode="bilinear")
    if match_dim1:
        if match_shape and len(shape) < 4:
            raise Exception(f"match_dim1 cannot be True if shape is under 4 dims; was {len(shape)}.")
        mask = torch.cat([mask] * shape[1], dim=1)
    if match_shape and len(shape) == 3 and len(mask.shape) != 3:
        mask = mask.squeeze(1)
    return mask


# applies min-max normalization, from:
# https://stackoverflow.com/questions/68791508/min-max-normalization-of-a-tensor-in-pytorch
def normalize_min_max(x: Tensor, new_min = 0.0, new_max = 1.0):
    x_min, x_max = x.min(), x.max()
    return (((x - x_min)/(x_max - x_min)) * (new_max - new_min)) + new_min

def linear_conversion(x, x_min=0.0, x_max=1.0, new_min=0.0, new_max=1.0):
    return (((x - x_min)/(x_max - x_min)) * (new_max - new_min)) + new_min

def extend_to_batch_size(tensor: Tensor, batch_size: int):
    if tensor.shape[0] > batch_size:
        return tensor[:batch_size]
    elif tensor.shape[0] < batch_size:
        remainder = batch_size-tensor.shape[0]
        return torch.cat([tensor] + [tensor[-1:]]*remainder, dim=0)
    return tensor

def broadcast_image_to_extend(tensor, target_batch_size, batched_number, except_one=True):
    current_batch_size = tensor.shape[0]
    #print(current_batch_size, target_batch_size)
    if except_one and current_batch_size == 1:
        return tensor

    per_batch = target_batch_size // batched_number
    tensor = tensor[:per_batch]

    if per_batch > tensor.shape[0]:
        tensor = extend_to_batch_size(tensor=tensor, batch_size=per_batch)

    current_batch_size = tensor.shape[0]
    if current_batch_size == target_batch_size:
        return tensor
    else:
        return torch.cat([tensor] * batched_number, dim=0)


# from https://stackoverflow.com/a/24621200
def deepcopy_with_sharing(obj, shared_attribute_names, memo=None):
    '''
    Deepcopy an object, except for a given list of attributes, which should
    be shared between the original object and its copy.

    obj is some object
    shared_attribute_names: A list of strings identifying the attributes that
        should be shared between the original and its copy.
    memo is the dictionary passed into __deepcopy__.  Ignore this argument if
        not calling from within __deepcopy__.
    '''
    assert isinstance(shared_attribute_names, (list, tuple))

    shared_attributes = {k: getattr(obj, k) for k in shared_attribute_names}

    if hasattr(obj, '__deepcopy__'):
        # Do hack to prevent infinite recursion in call to deepcopy
        deepcopy_method = obj.__deepcopy__
        obj.__deepcopy__ = None

    for attr in shared_attribute_names:
        del obj.__dict__[attr]

    clone = deepcopy(obj)

    for attr, val in shared_attributes.items():
        setattr(obj, attr, val)
        setattr(clone, attr, val)

    if hasattr(obj, '__deepcopy__'):
        # Undo hack
        obj.__deepcopy__ = deepcopy_method
        del clone.__deepcopy__

    return clone


def get_sorted_list_via_attr(objects: list, attr: str) -> list:
    if not objects:
        return objects
    elif len(objects) <= 1:
        return [x for x in objects]
    # now that we know we have to sort, do it following these rules:
    # a) if objects have same value of attribute, maintain their relative order
    # b) perform sorting of the groups of objects with same attributes
    unique_attrs = {}
    for o in objects:
        val_attr = getattr(o, attr)
        attr_list: list = unique_attrs.get(val_attr, list())
        attr_list.append(o)
        if val_attr not in unique_attrs:
            unique_attrs[val_attr] = attr_list
    # now that we have the unique attr values grouped together in relative order, sort them by key
    sorted_attrs = dict(sorted(unique_attrs.items()))
    # now flatten out the dict into a list to return
    sorted_list = []
    for object_list in sorted_attrs.values():
        sorted_list.extend(object_list)
    return sorted_list


# DFS Search for Torch.nn.Module, Written by Lvmin
def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class WeightTypeException(TypeError):
    "Raised when weight not compatible with AdvancedControlBase object"
    pass


class AdvancedControlBase:
    def __init__(self, base: ControlBase, timestep_keyframes: TimestepKeyframeGroup, weights_default: ControlWeights, require_model=False, require_vae=False, allow_condhint_latents=False):
        self.base = base
        self.compatible_weights = [ControlWeightType.UNIVERSAL, ControlWeightType.DEFAULT]
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
        self.t: float = None
        self.prev_t: float = None
        self.batched_number: Union[int, IntWithCondOrUncond] = None
        self.batch_size: int = 0
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
        self.control_merge = self.control_merge_inject
        self.pre_run = self.pre_run_inject
        self.cleanup = self.cleanup_inject
        self.set_previous_controlnet = self.set_previous_controlnet_inject
        self.set_cond_hint = self.set_cond_hint_inject
        # vae to store
        self.adv_vae = None
        # require model/vae to be passed into Apply Advanced ControlNet 🛂🅐🅒🅝 node
        self.require_model = require_model
        self.require_vae = require_vae
        self.allow_condhint_latents = allow_condhint_latents
        # disarm - when set to False, used to force usage of Apply Advanced ControlNet 🛂🅐🅒🅝 node (which will set it to True)
        self.disarmed = not require_model
    
    def patch_model(self, model: ModelPatcher):
        pass

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
                    msg = f"Weight on Timestep Keyframe with start_percent={tk.start_percent} is type " + \
                        f"{tk.control_weights.weight_type}, but loaded {type(self).__name__} only supports {self.compatible_weights} weights."
                    raise WeightTypeException(msg)

    def set_timestep_keyframes(self, timestep_keyframes: TimestepKeyframeGroup):
        self.timestep_keyframes = timestep_keyframes if timestep_keyframes else TimestepKeyframeGroup()
        # prepare first timestep_keyframe related stuff
        self._current_timestep_keyframe = None
        self._current_timestep_index = -1
        self._current_used_steps = 0
        self.weights = None
        self.latent_keyframes = None

    def prepare_current_timestep(self, t: Tensor, batched_number: int=1):
        self.t = float(t[0])
        # check if t has changed (otherwise do nothing, as step already accounted for)
        if self.t == self.prev_t:
            return
        # get current step percent
        curr_t: float = self.t
        prev_index = self._current_timestep_index
        # if met guaranteed steps (or no current keyframe), look for next keyframe in case need to switch
        if self._current_timestep_keyframe is None or self._current_used_steps >= self._current_timestep_keyframe.guarantee_steps:
            # if has next index, loop through and see if need to switch
            if self.timestep_keyframes.has_index(self._current_timestep_index+1):
                for i in range(self._current_timestep_index+1, len(self.timestep_keyframes)):
                    eval_tk = self.timestep_keyframes[i]
                    # check if start percent is less or equal to curr_t
                    if eval_tk.start_t >= curr_t:
                        self._current_timestep_index = i
                        self._current_timestep_keyframe = eval_tk
                        self._current_used_steps = 0
                        # keep track of control weights, latent keyframes, and masks,
                        # accounting for inherit_missing
                        if self._current_timestep_keyframe.has_control_weights():
                            self.weights = self._current_timestep_keyframe.control_weights
                        elif not self._current_timestep_keyframe.inherit_missing:
                            self.weights = self.weights_default
                        if self._current_timestep_keyframe.has_latent_keyframes():
                            self.latent_keyframes = self._current_timestep_keyframe.latent_keyframes
                        elif not self._current_timestep_keyframe.inherit_missing:
                            self.latent_keyframes = None
                        if self._current_timestep_keyframe.has_mask_hint():
                            self.tk_mask_cond_hint_original = self._current_timestep_keyframe.mask_hint_orig
                        elif not self._current_timestep_keyframe.inherit_missing:
                            del self.tk_mask_cond_hint_original
                            self.tk_mask_cond_hint_original = None
                        # if guarantee_steps greater than zero, stop searching for other keyframes
                        if self._current_timestep_keyframe.guarantee_steps > 0:
                            break
                    # if eval_tk is outside of percent range, stop looking further
                    else:
                        break
        # update prev_t
        self.prev_t = self.t
        # update steps current keyframe is used
        self._current_used_steps += 1
        # if index changed, apply overrides
        if prev_index != self._current_timestep_index:
            if self.weights_override is not None:
                self.weights = self.weights_override
            if self.latent_keyframe_override is not None:
                self.latent_keyframes = self.latent_keyframe_override

        # make sure weights and latent_keyframes are in a workable state
        # Note: each AdvancedControlBase should create their own get_universal_weights class
        self.prepare_weights()
    
    def prepare_weights(self):
        if self.weights is None:
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

    def set_cond_hint_inject(self, *args, **kwargs):
        to_return = self.base.set_cond_hint(*args, **kwargs)
        # if vae required, look in args and kwargs for it
        if self.require_vae:
            # check args first, as that's the default way vae param is used in ComfyUI
            for arg in args:
                if isinstance(arg, VAE):
                    self.adv_vae = arg
                    break
            # if not in args, check kwargs now
            if self.adv_vae is None:
                if 'vae' in kwargs:
                    self.adv_vae = kwargs['vae']
        return to_return

    def pre_run_inject(self, model, percent_to_timestep_function):
        self.base.pre_run(model, percent_to_timestep_function)
        self.pre_run_advanced(model, percent_to_timestep_function)
    
    def pre_run_advanced(self, model, percent_to_timestep_function):
        # for each timestep keyframe, calculate the start_t
        for tk in self.timestep_keyframes.keyframes:
            tk.start_t = percent_to_timestep_function(tk.start_percent)
        # clear variables
        self.cleanup_advanced()

    def set_previous_controlnet_inject(self, *args, **kwargs):
        to_return = self.base.set_previous_controlnet(*args, **kwargs)
        if not self.disarmed:
            raise Exception(f"Type '{type(self).__name__}' must be used with Apply Advanced ControlNet 🛂🅐🅒🅝 node (with model_optional passed in); otherwise, it will not work.")
        return to_return
    
    def disarm(self):
        self.disarmed = True

    def should_run(self):
        if math.isclose(self.strength, 0.0) or math.isclose(self._current_timestep_keyframe.strength, 0.0):
            return False
        if self.timestep_range is not None:
            if self.t > self.timestep_range[0] or self.t < self.timestep_range[1]:
                return False
        return True

    def get_control_inject(self, x_noisy, t, cond, batched_number):
        self.batched_number = batched_number
        self.batch_size = len(t)
        # prepare timestep and everything related
        self.prepare_current_timestep(t=t, batched_number=batched_number)
        # if should not perform any actions for the controlnet, exit without doing any work
        if self.strength == 0.0 or self._current_timestep_keyframe.strength == 0.0:
            return self.default_control_actions(x_noisy, t, cond, batched_number)
        # otherwise, perform normal function
        return self.get_control_advanced(x_noisy, t, cond, batched_number)

    def get_control_advanced(self, x_noisy, t, cond, batched_number):
        return self.default_control_actions(x_noisy, t, cond, batched_number)

    def default_control_actions(self, x_noisy, t, cond, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)
        return control_prev

    def calc_weight(self, idx: int, x: Tensor, control: dict[str, list[Tensor]], key: str) -> Union[float, Tensor]:
        if self.weights.weight_mask is not None:
            # prepare weight mask
            self.prepare_weight_mask_cond_hint(x, self.batched_number)
            # adjust mask for current layer and return
            return torch.pow(self.weight_mask_cond_hint, self.get_calc_pow(idx=idx, control=control, key=key))
        return self.weights.get(idx=idx, control=control, key=key)
    
    def get_calc_pow(self, idx: int, control: dict[str, list[Tensor]], key: str) -> int:
        if key == "middle":
            return 0
        else:
            c_len = len(control[key])
            real_idx = c_len-idx
            if key == "input":
                real_idx = c_len - real_idx + 1
            return real_idx

    def calc_latent_keyframe_mults(self, x: Tensor, batched_number: int) -> Tensor:
        # apply strengths, and get batch indeces to null out
        # AKA latents that should not be influenced by ControlNet
        final_mults = [1.0] * x.shape[0]
        if self.latent_keyframes:
            latent_count = x.shape[0] // batched_number
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
                    final_mults[(latent_count*b)+real_index] = keyframe.strength
            # null them out by multiplying by null_latent_kf_strength
            for batch_index in indeces_to_null:
                # apply null for each batched cond/uncond
                for b in range(batched_number):
                    final_mults[(latent_count*b)+batch_index] = self._current_timestep_keyframe.null_latent_kf_strength
        # convert final_mults into tensor and match expected dimension count
        final_tensor = torch.tensor(final_mults, dtype=x.dtype, device=x.device)
        while len(final_tensor.shape) < len(x.shape):
            final_tensor = final_tensor.unsqueeze(-1)
        return final_tensor

    def apply_advanced_strengths_and_masks(self, x: Tensor, batched_number: int, flux_shape: tuple=None):
        # handle weight's uncond_multiplier, if applicable
        if self.weights.has_uncond_multiplier:
            cond_or_uncond = self.batched_number.cond_or_uncond
            actual_length = x.size(0) // batched_number
            for idx, cond_type in enumerate(cond_or_uncond):
                # if uncond, set to weight's uncond_multiplier
                if cond_type == 1:
                    x[actual_length*idx:actual_length*(idx+1)] *= self.weights.uncond_multiplier
        if self.weights.has_uncond_mask:
            pass

        if self.latent_keyframes is not None:
            x[:] = x[:] * self.calc_latent_keyframe_mults(x=x, batched_number=batched_number)
        # apply masks, resizing mask to required dims
        if self.mask_cond_hint is not None:
            masks = prepare_mask_batch(self.mask_cond_hint, x.shape, match_shape=True, flux_shape=flux_shape)
            x[:] = x[:] * masks
        if self.tk_mask_cond_hint is not None:
            masks = prepare_mask_batch(self.tk_mask_cond_hint, x.shape, match_shape=True, flux_shape=flux_shape)
            x[:] = x[:] * masks
        # apply timestep keyframe strengths
        if self._current_timestep_keyframe.strength != 1.0:
            x[:] *= self._current_timestep_keyframe.strength
    
    def control_merge_inject(self: 'AdvancedControlBase', control: dict[str, list[Tensor]], control_prev: dict, output_dtype):
        out = {'input':[], 'middle':[], 'output': []}

        for key in control:
            control_output = control[key]
            applied_to = set()
            for i in range(len(control_output)):
                x = control_output[i]
                if x is not None:
                    if self.global_average_pooling:
                        x = torch.mean(x, dim=(2, 3), keepdim=True).repeat(1, 1, x.shape[2], x.shape[3])

                    if x not in applied_to: #memory saving strategy, allow shared tensors and only apply strength to shared tensors once
                        applied_to.add(x)
                        self.apply_advanced_strengths_and_masks(x, self.batched_number)
                        x *= self.strength * self.calc_weight(i, x, control, key)

                    if output_dtype is not None and x.dtype != output_dtype:
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
                                o[i] = prev_val + o[i]  # TODO from base ComfyUI: change back to inplace add if shared tensors stop being an issue
        return out

    def prepare_mask_cond_hint(self, x_noisy: Tensor, t, cond, batched_number, dtype=None, direct_attn=False):
        self._prepare_mask("mask_cond_hint", self.mask_cond_hint_original, x_noisy, t, cond, batched_number, dtype, direct_attn=direct_attn)
        self.prepare_tk_mask_cond_hint(x_noisy, t, cond, batched_number, dtype, direct_attn=direct_attn)

    def prepare_tk_mask_cond_hint(self, x_noisy: Tensor, t, cond, batched_number, dtype=None, direct_attn=False):
        return self._prepare_mask("tk_mask_cond_hint", self._current_timestep_keyframe.mask_hint_orig, x_noisy, t, cond, batched_number, dtype, direct_attn=direct_attn)

    def prepare_weight_mask_cond_hint(self, x_noisy: Tensor, batched_number, dtype=None):
        return self._prepare_mask("weight_mask_cond_hint", self.weights.weight_mask, x_noisy, t=None, cond=None, batched_number=batched_number, dtype=dtype, direct_attn=True)

    def _prepare_mask(self, attr_name, orig_mask: Tensor, x_noisy: Tensor, t, cond, batched_number, dtype=None, direct_attn=False):
        # make mask appropriate dimensions, if present
        if orig_mask is not None:
            out_mask = getattr(self, attr_name)
            multiplier = 1 if direct_attn else 8
            if self.sub_idxs is not None or out_mask is None or x_noisy.shape[2] * multiplier != out_mask.shape[1] or x_noisy.shape[3] * multiplier != out_mask.shape[2]:
                self._reset_attr(attr_name)
                del out_mask
                # TODO: perform upscale on only the sub_idxs masks at a time instead of all to conserve RAM
                # resize mask and match batch count
                out_mask = prepare_mask_batch(orig_mask, x_noisy.shape, multiplier=multiplier, match_shape=True)
                actual_latent_length = x_noisy.shape[0] // batched_number
                out_mask = extend_to_batch_size(out_mask, actual_latent_length if self.sub_idxs is None else self.full_latent_length)
                if self.sub_idxs is not None:
                    out_mask = out_mask[self.sub_idxs]
            # make cond_hint_mask length match x_noise
            if x_noisy.shape[0] != out_mask.shape[0]:
                out_mask = broadcast_image_to_extend(out_mask, x_noisy.shape[0], batched_number)
            # default dtype to be same as x_noisy
            if dtype is None:
                dtype = x_noisy.dtype
            setattr(self, attr_name, out_mask.to(dtype=dtype).to(x_noisy.device))
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
        self.prev_t = None
        self.batched_number = None
        self.batch_size = 0
        self.weights = None
        self.latent_keyframes = None
        # timestep stuff
        self._current_timestep_keyframe = None
        self._current_timestep_index = -1
        self._current_used_steps = 0
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
        copied.adv_vae = self.adv_vae
        copied.require_vae = self.require_vae
        copied.allow_condhint_latents = self.allow_condhint_latents
        copied.disarmed = self.disarmed
