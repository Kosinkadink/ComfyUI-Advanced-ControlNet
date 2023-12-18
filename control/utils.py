from typing import Callable, Union
import torch
from torch import Tensor
import torch.nn.functional as F
import comfy.ops
import comfy.utils


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
