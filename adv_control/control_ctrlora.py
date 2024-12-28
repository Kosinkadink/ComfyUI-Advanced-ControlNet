# Core code adapted from CtrLoRA github repo:
# https://github.com/xyfJASON/ctrlora
import torch
from torch import Tensor

from comfy.cldm.cldm import ControlNet as ControlNetCLDM
import comfy.model_detection
import comfy.model_management
import comfy.ops
import comfy.utils

from comfy.ldm.modules.diffusionmodules.util import (
    zero_module,
    timestep_embedding,
)

from .control import ControlNetAdvanced
from .utils import TimestepKeyframeGroup
from .logger import logger


class ControlNetCtrLoRA(ControlNetCLDM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # delete input hint block
        del self.input_hint_block
    
    def forward(self, x: Tensor, hint: Tensor, timesteps, context, y=None, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
        emb = self.time_embed(t_emb)

        out_output = []
        out_middle = []

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)
        
        h = hint.to(dtype=x.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            out_output.append(zero_conv(h, emb, context))
        
        h = self.middle_block(h, emb, context)
        out_middle.append(self.middle_block_out(h, emb, context))

        return {"middle": out_middle, "output": out_output}


class CtrLoRAAdvanced(ControlNetAdvanced):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.require_vae = True
        self.mult_by_ratio_when_vae = False

    def pre_run_advanced(self, model, percent_to_timestep_function):
        super().pre_run_advanced(model, percent_to_timestep_function)
        self.latent_format = model.latent_format  # LatentFormat object, used to process_in latent cond hint

    def cleanup_advanced(self):
        super().cleanup_advanced()
        if self.latent_format is not None:
            del self.latent_format
            self.latent_format = None

    def copy(self):
        c = CtrLoRAAdvanced(self.control_model, self.timestep_keyframes, global_average_pooling=self.global_average_pooling, load_device=self.load_device, manual_cast_dtype=self.manual_cast_dtype)
        c.control_model = self.control_model
        c.control_model_wrapped = self.control_model_wrapped
        self.copy_to(c)
        self.copy_to_advanced(c)
        return c


def load_ctrlora(base_path: str, lora_path: str,
                 base_data: dict[str, Tensor]=None, lora_data: dict[str, Tensor]=None,
                 timestep_keyframe: TimestepKeyframeGroup=None, model=None, model_options={}):
    if base_data is None:
        base_data = comfy.utils.load_torch_file(base_path, safe_load=True)
    controlnet_data = base_data

    # first, check that base_data contains keys with lora_layer
    contains_lora_layers = False
    for key in base_data:
        if "lora_layer" in key:
            contains_lora_layers = True
    if not contains_lora_layers:
        raise Exception(f"File '{base_path}' is not a valid CtrLoRA base model; does not contain any lora_layer keys.")
    
    controlnet_config = None
    supported_inference_dtypes = None

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
        raise Exception("")
        net = load_t2i_adapter(controlnet_data, model_options=model_options)
        if net is None:
            logging.error("error could not detect control model type.")
        return net

    if controlnet_config is None:
        model_config = comfy.model_detection.model_config_from_unet(controlnet_data, prefix, True)
        supported_inference_dtypes = list(model_config.supported_inference_dtypes)
        controlnet_config = model_config.unet_config

    unet_dtype = model_options.get("dtype", None)
    if unet_dtype is None:
        weight_dtype = comfy.utils.weight_dtype(controlnet_data)

        if supported_inference_dtypes is None:
            supported_inference_dtypes = [comfy.model_management.unet_dtype()]

        if weight_dtype is not None:
            supported_inference_dtypes.append(weight_dtype)

        unet_dtype = comfy.model_management.unet_dtype(model_params=-1, supported_dtypes=supported_inference_dtypes)

    load_device = comfy.model_management.get_torch_device()

    manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, load_device)
    operations = model_options.get("custom_operations", None)
    if operations is None:
        operations = comfy.ops.pick_operations(unet_dtype, manual_cast_dtype)

    controlnet_config["operations"] = operations
    controlnet_config["dtype"] = unet_dtype
    controlnet_config["device"] = comfy.model_management.unet_offload_device()
    controlnet_config.pop("out_channels")
    controlnet_config["hint_channels"] = 3
    #controlnet_config["hint_channels"] = controlnet_data["{}input_hint_block.0.weight".format(prefix)].shape[1]
    control_model = ControlNetCtrLoRA(**controlnet_config)

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
                logger.warning("WARNING: Loaded a diff controlnet without a model. It will very likely not work.")

        class WeightsLoader(torch.nn.Module):
            pass
        w = WeightsLoader()
        w.control_model = control_model
        missing, unexpected = w.load_state_dict(controlnet_data, strict=False)
    else:
        missing, unexpected = control_model.load_state_dict(controlnet_data, strict=False)

    if len(missing) > 0:
        logger.warning("missing controlnet keys: {}".format(missing))

    if len(unexpected) > 0:
        logger.debug("unexpected controlnet keys: {}".format(unexpected))

    global_average_pooling = model_options.get("global_average_pooling", False)
    control = CtrLoRAAdvanced(control_model, timestep_keyframe, global_average_pooling=global_average_pooling,
                                 load_device=load_device, manual_cast_dtype=manual_cast_dtype)
    # load lora data onto the controlnet
    if lora_path is not None:
        load_lora_data(control, lora_path)

    return control


def load_lora_data(control: CtrLoRAAdvanced, lora_path: str, loaded_data: dict[str, Tensor]=None, lora_strength=1.0):
    if loaded_data is None:
        loaded_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
    # check that lora_data contains keys with lora_layer
    contains_lora_layers = False
    for key in loaded_data:
        if "lora_layer" in key:
            contains_lora_layers = True
    if not contains_lora_layers:
        raise Exception(f"File '{lora_path}' is not a valid CtrLoRA lora model; does not contain any lora_layer keys.")

    # now that we know we have a ctrlora file, separate keys into 'set' and 'lora' keys
    data_set: dict[str, Tensor] = {}
    data_lora: dict[str, Tensor] = {}

    for key in list(loaded_data.keys()):
        if 'lora_layer' in key:
            data_lora[key] = loaded_data.pop(key)
        else:
            data_set[key] = loaded_data.pop(key)
    # no keys should be left over
    if len(loaded_data) > 0:
        logger.warning("Not all keys from CtrlLoRA lora model's loaded data were parsed!")
    
    # turn set/lora data into corresponding patches;
    patches = {}
    # set will replace the values
    for key, value in data_set.items():
        # prase model key from key;
        # remove "control_model."
        model_key = key.replace("control_model.", "")
        patches[model_key] = ("set", (value,))
    # lora will do mm of up and down tensors
    for down_key in data_lora:
        # only process lora down keys; we will process both up+down at the same time
        if ".up." in key:
            continue
        # get up version of down key
        up_key = down_key.replace(".down.", ".up.")
        # get key that will match up with model key;
        # remove "lora_layer.down." and "control_model."
        model_key = down_key.replace("lora_layer.down.", "").replace("control_model.", "")
        
        weight_down = data_lora[down_key]
        weight_up = data_lora[up_key]
        # currently, ComfyUI expects 6 elements in 'lora' type, but for future-proofing add a bunch more with None
        patches[model_key] = ("lora", (weight_up, weight_down, None, None, None, None,
                                       None, None, None, None, None, None, None, None))
    
    # now that patches are made, add them to model
    control.control_model_wrapped.add_patches(patches, strength_patch=lora_strength)
