import folder_paths

from .control_ctrlora import load_ctrlora


class CtrLoRALoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base": (folder_paths.get_filename_list("controlnet"), ),
                "lora": (folder_paths.get_filename_list("controlnet"), ),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet_plusplus"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/CtrLoRA"

    def load_controlnet_plusplus(self, base: str, lora: str):
        base_path = folder_paths.get_full_path("controlnet", base)
        lora_path = folder_paths.get_full_path("controlnet", lora)
        controlnet = load_ctrlora(base_path, lora_path)
        return (controlnet,)
