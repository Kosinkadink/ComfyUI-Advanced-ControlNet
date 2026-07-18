from comfy_api.latest import io
import folder_paths

from .control_ctrlora import load_ctrlora

class CtrLoRALoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ACN_CtrLoRALoader',
            display_name='Load CtrLoRA Model 🛂🅐🅒🅝',
            category='Adv-ControlNet 🛂🅐🅒🅝/CtrLoRA',
            inputs=[
                io.Combo.Input('base', options=folder_paths.get_filename_list("controlnet")),
                io.Combo.Input('lora', options=folder_paths.get_filename_list("controlnet"))
            ],
            outputs=[
                io.ControlNet.Output('CONTROL_NET', is_output_list=False)
            ]
        )
    

    @classmethod
    def execute(cls, base: str, lora: str):
        base_path = folder_paths.get_full_path("controlnet", base)
        lora_path = folder_paths.get_full_path("controlnet", lora)
        controlnet = load_ctrlora(base_path, lora_path)
        return io.NodeOutput(controlnet,)
