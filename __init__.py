from .adv_control.nodes import AdvancedControlNetExtension
from .adv_control.dinklink import init_dinklink
from .adv_control.sampling import prepare_dinklink_acn_wrapper

init_dinklink()
prepare_dinklink_acn_wrapper()


async def comfy_entrypoint() -> AdvancedControlNetExtension:
    return AdvancedControlNetExtension()
