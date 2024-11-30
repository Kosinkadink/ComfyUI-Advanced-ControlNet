from .adv_control.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .adv_control import documentation
from .adv_control.dinklink import init_dinklink
from .adv_control.sampling import prepare_dinklink_acn_wrapper

WEB_DIRECTORY = "./web"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]
documentation.format_descriptions(NODE_CLASS_MAPPINGS)

init_dinklink()
prepare_dinklink_acn_wrapper()
