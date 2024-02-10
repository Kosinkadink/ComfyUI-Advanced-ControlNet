from comfy.cldm.cldm import ControlNet as ControlNetCLDM


class SVDControlNet(ControlNetCLDM):
    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)