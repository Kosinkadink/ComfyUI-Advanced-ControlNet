class AnimateDiffLoaderWithContext:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("MODEL",)
    CATEGORY = ""