import gc

try:
    import torch
except Exception:
    torch = None


class RAMCleanup:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": ("ANY",),
                "clean_file_cache": ("BOOLEAN", {"default": True}),
                "clean_processes": ("BOOLEAN", {"default": True}),
                "clean_dlls": ("BOOLEAN", {"default": True}),
                "retry_times": ("INT", {"default": 3, "min": 0, "max": 10}),
            }
        }

    RETURN_TYPES = ("ANY",)
    FUNCTION = "cleanup"
    CATEGORY = "VanillaMotion"

    def cleanup(
        self,
        anything,
        clean_file_cache=True,
        clean_processes=True,
        clean_dlls=True,
        retry_times=3,
    ):
        _ = (clean_file_cache, clean_processes, clean_dlls, retry_times)
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (anything,)


class UnloadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("ANY",),
                "model": ("ANY",),
            }
        }

    RETURN_TYPES = ("ANY",)
    FUNCTION = "unload"
    CATEGORY = "VanillaMotion"

    def unload(self, value, model):
        _ = model
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (value,)


NODE_CLASS_MAPPINGS = {
    "RAMCleanup": RAMCleanup,
    "UnloadModel": UnloadModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RAMCleanup": "RAMCleanup",
    "UnloadModel": "UnloadModel",
}
