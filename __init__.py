from .sapsan_vfi_nodes import SnJakeSapsanVFICheckpointLoader, SnJakeSapsanVFIInterpolate

NODE_CLASS_MAPPINGS = {
    "SnJakeSapsanVFICheckpointLoader": SnJakeSapsanVFICheckpointLoader,
    "SnJakeSapsanVFIInterpolate": SnJakeSapsanVFIInterpolate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SnJakeSapsanVFICheckpointLoader": "😎 Sapsan-VFI Loader",
    "SnJakeSapsanVFIInterpolate": "😎 Sapsan-VFI Interpolate",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

