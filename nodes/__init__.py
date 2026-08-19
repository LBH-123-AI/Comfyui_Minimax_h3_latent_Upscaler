import logging

# 2D node: legacy API
from .minimax_h3_latent_upscaler_2d import (
    NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_2D,
    NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_2D,
)

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_2D)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(NODE_DISPLAY_NAME_MAPPINGS_2D)

# 3D node: legacy API (same registration mechanism as 2D)
try:
    from .minimax_h3_latent_upscaler_3d import (
        NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_3D,
        NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_3D,
    )
    NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_3D)
    NODE_DISPLAY_NAME_MAPPINGS.update(NODE_DISPLAY_NAME_MAPPINGS_3D)
except Exception as e:
    logging.error(f"[MinimaxH3] Failed to import 3D node: {e}")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
