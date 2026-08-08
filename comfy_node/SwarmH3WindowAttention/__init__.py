"""ComfyUI registration for SwarmUI's MiniMax H3 window-attention prototype."""

from .window_attention import H3WindowAttentionPatch


NODE_CLASS_MAPPINGS = {
    "H3WindowAttentionPatch": H3WindowAttentionPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3WindowAttentionPatch": "MiniMax H3 Window Attention (Prototype)",
}
