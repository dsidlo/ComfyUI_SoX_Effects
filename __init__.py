import sys
import os

# Conditional import: relative if in package context, absolute otherwise
if __package__:
    # Normal package import (e.g., by ComfyUI)
    import importlib
    from .src import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    # Direct script run: add current dir to path, use absolute
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

if __name__ == "__main__":
    print("Root __init__.py executed! Mappings loaded.")
    print(f"NODE_CLASS_MAPPINGS keys: {list(NODE_CLASS_MAPPINGS.keys())}")
