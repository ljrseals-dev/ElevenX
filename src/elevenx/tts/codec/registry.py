"""String-name codec registry with lazy built-in discovery."""
from typing import Dict, Type
from .interface import BaseCodec

_REGISTRY: Dict[str, Type[BaseCodec]] = {}

def register_codec(name: str):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("codec name must be a non-empty string")
    key = name.strip().lower()
    def wrapper(cls):
        _REGISTRY[key] = cls
        return cls
    return wrapper

def get_codec(name: str) -> Type[BaseCodec]:
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"Codec '{name}' not registered. Available: {list_codecs()}")
    return _REGISTRY[key]

def list_codecs():
    return sorted(_REGISTRY.keys())

def autoload() -> None:
    import importlib
    for mod in ("elevenx.tts.codec.baseline", "elevenx.tts.codec.harmonia"):
        try:
            importlib.import_module(mod)
        except ImportError:
            continue
