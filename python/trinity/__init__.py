"""Trinity 2.0 Python façade over the native C++ Howard engine."""

from .defaults import default_config

__all__ = ["Engine", "default_config"]


def __getattr__(name: str):
    if name == "Engine":
        from .engine import Engine
        return Engine
    raise AttributeError(name)
