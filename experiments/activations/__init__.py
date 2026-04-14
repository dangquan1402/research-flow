"""Activation function registry."""

ACTIVATION_REGISTRY = {}


def register_activation(name):
    def decorator(fn):
        ACTIVATION_REGISTRY[name] = fn
        return fn
    return decorator


def get_activation(name):
    if name not in ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unknown activation: {name}. Available: {list(ACTIVATION_REGISTRY.keys())}"
        )
    return ACTIVATION_REGISTRY[name]


from . import gelu as gelu, swiglu as swiglu, relu_squared as relu_squared  # noqa: E402
