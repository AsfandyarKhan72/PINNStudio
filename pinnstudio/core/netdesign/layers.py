"""The layers a stack block can hold: their settings and the width each gives.

A PINN's network maps coordinates to fields, so every value in it is a vector and a layer is described
by the width it gives for the width it takes. How a layer runs is in `runtime.py` (the PyTorch source the
generated scripts embed); the two are kept in step by tests/test_netdesign_torch.py, which builds every
layer and compares its output width with `out_width` here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pinnstudio.core.netdesign.params import Bool, Choice, Float, Int, JsonDict, ParamError, Text, validate

# DeepXDE's activation names, as the Neural network page offers them, plus "inherit" (the problem's own
# activation, set on the Neural network page) and "linear" (none).
ACTIVATIONS = ("inherit", "tanh", "sin", "sigmoid", "relu", "silu", "swish", "elu", "gelu", "selu", "linear")
INITIALIZERS = ("inherit", "Glorot uniform", "Glorot normal", "He uniform", "He normal", "zeros")
WIDTH_MAX = 4096

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"


@dataclass(frozen=True)
class LayerType:
    name: str
    title: str
    params: tuple
    rule: Callable[[int, dict], int]
    description: str = ""

    def out_width(self, width_in: int, params: dict) -> int:
        """The width this layer gives for `width_in`; ParamError with the reason when it cannot take it."""
        if width_in is None or width_in < 1:
            raise ParamError(f"{self.title} needs an input")
        w = self.rule(width_in, params)
        if w < 1:
            raise ParamError(f"{self.title} would give no values")
        return w


def _same(w, p):
    return w


def _custom_width(w, p):
    spec = str(p.get("width", "same")).strip().lower()
    if spec in ("", "same"):
        return w
    try:
        n = int(spec)
    except ValueError:
        raise ParamError("Custom layer: Output width must be 'same' or a whole number")
    if n < 1:
        raise ParamError("Custom layer: Output width must be at least 1")
    return n


def _periodic_width(w, p):
    return 2 * p["harmonics"] * w


_ACT = Choice("activation", "Activation", "inherit", "inherit: the Neural network page's activation", ACTIVATIONS)

LAYERS: dict[str, LayerType] = {lt.name: lt for lt in (
    LayerType("dense", "Dense", (
        Int("units", "Units", 64, "the layer's width", min=1, max=WIDTH_MAX),
        _ACT,
        Bool("bias", "Bias", True),
        Choice("init", "Initializer", "inherit", "inherit: the Neural network page's initializer", INITIALIZERS),
    ), lambda w, p: p["units"], "x W + b, then the activation"),
    LayerType("activation", "Activation", (
        Choice("activation", "Activation", "tanh", "", ACTIVATIONS[1:]),
    ), _same, "the activation alone"),
    LayerType("layer_norm", "Layer normalisation", (), _same,
              "each sample's values to mean 0 and spread 1, then a learned scale and shift"),
    LayerType("scale_shift", "Scale and shift", (
        Float("scale", "Scale", 1.0),
        Float("shift", "Shift", 0.0),
    ), _same, "each value × scale + shift (scale −1 and shift 1 make 1 − z)"),
    LayerType("fourier", "Fourier features", (
        Int("features", "Features", 64, "m: the output is [cos(x B), sin(x B)], 2m values", min=1, max=WIDTH_MAX // 2),
        Float("sigma", "Sigma (σ)", 1.0, "B ~ N(0, σ²): larger σ represents faster oscillations", positive=True),
        Bool("trainable", "Trainable B", False, "off: B is fixed at its random start, as in Wang et al. 2021"),
    ), lambda w, p: 2 * p["features"], "random Fourier features (Tancik et al. 2020, Wang, Wang & Perdikaris 2021)"),
    LayerType("periodic", "Periodic features", (
        Int("harmonics", "Harmonics", 1, "k = 1..K: cos(2πk x / L) and sin(2πk x / L) per input", min=1, max=64),
        Float("period", "Period (L)", 1.0, "the length over which the inputs repeat (the domain's length)",
              positive=True),
    ), _periodic_width, "periodic features: the output repeats exactly with period L"),
    LayerType("adaptive", "Adaptive activation", (
        Choice("activation", "Activation", "tanh", "", ACTIVATIONS[1:-1]),
        Float("scale", "Scale factor (n)", 10.0, "n in σ(n·a·x)", positive=True),
        Float("init", "Starting slope (a₀)", 0.1, "a starts at a₀ so that n·a₀ = 1 by default", positive=True),
    ), _same, "σ(n·a·x) with a learned slope a (Jagtap et al. 2020)"),
    LayerType("custom", "Custom layer", (
        Text("class_name", "Class", "MyLayer", "a torch.nn.Module class in the design's code",
             max_len=80, pattern=_IDENT, pattern_help="a Python class name"),
        JsonDict("kwargs", "Keyword arguments", "{}", "JSON, passed to the class"),
        Text("width", "Output width", "same", "'same' or a whole number", max_len=12),
    ), _custom_width, "a layer in your own PyTorch code (runs only once trusted)"),
)}


def layer_type(name: str) -> LayerType:
    try:
        return LAYERS[name]
    except KeyError:
        raise ParamError(f"unknown layer {name!r}")


def check_params(name: str, params: dict) -> dict:
    lt = layer_type(name)
    return validate(lt.params, params, where=f"{lt.title}: ")


def stack_width(layers, width_in: int) -> int:
    """The width a list of {type, params} layers gives for `width_in`."""
    w = width_in
    for spec in layers:
        lt = layer_type(spec["type"])
        w = lt.out_width(w, check_params(spec["type"], spec.get("params") or {}))
    return w
