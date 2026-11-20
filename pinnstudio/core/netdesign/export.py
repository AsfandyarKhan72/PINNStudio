"""A designed network as a standalone Python module.

The module defines the network exactly as the generated scripts do (the same `_DesignedNet` source, so
a model PINNStudio trained loads into it), plus `build_net()` and `load_trained(path)`. It needs PyTorch
and DeepXDE (the network subclasses `dde.nn.NN`), not PINNStudio.
"""
from __future__ import annotations

from pinnstudio.core.netdesign import runtime as R


def module_source(design, input_names, output_names, activation="tanh", kernel_initializer="Glorot uniform",
                  weight_decay=0.0) -> str:
    body = R.net_source(design, input_names, output_names)
    return f'''"""A neural network designed in PINNStudio's Network tab.

Inputs:  {", ".join(input_names)}
Outputs: {", ".join(output_names)}

    net = build_net()                       # new weights
    net = load_trained("model-10000.pt")    # the weights a PINNStudio run saved

The network is a deepxde.nn.NN: give it to dde.Model(data, net) to train it with DeepXDE.
"""
import os

os.environ.setdefault("DDE_BACKEND", "pytorch")
import deepxde as dde  # noqa: E402
import torch  # noqa: E402

{body}

ACTIVATION = {activation!r}
KERNEL_INITIALIZER = {kernel_initializer!r}
WEIGHT_DECAY = {float(weight_decay)!r}


def build_net():
    """The designed network with new weights."""
    return _DesignedNet(_DESIGN, ACTIVATION, KERNEL_INITIALIZER, WEIGHT_DECAY)


def load_trained(path):
    """The designed network with the weights of a checkpoint PINNStudio saved (model-<step>.pt)."""
    net = build_net()
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("model_state_dict", state)
    net.load_state_dict(state)
    net.eval()
    return net


if __name__ == "__main__":
    _net = build_net()
    print(_net)
    print("trainable parameters:", _net.num_trainable_parameters())
'''
