# SPDX-License-Identifier: LGPL-2.1-only
#
# Multi-scale Fourier feature networks (MsFFN, STMsFFN) for PINNStudio's generated scripts.
#
# This file is NOT under PINNStudio's MIT License. It is distributed under the GNU Lesser General Public
# License version 2.1 (LGPL-2.1); the full text is in LICENSES/LGPL-2.1.txt. The rest of PINNStudio is
# MIT-licensed (LICENSE); see THIRD_PARTY_LICENSES.md.
#
# Provenance. The network class in MSFFN_CODE implements the method of S. Wang, H. Wang and P. Perdikaris,
# "On the eigenvector bias of Fourier feature networks: From regression to solving multi-scale PDEs with
# physics-informed neural networks", Comput. Methods Appl. Mech. Eng. 384 (2021) 113938. It follows the
# interface and structure of DeepXDE's MsFFN and STMsFFN (deepxde/nn/pytorch/msffn.py,
# https://github.com/lululxvi/deepxde, by the DeepXDE developers, LGPL-2.1). It was generated with an AI
# assistant during PINNStudio's development, and it is not known whether DeepXDE's source was consulted, so
# it is treated as possibly derived from DeepXDE and distributed under DeepXDE's license. This does not
# state that DeepXDE code was copied.
#
# Changes compared with DeepXDE's implementation (PINNStudio, 2026-10-12): one class for both variants; the
# fixed Fourier matrices are registered as buffers, so a saved model restores with the same features; the
# output layer is sized for any number of scales; no dropout. It was written so that DeepXDE releases
# before 1.15.1, whose PyTorch backend has no MsFFN, work too.
#
# This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser
# General Public License version 2.1 as published by the Free Software Foundation. This library is
# distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
# for more details.
#
# MSFFN_CODE is source code that pinnstudio/core/codegen.py writes, unchanged and with the notice at its
# top, into every generated script that uses an MsFFN or STMsFFN network.

MSFFN_CODE = """
# ---- LGPL-2.1 section: class _MsFFN (multi-scale Fourier feature networks) ----
# This class is not under PINNStudio's MIT License or the terms of the rest of this script. It is
# distributed under the GNU Lesser General Public License version 2.1 (LGPL-2.1),
# https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html (also in PINNStudio's LICENSES/LGPL-2.1.txt),
# WITHOUT ANY WARRANTY. It implements the method of Wang, Wang & Perdikaris, CMAME 384 (2021) 113938, and
# follows DeepXDE's MsFFN/STMsFFN (deepxde/nn/pytorch/msffn.py, https://github.com/lululxvi/deepxde, the
# DeepXDE developers, LGPL-2.1), from which it may be derived. Changes by PINNStudio (2026-10-12): one class
# for both variants, buffers for the fixed Fourier matrices, an output layer sized for any number of scales,
# no dropout. Keep this notice with the class. See PINNStudio's THIRD_PARTY_LICENSES.md.
class _MsFFN(dde.nn.NN):
    # Input -> [cos(x B_k), sin(x B_k)] for each scale sigma_k (B_k ~ N(0, sigma_k^2),
    # fixed) -> the same hidden layers for every scale -> concatenated -> linear output.
    # Space-time version (sigmas_t given): x and t get their own features,
    # and every (space, time) pair of branches is multiplied point-wise.
    def __init__(self, layer_sizes, activation, kernel_initializer, sigmas, sigmas_t=None):
        super().__init__()
        self.activation = dde.nn.activations.get(activation)
        _init = dde.nn.initializers.get(kernel_initializer)
        _zero = dde.nn.initializers.get("zeros")
        self.space_time = bool(sigmas_t)
        _n_ff = layer_sizes[1] // 2
        _n_x = layer_sizes[0] - (1 if self.space_time else 0)
        self.n_x_scales = len(sigmas)
        for _k, _s in enumerate(sigmas):
            self.register_buffer(f"B_x{_k}", torch.randn(_n_x, _n_ff) * float(_s))
        self.n_t_scales = len(sigmas_t or [])
        for _k, _s in enumerate(sigmas_t or []):
            self.register_buffer(f"B_t{_k}", torch.randn(1, _n_ff) * float(_s))
        self.linears = torch.nn.ModuleList()
        for _i in range(2, len(layer_sizes) - 1):
            _lin = torch.nn.Linear(layer_sizes[_i - 1], layer_sizes[_i])
            _init(_lin.weight); _zero(_lin.bias)
            self.linears.append(_lin)
        _n_branches = self.n_x_scales * (self.n_t_scales if self.space_time else 1)
        self.dense = torch.nn.Linear(layer_sizes[-2] * _n_branches, layer_sizes[-1])
        _init(self.dense.weight); _zero(self.dense.bias)

    def _branch(self, _x, _B):
        _p = _x @ _B
        _y = torch.cat([torch.cos(_p), torch.sin(_p)], dim=1)
        for _lin in self.linears:
            _y = self.activation(_lin(_y))
        return _y

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        if self.space_time:
            _yx = [self._branch(x[:, :-1], getattr(self, f"B_x{_k}")) for _k in range(self.n_x_scales)]
            _yt = [self._branch(x[:, -1:], getattr(self, f"B_t{_k}")) for _k in range(self.n_t_scales)]
            _ys = [_a * _b for _a in _yx for _b in _yt]
        else:
            _ys = [self._branch(x, getattr(self, f"B_x{_k}")) for _k in range(self.n_x_scales)]
        y = self.dense(torch.cat(_ys, dim=1))
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y
# ---- end of the LGPL-2.1 section ----
"""
