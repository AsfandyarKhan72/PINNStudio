"""The network designer's domain: a neural network drawn as blocks and connections.

UI-free (tests/test_mvc.py, rule R1): the design model (`design`), the layer types and their parameters
(`layers`, `params`), the widths each block gives and whether a connection is legal (`shapes`), the
ready-made designs (`presets`), the automatic layout (`layout`), the PyTorch source the generated
scripts embed (`runtime`), the standalone export (`export`), and the files around a design: the block
library (`library`) and the record of trusted code and files (`trust`).

A design is stored in PINNConfig.network_design as JSON (a plain dict); `design.Design` is its checked
form. The network maps the problem's inputs (x, y, z, t, or the features of an input transform) to its
outputs (u, v, p, ...): every value in it is a vector, so a block's shape is just its width.
"""
