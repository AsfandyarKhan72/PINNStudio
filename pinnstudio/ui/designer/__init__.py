"""The network designer's views: the canvas, the palette, the inspector and the panel that holds them.

Views draw and collect input; every change goes through `pinnstudio.ui.controllers.designer` (rule R5
of tests/test_mvc.py). They read the state (`pinnstudio.ui.state.design_state`) and redraw on its
signals.
"""
