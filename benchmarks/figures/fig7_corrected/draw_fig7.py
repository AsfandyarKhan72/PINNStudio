"""Draws Figure 7 from fig7_corrected.json with the approved figure tool, unchanged:
make_methods_figure_070b284.py is tools/make_methods_figure.py at commit 070b284 (branch review/figures, the version
that drew the manuscript's figure; SHA-256 653d006acbf9ce027082335b7fa0c03ce5f0b936a0a47c6ced5e57fbf039404c).
Only the data file and the output folder are set from here; layout, size, fonts, colours, markers, legend and the
axis-limit rule are the tool's own.

  python benchmarks/figures/fig7_corrected/draw_fig7.py [data.json] [output name]

Defaults: fig7_corrected.json -> paper_figure7_corrected.{pdf,png} in this folder. With
benchmarks/figures/fig7_v2.json at 5d9dac0 as data it redraws the approved figure (zero pixel difference at
150 dpi).
"""
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_methods_figure_070b284 as m  # noqa: E402

data = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "fig7_corrected.json")
name = sys.argv[2] if len(sys.argv) > 2 else "paper_figure7_corrected"
m.RERUN = data
m.PREFIX = {data: "f7v2"}
m.OUT = HERE
ex = m.rerun_examples(path=data, setups=[(s[0],) for s in m.FIG7_SETUPS])
m.fig7(ex, name=name)
print("wrote", os.path.join(HERE, name + ".pdf / .png"))
