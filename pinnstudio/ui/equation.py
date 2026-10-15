"""Equation boxes: the equation typeset as you type, and names it doesn't know.

`to_latex` turns an equation in PINNStudio's syntax (du_xx, u**3,
sin(pi*x), ...) into LaTeX; `EquationPreview` shows it rendered (by
matplotlib's mathtext) under the box, with a warning when the equation
uses a name that is neither an output, a derivative of one, a coordinate,
a constant, an inverse variable nor a math function.
"""
import ast
import re

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from pinnstudio.ui import theme as _theme

FUNCS = {"sin": r"\sin", "cos": r"\cos", "tan": r"\tan", "sinh": r"\sinh", "cosh": r"\cosh", "tanh": r"\tanh",
         "arcsin": r"\arcsin", "arccos": r"\arccos", "arctan": r"\arctan", "exp": r"\exp", "log": r"\ln",
         "log10": r"\log_{10}", "sqrt": None, "abs": None, "frac": None, "integral": None, "gamma": r"\Gamma",
         "lgamma": r"\ln\Gamma", "ceil": None, "floor": None, "erf": r"\mathrm{erf}", "sign": r"\mathrm{sgn}",
         "relu": r"\mathrm{ReLU}", "sigmoid": r"\sigma"}
CONSTS = {"pi": r"\pi", "e": "e"}
COORDS = ("x", "y", "z", "t", "s")
_DERIV = re.compile(r"^d([A-Za-z][A-Za-z0-9]*)_([xyzt]+)$")


def _name(n, outputs):
    m = _DERIV.match(n)
    if m and (not outputs or m.group(1) in outputs):
        var, wrt = m.group(1), m.group(2)
        if len(wrt) == 1:
            return rf"\frac{{\partial {_sym(var)}}}{{\partial {wrt}}}"
        counts = []
        for ch in "xyzt":
            k = wrt.count(ch)
            if k:
                counts.append(rf"\partial {ch}" + (f"^{{{k}}}" if k > 1 else ""))
        return rf"\frac{{\partial^{{{len(wrt)}}} {_sym(var)}}}{{{' '.join(counts)}}}"
    if n in CONSTS:
        return CONSTS[n]
    return _sym(n)


_GREEK = {"alpha", "beta", "gamma", "delta", "epsilon", "lambda", "mu", "nu", "rho", "sigma", "tau", "phi", "psi",
          "omega", "kappa", "theta", "eta", "xi", "zeta", "chi"}


def _sym(n):
    base, _, sub = n.partition("_")
    b = rf"\{base}" if base in _GREEK else (base if len(base) == 1 else rf"\mathrm{{{base}}}")
    return f"{b}_{{{sub}}}" if sub else b


_PREC = {ast.Add: 1, ast.Sub: 1, ast.Mult: 2, ast.Div: 2, ast.Pow: 4}


def _tex(node, outputs, parent=0):
    if isinstance(node, ast.Expression):
        return _tex(node.body, outputs)
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, float):
            s = f"{v:g}"
            if "e" in s:
                mant, ex = s.split("e")
                return rf"{mant} \times 10^{{{int(ex)}}}"
            return s
        return str(v)
    if isinstance(node, ast.Name):
        return _name(node.id, outputs)
    if isinstance(node, ast.UnaryOp):
        inner = _tex(node.operand, outputs, 3)
        return f"-{inner}" if isinstance(node.op, ast.USub) else inner
    if isinstance(node, ast.BinOp):
        p = _PREC.get(type(node.op), 2)
        if isinstance(node.op, ast.Div):
            out = rf"\frac{{{_tex(node.left, outputs)}}}{{{_tex(node.right, outputs)}}}"
            return out
        if isinstance(node.op, ast.Pow):
            base = _tex(node.left, outputs, 5)
            return f"{base}^{{{_tex(node.right, outputs)}}}"
        if isinstance(node.op, ast.Mult):   # 1*u, u*1.0 -> u
            for a, b in ((node.left, node.right), (node.right, node.left)):
                if isinstance(a, ast.Constant) and isinstance(a.value, (int, float)) and a.value == 1:
                    return _tex(b, outputs, parent)
        left = _tex(node.left, outputs, p)
        right = _tex(node.right, outputs, p + (1 if isinstance(node.op, ast.Sub) else 0))
        op = {ast.Add: " + ", ast.Sub: " - ", ast.Mult: r" \, "}.get(type(node.op), " ? ")
        if isinstance(node.op, ast.Mult) and (isinstance(node.right, ast.Constant) or
                                              (isinstance(node.left, ast.Constant) and
                                               isinstance(node.right, ast.Constant))):
            op = r" \cdot "
        out = f"{left}{op}{right}"
        return rf"\left({out}\right)" if p < parent else out
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        f = node.func.id
        args = [_tex(a, outputs) for a in node.args]
        a0 = args[0] if args else ""
        if f == "sqrt":
            return rf"\sqrt{{{a0}}}"
        if f == "abs":
            return rf"\left|{a0}\right|"
        if f == "frac":
            return rf"(-\Delta)^{{\alpha/2}} {a0}"
        if f == "integral":
            return rf"\int_{{t_0}}^{{t}} {a0} \, ds"
        if f in ("ceil", "floor"):
            return rf"\l{f}\left({a0}\right)\r{f}"
        name = FUNCS.get(f) or rf"\mathrm{{{f}}}"
        return rf"{name}\left({', '.join(args)}\right)"
    if isinstance(node, ast.Compare):
        return r"\,?\,"
    raise ValueError("unsupported")


def to_latex(expr, outputs=()):
    tree = ast.parse((expr or "").strip(), mode="eval")
    return _tex(tree, set(outputs))


def unknown_names(expr, known):
    """Names the equation uses that PINNStudio wouldn't know."""
    try:
        tree = ast.parse((expr or "").strip() or "0", mode="eval")
    except SyntaxError:
        return None
    outputs = known.get("outputs", set())
    bad = []
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    for n in ast.walk(tree):
        if not isinstance(n, ast.Name):
            continue
        name = n.id
        if name in calls and name in FUNCS:
            continue
        if name in outputs or name in COORDS or name in CONSTS or name in known.get("other", set()):
            continue
        m = _DERIV.match(name)
        if m and m.group(1) in outputs and all(c in known.get("coords", "xyzt") for c in m.group(2)):
            continue
        if name in ("np", "torch"):
            continue
        if name not in bad:
            bad.append(name)
    return bad


def render(latex, color="#1f2328", size=22):
    """LaTeX (matplotlib mathtext) -> QPixmap, or None."""
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except Exception:
        return None
    fig = Figure(figsize=(0.01, 0.01), dpi=200)
    fig.patch.set_alpha(0)
    fig.text(0, 0, f"${latex} = 0$", fontsize=size * 0.75, color=color, va="bottom", ha="left")
    canvas = FigureCanvasAgg(fig)
    try:
        canvas.draw()
        bbox = fig.texts[0].get_window_extent()
    except Exception:
        return None
    w, h = int(bbox.width) + 8, int(bbox.height) + 8
    fig.set_size_inches(w / 200, h / 200)
    fig.texts[0].set_position((4 / w, 4 / h))
    try:
        canvas.draw()
    except Exception:
        return None
    buf = canvas.buffer_rgba()
    img = QImage(bytes(buf), canvas.get_width_height()[0], canvas.get_width_height()[1],
                 QImage.Format.Format_RGBA8888)
    pm = QPixmap.fromImage(img.copy())
    pm.setDevicePixelRatio(2.0)
    return pm


class EquationPreview(QWidget):
    """Under an equation box: the equation typeset, and a note about names
    it doesn't know. `known_fn()` -> {"outputs": set, "other": set, "coords": str}."""

    def __init__(self, line_edit, known_fn, parent=None):
        super().__init__(parent)
        self.edit = line_edit
        self.known_fn = known_fn
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 0, 2, 4)
        lay.setSpacing(2)
        self.math = QLabel()
        self.math.setObjectName("equationPreview")
        self.math.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        # a long equation is shrunk to fit rather than widening the panel
        self.math.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._pm = None
        lay.addWidget(self.math)
        self.note = QLabel()
        self.note.setObjectName("equationNote")
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(350)
        self._timer.timeout.connect(self.refresh)
        line_edit.textChanged.connect(lambda _t: self._timer.start())
        QTimer.singleShot(0, self.refresh)

    def refresh(self):
        try:
            text = self.edit.text()
        except RuntimeError:
            return
        t = _theme.current()
        try:
            known = self.known_fn() or {}
        except Exception:
            known = {}
        bad = unknown_names(text, known)
        if bad is None:
            self.note.setText("This isn't a complete expression yet.")
            self.note.setStyleSheet(f"color: {t['warning']};")
            self.note.setVisible(True)
            self._pm = None
            self.math.clear()
            return
        if bad:
            self.note.setText("Unknown name" + ("s" if len(bad) > 1 else "") + ": " + ", ".join(bad) +
                              ". Add it as a constant, or check the spelling.")
            self.note.setStyleSheet(f"color: {t['warning']};")
        else:
            self.note.setText("")
        self.note.setVisible(bool(bad))
        try:
            latex = to_latex(text, known.get("outputs", ()))
        except Exception:
            self._pm = None
            self.math.clear()
            return
        self._pm = render(latex, t["text"])
        self._show()

    def _show(self):
        pm = self._pm
        if pm is None:
            self.math.clear()
            return
        avail = max(40, self.width() - 6)
        logical = pm.width() / pm.devicePixelRatio()
        if logical > avail:
            dpr = pm.devicePixelRatio()
            pm = pm.scaledToWidth(int(avail * dpr), Qt.TransformationMode.SmoothTransformation)
            pm.setDevicePixelRatio(dpr)
        self.math.setPixmap(pm)
        self.math.setMinimumHeight(int(pm.height() / pm.devicePixelRatio()))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._pm is not None:
            self._show()
