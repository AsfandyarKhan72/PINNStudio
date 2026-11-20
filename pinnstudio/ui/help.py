"""In-app help: the tutorial list, and the Quick Reference window that works offline.

The documentation website holds the full user guide and the tutorials; the Help menu and the start page
link to them. The Quick Reference is the part a user needs while typing, available without a network
connection: the expression syntax, the boundary-condition kinds, the usual workflow and the keyboard
shortcuts (read from the window's own actions, so it always lists the shortcuts that really work).

TUTORIALS must match the table in website/tutorials/index.md (tests/test_help_menu.py checks it).
"""
import html

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit, QTabWidget,
                               QTextBrowser, QVBoxLayout)

DOCS_URL = "https://asfandyarkhan72.github.io/PINNStudio/"
ISSUES_URL = "https://github.com/AsfandyarKhan72/PINNStudio/issues/new/choose"

# (number, title, page) -- the page under tutorials/ on the documentation website
TUTORIALS = (
    (1, "First forward ODE", "01-ode"),
    (2, "1D Burgers equation", "02-burgers"),
    (3, "2D heat equation", "03-heat-2d"),
    (4, "Coupled PDE system", "04-coupled"),
    (5, "Phase-field (Allen–Cahn)", "05-phase-field"),
    (6, "Inverse parameter estimation", "06-inverse"),
    (7, "Importing observational data", "07-data"),
    (8, "Custom geometry", "08-geometry"),
    (9, "Exact constraints", "09-exact"),
    (10, "RAR", "10-rar"),
    (11, "RBA and SA-PINN", "11-point-weights"),
    (12, "Causal and time-window training", "12-causal-windows"),
    (13, "Comparing configurations", "13-compare"),
    (14, "Bayesian hyperparameter search", "14-bayesian"),
    (15, "Restoring a trained model", "15-restore"),
    (16, "Standalone scripts", "16-scripts"),
    (17, "Error analysis", "17-error-analysis"),
    (18, "Your own PDE (Fisher-KPP)", "18-own-problem"),
    (19, "Steady 2D flow (Kovasznay)", "19-kovasznay"),
    (20, "Diagnosing a run that does not converge", "20-diagnosing"),
    (21, "A 3D problem (Poisson in a sphere)", "21-3d-poisson"),
)

# Other pages of the documentation the Help menu links to: (menu text, page)
DOC_PAGES = (
    ("Getting Started", "getting-started/first-run/"),
    ("User Guide", "guide/"),
    ("Choosing a Strategy", "guide/choosing-a-strategy/"),
    ("Troubleshooting", "help/troubleshooting/"),
    ("FAQ", "help/faq/"),
)


def tutorial_url(page):
    return f"{DOCS_URL}tutorials/{page}/"


def doc_url(page):
    return DOCS_URL + page


# ── the Quick Reference ──────────────────────────────────────────────────
_EXPRESSIONS = """
<h3>Equations are residuals</h3>
<p>Each equation is an expression that is zero when the equation holds. The Burgers equation
u<sub>t</sub> + u u<sub>x</sub> = &nu; u<sub>xx</sub> is typed as</p>
<pre>du_t + u*du_x - (0.01/pi)*du_xx</pre>
<p>Expressions use Python syntax: <code>*</code> to multiply, <code>**</code> for powers
(<code>u**2</code>, not <code>u^2</code>).</p>

<h3>Derivatives: d&lt;output&gt;_&lt;variables&gt;</h3>
<table>
<tr><th>You type</th><th>Means</th></tr>
<tr><td><code>du_x</code>, <code>du_y</code>, <code>du_z</code>, <code>du_t</code></td><td>first derivatives</td></tr>
<tr><td><code>du_xx</code>, <code>du_yy</code>, <code>du_tt</code></td><td>second derivatives</td></tr>
<tr><td><code>du_xy</code>, <code>du_xt</code>, <code>du_yt</code></td><td>mixed second derivatives</td></tr>
<tr><td><code>du_xxx</code></td><td>third derivative in x (1D only)</td></tr>
<tr><td><code>du_xxxx</code>, <code>du_xxyy</code>, <code>du_xxtt</code></td><td>fourth derivatives</td></tr>
<tr><td><code>du_t</code> … <code>du_tttt</code></td><td>ODE problems (t is the only variable)</td></tr>
</table>
<p>Any other derivative (for example <code>du_xxy</code>) is refused at Solve. Write a higher derivative with
an extra output instead: an output <code>v</code> with the equation <code>v - du_xx</code>.</p>

<h3>Names you can use</h3>
<ul>
<li>the coordinates <code>x</code>, <code>y</code>, <code>z</code>, <code>t</code> (as the problem has them);</li>
<li>the output names (Equations page) and the named constants (<b>Add constant</b>);</li>
<li>the unknown parameters of an inverse problem;</li>
<li><code>sin</code>, <code>cos</code>, <code>exp</code>, <code>log</code>, <code>sqrt</code>, <code>tanh</code>,
<code>pi</code>;</li>
<li>in boundary conditions also the domain limits <code>x_min</code>, <code>x_max</code>, <code>y_min</code>, …</li>
</ul>
<p>A name the page does not know is listed in orange under the equation (<i>Unknown names: …</i>).</p>

<h3>Special terms</h3>
<ul>
<li><code>integral(f)</code> (ODE): the integral of f from the start of the time interval to t, with
<code>s</code> as the integration variable.</li>
<li><code>frac(u)</code>: a fractional operator of order &alpha; applied to output u.</li>
</ul>
"""

_CONDITIONS = """
<h3>Boundary-condition kinds</h3>
<table>
<tr><th>Kind</th><th>Sets</th></tr>
<tr><td>Dirichlet</td><td>the value of an output on a boundary</td></tr>
<tr><td>Neumann</td><td>its derivative across the boundary (along the outward normal)</td></tr>
<tr><td>Robin</td><td>a combination of the value and the normal derivative</td></tr>
<tr><td>Periodic</td><td>the same value (and, if chosen, derivative) on two opposite sides</td></tr>
<tr><td>Point set</td><td>known or measured values at points read from a file</td></tr>
<tr><td>Operator</td><td>any expression of the outputs and their derivatives on a boundary</td></tr>
<tr><td>Interface 2D</td><td>joins two edges of a sketched domain</td></tr>
</table>

<h3>Where a condition applies</h3>
<p><b>Where</b> picks a side (<i>left</i>, <i>right</i>, <i>bottom</i>, <i>top</i>, …, or <i>all</i>), or a
location expression such as <code>np.isclose(x, x_max)</code> or <code>x &lt; 0.5</code>. Click the examples
button next to <b>Where</b> for more.</p>

<h3>Soft or exact</h3>
<p>Every condition row is a loss term, met only as well as its weight asks. To make a condition hold
exactly, build it into the network instead: <b>Exact</b> initial conditions (Initial conditions page),
exact periodicity, or an output transform formula (Neural network page).</p>

<h3>Initial conditions</h3>
<p>One expression per output in x, y, z, for example <code>-sin(pi*x)</code>. Initial-condition
pre-training (Training page) first fits the network to the initial condition alone.</p>
"""

_WORKFLOW = """
<h3>The usual path</h3>
<ol>
<li><b>Problem</b>: forward or inverse, the dimension, steady or time-dependent.</li>
<li><b>Equations</b>: set <b>Outputs</b> first (changing it resets the boxes), name the outputs, type the
residuals, add constants.</li>
<li><b>Geometry</b>: the domain and the time interval.</li>
<li><b>Initial conditions</b> and <b>Boundary conditions</b>: one row per condition.</li>
<li><b>Collocation points</b>: the examples' numbers are a good start.</li>
<li><b>Neural network</b>: 3–6 layers of 32–100 neurons, tanh.</li>
<li><b>Training</b>: Adam (learning rate about 0.001), then L-BFGS to refine.</li>
<li><b>Solve</b> (F5). Watch the loss curve; <b>Stop</b> (Shift+F5) keeps what has been trained.</li>
<li><b>Results</b>: the figures and files of each run; <b>Error Analysis</b> against a reference.</li>
</ol>

<h3>When training does not converge</h3>
<ul>
<li>Check the residual signs: the expression must be <i>zero</i> when the equation holds.</li>
<li>Look at the loss terms separately (Results): one term far above the others usually needs a larger
weight on the others, or <b>automatic weighting</b> (Loss weights page).</li>
<li>Sharp fronts: more domain points, <b>RAR</b> (adaptive refinement) or a wider network.</li>
<li>Long time intervals: <b>causal</b> weighting or <b>time windows</b> (Training methods).</li>
<li>Very different output sizes: PFNN or an output transform that rescales them.</li>
<li>Very small target errors: float64 precision and a longer L-BFGS phase.</li>
</ul>
<p>Each node of the model tree has a <b>?</b> button (or press F1) that opens its page of the user guide.
Press <b>Ctrl+K</b> and type a few letters to go to any page or section, or to run any menu command.</p>
"""


def _shortcuts_html(window):
    """The window's actions that have keyboard shortcuts, with their menu, as a table."""
    from PySide6.QtGui import QKeySequence
    rows = []
    menubar = window.menuBar() if window is not None else None

    def walk(menu, path):
        for act in menu.actions():
            if act.isSeparator():
                continue
            sub = act.menu()
            if sub is not None:
                walk(sub, path + [act.text()])
                continue
            keys = [k.toString(QKeySequence.SequenceFormat.NativeText) for k in act.shortcuts() if not k.isEmpty()]
            if keys:
                label = " › ".join(path + [act.text().replace("&", "")])
                rows.append((label, ", ".join(keys)))
    if menubar is not None:
        for top in menubar.actions():
            if top.menu() is not None:
                walk(top.menu(), [top.text().replace("&", "")])
    body = "".join(f"<tr><td>{html.escape(a)}</td><td><code>{html.escape(k)}</code></td></tr>" for a, k in rows)
    return ("<h3>Keyboard shortcuts</h3><table><tr><th>Action</th><th>Keys</th></tr>" + body + "</table>"
            "<p>In the model tree, the arrow keys move between pages; in a page, Tab moves between fields.</p>")


def _mono():
    from pinnstudio.ui.theme import mono_font_family
    return mono_font_family()


def _css(t):
    return (f"body {{ color: {t['text']}; }} h3 {{ margin-top: 14px; margin-bottom: 4px; }} "
            f"code, pre {{ font-family: '{_mono()}'; color: {t['accent_text']}; }} "
            f"pre {{ background: {t['surface']}; padding: 6px; }} "
            f"th {{ text-align: left; color: {t['text_muted']}; padding: 3px 12px 3px 0; }} "
            f"td {{ padding: 3px 12px 3px 0; }} a {{ color: {t['accent_text']}; }}")


class QuickReferenceDialog(QDialog):
    """Help › Quick Reference: syntax, conditions, workflow and shortcuts, with a filter box."""

    SECTIONS = (("Expressions", _EXPRESSIONS), ("Conditions", _CONDITIONS), ("Workflow", _WORKFLOW))

    def __init__(self, window=None, parent=None):
        super().__init__(parent or window)
        self.setObjectName("quickReference")
        self.setWindowTitle("PINNStudio Quick Reference")
        self.resize(720, 620)
        from pinnstudio.ui import theme as _theme
        t = _theme.current()
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Find in the quick reference (for example: periodic, du_xx, F5)")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._find)
        top.addWidget(self.filter, 1)
        self.status = QLabel("")
        self.status.setObjectName("pageDesc")
        top.addWidget(self.status)
        lay.addLayout(top)
        self.tabs = QTabWidget()
        self.browsers = []
        sections = list(self.SECTIONS) + [("Shortcuts", _shortcuts_html(window))]
        for name, body in sections:
            b = QTextBrowser()
            b.setObjectName("helpBrowser")
            b.setOpenExternalLinks(True)
            b.document().setDefaultStyleSheet(_css(t))
            b.setHtml(body)
            self.tabs.addTab(b, name)
            self.browsers.append(b)
        lay.addWidget(self.tabs, 1)
        link = f"color: {t['accent_text']};"
        more = QLabel(f'Everything else is in the <a style="{link}" href="{DOCS_URL}guide/">user guide</a> and '
                      f'the <a style="{link}" href="{DOCS_URL}tutorials/">tutorials</a>.')
        more.setOpenExternalLinks(True)
        more.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        lay.addWidget(more)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _find(self, text):
        """Show the first tab that contains `text` and select its first occurrence there."""
        text = text.strip()
        if not text:
            self.status.setText("")
            return
        order = [self.tabs.currentIndex()] + [i for i in range(len(self.browsers)) if i != self.tabs.currentIndex()]
        for i in order:
            b = self.browsers[i]
            cur = b.document().find(text)
            if not cur.isNull():
                self.tabs.setCurrentIndex(i)
                b.setTextCursor(cur)
                b.ensureCursorVisible()
                self.status.setText("")
                return
        self.status.setText("Not found")
