import sys
import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QPushButton,
    QTextEdit, QGroupBox, QComboBox, QSplitter, QLineEdit,
    QFileDialog, QCheckBox, QRadioButton, QButtonGroup,
    QDialog, QMenuBar, QMenu, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QFont, QAction, QColor
from core.config import PINNConfig
from core.runner import run_pinn


class SciLineEdit(QLineEdit):
    """QLineEdit that accepts scientific notation like 1e-5, 1.5e3."""
    def __init__(self, value=1.0, parent=None):
        super().__init__(parent)
        self._value = float(value)
        self.setText(self._format(value))
        self.editingFinished.connect(self._on_edited)
        self.setFixedHeight(28)

    def _format(self, v):
        v = float(v)
        if v == 0:
            return "0"
        if abs(v) < 0.001 or abs(v) >= 1e6:
            return f"{v:.2e}"
        return f"{v:g}"

    def _on_edited(self):
        try:
            self._value = float(self.text().strip())
            self.setText(self._format(self._value))
            self.setStyleSheet("")
        except ValueError:
            self.setStyleSheet("border: 1px solid red;")

    def value(self):
        try:
            return float(self.text().strip())
        except ValueError:
            return self._value

    def setValue(self, v):
        self._value = float(v)
        self.setText(self._format(v))
        self.setStyleSheet("")


# ── Background worker thread ─────────────────────────────────
class SolverThread(QThread):
    output_signal = pyqtSignal(str)
    done_signal   = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.process = None

    def run(self):
        result = run_pinn(
            self.config,
            on_output=lambda line: self.output_signal.emit(line),
            set_process=self._set_process
        )
        self.done_signal.emit(result)

    def _set_process(self, proc):
        self.process = proc

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()


# ── Main Window ──────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepXDE GUI — PINN Solver")
        self.setMinimumSize(1100, 750)
        self._apply_theme()
        self._build_ui()

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background: #1a1a2e; }
            QWidget { background: #1a1a2e; color: #e0e0e0; font-family: 'Segoe UI', Arial; font-size: 12px; }
            QGroupBox {
                border: 1px solid #3a3a5c;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 4px;
                font-weight: bold;
                color: #a0c4ff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
                background: #16213e;
                border: 1px solid #3a3a5c;
                border-radius: 4px;
                padding: 2px 6px;
                color: #e0e0e0;
            }
            QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
                border: 1px solid #a0c4ff;
            }
            QCheckBox { color: #c0c0c0; spacing: 6px; }
            QCheckBox::indicator {
                width: 14px; height: 14px;
                border: 1px solid #3a3a5c;
                border-radius: 3px;
                background: #16213e;
            }
            QCheckBox::indicator:checked { background: #a0c4ff; border-color: #a0c4ff; }
            QRadioButton { color: #c0c0c0; spacing: 6px; }
            QRadioButton::indicator {
                width: 14px; height: 14px;
                border: 1px solid #3a3a5c;
                border-radius: 7px;
                background: #16213e;
            }
            QRadioButton::indicator:checked { background: #a0c4ff; border-color: #a0c4ff; }
            QLabel { color: #c0c0c0; }
            QScrollArea { border: none; background: #1a1a2e; }
            QScrollBar:vertical {
                background: #16213e; width: 8px; border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #3a3a5c; border-radius: 4px; min-height: 20px;
            }
            QTextEdit {
                background: #0f0f23;
                border: 1px solid #3a3a5c;
                border-radius: 6px;
                color: #a0ffb0;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
            QSplitter::handle { background: #3a3a5c; width: 2px; }
            QMenuBar { background: #16213e; color: #c0c0c0; border-bottom: 1px solid #3a3a5c; }
            QMenuBar::item:selected { background: #3a3a5c; }
            QMenu { background: #16213e; border: 1px solid #3a3a5c; }
            QMenu::item:selected { background: #3a3a5c; }
        """)

    def _build_ui(self):
        from PyQt6.QtWidgets import QScrollArea

        # ── Menu bar ─────────────────────────────────────────
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        settings_menu = menubar.addMenu("⚙ Settings")
        lbfgs_action = QAction("L-BFGS Options", self)
        lbfgs_action.triggered.connect(self._on_lbfgs_settings)
        settings_menu.addAction(lbfgs_action)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left panel ───────────────────────────────────────
        left_inner = QWidget()
        left_layout = QVBoxLayout(left_inner)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(10, 8, 10, 8)

        left_scroll = QScrollArea()
        left_scroll.setWidget(left_inner)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMaximumWidth(400)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Title
        title = QLabel("🔥 DeepXDE GUI")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #a0c4ff; margin-bottom: 2px;")
        left_layout.addWidget(title)

        subtitle = QLabel("Physics-Informed Neural Network Solver")
        subtitle.setStyleSheet("color: #7070a0; font-size: 11px; margin-bottom: 6px;")
        left_layout.addWidget(subtitle)

        # ── Dimension selector ────────────────────────────────
        dim_group = QGroupBox("Problem Dimension")
        dim_layout = QHBoxLayout(dim_group)
        self.radio_1d = QRadioButton("1D  (x, t)")
        self.radio_2d = QRadioButton("2D  (x, y, t)")
        self.radio_1d.setChecked(True)
        self.dim_group_btn = QButtonGroup()
        self.dim_group_btn.addButton(self.radio_1d)
        self.dim_group_btn.addButton(self.radio_2d)
        dim_layout.addWidget(self.radio_1d)
        dim_layout.addWidget(self.radio_2d)
        left_layout.addWidget(dim_group)
        self.radio_1d.toggled.connect(self._on_dim_changed)

        # ── Problem type ──────────────────────────────────────
        type_group = QGroupBox("Problem Type")
        type_layout = QHBoxLayout(type_group)
        self.radio_forward = QRadioButton("Forward")
        self.radio_inverse = QRadioButton("Inverse")
        self.radio_forward.setChecked(True)
        self.problem_type_group = QButtonGroup()
        self.problem_type_group.addButton(self.radio_forward)
        self.problem_type_group.addButton(self.radio_inverse)
        type_layout.addWidget(self.radio_forward)
        type_layout.addWidget(self.radio_inverse)
        left_layout.addWidget(type_group)
        self.radio_forward.toggled.connect(self._on_problem_type_changed)

        # ── Number of PDEs ────────────────────────────────────
        nout_group = QGroupBox("Number of PDEs / Outputs")
        nout_layout = QHBoxLayout(nout_group)
        nout_layout.addWidget(QLabel("How many PDEs:"))
        self.num_outputs_spin = QSpinBox()
        self.num_outputs_spin.setRange(1, 4)
        self.num_outputs_spin.setValue(1)
        self.num_outputs_spin.setFixedHeight(30)
        self.num_outputs_spin.setFixedWidth(60)
        self.num_outputs_spin.valueChanged.connect(self._on_num_outputs_changed)
        nout_layout.addWidget(self.num_outputs_spin)
        nout_layout.addStretch()
        left_layout.addWidget(nout_group)

        # ── PDE input ─────────────────────────────────────────
        self.pde_group = QGroupBox("PDE Definition")
        self.pde_main_layout = QVBoxLayout(self.pde_group)
        self.pde_main_layout.setSpacing(6)
        self.pde_inputs = []
        self.output_name_inputs = []
        self._build_pde_inputs(1)
        left_layout.addWidget(self.pde_group)

        # ── Domain ────────────────────────────────────────────
        domain_group = QGroupBox("Domain")
        domain_layout = QVBoxLayout(domain_group)
        domain_layout.setSpacing(6)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("x:"))
        self.x_min = QDoubleSpinBox()
        self.x_min.setRange(-1e6, 1e6); self.x_min.setValue(0.0); self.x_min.setSingleStep(0.5)
        self.x_max = QDoubleSpinBox()
        self.x_max.setRange(-1e6, 1e6); self.x_max.setValue(1.0); self.x_max.setSingleStep(0.5)
        row1.addWidget(self.x_min); row1.addWidget(QLabel("to")); row1.addWidget(self.x_max)
        domain_layout.addLayout(row1)

        self.y_row_widget = QWidget()
        y_row = QHBoxLayout(self.y_row_widget)
        y_row.setContentsMargins(0, 0, 0, 0)
        y_row.addWidget(QLabel("y:"))
        self.y_min = QDoubleSpinBox()
        self.y_min.setRange(-1e6, 1e6); self.y_min.setValue(0.0); self.y_min.setSingleStep(0.5)
        self.y_max = QDoubleSpinBox()
        self.y_max.setRange(-1e6, 1e6); self.y_max.setValue(1.0); self.y_max.setSingleStep(0.5)
        y_row.addWidget(self.y_min); y_row.addWidget(QLabel("to")); y_row.addWidget(self.y_max)
        self.y_row_widget.setVisible(False)
        domain_layout.addWidget(self.y_row_widget)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("t:"))
        self.t_min = QDoubleSpinBox()
        self.t_min.setRange(0.0, 1e6); self.t_min.setValue(0.0); self.t_min.setSingleStep(0.5)
        self.t_max = QDoubleSpinBox()
        self.t_max.setRange(0.0, 1e6); self.t_max.setValue(1.0); self.t_max.setSingleStep(0.5)
        row2.addWidget(self.t_min); row2.addWidget(QLabel("to")); row2.addWidget(self.t_max)
        domain_layout.addLayout(row2)
        left_layout.addWidget(domain_group)

        # ── Collocation Points ────────────────────────────────
        points_group = QGroupBox("Collocation Points")
        points_layout = QVBoxLayout(points_group)
        points_layout.setSpacing(4)

        def _pts_row(label, default, min_v, max_v, step):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            sb = QSpinBox()
            sb.setRange(min_v, max_v); sb.setSingleStep(step); sb.setValue(default)
            sb.setFixedHeight(28); sb.setFixedWidth(100)
            row.addStretch(); row.addWidget(sb)
            points_layout.addLayout(row)
            return sb

        self.num_domain   = _pts_row("Domain points:",   2000, 100, 100000, 500)
        self.num_boundary = _pts_row("Boundary points:", 200,  10,  10000,  100)
        self.num_initial  = _pts_row("Initial points:",  200,  10,  10000,  100)
        self.num_test     = _pts_row("Test points:",     1000, 100, 50000,  500)
        left_layout.addWidget(points_group)

        # ── Boundary & Initial conditions ─────────────────────
        self.bc_group = QGroupBox("Boundary & Initial Conditions")
        self.bc_main_layout = QVBoxLayout(self.bc_group)
        self.bc_main_layout.setSpacing(4)
        self.bc_left_types = [];  self.bc_left_vals = [];   self.bc_left_active = [];  self.bc_left_deriv = []
        self.bc_right_types = []; self.bc_right_vals = [];  self.bc_right_active = []; self.bc_right_deriv = []
        self.bc_bottom_types = []; self.bc_bottom_vals = []; self.bc_bottom_active = []; self.bc_bottom_deriv = []
        self.bc_top_types = [];   self.bc_top_vals = [];    self.bc_top_active = [];   self.bc_top_deriv = []
        self.ic_inputs = [];      self.ic_active = []
        self._2d_bc_widgets = []
        self._build_bc_inputs(1)
        left_layout.addWidget(self.bc_group)

        # ── Neural Network ────────────────────────────────────
        nn_group = QGroupBox("Neural Network")
        nn_layout = QVBoxLayout(nn_group)
        nn_layout.setSpacing(6)

        def _nn_row(label, widget):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addStretch()
            row.addWidget(widget)
            nn_layout.addLayout(row)

        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 8); self.layers_spin.setValue(3)
        self.layers_spin.setFixedWidth(100); self.layers_spin.setFixedHeight(28)
        _nn_row("Hidden layers:", self.layers_spin)

        self.neurons_spin = QSpinBox()
        self.neurons_spin.setRange(8, 512); self.neurons_spin.setSingleStep(8); self.neurons_spin.setValue(64)
        self.neurons_spin.setFixedWidth(100); self.neurons_spin.setFixedHeight(28)
        _nn_row("Neurons per layer:", self.neurons_spin)

        self.activation_combo = QComboBox()
        self.activation_combo.addItems(["tanh", "relu", "sigmoid", "swish"])
        self.activation_combo.setFixedWidth(100); self.activation_combo.setFixedHeight(28)
        _nn_row("Activation:", self.activation_combo)
        left_layout.addWidget(nn_group)

        # ── Training ──────────────────────────────────────────
        train_group = QGroupBox("Training")
        train_layout = QVBoxLayout(train_group)
        train_layout.setSpacing(5)
        train_layout.setContentsMargins(10, 10, 10, 10)

        def _train_row(label, widget):
            train_layout.addWidget(QLabel(label))
            train_layout.addWidget(widget)

        self.opt1_combo = QComboBox()
        self.opt1_combo.addItems(["adam", "sgd", "rmsprop"]); self.opt1_combo.setFixedHeight(28)
        _train_row("Phase 1 — Optimizer:", self.opt1_combo)

        self.iter1_spin = QSpinBox()
        self.iter1_spin.setRange(0, 1000000); self.iter1_spin.setSingleStep(1000)
        self.iter1_spin.setValue(10000); self.iter1_spin.setFixedHeight(28)
        _train_row("Phase 1 — Iterations:", self.iter1_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1.0); self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(0.0001); self.lr_spin.setValue(0.001); self.lr_spin.setFixedHeight(28)
        _train_row("Phase 1 — Learning Rate:", self.lr_spin)

        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["MSE", "MAE", "mean l2 relative error",
                                   "mean absolute percentage error", "softplus"])
        self.loss_combo.setFixedHeight(28)
        _train_row("Loss function:", self.loss_combo)

        div = QLabel("─── Phase 2 (optional) ───")
        div.setStyleSheet("color: #505080; font-size: 11px;")
        train_layout.addWidget(div)

        self.opt2_combo = QComboBox()
        self.opt2_combo.addItems(["none", "lbfgs"]); self.opt2_combo.setFixedHeight(28)
        _train_row("Phase 2 — Optimizer:", self.opt2_combo)

        self.iter2_spin = QSpinBox()
        self.iter2_spin.setRange(0, 100000); self.iter2_spin.setSingleStep(1000)
        self.iter2_spin.setValue(5000); self.iter2_spin.setFixedHeight(28)
        _train_row("Phase 2 — Iterations:", self.iter2_spin)

        # L-BFGS settings
        self.lbfgs_widget = QWidget()
        lbfgs_layout = QVBoxLayout(self.lbfgs_widget)
        lbfgs_layout.setSpacing(4); lbfgs_layout.setContentsMargins(0, 0, 0, 0)

        self.lbfgs_use_default_cb = QCheckBox("Use DeepXDE L-BFGS defaults")
        self.lbfgs_use_default_cb.setChecked(True)
        self.lbfgs_use_default_cb.stateChanged.connect(self._on_lbfgs_default_changed)
        lbfgs_layout.addWidget(self.lbfgs_use_default_cb)

        self.lbfgs_manual_widget = QWidget()
        lbfgs_manual_layout = QVBoxLayout(self.lbfgs_manual_widget)
        lbfgs_manual_layout.setSpacing(4); lbfgs_manual_layout.setContentsMargins(0, 0, 0, 0)

        def _lbfgs_row(label, val):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            sb = SciLineEdit(val); sb.setFixedWidth(120)
            row.addStretch(); row.addWidget(sb)
            lbfgs_manual_layout.addLayout(row)
            return sb

        self.lbfgs_maxcor  = _lbfgs_row("maxcor:",  100)
        self.lbfgs_ftol    = _lbfgs_row("ftol:",     1e-12)
        self.lbfgs_gtol    = _lbfgs_row("gtol:",     1e-8)
        self.lbfgs_maxiter = _lbfgs_row("maxiter:",  15000)
        self.lbfgs_maxfun  = _lbfgs_row("maxfun:",   15000)
        self.lbfgs_maxls   = _lbfgs_row("maxls:",    50)

        self.lbfgs_manual_widget.setVisible(False)
        lbfgs_layout.addWidget(self.lbfgs_manual_widget)
        self.lbfgs_widget.setVisible(False)
        train_layout.addWidget(self.lbfgs_widget)
        self.opt2_combo.currentTextChanged.connect(self._on_opt2_changed)
        left_layout.addWidget(train_group)

        # ── Loss Weights ──────────────────────────────────────
        self.weights_group = QGroupBox("Loss Weights")
        self.weights_main_layout = QVBoxLayout(self.weights_group)
        self.weights_main_layout.setSpacing(4)
        self.weight_widgets = {}
        self._build_weight_inputs(1)
        left_layout.addWidget(self.weights_group)

        # ── Adaptive Training ─────────────────────────────────
        adapt_group = QGroupBox("Adaptive Training")
        adapt_layout = QVBoxLayout(adapt_group)
        adapt_layout.setSpacing(5)

        row_a1 = QHBoxLayout()
        row_a1.addWidget(QLabel("Method:"))
        self.adapt_combo = QComboBox()
        self.adapt_combo.addItems(["None", "RAR", "Time Adaptive"])
        self.adapt_combo.setFixedHeight(28)
        self.adapt_combo.currentTextChanged.connect(self._on_adapt_changed)
        row_a1.addWidget(self.adapt_combo)
        adapt_layout.addLayout(row_a1)

        # RAR widget
        self.rar_widget = QWidget()
        rar_layout = QVBoxLayout(self.rar_widget)
        rar_layout.setSpacing(4); rar_layout.setContentsMargins(0, 0, 0, 0)

        def _rar_row(label, default, min_v, max_v, step):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            sb = QSpinBox(); sb.setRange(min_v, max_v); sb.setSingleStep(step); sb.setValue(default)
            sb.setFixedHeight(28); sb.setFixedWidth(100)
            row.addStretch(); row.addWidget(sb)
            rar_layout.addLayout(row)
            return sb

        self.rar_cycles     = _rar_row("RAR cycles:",         3,     1,    20,     1)
        self.rar_candidates = _rar_row("Candidate points:",   50000, 1000, 200000, 5000)
        self.rar_add_points = _rar_row("Points per cycle:",   500,   10,   10000,  100)
        self.rar_adam_iters = _rar_row("Adam iters/cycle:",   5000,  100,  50000,  1000)
        self.rar_lbfgs_iters= _rar_row("L-BFGS iters/cycle:", 0,    0,    50000,  1000)
        self.rar_widget.setVisible(False)
        adapt_layout.addWidget(self.rar_widget)

        # Time Adaptive widget
        self.ta_widget = QWidget()
        ta_layout = QVBoxLayout(self.ta_widget)
        ta_layout.setSpacing(4); ta_layout.setContentsMargins(0, 0, 0, 0)

        row_ta1 = QHBoxLayout()
        row_ta1.addWidget(QLabel("Time steps:"))
        self.ta_steps = QSpinBox(); self.ta_steps.setRange(2, 50); self.ta_steps.setValue(5)
        self.ta_steps.setFixedHeight(28); self.ta_steps.setFixedWidth(100)
        row_ta1.addStretch(); row_ta1.addWidget(self.ta_steps)
        ta_layout.addLayout(row_ta1)

        row_ta2 = QHBoxLayout()
        row_ta2.addWidget(QLabel("IC grid resolution:"))
        self.ta_grid = QComboBox(); self.ta_grid.addItems(["101", "51", "21", "11"])
        self.ta_grid.setFixedHeight(28); self.ta_grid.setFixedWidth(100)
        row_ta2.addStretch(); row_ta2.addWidget(self.ta_grid)
        ta_layout.addLayout(row_ta2)
        self.ta_widget.setVisible(False)
        adapt_layout.addWidget(self.ta_widget)
        left_layout.addWidget(adapt_group)

        # ── Inverse PINN panel ────────────────────────────────
        self.inverse_group = QGroupBox("Inverse PINN Settings")
        inv_layout = QVBoxLayout(self.inverse_group)
        inv_layout.setSpacing(5)

        inv_layout.addWidget(QLabel("Unknown parameter name:"))
        self.inv_param_name = QLineEdit(); self.inv_param_name.setText("beta"); self.inv_param_name.setFixedHeight(28)
        inv_layout.addWidget(self.inv_param_name)

        inv_layout.addWidget(QLabel("Initial guess:"))
        self.inv_param_init = QDoubleSpinBox()
        self.inv_param_init.setRange(-1e6, 1e6); self.inv_param_init.setDecimals(6)
        self.inv_param_init.setValue(1.0); self.inv_param_init.setFixedHeight(28)
        inv_layout.addWidget(self.inv_param_init)

        inv_layout.addWidget(QLabel("Measured data file (x, t, u):"))
        inv_data_row = QHBoxLayout()
        self.inv_data_path = QLineEdit(); self.inv_data_path.setPlaceholderText("Browse..."); self.inv_data_path.setFixedHeight(28)
        inv_data_row.addWidget(self.inv_data_path)
        self.inv_data_browse = QPushButton("Browse"); self.inv_data_browse.setFixedHeight(28); self.inv_data_browse.setFixedWidth(65)
        self.inv_data_browse.clicked.connect(self._on_browse_inv_data)
        inv_data_row.addWidget(self.inv_data_browse)
        inv_layout.addLayout(inv_data_row)

        inv_layout.addWidget(QLabel("IC type:"))
        self.inv_ic_type = QComboBox(); self.inv_ic_type.addItems(["Expression", "File (x, t, u)"])
        self.inv_ic_type.setFixedHeight(28); self.inv_ic_type.currentTextChanged.connect(self._on_inv_ic_type_changed)
        inv_layout.addWidget(self.inv_ic_type)

        self.inv_ic_file_label = QLabel("IC data file:"); self.inv_ic_file_label.setVisible(False)
        inv_layout.addWidget(self.inv_ic_file_label)
        inv_ic_row = QHBoxLayout()
        self.inv_ic_path = QLineEdit(); self.inv_ic_path.setPlaceholderText("Browse..."); self.inv_ic_path.setFixedHeight(28); self.inv_ic_path.setVisible(False)
        inv_ic_row.addWidget(self.inv_ic_path)
        self.inv_ic_browse = QPushButton("Browse"); self.inv_ic_browse.setFixedHeight(28); self.inv_ic_browse.setFixedWidth(65); self.inv_ic_browse.setVisible(False)
        self.inv_ic_browse.clicked.connect(self._on_browse_inv_ic)
        inv_ic_row.addWidget(self.inv_ic_browse)
        inv_layout.addLayout(inv_ic_row)
        inv_layout.addWidget(QLabel("Observed data loss weight:"))
        self.inv_obs_weight = SciLineEdit(1.0)
        self.inv_obs_weight.setFixedHeight(28)
        inv_layout.addWidget(self.inv_obs_weight)
        self.inv_param_log_scale = QCheckBox("Log scale for parameter convergence plot")
        self.inv_param_log_scale.setChecked(False)
        inv_layout.addWidget(self.inv_param_log_scale)

        self.inverse_group.setVisible(False)
        left_layout.addWidget(self.inverse_group)

        # ── Solve / Stop buttons ──────────────────────────────
        self.solve_btn = QPushButton("▶  Solve")
        self.solve_btn.setMinimumHeight(44)
        self.solve_btn.setStyleSheet("""
            QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0078d4,stop:1 #005a9e);
                          color: white; font-size: 14px; font-weight: bold; border-radius: 6px; border: none; }
            QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1a8ae8,stop:1 #0070c0); }
            QPushButton:disabled { background: #333355; color: #666; }
        """)
        self.solve_btn.clicked.connect(self._on_solve)
        left_layout.addWidget(self.solve_btn)

        self.stop_btn = QPushButton("⏹  Stop")
        self.stop_btn.setMinimumHeight(36)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton { background: #6b1f1f; color: white; font-size: 13px; font-weight: bold; border-radius: 6px; border: none; }
            QPushButton:hover { background: #8b2f2f; }
            QPushButton:disabled { background: #333355; color: #666; }
        """)
        self.stop_btn.clicked.connect(self._on_stop)
        left_layout.addWidget(self.stop_btn)

        # ── Parametric Study ──────────────────────────────────
        param_group = QGroupBox("Parametric Study")
        param_layout = QVBoxLayout(param_group)
        param_layout.setSpacing(5)

        self.param_check = QCheckBox("Enable Parametric Study")
        self.param_check.setFixedHeight(28)
        self.param_check.stateChanged.connect(self._on_param_changed)
        param_layout.addWidget(self.param_check)

        self.param_widget = QWidget()
        pw_layout = QVBoxLayout(self.param_widget)
        pw_layout.setSpacing(4); pw_layout.setContentsMargins(0, 0, 0, 0)

        pw_layout.addWidget(QLabel("Parameter to vary:"))
        self.param_combo = QComboBox()
        self.param_combo.addItems(["learning_rate","ic_weight","pde_weight","bc_left_weight",
                                    "bc_right_weight","hidden_layers","neurons_per_layer","phase1_iterations"])
        self.param_combo.setFixedHeight(28)
        pw_layout.addWidget(self.param_combo)

        pw_layout.addWidget(QLabel("Values (comma separated):"))
        self.param_values_input = QLineEdit(); self.param_values_input.setPlaceholderText("e.g. 1, 5, 10, 50")
        self.param_values_input.setFixedHeight(28)
        pw_layout.addWidget(self.param_values_input)

        self.param_widget.setVisible(False)
        param_layout.addWidget(self.param_widget)
        left_layout.addWidget(param_group)

        left_layout.addStretch()
        splitter.addWidget(left_scroll)

        # ── Right panel ───────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)

        # Log box
        log_label = QLabel("📋 Training Log")
        log_label.setStyleSheet("color: #a0c4ff; font-weight: bold; font-size: 12px;")
        right_layout.addWidget(log_label)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(140)
        self.log_box.setMaximumHeight(170)
        self.log_box.setPlaceholderText("Training log will appear here...")
        right_layout.addWidget(self.log_box)

        # Controls row
        ctrl_row = QHBoxLayout()

        # Output selector
        ctrl_row.addWidget(QLabel("Plot output:"))
        self.plot_output_combo = QComboBox()
        self.plot_output_combo.addItems(["Output 1 (u)"])
        self.plot_output_combo.setFixedHeight(28)
        self.plot_output_combo.setFixedWidth(130)
        ctrl_row.addWidget(self.plot_output_combo)

        ctrl_row.addWidget(QLabel("  Plot type:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Surface", "Line (time steps)"])
        self.plot_type_combo.setFixedHeight(28)
        self.plot_type_combo.setFixedWidth(140)
        self.plot_type_combo.currentTextChanged.connect(self._on_plot_type_changed)
        ctrl_row.addWidget(self.plot_type_combo)

        self.timesteps_label = QLabel("  Steps:")
        self.timesteps_label.setVisible(False)
        ctrl_row.addWidget(self.timesteps_label)
        self.timesteps_spin = QSpinBox()
        self.timesteps_spin.setRange(2, 20); self.timesteps_spin.setValue(4)
        self.timesteps_spin.setFixedHeight(28); self.timesteps_spin.setFixedWidth(55)
        self.timesteps_spin.setVisible(False)
        ctrl_row.addWidget(self.timesteps_spin)

        ctrl_row.addStretch()
        right_layout.addLayout(ctrl_row)

        # Save / export row
        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("Save to:"))
        self.save_dir_input = QLineEdit()
        self.save_dir_input.setPlaceholderText("Optional save directory")
        self.save_dir_input.setFixedHeight(28)
        save_row.addWidget(self.save_dir_input)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setFixedHeight(28); self.browse_btn.setFixedWidth(65)
        self.browse_btn.clicked.connect(self._on_browse)
        save_row.addWidget(self.browse_btn)

        save_row.addWidget(QLabel("  Grid:"))
        self.export_grid_combo = QComboBox()
        self.export_grid_combo.addItems(["101", "51", "21", "11"])
        self.export_grid_combo.setFixedHeight(28); self.export_grid_combo.setFixedWidth(55)
        save_row.addWidget(self.export_grid_combo)

        save_row.addWidget(QLabel("t steps:"))
        self.export_tsteps_spin = QSpinBox()
        self.export_tsteps_spin.setRange(2, 50); self.export_tsteps_spin.setValue(11)
        self.export_tsteps_spin.setFixedHeight(28); self.export_tsteps_spin.setFixedWidth(50)
        save_row.addWidget(self.export_tsteps_spin)
        right_layout.addLayout(save_row)

        # Plot area
        plots_layout = QHBoxLayout()
        plots_layout.setSpacing(6)

        self.loss_label = QLabel()
        self.loss_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loss_label.setText("📉 Loss plot")
        self.loss_label.setStyleSheet("border: 1px solid #3a3a5c; border-radius: 6px; color: #505080; background: #16213e;")
        self.loss_label.setMinimumSize(420, 380)

        self.solution_label = QLabel()
        self.solution_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.solution_label.setText("🗺 Solution plot")
        self.solution_label.setStyleSheet("border: 1px solid #3a3a5c; border-radius: 6px; color: #505080; background: #16213e;")
        self.solution_label.setMinimumSize(420, 380)

        plots_layout.addWidget(self.loss_label)
        plots_layout.addWidget(self.solution_label)
        right_layout.addLayout(plots_layout)

        splitter.addWidget(right)
        splitter.setSizes([390, 720])

    # ── Dimension change ──────────────────────────────────────
    def _on_dim_changed(self):
        is_2d = self.radio_2d.isChecked()
        self.y_row_widget.setVisible(is_2d)
        for w in self._2d_bc_widgets:
            w.setVisible(is_2d)
        self._build_pde_inputs(self.num_outputs_spin.value())
        self._build_bc_inputs(self.num_outputs_spin.value())
        self._build_weight_inputs(self.num_outputs_spin.value())

    # ── Plot type change ──────────────────────────────────────
    def _on_plot_type_changed(self, text):
        is_line = text == "Line (time steps)"
        self.timesteps_label.setVisible(is_line)
        self.timesteps_spin.setVisible(is_line)

    def _on_adapt_changed(self, text):
        self.rar_widget.setVisible(text == "RAR")
        self.ta_widget.setVisible(text == "Time Adaptive")

    def _on_param_changed(self, state):
        self.param_widget.setVisible(state == 2)

    def _on_browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if folder:
            self.save_dir_input.setText(folder)

    def _on_opt2_changed(self, text):
        self.lbfgs_widget.setVisible(text == "lbfgs")

    def _on_lbfgs_default_changed(self, state):
        self.lbfgs_manual_widget.setVisible(state != 2)

    def _on_bc_left_changed(self, text):
        pass

    def _on_bc_right_changed(self, text):
        pass

    # ── Build PDE inputs ──────────────────────────────────────
    def _build_pde_inputs(self, n):
        for i in reversed(range(self.pde_main_layout.count())):
            w = self.pde_main_layout.itemAt(i).widget()
            if w:
                w.deleteLater()
        self.pde_inputs.clear()
        self.output_name_inputs.clear()

        is_2d = self.radio_2d.isChecked() if hasattr(self, 'radio_2d') else False

        for i in range(n):
            name_row = QHBoxLayout()
            name_row.addWidget(QLabel(f"Output {i+1} name:"))
            name_input = QLineEdit()
            name_input.setText(["u", "v", "w", "p"][i] if i < 4 else f"u{i+1}")
            name_input.setFixedHeight(26); name_input.setFixedWidth(55)
            name_row.addWidget(name_input); name_row.addStretch()
            self.output_name_inputs.append(name_input)
            nw = QWidget(); nw.setLayout(name_row)
            self.pde_main_layout.addWidget(nw)

            self.pde_main_layout.addWidget(QLabel(f"PDE {i+1} residual = 0:"))
            pde_inp = QLineEdit()
            pde_inp.setText("du_t - 0.4 * du_xx" if i == 0 else "dv_t - 0.1 * dv_xx")
            pde_inp.setFixedHeight(28)
            self.pde_inputs.append(pde_inp)
            self.pde_main_layout.addWidget(pde_inp)

        names_ex = ", ".join([["u","v","w","p"][i] if i < 4 else f"u{i+1}" for i in range(n)])
        if is_2d:
            hint_text = (f"2D outputs: {names_ex}\n"
                         f"Derivatives: d[name]_x, d[name]_y, d[name]_t, d[name]_xx, d[name]_yy, d[name]_xy\n"
                         f"IC example: np.sin(np.pi*x[:,0])*np.sin(np.pi*x[:,1])")
        else:
            hint_text = (f"1D outputs: {names_ex}\n"
                         f"Derivatives: d[name]_x, d[name]_t, d[name]_xx, d[name]_tt, d[name]_xt")
        hint = QLabel(hint_text)
        hint.setStyleSheet("color: #505080; font-size: 10px;")
        hint.setWordWrap(True)
        self.pde_main_layout.addWidget(hint)

    # ── Build BC inputs ───────────────────────────────────────
    def _build_bc_inputs(self, n):
        for i in reversed(range(self.bc_main_layout.count())):
            w = self.bc_main_layout.itemAt(i).widget()
            if w:
                w.deleteLater()
        self.bc_left_types.clear();   self.bc_left_vals.clear();   self.bc_left_active.clear();   self.bc_left_deriv.clear()
        self.bc_right_types.clear();  self.bc_right_vals.clear();  self.bc_right_active.clear();  self.bc_right_deriv.clear()
        self.bc_bottom_types.clear(); self.bc_bottom_vals.clear(); self.bc_bottom_active.clear(); self.bc_bottom_deriv.clear()
        self.bc_top_types.clear();    self.bc_top_vals.clear();    self.bc_top_active.clear();    self.bc_top_deriv.clear()
        self.ic_inputs.clear();       self.ic_active.clear()
        self._2d_bc_widgets.clear()

        is_2d = self.radio_2d.isChecked() if hasattr(self, 'radio_2d') else False

        def _make_bc_block(label_txt, types_list, vals_list, active_list, deriv_list, x_pos):
            """Create a BC row: checkbox + type combo + value spinbox + deriv checkbox"""
            active = QCheckBox(label_txt); active.setChecked(True)
            active_list.append(active)
            self.bc_main_layout.addWidget(active)

            bc_type = QComboBox(); bc_type.addItems(["Dirichlet", "Neumann", "Periodic"]); bc_type.setFixedHeight(26)
            types_list.append(bc_type)
            self.bc_main_layout.addWidget(bc_type)

            bc_val = QDoubleSpinBox(); bc_val.setRange(-1000, 1000); bc_val.setValue(0.0); bc_val.setFixedHeight(26)
            vals_list.append(bc_val)
            self.bc_main_layout.addWidget(bc_val)

            bc_deriv = QCheckBox("  + derivative periodic BC"); bc_deriv.setChecked(False); bc_deriv.setVisible(False)
            deriv_list.append(bc_deriv)
            self.bc_main_layout.addWidget(bc_deriv)

            bc_type.currentTextChanged.connect(
                lambda t, vw=bc_val, dw=bc_deriv: (
                    vw.setVisible(t != "Periodic"),
                    dw.setVisible(t == "Periodic"),
                    self._build_weight_inputs(self.num_outputs_spin.value())
                )
            )

        for i in range(n):
            name = ["u", "v", "w", "p"][i] if i < 4 else f"u{i+1}"

            sep = QLabel(f"── Output {i+1} ({name}) ──")
            sep.setStyleSheet("color: #505080; font-size: 10px; margin-top: 4px;")
            self.bc_main_layout.addWidget(sep)

            _make_bc_block(f"BC left (x=xmin) for {name}", self.bc_left_types, self.bc_left_vals, self.bc_left_active, self.bc_left_deriv, "x_min")
            _idx = len(self.bc_left_types) - 1
            _make_bc_block(f"BC right (x=xmax) for {name}", self.bc_right_types, self.bc_right_vals, self.bc_right_active, self.bc_right_deriv, "x_max")
            # Hide right BC widgets when left is Periodic
            _right_active = self.bc_right_active[-1]
            _right_type   = self.bc_right_types[-1]
            _right_val    = self.bc_right_vals[-1]
            _right_deriv  = self.bc_right_deriv[-1]
            def _sync_right(t, ra=_right_active, rt=_right_type, rv=_right_val, rd=_right_deriv):
                ra.setVisible(t != "Periodic")
                rt.setVisible(t != "Periodic")
                rv.setVisible(t != "Periodic")
                rd.setVisible(False)
            self.bc_left_types[-1].currentTextChanged.connect(_sync_right)

            # 2D BCs — bottom and top
            _2d_w = QWidget()
            _2d_l = QVBoxLayout(_2d_w); _2d_l.setContentsMargins(0,0,0,0); _2d_l.setSpacing(3)

            # Temporarily redirect bc_main_layout to _2d_l
            _orig_layout = self.bc_main_layout
            self.bc_main_layout = _2d_l
            _make_bc_block(f"BC bottom (y=ymin) for {name}", self.bc_bottom_types, self.bc_bottom_vals, self.bc_bottom_active, self.bc_bottom_deriv, "y_min")
            _make_bc_block(f"BC top (y=ymax) for {name}", self.bc_top_types, self.bc_top_vals, self.bc_top_active, self.bc_top_deriv, "y_max")
            self.bc_main_layout = _orig_layout
            # Hide top BC widgets when bottom is Periodic
            _top_active = self.bc_top_active[-1]
            _top_type   = self.bc_top_types[-1]
            _top_val    = self.bc_top_vals[-1]
            _top_deriv  = self.bc_top_deriv[-1]
            def _sync_top(t, ta=_top_active, tt=_top_type, tv=_top_val, td=_top_deriv):
                ta.setVisible(t != "Periodic")
                tt.setVisible(t != "Periodic")
                tv.setVisible(t != "Periodic")
                td.setVisible(False)
            self.bc_bottom_types[-1].currentTextChanged.connect(_sync_top)

            _2d_w.setVisible(is_2d)
            self._2d_bc_widgets.append(_2d_w)
            self.bc_main_layout.addWidget(_2d_w)

            # IC
            ic_act = QCheckBox(f"IC for {name}"); ic_act.setChecked(True)
            self.ic_active.append(ic_act)
            self.bc_main_layout.addWidget(ic_act)

            ic_inp = QLineEdit()
            ic_inp.setText("np.sin(np.pi * x[:, 0])" if i == 0 else "np.cos(np.pi * x[:, 0])")
            ic_inp.setFixedHeight(26)
            self.ic_inputs.append(ic_inp)
            self.bc_main_layout.addWidget(ic_inp)

    # ── Build weight inputs ───────────────────────────────────
    def _build_weight_inputs(self, n):
        for i in reversed(range(self.weights_main_layout.count())):
            w = self.weights_main_layout.itemAt(i).widget()
            if w:
                w.deleteLater()
        self.weight_widgets.clear()

        def _w_row(label, key):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            w = SciLineEdit(1.0); w.setFixedWidth(100)
            row.addStretch(); row.addWidget(w)
            self.weight_widgets[key] = w
            ww = QWidget(); ww.setLayout(row)
            self.weights_main_layout.addWidget(ww)

        for i in range(n):
            name = self.output_name_inputs[i].text() if i < len(self.output_name_inputs) else f"u{i+1}"
            _w_row(f"PDE {i+1} ({name}):", f"pde_{i}")
            if i < len(self.bc_left_active) and self.bc_left_active[i].isChecked():
                _w_row(f"BC left {i+1} ({name}):", f"bc_left_{i}")
            _blt_is_per = i < len(self.bc_left_types) and self.bc_left_types[i].currentText() == "Periodic"
            _brt_is_per = i < len(self.bc_right_types) and self.bc_right_types[i].currentText() == "Periodic"
            if i < len(self.bc_right_active) and self.bc_right_active[i].isChecked() and not _brt_is_per and not _blt_is_per:
                _w_row(f"BC right {i+1} ({name}):", f"bc_right_{i}")
            is_2d = self.radio_2d.isChecked() if hasattr(self, 'radio_2d') else False
            if is_2d:
                if i < len(self.bc_bottom_active) and self.bc_bottom_active[i].isChecked():
                    _w_row(f"BC bottom {i+1} ({name}):", f"bc_bottom_{i}")
                _bbt_is_per = i < len(self.bc_bottom_types) and self.bc_bottom_types[i].currentText() == "Periodic"
                _btt_is_per = i < len(self.bc_top_types) and self.bc_top_types[i].currentText() == "Periodic"
                if i < len(self.bc_top_active) and self.bc_top_active[i].isChecked() and not _btt_is_per and not _bbt_is_per:
                    _w_row(f"BC top {i+1} ({name}):", f"bc_top_{i}")
            if i < len(self.ic_active) and self.ic_active[i].isChecked():
                _w_row(f"IC {i+1} ({name}):", f"ic_{i}")

    # ── Build config ──────────────────────────────────────────
    def _build_config(self):
        n = self.layers_spin.value()
        w = self.neurons_spin.value()
        n_out = self.num_outputs_spin.value()
        is_2d = self.radio_2d.isChecked()
        input_size = 3 if is_2d else 2
        layers = [input_size] + [w] * n + [n_out]

        def _safe_val(lst, i, default=0.0):
            try:
                return lst[i].value()
            except Exception:
                return default

        def _safe_text(lst, i, default="Dirichlet"):
            try:
                return lst[i].currentText()
            except Exception:
                return default

        return PINNConfig(
            problem_dim="2D" if is_2d else "1D",
            num_outputs=n_out,
            output_names=",".join([self.output_name_inputs[i].text() for i in range(n_out)]),
            pde_expressions="|".join([self.pde_inputs[i].text() for i in range(n_out)]),

            bc_left_types=",".join([_safe_text(self.bc_left_types, i) for i in range(n_out)]),
            bc_right_types=",".join([_safe_text(self.bc_right_types, i) for i in range(n_out)]),
            bc_left_values=",".join([str(_safe_val(self.bc_left_vals, i)) for i in range(n_out)]),
            bc_right_values=",".join([str(_safe_val(self.bc_right_vals, i)) for i in range(n_out)]),
            bc_left_active=",".join([str(self.bc_left_active[i].isChecked()) for i in range(n_out)]),
            bc_right_active=",".join([
                "False" if (i < len(self.bc_left_types) and self.bc_left_types[i].currentText() == "Periodic")
                else str(self.bc_right_active[i].isChecked())
                for i in range(n_out)
            ]),
            bc_left_deriv=",".join([str(self.bc_left_deriv[i].isChecked()) for i in range(n_out)]),
            bc_right_deriv=",".join([str(self.bc_right_deriv[i].isChecked()) for i in range(n_out)]),

            bc_bottom_types=",".join([_safe_text(self.bc_bottom_types, i) for i in range(n_out)]) if is_2d else "Dirichlet",
            bc_top_types=",".join([_safe_text(self.bc_top_types, i) for i in range(n_out)]) if is_2d else "Dirichlet",
            bc_bottom_values=",".join([str(_safe_val(self.bc_bottom_vals, i)) for i in range(n_out)]) if is_2d else "0.0",
            bc_top_values=",".join([str(_safe_val(self.bc_top_vals, i)) for i in range(n_out)]) if is_2d else "0.0",
            bc_bottom_active=",".join([str(self.bc_bottom_active[i].isChecked()) for i in range(n_out)]) if is_2d else "True",
            bc_top_active=",".join([
                "False" if (i < len(self.bc_bottom_types) and self.bc_bottom_types[i].currentText() == "Periodic")
                else str(self.bc_top_active[i].isChecked())
                for i in range(n_out)
            ]) if is_2d else "True",
            bc_bottom_deriv=",".join([str(self.bc_bottom_deriv[i].isChecked()) for i in range(n_out)]) if is_2d else "False",
            bc_top_deriv=",".join([str(self.bc_top_deriv[i].isChecked()) for i in range(n_out)]) if is_2d else "False",

            ic_expressions="|".join([self.ic_inputs[i].text() for i in range(n_out)]),
            ic_active=",".join([str(self.ic_active[i].isChecked()) for i in range(n_out)]),

            loss_weights_multi=",".join([
                str(self.weight_widgets[k].value())
                for i in range(n_out)
                for k in ([f"pde_{i}", f"bc_left_{i}", f"bc_right_{i}", f"bc_bottom_{i}", f"bc_top_{i}", f"ic_{i}"]
                           if is_2d else
                           [f"pde_{i}", f"bc_left_{i}", f"bc_right_{i}", f"ic_{i}"])
                if k in self.weight_widgets
            ]),
            plot_output_idx=self.plot_output_combo.currentIndex(),

            pde_expression=self.pde_inputs[0].text() if self.pde_inputs else "du_t - 0.4 * du_xx",
            bc_left=_safe_val(self.bc_left_vals, 0),
            bc_right=_safe_val(self.bc_right_vals, 0),
            bc_left_type=_safe_text(self.bc_left_types, 0),
            bc_right_type=_safe_text(self.bc_right_types, 0),
            ic_expression=self.ic_inputs[0].text() if self.ic_inputs else "np.sin(np.pi * x[:, 0])",
            loss_weights=[
                self.weight_widgets.get("pde_0", SciLineEdit(1.0)).value(),
                self.weight_widgets.get("bc_left_0", SciLineEdit(1.0)).value(),
                self.weight_widgets.get("bc_right_0", SciLineEdit(1.0)).value(),
                self.weight_widgets.get("ic_0", SciLineEdit(1.0)).value(),
            ],
            loss_weight_obs=self.inv_obs_weight.value(),
            inv_param_log_scale=self.inv_param_log_scale.isChecked(),

            x_min=self.x_min.value(), x_max=self.x_max.value(),
            y_min=self.y_min.value(), y_max=self.y_max.value(),
            t_min=self.t_min.value(), t_max=self.t_max.value(),
            layers=layers,
            activation=self.activation_combo.currentText(),
            iterations=self.iter1_spin.value(),
            optimizer=self.opt1_combo.currentText(),
            optimizer2=self.opt2_combo.currentText(),
            iterations2=self.iter2_spin.value(),
            num_domain=self.num_domain.value(),
            num_boundary=self.num_boundary.value(),
            num_initial=self.num_initial.value(),
            num_test=self.num_test.value(),
            plot_type=self.plot_type_combo.currentText(),
            num_timesteps=self.timesteps_spin.value(),
            save_dir=self.save_dir_input.text(),
            adapt_method=self.adapt_combo.currentText(),
            rar_cycles=self.rar_cycles.value(),
            rar_candidates=self.rar_candidates.value(),
            rar_add_points=self.rar_add_points.value(),
            rar_adam_iters=self.rar_adam_iters.value(),
            rar_lbfgs_iters=self.rar_lbfgs_iters.value(),
            time_adaptive=self.adapt_combo.currentText() == "Time Adaptive",
            ta_num_steps=self.ta_steps.value(),
            ta_grid_size=int(self.ta_grid.currentText()),
            learning_rate=self.lr_spin.value(),
            loss_type=self.loss_combo.currentText(),
            parametric_study=self.param_check.isChecked(),
            parametric_param=self.param_combo.currentText(),
            parametric_values=self.param_values_input.text(),
            problem_type="Inverse" if self.radio_inverse.isChecked() else "Forward",
            inverse_param_name=self.inv_param_name.text().strip(),
            inverse_param_init=self.inv_param_init.value(),
            inverse_data_file=self.inv_data_path.text().strip(),
            inverse_ic_type=self.inv_ic_type.currentText(),
            inverse_ic_file=self.inv_ic_path.text().strip(),
            export_grid_size=int(self.export_grid_combo.currentText()),
            export_t_steps=self.export_tsteps_spin.value(),
            lbfgs_use_default=self.lbfgs_use_default_cb.isChecked(),
            lbfgs_maxcor=int(self.lbfgs_maxcor.value()),
            lbfgs_ftol=self.lbfgs_ftol.value(),
            lbfgs_gtol=self.lbfgs_gtol.value(),
            lbfgs_maxiter=int(self.lbfgs_maxiter.value()),
            lbfgs_maxfun=int(self.lbfgs_maxfun.value()),
            lbfgs_maxls=int(self.lbfgs_maxls.value()),
        )

    def _on_num_outputs_changed(self, n):
        self._build_pde_inputs(n)
        self._build_bc_inputs(n)
        self._build_weight_inputs(n)
        self.plot_output_combo.clear()
        for i in range(n):
            name = self.output_name_inputs[i].text() if i < len(self.output_name_inputs) else f"u{i+1}"
            self.plot_output_combo.addItem(f"Output {i+1} ({name})")

    def _on_problem_type_changed(self, checked):
        self.inverse_group.setVisible(self.radio_inverse.isChecked())

    def _on_browse_inv_data(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select measured data file", "", "Data files (*.txt *.csv *.dat)")
        if f:
            self.inv_data_path.setText(f)

    def _on_browse_inv_ic(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select IC data file", "", "Data files (*.txt *.csv *.dat)")
        if f:
            self.inv_ic_path.setText(f)

    def _on_inv_ic_type_changed(self, text):
        is_file = text == "File (x, t, u)"
        self.inv_ic_file_label.setVisible(is_file)
        self.inv_ic_path.setVisible(is_file)
        self.inv_ic_browse.setVisible(is_file)

    def _on_solve(self):
        self.solve_btn.setEnabled(False)
        self.solve_btn.setText("⏳  Solving...")
        self.stop_btn.setEnabled(True)
        self.log_box.clear()
        self.loss_label.setText("⏳ Training...")
        self.solution_label.setText("⏳ Training...")
        for _p in ["/tmp/loss_plot.png", "/tmp/solution_plot.png", "/tmp/param_plot.png"]:
            if os.path.exists(_p):
                os.remove(_p)
        config = self._build_config()
        self.thread = SolverThread(config)
        self.thread.output_signal.connect(self._on_output)
        self.thread.done_signal.connect(self._on_done)
        self.thread.start()

    def _on_stop(self):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait(3000)
            self.log_box.append("\n⏹ Stopped by user.")
            self.solve_btn.setEnabled(True)
            self.solve_btn.setText("▶  Solve")
            self.stop_btn.setEnabled(False)

    def _on_output(self, line):
        self.log_box.append(line)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def _on_done(self, result):
        self.solve_btn.setEnabled(True)
        self.solve_btn.setText("▶  Solve")
        self.stop_btn.setEnabled(False)

        if result == "DONE":
            self.log_box.append("\n✅ Training complete!")
            self._last_config = self._build_config()

            save_dir = self.save_dir_input.text().strip()
            loss_path = os.path.join(save_dir, "loss_plot.png") if save_dir else "/tmp/loss_plot.png"
            solution_path = os.path.join(save_dir, "solution_plot.png") if save_dir else "/tmp/solution_plot.png"

            if os.path.exists(loss_path):
                self.loss_label.setPixmap(QPixmap(loss_path).scaled(
                    500, 420, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
            if os.path.exists(solution_path):
                self.solution_label.setPixmap(QPixmap(solution_path).scaled(
                    500, 420, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
            if save_dir:
                self.log_box.append(f"💾 Results saved to: {save_dir}")

            param_plot = "/tmp/param_plot.png"
            if os.path.exists(param_plot) and self.radio_inverse.isChecked():
                self.solution_label.setPixmap(QPixmap(param_plot).scaled(
                    500, 420, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
        else:
            self.log_box.append("\n❌ Error during training. Check log above.")
            self.log_box.append("💡 Tip: Check PDE/IC syntax — use * for multiplication (e.g. 5*u not 5u)")

    def _on_lbfgs_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("L-BFGS Settings")
        dialog.setMinimumWidth(340)
        layout = QVBoxLayout(dialog)

        use_default = QCheckBox("Use DeepXDE defaults")
        use_default.setChecked(self.lbfgs_use_default_cb.isChecked())
        layout.addWidget(use_default)

        manual_widget = QWidget()
        manual_layout = QVBoxLayout(manual_widget)
        manual_layout.setSpacing(6)

        def _make_row(label, val):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            sb = SciLineEdit(val); sb.setFixedWidth(130)
            row.addStretch(); row.addWidget(sb)
            manual_layout.addLayout(row)
            return sb

        sb_maxcor  = _make_row("maxcor:",  self.lbfgs_maxcor.value())
        sb_ftol    = _make_row("ftol:",    self.lbfgs_ftol.value())
        sb_gtol    = _make_row("gtol:",    self.lbfgs_gtol.value())
        sb_maxiter = _make_row("maxiter:", self.lbfgs_maxiter.value())
        sb_maxfun  = _make_row("maxfun:",  self.lbfgs_maxfun.value())
        sb_maxls   = _make_row("maxls:",   self.lbfgs_maxls.value())

        manual_widget.setVisible(not use_default.isChecked())
        layout.addWidget(manual_widget)
        use_default.stateChanged.connect(lambda s: manual_widget.setVisible(s != 2))

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK"); cancel_btn = QPushButton("Cancel")
        btn_row.addStretch(); btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        cancel_btn.clicked.connect(dialog.reject)

        def _on_ok():
            self.lbfgs_use_default_cb.setChecked(use_default.isChecked())
            self.lbfgs_maxcor.setValue(sb_maxcor.value())
            self.lbfgs_ftol.setValue(sb_ftol.value())
            self.lbfgs_gtol.setValue(sb_gtol.value())
            self.lbfgs_maxiter.setValue(sb_maxiter.value())
            self.lbfgs_maxfun.setValue(sb_maxfun.value())
            self.lbfgs_maxls.setValue(sb_maxls.value())
            dialog.accept()

        ok_btn.clicked.connect(_on_ok)
        dialog.exec()