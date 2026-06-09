import sys
import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QPushButton,
    QTextEdit, QGroupBox, QComboBox, QSplitter, QLineEdit,
    QFileDialog, QCheckBox, QRadioButton, QButtonGroup,
    QDialog, QMenuBar, QMenu, QFrame, QApplication
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QFont, QAction, QColor
from pinnstudio.core.config import PINNConfig
from pinnstudio.core.runner import run_pinn


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
        self._font_size = 16
        self._log_font_size = 16
        self._theme = "Solarized Dark"
        self._accent = "Blue (#a0c4ff)"
        self._float_type = "float32"
        self._apply_theme()
        self._build_ui()
        self._apply_display_settings()

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background: #002b36; }
            QWidget { background: #002b36; color: #e0e0e0; font-family: 'Segoe UI', Arial; font-size: 16px; }
            QGroupBox {
                border: 1px solid #586e75;
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
                background: #252526;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 2px 6px;
                color: #e0e0e0;
            }
            QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
                border: 1px solid #a0c4ff;
            }
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background: #586e75;
                border: none;
                width: 16px;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 6px solid #ffffff;
                width: 0px; height: 0px;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #ffffff;
                width: 0px; height: 0px;
            }
            QComboBox::drop-down {
                background: #586e75;
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #ffffff;
                width: 0px; height: 0px;
            }
            QCheckBox { color: #c0c0c0; spacing: 6px; }
            QCheckBox::indicator {
                width: 14px; height: 14px;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                background: #252526;
            }
            QCheckBox::indicator:checked { background: #a0c4ff; border-color: #a0c4ff; }
            QRadioButton { color: #c0c0c0; spacing: 6px; }
            QRadioButton::indicator {
                width: 14px; height: 14px;
                border: 1px solid #3e3e42;
                border-radius: 7px;
                background: #252526;
            }
            QRadioButton::indicator:checked { background: #a0c4ff; border-color: #a0c4ff; }
            QLabel { color: #c0c0c0; }
            QScrollArea { border: none; background: #1e1e1e; }
            QScrollBar:vertical {
                background: #252526; width: 14px; border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #586e75; border-radius: 6px; min-height: 30px;
            }

            QTextEdit {
                background: #0f0f23;
                border: 1px solid #3e3e42;
                border-radius: 6px;
                color: #a0ffb0;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
            QSplitter::handle { background: #3e3e42; width: 2px; }
            QMenuBar { background: #252526; color: #c0c0c0; border-bottom: 1px solid #3e3e42; }
            QMenuBar::item:selected { background: #3e3e42; }
            QMenu { background: #252526; border: 1px solid #3e3e42; }
            QMenu::item:selected { background: #3e3e42; }
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

        float_action = QAction("Float Precision...", self)
        float_action.triggered.connect(self._on_float_settings)
        settings_menu.addAction(float_action)

        view_menu = menubar.addMenu("🎨 Display")
        display_action = QAction("Display Settings...", self)
        display_action.triggered.connect(self._on_display_settings)
        view_menu.addAction(display_action)

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
        left_scroll.setMinimumWidth(300)
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

        # ── Quick Examples (shown right after dimension) ──────
        examples_group = QGroupBox("📋 Quick Examples")
        examples_layout = QHBoxLayout(examples_group)
        examples_layout.addWidget(QLabel("Load example:"))
        self.quick_examples_combo = QComboBox()
        self.quick_examples_combo.addItems(["── Select ──", "1D Heat", "1D Allen-Cahn", "1D Cahn-Hilliard"])
        self.quick_examples_combo.setFixedHeight(28)
        self.quick_examples_combo.currentTextChanged.connect(self._on_quick_example_selected)
        examples_layout.addWidget(self.quick_examples_combo)
        left_layout.addWidget(examples_group)

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
        self.num_initial  = _pts_row("Initial points:",  200,  0,  10000,  100)
        self.num_test     = _pts_row("Test points:",     1000, 100, 50000,  500)

        pts_dist_row = QHBoxLayout()
        pts_dist_row.addWidget(QLabel("Point distribution:"))
        self.pts_dist_combo = QComboBox()
        self.pts_dist_combo.addItems(["Hammersley", "uniform", "Halton", "LHS", "Sobol", "pseudorandom"])
        self.pts_dist_combo.setFixedHeight(28)
        pts_dist_row.addStretch()
        pts_dist_row.addWidget(self.pts_dist_combo)
        points_layout.addLayout(pts_dist_row)

        self.view_domain_check = QCheckBox("👁  View domain & point distribution (2D only)")
        self.view_domain_check.setChecked(False)
        self.view_domain_check.setVisible(False)
        self.view_domain_check.stateChanged.connect(self._on_view_domain_changed)
        points_layout.addWidget(self.view_domain_check)
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

        # ── Mini-batch ────────────────────────────────────────
        batch_group = QGroupBox("Mini-batch Training")
        batch_layout = QVBoxLayout(batch_group)
        batch_layout.setSpacing(5)

        self.batch_check = QCheckBox("Enable mini-batch training")
        self.batch_check.setChecked(True)
        self.batch_check.stateChanged.connect(self._on_batch_changed)
        batch_layout.addWidget(self.batch_check)

        self.batch_widget = QWidget()
        bw_layout = QHBoxLayout(self.batch_widget)
        bw_layout.setContentsMargins(0, 0, 0, 0)
        bw_layout.addWidget(QLabel("Batch size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(16, 10000)
        self.batch_spin.setSingleStep(16)
        self.batch_spin.setValue(32)
        self.batch_spin.setFixedWidth(100)
        self.batch_spin.setFixedHeight(28)
        bw_layout.addStretch()
        bw_layout.addWidget(self.batch_spin)
        self.batch_widget.setVisible(True)
        batch_layout.addWidget(self.batch_widget)

        note = QLabel("Splits collocation points into mini-batches per iteration.")
        note.setStyleSheet("color: #586e75; font-size: 11px;")
        note.setWordWrap(True)
        batch_layout.addWidget(note)
        left_layout.addWidget(batch_group)

        # ── Training ──────────────────────────────────────────
        train_group = QGroupBox("Training")
        train_layout = QVBoxLayout(train_group)
        train_layout.setSpacing(5)
        train_layout.setContentsMargins(10, 10, 10, 10)

        def _train_row(label, widget):
            train_layout.addWidget(QLabel(label))
            train_layout.addWidget(widget)

        div0 = QLabel("─── IC Pre-Training (optional) ───")
        div0.setStyleSheet("color: #505080; font-size: 11px;")
        train_layout.addWidget(div0)

        self.ic_pretrain_cb = QCheckBox("Enable IC-guided pre-training")
        self.ic_pretrain_cb.setChecked(False)
        self.ic_pretrain_cb.setStyleSheet("color: #69db7c; font-size: 12px;")
        self.ic_pretrain_cb.stateChanged.connect(self._on_ic_pretrain_changed)
        train_layout.addWidget(self.ic_pretrain_cb)

        self.ic_pretrain_widget = QWidget()
        ic_pt_layout = QVBoxLayout(self.ic_pretrain_widget)
        ic_pt_layout.setSpacing(4); ic_pt_layout.setContentsMargins(0, 0, 0, 0)
        ic_pt_layout.addWidget(QLabel("IC pre-train optimizer:"))
        self.ic_pretrain_opt = QComboBox()
        self.ic_pretrain_opt.addItems(["adam"])
        self.ic_pretrain_opt.setFixedHeight(28)
        ic_pt_layout.addWidget(self.ic_pretrain_opt)
        ic_pt_layout.addWidget(QLabel("IC pre-train iterations:"))
        self.ic_pretrain_iters = QSpinBox()
        self.ic_pretrain_iters.setRange(100, 500000)
        self.ic_pretrain_iters.setSingleStep(1000)
        self.ic_pretrain_iters.setValue(200000)
        self.ic_pretrain_iters.setFixedHeight(28)
        ic_pt_layout.addWidget(self.ic_pretrain_iters)

        # Test points
        ic_test_row = QHBoxLayout()
        ic_test_row.addWidget(QLabel("Test points:"))
        self.ic_pretrain_test = QSpinBox()
        self.ic_pretrain_test.setRange(100, 100000)
        self.ic_pretrain_test.setSingleStep(1000)
        self.ic_pretrain_test.setValue(10000)
        self.ic_pretrain_test.setFixedHeight(28)
        self.ic_pretrain_test.setFixedWidth(100)
        ic_test_row.addStretch(); ic_test_row.addWidget(self.ic_pretrain_test)
        ic_pt_layout.addLayout(ic_test_row)

        # Initial points (only shown when IC is from expression, not file)
        self.ic_pretrain_init_widget = QWidget()
        ic_init_row = QHBoxLayout(self.ic_pretrain_init_widget)
        ic_init_row.setContentsMargins(0, 0, 0, 0)
        ic_init_row.addWidget(QLabel("Initial points:"))
        self.ic_pretrain_init = QSpinBox()
        self.ic_pretrain_init.setRange(0, 10000)
        self.ic_pretrain_init.setSingleStep(100)
        self.ic_pretrain_init.setValue(1000)
        self.ic_pretrain_init.setFixedHeight(28)
        self.ic_pretrain_init.setFixedWidth(100)
        ic_init_row.addStretch(); ic_init_row.addWidget(self.ic_pretrain_init)
        ic_pt_layout.addWidget(self.ic_pretrain_init_widget)

        # Restore option
        ic_restore_cb = QCheckBox("🔄 Restore from saved IC pre-train model")
        ic_restore_cb.setChecked(False)
        ic_restore_cb.setStyleSheet("color: #69db7c; font-size: 12px;")
        ic_pt_layout.addWidget(ic_restore_cb)
        self.ic_pretrain_restore_cb = ic_restore_cb

        self.ic_pretrain_restore_widget = QWidget()
        ic_restore_layout = QHBoxLayout(self.ic_pretrain_restore_widget)
        ic_restore_layout.setContentsMargins(0, 0, 0, 0)
        self.ic_pretrain_restore_path = QLineEdit()
        self.ic_pretrain_restore_path.setPlaceholderText("Browse for IC pre-train .pt file...")
        self.ic_pretrain_restore_path.setFixedHeight(26)
        ic_restore_layout.addWidget(self.ic_pretrain_restore_path)
        ic_restore_browse = QPushButton("Browse")
        ic_restore_browse.setFixedHeight(26); ic_restore_browse.setFixedWidth(65)
        ic_restore_browse.clicked.connect(lambda: self.ic_pretrain_restore_path.setText(
            QFileDialog.getOpenFileName(None, "Select IC pre-train model", "", "Model (*.pt)")[0]))
        ic_restore_layout.addWidget(ic_restore_browse)
        self.ic_pretrain_restore_widget.setVisible(False)
        ic_pt_layout.addWidget(self.ic_pretrain_restore_widget)
        ic_restore_cb.stateChanged.connect(
            lambda s: self.ic_pretrain_restore_widget.setVisible(s == 2))

        ic_note = QLabel("Trains IC loss only before main training.\nFirst step only for time-adaptive.")
        ic_note.setStyleSheet("color: #586e75; font-size: 11px;")
        ic_note.setWordWrap(True)
        ic_pt_layout.addWidget(ic_note)

        self.ic_pretrain_widget.setVisible(False)
        train_layout.addWidget(self.ic_pretrain_widget)

        div_phase1 = QLabel("─── Phase 1 ───")
        div_phase1.setStyleSheet("color: #505080; font-size: 11px;")
        train_layout.addWidget(div_phase1)

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

        # ── Optimizer Scheduler ───────────────────────────────
        div_sched = QLabel("─── Optimizer Scheduler (optional) ───")
        div_sched.setStyleSheet("color: #505080; font-size: 11px;")
        train_layout.addWidget(div_sched)

        self.sched_cb = QCheckBox("Enable Optimizer Scheduler")
        self.sched_cb.setChecked(False)
        self.sched_cb.setStyleSheet("color: #ffa94d; font-size: 12px;")
        self.sched_cb.stateChanged.connect(self._on_scheduler_changed)
        train_layout.addWidget(self.sched_cb)

        self.sched_widget = QWidget()
        sched_layout = QVBoxLayout(self.sched_widget)
        sched_layout.setSpacing(4)
        sched_layout.setContentsMargins(0, 0, 0, 0)

        # Same weights checkbox
        self.sched_same_weights_cb = QCheckBox("Use same weights for all phases")
        self.sched_same_weights_cb.setChecked(True)
        self.sched_same_weights_cb.setStyleSheet("color: #69db7c; font-size: 12px;")
        self.sched_same_weights_cb.stateChanged.connect(
            lambda s: self._build_weight_inputs(self.num_outputs_spin.value()))
        sched_layout.addWidget(self.sched_same_weights_cb)

        # Phase list container
        self.sched_phases_widget = QWidget()
        self.sched_phases_layout = QVBoxLayout(self.sched_phases_widget)
        self.sched_phases_layout.setSpacing(4)
        self.sched_phases_layout.setContentsMargins(0, 0, 0, 0)
        sched_layout.addWidget(self.sched_phases_widget)
        self.sched_phase_list = []  # list of dicts with widgets

        # Add phase button
        add_phase_btn = QPushButton("➕ Add Phase")
        add_phase_btn.setStyleSheet(
            "QPushButton { color: #69db7c; background: transparent; "
            "border: 1px solid #2a6a4a; border-radius: 4px; padding: 2px 8px; }")
        add_phase_btn.clicked.connect(self._add_scheduler_phase)
        sched_layout.addWidget(add_phase_btn)

        self.sched_widget.setVisible(False)
        train_layout.addWidget(self.sched_widget)

        # L-BFGS settings
        self.lbfgs_widget = QWidget()
        lbfgs_layout = QVBoxLayout(self.lbfgs_widget)
        lbfgs_layout.setSpacing(4); lbfgs_layout.setContentsMargins(0, 0, 0, 0)

        self.lbfgs_use_default_cb = QCheckBox("Use L-BFGS recommended settings")
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
        self.lbfgs_gtol    = _lbfgs_row("gtol:",     1e-10)
        self.lbfgs_maxiter = _lbfgs_row("maxiter:",  15000)
        self.lbfgs_maxfun  = _lbfgs_row("maxfun:",   18750)
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

        # Transfer learning
        self.ta_transfer_cb = QCheckBox("Enable transfer learning (warm start from previous step)")
        self.ta_transfer_cb.setChecked(False)
        self.ta_transfer_cb.setStyleSheet("color: #69db7c; font-size: 12px;")
        ta_layout.addWidget(self.ta_transfer_cb)

        self.ta_transfer_opt_widget = QWidget()
        tl_row = QHBoxLayout(self.ta_transfer_opt_widget)
        tl_row.setContentsMargins(0, 0, 0, 0)
        tl_row.addWidget(QLabel("Transfer optimizer:"))
        self.ta_transfer_opt = QComboBox()
        self.ta_transfer_opt.addItems(["adam", "lbfgs"])
        self.ta_transfer_opt.setFixedHeight(26); self.ta_transfer_opt.setFixedWidth(80)
        tl_row.addStretch(); tl_row.addWidget(self.ta_transfer_opt)
        self.ta_transfer_opt_widget.setVisible(False)
        ta_layout.addWidget(self.ta_transfer_opt_widget)
        self.ta_transfer_cb.stateChanged.connect(
            lambda s: self.ta_transfer_opt_widget.setVisible(s == 2))

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

        # ── Model Restore & Visualization ─────────────────────
        restore_group = QGroupBox("Model Restore && Visualization")
        restore_layout = QVBoxLayout(restore_group)
        restore_layout.setSpacing(5)

        restore_toggle = QCheckBox("🔄 Enable Model Restore && Visualization")
        restore_toggle.setChecked(False)
        restore_toggle.setStyleSheet("color: #a0c4ff; font-weight: bold; font-size: 13px;")
        restore_layout.addWidget(restore_toggle)

        restore_content = QWidget()
        restore_content_layout = QVBoxLayout(restore_content)
        restore_content_layout.setContentsMargins(0, 0, 0, 0)
        restore_content_layout.setSpacing(5)
        restore_content.setVisible(False)
        restore_toggle.stateChanged.connect(lambda s: restore_content.setVisible(s == 2))
        restore_layout.addWidget(restore_content)

        restore_content_layout.addWidget(QLabel("Model file (.pt):"))
        restore_path_row = QHBoxLayout()
        self.restore_model_path = QLineEdit()
        self.restore_model_path.setPlaceholderText("Browse for model .pt file...")
        self.restore_model_path.setFixedHeight(28)
        restore_path_row.addWidget(self.restore_model_path)
        self.restore_browse_btn = QPushButton("Browse")
        self.restore_browse_btn.setFixedHeight(28); self.restore_browse_btn.setFixedWidth(65)
        self.restore_browse_btn.clicked.connect(self._on_browse_restore_model)
        restore_path_row.addWidget(self.restore_browse_btn)
        restore_content_layout.addLayout(restore_path_row)

        restore_content_layout.addWidget(QLabel("Config file (model_config.json):"))
        config_path_row = QHBoxLayout()
        self.restore_config_path = QLineEdit()
        self.restore_config_path.setPlaceholderText("Auto-detected or browse...")
        self.restore_config_path.setFixedHeight(28)
        config_path_row.addWidget(self.restore_config_path)
        self.restore_config_browse_btn = QPushButton("Browse")
        self.restore_config_browse_btn.setFixedHeight(28); self.restore_config_browse_btn.setFixedWidth(65)
        self.restore_config_browse_btn.clicked.connect(self._on_browse_restore_config)
        config_path_row.addWidget(self.restore_config_browse_btn)
        restore_content_layout.addLayout(config_path_row)

        restore_content_layout.addWidget(QLabel("Optimizer used for this model:"))
        self.restore_optimizer_combo = QComboBox()
        self.restore_optimizer_combo.addItems(["adam", "lbfgs"])
        self.restore_optimizer_combo.setFixedHeight(28)
        restore_content_layout.addWidget(self.restore_optimizer_combo)

        restore_content_layout.addWidget(QLabel("Visualization type:"))
        self.restore_viz_combo = QComboBox()
        self.restore_viz_combo.addItems(["Surface", "Line (time steps)", "Animation Line (GIF)", "Animation Surface (GIF)"])
        self.restore_viz_combo.setFixedHeight(28)
        self.restore_viz_combo.currentTextChanged.connect(self._on_restore_viz_changed)
        restore_content_layout.addWidget(self.restore_viz_combo)

        self.restore_tsteps_spin = QSpinBox()
        self.restore_tsteps_spin.setRange(2, 50); self.restore_tsteps_spin.setValue(10)
        self.restore_tsteps_spin.setVisible(False)

        self._restore_viz_settings = {
            'colormap': 'RdBu_r',
            'surface_time': 1.0,
            'n_steps': 10,
            'colorbar': True,
        }

        restore_content_layout.addWidget(QLabel("Output to plot:"))
        self.restore_output_combo = QComboBox()
        self.restore_output_combo.addItems(["Output 1 (u)"])
        self.restore_output_combo.setFixedHeight(28)
        restore_content_layout.addWidget(self.restore_output_combo)

        restore_content_layout.addWidget(QLabel("Save visualization to:"))
        restore_save_row = QHBoxLayout()
        self.restore_save_path = QLineEdit()
        self.restore_save_path.setPlaceholderText("Directory to save output...")
        self.restore_save_path.setFixedHeight(28)
        restore_save_row.addWidget(self.restore_save_path)
        self.restore_save_browse_btn = QPushButton("Browse")
        self.restore_save_browse_btn.setFixedHeight(28); self.restore_save_browse_btn.setFixedWidth(65)
        self.restore_save_browse_btn.clicked.connect(self._on_browse_restore_save)
        restore_save_row.addWidget(self.restore_save_browse_btn)
        restore_content_layout.addLayout(restore_save_row)

        self.restore_btn = QPushButton("🔄  Restore & Visualize")

        self.restore_btn.setMinimumHeight(38)
        self.restore_btn.setStyleSheet("""
            QPushButton { background: #1a5c3a; color: white; font-size: 13px;
                          font-weight: bold; border-radius: 6px; border: none; }
            QPushButton:hover { background: #2a7c4a; }
            QPushButton:disabled { background: #333355; color: #666; }
        """)
        self.restore_btn.clicked.connect(self._on_restore)
        restore_content_layout.addWidget(self.restore_btn)
        left_layout.addWidget(restore_group)

        left_layout.addStretch()
        splitter.addWidget(left_scroll)

        # ── Right panel ───────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 4, 8, 4)
        right_layout.setSpacing(3)

        # Vertical splitter for log + plots
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_layout.addWidget(right_splitter)

        # Top part — log
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(2)

        log_label = QLabel("📋 Training Log")
        log_label.setStyleSheet("color: #a0c4ff; font-weight: bold; font-size: 12px; margin-top: 2px;")
        log_label.setFixedHeight(22)
        log_layout.addWidget(log_label)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Training log will appear here...")
        log_layout.addWidget(self.log_box)
        right_splitter.addWidget(log_widget)

        # Bottom part — controls + plots
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(3)
        right_splitter.addWidget(bottom_widget)
        right_splitter.setSizes([150, 650])

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

        self.plot_settings_btn = QPushButton("⚙")
        self.plot_settings_btn.setFixedHeight(28)
        self.plot_settings_btn.setFixedWidth(28)
        self.plot_settings_btn.setToolTip("Plot settings")
        self.plot_settings_btn.setStyleSheet("""
            QPushButton { background: #3e3e42; color: #a0c4ff; font-size: 14px;
                          border-radius: 4px; border: 1px solid #586e75; }
            QPushButton:hover { background: #586e75; }
        """)
        self.plot_settings_btn.clicked.connect(self._on_plot_settings)
        ctrl_row.addWidget(self.plot_settings_btn)

        self.ea_btn = QPushButton("📊 Error Analysis")
        self.ea_btn.setFixedHeight(28)
        self.ea_btn.setStyleSheet("""
            QPushButton { background: #1a3a5a; color: #74c0fc; font-size: 12px;
                          font-weight: bold; border-radius: 4px; border: 1px solid #2a5a8a; padding: 0 8px; }
            QPushButton:hover { background: #2a5a8a; }
        """)
        self.ea_btn.clicked.connect(self._on_error_analysis_btn)
        ctrl_row.addWidget(self.ea_btn)

        self._plot_viz_settings = {
            'colormap': 'RdBu_r',
            'surface_time': 1.0,
            'n_steps': 4,
            'n_2d_snapshots': 2,
            'colorbar': True,
            'levels': 100,
            'resolution': 200,
            'dpi': 300,
            'auto_range': True,
            'vmin': -1.0,
            'vmax': 1.0,
            'linewidth': 2.0,
            'fps': 10,
        }

        self.export_btn = QPushButton("💾 Export Solution")
        self.export_btn.setFixedHeight(28)
        self.export_btn.setStyleSheet("""
            QPushButton { background: #1a4a3a; color: #69db7c; font-size: 12px;
                          font-weight: bold; border-radius: 4px; border: 1px solid #2a6a4a; padding: 0 8px; }
            QPushButton:hover { background: #2a6a4a; }
        """)
        self.export_btn.clicked.connect(self._on_export_settings)
        ctrl_row.addWidget(self.export_btn)

        self.param_save_label = QLabel("  Save parameter:")
        self.param_save_label.setVisible(False)
        ctrl_row.addWidget(self.param_save_label)
        self.param_save_combo = QComboBox()
        self.param_save_combo.addItems(["No", "Every 100 iters", "Every 1000 iters"])
        self.param_save_combo.setFixedHeight(28)
        self.param_save_combo.setFixedWidth(140)
        self.param_save_combo.setVisible(False)
        ctrl_row.addWidget(self.param_save_combo)

        self.timesteps_spin = QSpinBox()
        self.timesteps_spin.setRange(2, 20); self.timesteps_spin.setValue(4)
        self.timesteps_spin.setVisible(False)

        ctrl_row.addStretch()
        bottom_layout.addLayout(ctrl_row)

        # Save / export row
        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("Save to:"))
        self.save_dir_input = QLineEdit()
        self.save_dir_input.setPlaceholderText("Save directory — saves plots, logs, models & data")
        self.save_dir_input.setText("/home/asfandyarkhan/deepxde_gui/Results")
        self.save_dir_input.setFixedHeight(28)
        save_row.addWidget(self.save_dir_input)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setFixedHeight(28); self.browse_btn.setFixedWidth(65)
        self.browse_btn.clicked.connect(self._on_browse)
        save_row.addWidget(self.browse_btn)

        self.export_grid_combo = QComboBox()
        self.export_grid_combo.addItems(["101", "51", "21", "11"])
        self.export_grid_combo.setVisible(False)

        self.export_tsteps_spin = QSpinBox()
        self.export_tsteps_spin.setRange(2, 50); self.export_tsteps_spin.setValue(11)
        self.export_tsteps_spin.setVisible(False)
        bottom_layout.addLayout(save_row)

        # Plot area
        plots_layout = QHBoxLayout()
        plots_layout.setSpacing(6)

        self.loss_label = QLabel()
        self.loss_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loss_label.setText("📉 Loss plot")
        self.loss_label.setStyleSheet("border: 1px solid #3e3e42; border-radius: 6px; color: #505080; background: #252526;")
        self.loss_label.setMinimumSize(500, 450)

        self.solution_label = QLabel()
        self.solution_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.solution_label.setText("🗺 Solution plot")
        self.solution_label.setStyleSheet("border: 1px solid #3e3e42; border-radius: 6px; color: #505080; background: #252526;")
        self.solution_label.setMinimumSize(500, 450)

        plots_layout.addWidget(self.loss_label)
        plots_layout.addWidget(self.solution_label)
        bottom_layout.addLayout(plots_layout)

        splitter.addWidget(right)
        splitter.setSizes([390, 720])

    # ── Dimension change ──────────────────────────────────────
    def _on_dim_changed(self):
        is_2d = self.radio_2d.isChecked()
        # Update quick examples list to match dimension
        self.quick_examples_combo.blockSignals(True)
        self.quick_examples_combo.clear()
        if is_2d:
            self.quick_examples_combo.addItems([
                "── Select ──",
                "2D Heat (Dirichlet/Neumann)",
                "2D Allen-Cahn (Mattey)",
                "2D Allen-Cahn (Wight)",
                "2D Cahn-Hilliard (Wight)",
                "FeCr PINN"
            ])
        else:
            self.quick_examples_combo.addItems([
                "── Select ──",
                "1D Heat",
                "1D Allen-Cahn",
                "1D Cahn-Hilliard"
            ])
        self.quick_examples_combo.blockSignals(False)
        self.y_row_widget.setVisible(is_2d)
        self.view_domain_check.setVisible(is_2d)
        for w in self._2d_bc_widgets:
            w.setVisible(is_2d)
        self._build_pde_inputs(self.num_outputs_spin.value())
        self._build_bc_inputs(self.num_outputs_spin.value())
        self._build_weight_inputs(self.num_outputs_spin.value())
    
    def _on_quick_example_selected(self, text):
        if text.startswith("──"):
            return
        self._on_template_selected(text)
        self.quick_examples_combo.blockSignals(True)
        self.quick_examples_combo.setCurrentIndex(0)
        self.quick_examples_combo.blockSignals(False)

    def _auto_configure_ea(self, ref_dir):
        """Auto-configure error analysis when a template with ground truth files is loaded."""
        import glob, numpy as np
        if not ref_dir or not os.path.isdir(ref_dir):
            return
        txt_files = sorted(glob.glob(os.path.join(ref_dir, 't_*.txt')))
        if not txt_files:
            return
        valid_files = []
        for fp in txt_files:
            try:
                d = np.loadtxt(fp)
                if d.ndim == 1: d = d.reshape(1, -1)
                is_2d = self.radio_2d.isChecked()
                t_val = float(d[0, 2]) if is_2d else float(d[0, 1])
                valid_files.append((t_val, fp))
            except Exception:
                continue
        if not valid_files:
            return
        valid_files.sort(key=lambda x: x[0])
        self._ea_settings = {
            'files': valid_files,
            'do_line': True,
            'do_surface': True,
            'do_l2': True,
            'do_mse': True,
            'do_max': True,
        }
        self.log_box.append(f"✅ Error analysis auto-configured — {len(valid_files)} ground truth files from template")

    # ── Plot type change ──────────────────────────────────────
    def _on_plot_type_changed(self, text):
        if text == "Line (time steps)":
            self._on_line_plot_settings()
    

    def _on_line_plot_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Line Plot Settings")
        dialog.setMinimumWidth(300)
        layout = QVBoxLayout(dialog)

        info = QLabel("Select number of time steps to plot.")
        info.setStyleSheet("color: #74c0fc; font-size: 12px;")
        layout.addWidget(info)

        steps_row = QHBoxLayout()
        steps_row.addWidget(QLabel("Time steps to show:"))
        steps_spin = QSpinBox()
        steps_spin.setRange(2, 20); steps_spin.setValue(self.timesteps_spin.value())
        steps_spin.setFixedWidth(80)
        steps_row.addStretch(); steps_row.addWidget(steps_spin)
        layout.addLayout(steps_row)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK"); cancel_btn = QPushButton("Cancel")
        btn_row.addStretch(); btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        def _on_cancel():
            self.plot_type_combo.setCurrentText("Surface")
            dialog.reject()

        cancel_btn.clicked.connect(_on_cancel)

        def _on_ok():
            self.timesteps_spin.setValue(steps_spin.value())
            dialog.accept()

        ok_btn.clicked.connect(_on_ok)
        dialog.exec()

    def _on_error_analysis_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Error Analysis Settings")
        dialog.setMinimumWidth(420)
        layout = QVBoxLayout(dialog)

        # Source selection
        src_group = QGroupBox("Reference Data Source")
        src_layout = QVBoxLayout(src_group)
        self._ea_radio_expr = QRadioButton("Analytical Expression")
        self._ea_radio_csv  = QRadioButton("CSV File")
        self._ea_radio_expr.setChecked(True)
        src_layout.addWidget(self._ea_radio_expr)
        src_layout.addWidget(self._ea_radio_csv)
        layout.addWidget(src_group)

        # Expression section
        self._ea_expr_widget = QWidget()
        expr_layout = QVBoxLayout(self._ea_expr_widget)
        expr_layout.setContentsMargins(0, 0, 0, 0)
        expr_layout.addWidget(QLabel("Analytical expression (simplified syntax):"))
        self._ea_expr_input = QLineEdit()
        self._ea_expr_input.setPlaceholderText("e.g. sin(pi*x)*exp(-0.4*pi**2*t)")
        self._ea_expr_input.setFixedHeight(28)
        expr_layout.addWidget(self._ea_expr_input)
        expr_layout.addWidget(QLabel("Evaluation times (comma separated):"))
        self._ea_times_input = QLineEdit()
        self._ea_times_input.setText("0.25, 0.5, 0.75, 1.0")
        self._ea_times_input.setFixedHeight(28)
        expr_layout.addWidget(self._ea_times_input)
        layout.addWidget(self._ea_expr_widget)

        # CSV section
        self._ea_csv_widget = QWidget()
        csv_layout = QVBoxLayout(self._ea_csv_widget)
        csv_layout.setContentsMargins(0, 0, 0, 0)
        csv_layout.addWidget(QLabel("CSV file:"))
        csv_row = QHBoxLayout()
        self._ea_csv_path = QLineEdit()
        self._ea_csv_path.setPlaceholderText("Browse for CSV file...")
        self._ea_csv_path.setFixedHeight(28)
        csv_row.addWidget(self._ea_csv_path)
        csv_browse = QPushButton("Browse")
        csv_browse.setFixedHeight(28); csv_browse.setFixedWidth(65)
        csv_browse.clicked.connect(lambda: self._ea_csv_path.setText(
            QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV (*.csv *.txt)")[0]))
        csv_row.addWidget(csv_browse)
        csv_layout.addLayout(csv_row)
        info = QLabel("Expected format — 1D: x, t, u  |  2D: x, y, t, u")
        info.setStyleSheet("color: #586e75; font-size: 11px;")
        csv_layout.addWidget(info)
        self._ea_csv_widget.setVisible(False)
        layout.addWidget(self._ea_csv_widget)

        # Toggle visibility
        self._ea_radio_expr.toggled.connect(lambda c: (
            self._ea_expr_widget.setVisible(c),
            self._ea_csv_widget.setVisible(not c)
        ))

        # Time snapshot for 2D
        self._ea_2d_widget = QWidget()
        td_layout = QHBoxLayout(self._ea_2d_widget)
        td_layout.setContentsMargins(0, 0, 0, 0)
        td_layout.addWidget(QLabel("2D snapshot time:"))
        self._ea_2d_time = QDoubleSpinBox()
        self._ea_2d_time.setRange(0.0, 1e6); self._ea_2d_time.setValue(0.5)
        self._ea_2d_time.setFixedHeight(28); self._ea_2d_time.setFixedWidth(100)
        td_layout.addStretch(); td_layout.addWidget(self._ea_2d_time)
        self._ea_2d_widget.setVisible(self.radio_2d.isChecked())
        layout.addWidget(self._ea_2d_widget)

        # Error metrics
        metrics_group = QGroupBox("Error Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        self._ea_abs  = QCheckBox("Absolute Error plot"); self._ea_abs.setChecked(True)
        self._ea_l2   = QCheckBox("L2 Relative Error");   self._ea_l2.setChecked(True)
        self._ea_mse  = QCheckBox("MSE");                  self._ea_mse.setChecked(True)
        self._ea_max  = QCheckBox("Max Error");            self._ea_max.setChecked(True)
        for w in [self._ea_abs, self._ea_l2, self._ea_mse, self._ea_max]:
            metrics_layout.addWidget(w)
        layout.addWidget(metrics_group)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("Run Analysis")
        ok_btn.setStyleSheet("QPushButton { background: #1a4a6a; color: #74c0fc; font-weight: bold; border-radius: 4px; padding: 4px 12px; }")
        cancel_btn = QPushButton("Cancel")
        btn_row.addStretch(); btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        def _on_cancel():
            self.plot_type_combo.setCurrentText("Surface")
            dialog.reject()

        cancel_btn.clicked.connect(_on_cancel)
        ok_btn.clicked.connect(lambda: self._run_error_analysis(dialog))
        dialog.exec()

    def _run_error_analysis(self, dialog):
        save_dir = self.save_dir_input.text().strip()
        if not save_dir:
            self.log_box.append("❌ Please set a save directory first.")
            self.plot_type_combo.setCurrentText("Surface")
            dialog.reject()
            return

        use_expr = self._ea_radio_expr.isChecked()
        expr_raw = self._ea_expr_input.text().strip()
        csv_path = self._ea_csv_path.text().strip()

        if use_expr and not expr_raw:
            self.log_box.append("❌ Please enter an analytical expression."); return
        if not use_expr and not csv_path:
            self.log_box.append("❌ Please select a CSV file."); return

        self._ea_settings = {
            'use_expr': use_expr,
            'expr': expr_raw,
            'csv_path': csv_path,
            'times': self._ea_times_input.text().strip(),
            'snap_time': self._ea_2d_time.value(),
            'do_abs': self._ea_abs.isChecked(),
            'do_l2': self._ea_l2.isChecked(),
            'do_mse': self._ea_mse.isChecked(),
            'do_max': self._ea_max.isChecked(),
        }
        self.log_box.append("✅ Error analysis configured — will run after training completes.")
        dialog.accept()

    def _on_ea_done(self, success):
        save_dir = self.save_dir_input.text().strip()
        if success:
            self.log_box.append("✅ Error analysis complete!")
            plot_path = os.path.join(save_dir, "comparison_plot.png")
            if os.path.exists(plot_path):
                self.solution_label.setPixmap(QPixmap(plot_path).scaled(
                    500, 420, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
        else:
            self.log_box.append("❌ Error analysis failed — check log.")

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
            if is_2d:
                pde_inp.setText("du_t - 0.0001 * (du_xx + du_yy) + 1 * (u**3 - u)" if i == 0 else "dv_t - 0.0001 * (dv_xx + dv_yy) + 1 * (v**3 - v)")
            else:
                pde_inp.setText("du_t - 0.4 * du_xx" if i == 0 else "dv_t - 0.1 * dv_xx")
            pde_inp.setFixedHeight(28)
            self.pde_inputs.append(pde_inp)
            self.pde_main_layout.addWidget(pde_inp)

        names_ex = ", ".join([["u","v","w","p"][i] if i < 4 else f"u{i+1}" for i in range(n)])
        if is_2d:
            hint_text = (f"Outputs: {names_ex}\n"
                         f"du_x→∂u/∂x  du_y→∂u/∂y  du_t→∂u/∂t\n"
                         f"du_xx→∂²u/∂x²  du_yy→∂²u/∂y²  du_xy→∂²u/∂x∂y\n"
                         f"du_tt→∂²u/∂t²  du_xt→∂²u/∂x∂t  du_yt→∂²u/∂y∂t\n"
                         f"du_xxxx→∂⁴u/∂x⁴  du_yyyy→∂⁴u/∂y⁴\n"
                         f"du_xxyy→∂⁴u/∂x²∂y²  du_xxtt→∂⁴u/∂x²∂t²\n"
                         f"Functions: sin, cos, exp, log, sqrt, tanh, pi\n"
                         f"e.g. Allen-Cahn 2D: du_t - 0.001*(du_xx+du_yy) + u**3 - u\n"
                         f"e.g. sin(pi*u)*du_xx + cos(u)*du_yy")
        else:
            hint_text = (f"Outputs: {names_ex}\n"
                         f"du_x→∂u/∂x  du_t→∂u/∂t\n"
                         f"du_xx→∂²u/∂x²  du_tt→∂²u/∂t²  du_xt→∂²u/∂x∂t\n"
                         f"du_xxxx→∂⁴u/∂x⁴  du_tttt→∂⁴u/∂t⁴  du_xxtt→∂⁴u/∂x²∂t²\n"
                         f"Functions: sin, cos, exp, log, sqrt, tanh, pi\n"
                         f"e.g. Diffusion:      du_t - 0.4*du_xx\n"
                         f"e.g. Burgers:        du_t + u*du_x - 0.01*du_xx\n"
                         f"e.g. Cahn-Hilliard:  du_t - (du_xx - du_xxxx)\n"
                         f"e.g. Nonlinear:      du_t - sin(u)*du_xx")
            
        # Templates + derivative reference row
        tmpl_ref_row = QHBoxLayout()

        hint_toggle = QCheckBox("📖 Show derivative reference")
        hint_toggle.setChecked(False)
        hint_toggle.setStyleSheet("color: #74c0fc; font-size: 13px;")
        tmpl_ref_row.addWidget(hint_toggle)

        tmpl_ref_row.addStretch()

        tmpl_ref_row_widget = QWidget()
        tmpl_ref_row_widget.setLayout(tmpl_ref_row)
        self.pde_main_layout.addWidget(tmpl_ref_row_widget)

        hint = QLabel(hint_text)
        hint.setStyleSheet("color: #74c0fc; font-size: 13px;")
        hint.setWordWrap(True)
        hint.setVisible(False)
        self.pde_main_layout.addWidget(hint)

        hint_toggle.stateChanged.connect(lambda state, h=hint: h.setVisible(state == 2))

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
        if not hasattr(self, 'ic_from_file'): self.ic_from_file = []
        if not hasattr(self, 'ic_file_paths'): self.ic_file_paths = []
        self.ic_from_file.clear()
        self.ic_file_paths.clear()
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

            # Master toggle for outputs > 0
            if i > 0:
                _bc_enable_cb = QCheckBox(f"Enable boundary conditions for {name}")
                _bc_enable_cb.setChecked(True)
                _bc_enable_cb.setStyleSheet("color: #ffa94d; font-size: 12px;")
                self.bc_main_layout.addWidget(_bc_enable_cb)
            else:
                _bc_enable_cb = None

            # Container widget for all BC+IC of this output
            _bc_container = QWidget()
            _bc_cont_layout = QVBoxLayout(_bc_container)
            _bc_cont_layout.setSpacing(3)
            _bc_cont_layout.setContentsMargins(0, 0, 0, 0)
            _main_layout_save = self.bc_main_layout
            self.bc_main_layout = _bc_cont_layout

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
            _2d_save = self.bc_main_layout
            self.bc_main_layout = _2d_l
            _make_bc_block(f"BC bottom (y=ymin) for {name}", self.bc_bottom_types, self.bc_bottom_vals, self.bc_bottom_active, self.bc_bottom_deriv, "y_min")
            _make_bc_block(f"BC top (y=ymax) for {name}", self.bc_top_types, self.bc_top_vals, self.bc_top_active, self.bc_top_deriv, "y_max")
            self.bc_main_layout = _2d_save
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
            if is_2d:
                ic_inp.setText("sin(4*pi*x)*cos(4*pi*y)" if i == 0 else "cos(4*pi*x)*sin(4*pi*y)")
            else:
                ic_inp.setText("sin(pi*x)" if i == 0 else "cos(pi*x)")
            ic_inp.setFixedHeight(26)
            self.ic_inputs.append(ic_inp)
            self.bc_main_layout.addWidget(ic_inp)

            # IC reference toggle
            ic_hint_toggle = QCheckBox("📖 Show IC reference")
            ic_hint_toggle.setChecked(False)
            ic_hint_toggle.setStyleSheet("color: #74c0fc; font-size: 12px;")
            self.bc_main_layout.addWidget(ic_hint_toggle)

            if is_2d:
                ic_hint_text = (
                    "Use x, y as spatial variables, t for time.\n"
                    "sin(pi*x)*cos(pi*y)  →  sin wave 2D\n"
                    "exp(-(x**2+y**2))    →  Gaussian\n"
                    "sin(4*pi*x)*cos(4*pi*y)  →  higher freq\n"
                    "0                    →  zero IC\n"
                    "No need for np. or x[:,0] — handled automatically."
                )
            else:
                ic_hint_text = (
                    "Use x as spatial variable, t for time.\n"
                    "sin(pi*x)       →  sine wave\n"
                    "exp(-x**2)      →  Gaussian\n"
                    "sin(4*pi*x)     →  higher frequency\n"
                    "x*(1-x)         →  parabola\n"
                    "0               →  zero IC\n"
                    "No need for np. or x[:,0] — handled automatically."
                )
            ic_hint = QLabel(ic_hint_text)
            ic_hint.setStyleSheet("color: #74c0fc; font-size: 12px;")
            ic_hint.setWordWrap(True)
            ic_hint.setVisible(False)
            self.bc_main_layout.addWidget(ic_hint)
            ic_hint_toggle.stateChanged.connect(lambda state, h=ic_hint: h.setVisible(state == 2))

            # IC from file option (2D only)
            if is_2d:
                ic_file_cb = QCheckBox("📂 Load IC from file (x,y,t,c format)")
                ic_file_cb.setChecked(False)
                ic_file_cb.setStyleSheet("color: #ffa94d; font-size: 12px;")
                self.bc_main_layout.addWidget(ic_file_cb)
                self.ic_from_file.append(ic_file_cb)

                ic_file_widget = QWidget()
                ic_file_layout = QHBoxLayout(ic_file_widget)
                ic_file_layout.setContentsMargins(0, 0, 0, 0)
                ic_file_path = QLineEdit()
                ic_file_path.setPlaceholderText("Browse for IC file (x,y,t,c)...")
                ic_file_path.setFixedHeight(26)
                ic_file_layout.addWidget(ic_file_path)
                ic_file_browse = QPushButton("Browse")
                ic_file_browse.setFixedHeight(26); ic_file_browse.setFixedWidth(65)
                def _make_browse(path_edit):
                    def _browse():
                        f, _ = QFileDialog.getOpenFileName(None, "Select IC file", "", "Data files (*.txt *.csv *.dat)")
                        if f: path_edit.setText(f)
                    return _browse
                ic_file_browse.clicked.connect(_make_browse(ic_file_path))
                ic_file_layout.addWidget(ic_file_browse)
                ic_file_widget.setVisible(False)
                self.bc_main_layout.addWidget(ic_file_widget)
                self.ic_file_paths.append(ic_file_path)

                ic_file_cb.stateChanged.connect(
                    lambda state, w=ic_file_widget, inp=ic_inp, act=ic_act:
                    (w.setVisible(state == 2), inp.setVisible(state != 2),
                     act.setChecked(state != 2),
                     self.ic_pretrain_init_widget.setVisible(state != 2)
                     if hasattr(self, 'ic_pretrain_init_widget') else None)
                )
            else:
                self.ic_from_file.append(None)
                self.ic_file_paths.append(None)

            # Restore layout and add container
            self.bc_main_layout = _main_layout_save
            self.bc_main_layout.addWidget(_bc_container)

            # Wire master toggle
            if _bc_enable_cb is not None:
                def _make_toggle(cont, la, ra, ica):
                    def _tog(state):
                        cont.setVisible(state == 2)
                        la.setChecked(state == 2)
                        ra.setChecked(state == 2)
                        if ica: ica.setChecked(state == 2)
                        self._build_weight_inputs(self.num_outputs_spin.value())
                    return _tog
                _i_capture = i
                _bc_enable_cb.stateChanged.connect(_make_toggle(
                    _bc_container,
                    self.bc_left_active[_i_capture],
                    self.bc_right_active[_i_capture],
                    self.ic_active[_i_capture] if _i_capture < len(self.ic_active) else None
                ))

    # ── Build weight inputs ───────────────────────────────────
    def _build_weight_inputs(self, n):
        for i in reversed(range(self.weights_main_layout.count())):
            w = self.weights_main_layout.itemAt(i).widget()
            if w:
                w.deleteLater()
        self.weight_widgets.clear()

        # Check if per-phase weights needed
        _sched_enabled = hasattr(self, 'sched_cb') and self.sched_cb.isChecked()
        _same_weights = not _sched_enabled or (
            hasattr(self, 'sched_same_weights_cb') and self.sched_same_weights_cb.isChecked())
        _num_phases = 1 + (len(self.sched_phase_list) if _sched_enabled else 0)

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
            _ic_from_file_checked = (
                hasattr(self, 'ic_from_file') and
                i < len(self.ic_from_file) and
                self.ic_from_file[i] is not None and
                self.ic_from_file[i].isChecked()
            )
            if (i < len(self.ic_active) and self.ic_active[i].isChecked()) or _ic_from_file_checked:
                _w_row(f"IC {i+1} ({name}):", f"ic_{i}")

        # Per-phase weight rows if scheduler enabled and different weights
        if _sched_enabled and not _same_weights:
            for _pi, _ph in enumerate(self.sched_phase_list):
                _phase_num = _ph['phase_num']
                sep = QLabel(f"── Phase {_phase_num} weights ──")
                sep.setStyleSheet("color: #505080; font-size: 10px;")
                _sw = QWidget(); _sl = QHBoxLayout(_sw)
                _sl.setContentsMargins(0,0,0,0); _sl.addWidget(sep)
                self.weights_main_layout.addWidget(_sw)
                for _i in range(n):
                    _name = self.output_name_inputs[_i].text() if _i < len(self.output_name_inputs) else f"u{_i+1}"
                    _w_row(f"PDE {_i+1} ({_name}) P{_phase_num}:", f"pde_{_i}_p{_phase_num}")
                    if _i < len(self.bc_left_active) and self.bc_left_active[_i].isChecked():
                        _w_row(f"BC left {_i+1} P{_phase_num}:", f"bc_left_{_i}_p{_phase_num}")
                    _ic_ff = (hasattr(self, 'ic_from_file') and _i < len(self.ic_from_file)
                              and self.ic_from_file[_i] is not None and self.ic_from_file[_i].isChecked())
                    if (_i < len(self.ic_active) and self.ic_active[_i].isChecked()) or _ic_ff:
                        _w_row(f"IC {_i+1} ({_name}) P{_phase_num}:", f"ic_{_i}_p{_phase_num}")

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
            forward_ic_from_file=any(
                i < len(self.ic_from_file) and self.ic_from_file[i] is not None
                and self.ic_from_file[i].isChecked()
                for i in range(n_out)
            ),
            forward_ic_file=next(
                (self.ic_file_paths[i].text().strip()
                 for i in range(n_out)
                 if i < len(self.ic_from_file)
                 and self.ic_from_file[i] is not None
                 and self.ic_from_file[i].isChecked()
                 and self.ic_file_paths[i] is not None),
                ""
            ),

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
            inv_param_save=self.param_save_combo.currentText(),

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
            point_distribution=self.pts_dist_combo.currentText(),
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
            ta_transfer_learning=self.ta_transfer_cb.isChecked(),
            ta_transfer_optimizer=self.ta_transfer_opt.currentText(),
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
            template_type=getattr(self, '_current_template_type', ''),
            optimizer_scheduler=self.sched_cb.isChecked() if hasattr(self, 'sched_cb') else False,
            scheduler_same_weights=self.sched_same_weights_cb.isChecked() if hasattr(self, 'sched_same_weights_cb') else True,
            scheduler_phases=self._build_scheduler_phases_json(),
            lbfgs_maxcor=int(self.lbfgs_maxcor.value()),
            lbfgs_ftol=self.lbfgs_ftol.value(),
            lbfgs_gtol=self.lbfgs_gtol.value(),
            lbfgs_maxiter=int(self.lbfgs_maxiter.value()),
            lbfgs_maxfun=int(self.lbfgs_maxfun.value()),
            lbfgs_maxls=int(self.lbfgs_maxls.value()),
            float_type=getattr(self, '_float_type', 'float64'),
            batch_size=self.batch_spin.value() if self.batch_check.isChecked() else 0,
            ic_pretrain=self.ic_pretrain_cb.isChecked(),
            ic_pretrain_optimizer=self.ic_pretrain_opt.currentText(),
            ic_pretrain_iterations=self.ic_pretrain_iters.value(),
            ic_pretrain_num_test=self.ic_pretrain_test.value(),
            ic_pretrain_num_initial=self.ic_pretrain_init.value(),
            ic_pretrain_restore=self.ic_pretrain_restore_cb.isChecked(),
            ic_pretrain_restore_path=self.ic_pretrain_restore_path.text().strip(),
            plot_colormap=self._plot_viz_settings.get('colormap', 'RdBu_r'),
            plot_levels=self._plot_viz_settings.get('levels', 50),
            plot_resolution=self._plot_viz_settings.get('resolution', 100),
            plot_dpi=self._plot_viz_settings.get('dpi', 100),
            plot_colorbar=self._plot_viz_settings.get('colorbar', True),
            plot_auto_range=self._plot_viz_settings.get('auto_range', True),
            plot_vmin=self._plot_viz_settings.get('vmin', -1.0),
            plot_vmax=self._plot_viz_settings.get('vmax', 1.0),
            plot_linewidth=self._plot_viz_settings.get('linewidth', 2.0),
            plot_n_2d_snapshots=self._plot_viz_settings.get('n_2d_snapshots', 2),
            ea_files=repr(self._ea_settings.get('files', [])) if getattr(self, '_ea_settings', None) else "[]",
            ea_do_line=self._ea_settings.get('do_line', True) if getattr(self, '_ea_settings', None) else True,
            ea_do_surface=self._ea_settings.get('do_surface', True) if getattr(self, '_ea_settings', None) else True,
        )

    def _on_num_outputs_changed(self, n):
        self._build_pde_inputs(n)
        self._build_bc_inputs(n)
        self._build_weight_inputs(n)
        self.plot_output_combo.clear()
        self.restore_output_combo.clear()
        for i in range(n):
            name = self.output_name_inputs[i].text() if i < len(self.output_name_inputs) else f"u{i+1}"
            self.plot_output_combo.addItem(f"Output {i+1} ({name})")
            self.restore_output_combo.addItem(f"Output {i+1} ({name})")

    def _on_problem_type_changed(self, checked):
        is_inv = self.radio_inverse.isChecked()
        self.inverse_group.setVisible(is_inv)
        self.param_save_label.setVisible(is_inv)
        self.param_save_combo.setVisible(is_inv)

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
            sol_dir = os.path.join(save_dir, "solution_results") if save_dir else "/tmp"
            loss_path = os.path.join(sol_dir, "loss_plot.png")
            solution_path = os.path.join(sol_dir, "solution_plot.png")
            # Fallback to root save dir for older runs
            if not os.path.exists(loss_path):
                loss_path = os.path.join(save_dir, "loss_plot.png") if save_dir else "/tmp/loss_plot.png"
            if not os.path.exists(solution_path):
                solution_path = os.path.join(save_dir, "solution_plot.png") if save_dir else "/tmp/solution_plot.png"

            if os.path.exists(loss_path):
                self.loss_label.setPixmap(QPixmap(loss_path).scaled(
                    self.loss_label.width(), self.loss_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
            if os.path.exists(solution_path):
                self.solution_label.setPixmap(QPixmap(solution_path).scaled(
                    self.solution_label.width(), self.solution_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
            
            if save_dir:
                self.log_box.append(f"💾 Results saved to: {save_dir}")

            if hasattr(self, '_ea_settings') and self._ea_settings:
                self.log_box.append("✅ Error analysis ran inline — check error_analysis/ folder.")
                self._ea_settings = None
            else:
                self.log_box.append("ℹ️ No error analysis configured — click '📊 Error Analysis' before training.")

        else:
            self.log_box.append("\n❌ Error during training. Check log above.")
            self.log_box.append("💡 Tip: Check PDE/IC syntax — use * for multiplication (e.g. 5*u not 5u)")

    def _on_float_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Float Precision")
        dialog.setMinimumWidth(300)
        layout = QVBoxLayout(dialog)

        info = QLabel("Float64 recommended for L-BFGS convergence.\nFloat32 is faster but L-BFGS may stop early.")
        info.setStyleSheet("color: #74c0fc; font-size: 12px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        row = QHBoxLayout()
        row.addWidget(QLabel("Float type:"))
        self._float_combo = QComboBox()
        self._float_combo.addItems(["float64", "float32"])
        self._float_combo.setCurrentText(getattr(self, '_float_type', 'float32'))
        self._float_combo.setFixedWidth(100)
        row.addStretch(); row.addWidget(self._float_combo)
        layout.addLayout(row)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK"); cancel_btn = QPushButton("Cancel")
        btn_row.addStretch(); btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        cancel_btn.clicked.connect(dialog.reject)

        def _on_ok():
            self._float_type = self._float_combo.currentText()
            self.log_box.append(f"✅ Float precision set to: {self._float_type}")
            dialog.accept()

        ok_btn.clicked.connect(_on_ok)
        dialog.exec()

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
    
    def _on_view_domain_changed(self, state):
        if state == 2:
            self._preview_domain()
        else:
            self.loss_label.setText("📉 Loss plot")
            self.solution_label.setText("🗺 Solution plot")

    def _preview_domain(self):
        import tempfile, subprocess, sys
        x_min = self.x_min.value(); x_max = self.x_max.value()
        y_min = self.y_min.value(); y_max = self.y_max.value()
        t_min = self.t_min.value(); t_max = self.t_max.value()
        n_domain   = self.num_domain.value()
        n_boundary = self.num_boundary.value()
        n_initial  = self.num_initial.value()
        dist       = self.pts_dist_combo.currentText()

        try:
            self.pts_dist_combo.currentTextChanged.disconnect(self._on_dist_changed)
        except Exception:
            pass
        self.pts_dist_combo.currentTextChanged.connect(self._on_dist_changed)
        for sb in [self.num_domain, self.num_boundary, self.num_initial]:
            try:
                sb.valueChanged.disconnect(self._on_pts_changed)
            except Exception:
                pass
            sb.valueChanged.connect(self._on_pts_changed)

        script = f"""
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

geom  = dde.geometry.Rectangle([{x_min}, {y_min}], [{x_max}, {y_max}])
t_dom = dde.geometry.TimeDomain({t_min}, {t_max})
gt    = dde.geometry.GeometryXTime(geom, t_dom)

def pde(x, y): return y[:, 0:1] * 0
data = dde.data.TimePDE(gt, pde, [],
    num_domain={n_domain}, num_boundary={n_boundary},
    num_initial={n_initial}, num_test=10,
    train_distribution="{dist}")

pts = data.train_points()
t_range = {t_max} - {t_min}
tol = max(t_range * 0.05, 1e-6)
t_snap = {t_min}

plt.rcParams['figure.dpi'] = 120
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#1e1e1e')

for ax in axes:
    ax.set_facecolor('#252526')
    ax.set_xlim({x_min} - 0.05*({x_max}-{x_min}), {x_max} + 0.05*({x_max}-{x_min}))
    ax.set_ylim({y_min} - 0.05*({y_max}-{y_min}), {y_max} + 0.05*({y_max}-{y_min}))
    rect = plt.Rectangle(({x_min},{y_min}), {x_max}-{x_min}, {y_max}-{y_min},
        linewidth=2, edgecolor='#a0c4ff', facecolor='none')
    ax.add_patch(rect)
    ax.tick_params(colors='#c0c0c0')
    for sp in ax.spines.values(): sp.set_color('#3e3e42')

# Separate points by type using edge detection
ic_mask   = pts[:, 2] <= {t_min} + tol
on_left   = np.abs(pts[:, 0] - {x_min}) < 1e-10
on_right  = np.abs(pts[:, 0] - {x_max}) < 1e-10
on_bottom = np.abs(pts[:, 1] - {y_min}) < 1e-10
on_top    = np.abs(pts[:, 1] - {y_max}) < 1e-10
bnd_mask  = (on_left | on_right | on_bottom | on_top) & ~ic_mask
dom_mask  = ~ic_mask & ~bnd_mask
dom_pts = pts[dom_mask]
bnd_pts = pts[bnd_mask]
ic_pts  = pts[ic_mask]

# Left: spatial distribution (x,y)
ax = axes[0]
if len(dom_pts): ax.scatter(dom_pts[:,0], dom_pts[:,1], s=8, c='#74c0fc', alpha=0.7, label=f'Domain ({{len(dom_pts)}})')
if len(bnd_pts): ax.scatter(bnd_pts[:,0], bnd_pts[:,1], s=20, c='#f03e3e', alpha=1.0, label=f'Boundary ({{len(bnd_pts)}})')
if len(ic_pts):  ax.scatter(ic_pts[:,0],  ic_pts[:,1],  s=20, c='#2f9e44', alpha=1.0, label=f'IC ({{len(ic_pts)}})')
ax.set_xlabel('x', color='#e0e0e0', fontsize=11)
ax.set_ylabel('y', color='#e0e0e0', fontsize=11)
ax.set_title('Spatial (x,y) | {dist} | D={n_domain} B={n_boundary} IC={n_initial}', color='#74c0fc', fontsize=10, fontweight='bold')
ax.legend(fontsize=10, facecolor='#2a2a2a', labelcolor='#e0e0e0', edgecolor='#555', markerscale=1.5)

# Right: time distribution (x vs t)
ax = axes[1]
ax.set_xlim({x_min}, {x_max}); ax.set_ylim({t_min}, {t_max})
if len(dom_pts): ax.scatter(dom_pts[:,0], dom_pts[:,2], s=6, c='#74c0fc', alpha=0.5, label=f'Domain ({{len(dom_pts)}})')
if len(bnd_pts): ax.scatter(bnd_pts[:,0], bnd_pts[:,2], s=14, c='#f03e3e', alpha=0.9, label=f'Boundary ({{len(bnd_pts)}})')
if len(ic_pts):  ax.scatter(ic_pts[:,0],  ic_pts[:,2],  s=14, c='#2f9e44', alpha=1.0, label=f'IC ({{len(ic_pts)}})')
ax.set_xlabel('x', color='#e0e0e0', fontsize=11)
ax.set_ylabel('t', color='#e0e0e0', fontsize=11)
ax.set_title(f'Time Distribution (x vs t)\\ntotal={{len(pts)}} points', color='#74c0fc', fontsize=10, fontweight='bold')
ax.legend(fontsize=10, facecolor='#2a2a2a', labelcolor='#e0e0e0', edgecolor='#555', markerscale=1.5)
plt.tight_layout()
plt.savefig('/tmp/domain_preview.png', dpi=100, bbox_inches='tight', facecolor='#1e1e1e')
plt.close()
print("DOMAIN_PREVIEW_DONE")
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
            tf.write(script)
            tmp = tf.name

        self.loss_label.setText("⏳ Generating preview...")

        from PyQt6.QtCore import QThread, pyqtSignal as _sig

        class _PreviewThread(QThread):
            done_sig = _sig(bool)
            log_sig  = _sig(str)
            def __init__(self, tmp):
                super().__init__(); self._tmp = tmp
            def run(self):
                import subprocess, sys
                proc = subprocess.Popen([sys.executable, self._tmp],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    self.log_sig.emit(line.rstrip())
                proc.wait()
                os.unlink(self._tmp)
                self.done_sig.emit(proc.returncode == 0)

        self._preview_thread = _PreviewThread(tmp)
        self._preview_thread.done_sig.connect(self._on_preview_done)
        self._preview_thread.log_sig.connect(self.log_box.append)
        self._preview_thread.start()

    def _on_dist_changed(self, text):
        if self.view_domain_check.isChecked():
            self._preview_domain()

    def _on_pts_changed(self, val):
        if self.view_domain_check.isChecked():
            self._preview_domain()
    
    def _on_preview_done(self, success):
        if success and os.path.exists('/tmp/domain_preview.png'):
            # Split the wide image across both panels
            from PyQt6.QtGui import QPixmap as _QPix
            full = _QPix('/tmp/domain_preview.png')
            w = full.width(); h = full.height()
            left_half  = full.copy(0,       0, w//2, h)
            right_half = full.copy(w//2, 0, w//2, h)
            self.loss_label.setPixmap(left_half.scaled(
                500, 420, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            self.solution_label.setPixmap(right_half.scaled(
                500, 420, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        else:
            self.loss_label.setText("❌ Preview failed")
    
    def _on_display_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Display Settings")
        dialog.setMinimumWidth(380)
        layout = QVBoxLayout(dialog)

        # Font size
        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("UI Font size:"))
        font_spin = QSpinBox(); font_spin.setRange(9, 18); font_spin.setValue(self._font_size)
        font_spin.setFixedWidth(80)
        font_row.addStretch(); font_row.addWidget(font_spin)
        layout.addLayout(font_row)

        # Log font size
        log_font_row = QHBoxLayout()
        log_font_row.addWidget(QLabel("Log font size:"))
        log_font_spin = QSpinBox(); log_font_spin.setRange(8, 16); log_font_spin.setValue(self._log_font_size)
        log_font_spin.setFixedWidth(80)
        log_font_row.addStretch(); log_font_row.addWidget(log_font_spin)
        layout.addLayout(log_font_row)

        # Theme
        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel("Color theme:"))
        theme_combo = QComboBox()
        theme_combo.addItems(["Dark Grey", "GitHub Dark", "Monokai", "Solarized Dark", "Navy Blue", "White"])
        theme_combo.setCurrentText(self._theme)
        theme_combo.setFixedWidth(160)
        theme_row.addStretch(); theme_row.addWidget(theme_combo)
        layout.addLayout(theme_row)

        # Accent color
        accent_row = QHBoxLayout()
        accent_row.addWidget(QLabel("Accent color:"))
        accent_combo = QComboBox()
        accent_combo.addItems(["Blue (#a0c4ff)", "Green (#69db7c)", "Orange (#ffa94d)", "Purple (#cc5de8)", "Teal (#38d9a9)", "Black (#000000)"])
        accent_combo.setCurrentText(self._accent)
        accent_combo.setFixedWidth(160)
        accent_row.addStretch(); accent_row.addWidget(accent_combo)
        layout.addLayout(accent_row)

        # Preview button
        preview_btn = QPushButton("Preview")
        layout.addWidget(preview_btn)

        def _apply_preview():
            self._font_size = font_spin.value()
            self._log_font_size = log_font_spin.value()
            self._theme = theme_combo.currentText()
            self._accent = accent_combo.currentText()
            self._apply_display_settings()

        preview_btn.clicked.connect(_apply_preview)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK"); cancel_btn = QPushButton("Cancel")
        btn_row.addStretch(); btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        cancel_btn.clicked.connect(dialog.reject)
        ok_btn.clicked.connect(lambda: (_apply_preview(), dialog.accept()))
        dialog.exec()


    def _apply_display_settings(self):
        themes = {
            "Dark Grey":     ("#1e1e1e", "#252526", "#3e3e42"),
            "GitHub Dark":   ("#0d1117", "#161b22", "#30363d"),
            "Monokai":       ("#272822", "#1e1f1c", "#49483e"),
            "Solarized Dark":("#002b36", "#073642", "#586e75"),
            "Navy Blue":     ("#1a1a2e", "#16213e", "#3a3a5c"),
            "White":         ("#ffffff", "#f5f5f5", "#d0d0d0"),
        }
        accents = {
            "Blue (#a0c4ff)":   "#a0c4ff",
            "Green (#69db7c)":  "#69db7c",
            "Orange (#ffa94d)": "#ffa94d",
            "Purple (#cc5de8)": "#cc5de8",
            "Teal (#38d9a9)":   "#38d9a9",
            "Black (#000000)":  "#000000",
        }
        bg, widget_bg, border = themes.get(self._theme, themes["Dark Grey"])
        accent = accents.get(self._accent, "#a0c4ff")
        fs = self._font_size
        lfs = self._log_font_size
        is_white = self._theme == "White"
        text_color = "#1e1e1e" if is_white else "#e0e0e0"
        label_color = "#333333" if is_white else "#c0c0c0"
        arrow_color = "#333333" if is_white else "#ffffff"

        self.setStyleSheet(f"""
            QMainWindow {{ background: {bg}; }}
            QWidget {{ background: {bg}; color: {text_color}; font-family: 'Segoe UI', Arial; font-size: {fs}px; }}
            QGroupBox {{
                border: 1px solid {border};
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 4px;
                font-weight: bold;
                color: {accent};
            }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
            QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
                background: {widget_bg};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 2px 6px;
                color: {text_color};
            }}
            QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{
                border: 1px solid {accent};
            }}
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                background: {border};
                border: none;
                width: 16px;
            }}
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 6px solid {arrow_color};
                width: 0px; height: 0px;
            }}
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid {arrow_color};
                width: 0px; height: 0px;
            }}
            QComboBox::drop-down {{ background: {border}; border: none; width: 20px; }}
            QComboBox::down-arrow {{
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid {arrow_color};
                width: 0px; height: 0px;
            }}
            QCheckBox {{ color: {label_color}; spacing: 6px; }}
            QCheckBox::indicator {{
                width: 14px; height: 14px;
                border: 1px solid {border};
                border-radius: 3px;
                background: {widget_bg};
            }}
            QCheckBox::indicator:checked {{ background: {accent}; border-color: {accent}; }}
            QRadioButton {{ color: {label_color}; spacing: 6px; }}
            QRadioButton::indicator {{
                width: 14px; height: 14px;
                border: 1px solid {border};
                border-radius: 7px;
                background: {widget_bg};
            }}
            QRadioButton::indicator:checked {{ background: {accent}; border-color: {accent}; }}
            QLabel {{ color: {label_color}; }}
            QScrollArea {{ border: none; background: {bg}; }}
            QScrollBar:vertical {{ background: {widget_bg}; width: 14px; border-radius: 6px; }}
            QScrollBar::handle:vertical {{ background: {border}; border-radius: 6px; min-height: 30px; }}
            QTextEdit {{
                background: {'#f8f8f8' if is_white else '#0f0f23'};
                border: 1px solid {border};
                border-radius: 6px;
                color: {'#1e1e1e' if is_white else '#a0ffb0'};
                font-family: 'Courier New', monospace;
                font-size: {lfs}px;
            }}
            QSplitter::handle {{ background: {border}; width: 2px; }}
            QMenuBar {{ background: {widget_bg}; color: {label_color}; border-bottom: 1px solid {border}; }}
            QMenuBar::item:selected {{ background: {border}; }}
            QMenu {{ background: {widget_bg}; border: 1px solid {border}; }}
            QMenu::item:selected {{ background: {border}; }}
        """)
        self.loss_label.setStyleSheet(f"border: 1px solid {border}; border-radius: 6px; color: #505080; background: {widget_bg};")
        self.solution_label.setStyleSheet(f"border: 1px solid {border}; border-radius: 6px; color: #505080; background: {widget_bg};")

    def _on_export_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Solution Data")
        dialog.setMinimumWidth(320)
        layout = QVBoxLayout(dialog)

        info = QLabel("Saves solution as CSV files (one per time step)\nfor forward problems after training.")
        info.setStyleSheet("color: #74c0fc; font-size: 12px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        grid_row = QHBoxLayout()
        grid_row.addWidget(QLabel("Grid size (points per axis):"))
        grid_combo = QComboBox()
        grid_combo.addItems(["101", "51", "21", "11"])
        grid_combo.setCurrentText(self.export_grid_combo.currentText())
        grid_combo.setFixedWidth(80)
        grid_row.addStretch(); grid_row.addWidget(grid_combo)
        layout.addLayout(grid_row)

        tsteps_row = QHBoxLayout()
        tsteps_row.addWidget(QLabel("Number of time snapshots:"))
        tsteps_spin = QSpinBox()
        tsteps_spin.setRange(2, 50); tsteps_spin.setValue(self.export_tsteps_spin.value())
        tsteps_spin.setFixedWidth(80)
        tsteps_row.addStretch(); tsteps_row.addWidget(tsteps_spin)
        layout.addLayout(tsteps_row)

        note = QLabel("Output: solution_data/solution_t{time}.txt\nColumns: x, t, u (1D) or x, y, t, u (2D)")
        note.setStyleSheet("color: #586e75; font-size: 11px;")
        note.setWordWrap(True)
        layout.addWidget(note)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK"); cancel_btn = QPushButton("Cancel")
        btn_row.addStretch(); btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        cancel_btn.clicked.connect(dialog.reject)

        def _on_ok():
            self.export_grid_combo.setCurrentText(grid_combo.currentText())
            self.export_tsteps_spin.setValue(tsteps_spin.value())
            dialog.accept()

        ok_btn.clicked.connect(_on_ok)
        dialog.exec()
    
    def _execute_error_analysis(self, ea, config):
        import tempfile, glob, json
        save_dir = self.save_dir_input.text().strip()
        is_2d = config.problem_dim == "2D"

        # Find most recently modified model
        model_path = ""
        all_models = glob.glob(os.path.join(save_dir, "model_lbfgs-*.pt")) + \
                     glob.glob(os.path.join(save_dir, "model_adam-*.pt"))
        if all_models:
            model_path = max(all_models, key=os.path.getmtime)
            self.log_box.append(f"📂 Using model: {os.path.basename(model_path)}")

        if not model_path:
            self.log_box.append("❌ No saved model found — set save directory before training."); return

        config_path = model_path.replace(".pt", ".json")
        if not os.path.exists(config_path):
            config_path = os.path.join(save_dir, "model_config.json")

        try:
            with open(config_path) as f:
                cfg = json.load(f)
        except Exception as e:
            self.log_box.append(f"❌ Could not read model config: {e}"); return

        layers = cfg["layers"]; activation = cfg["activation"]
        x_min = cfg["x_min"]; x_max = cfg["x_max"]
        y_min = cfg.get("y_min", 0.0); y_max = cfg.get("y_max", 1.0)
        t_min = cfg["t_min"]; t_max = cfg["t_max"]
        loss_type = cfg.get("loss_type", "MSE")
        compile_opt = "lbfgs" if "lbfgs" in model_path else "adam"

        use_expr = ea['use_expr']
        expr_raw = ea['expr']
        csv_path = ea['csv_path']
        times_str = ea['times']
        snap_time = ea['snap_time']

        from pinnstudio.core.codegen import _simplify_expr
        expr_converted = _simplify_expr(expr_raw, is_2d) if use_expr else ""

        try:
            times = [float(t.strip()) for t in times_str.split(",") if t.strip()]
        except Exception:
            times = [0.5]

        script = f"""
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if {str(is_2d)}:
    geom = dde.geometry.Rectangle([{x_min}, {y_min}], [{x_max}, {y_max}])
else:
    geom = dde.geometry.Interval({x_min}, {x_max})
td = dde.geometry.TimeDomain({t_min}, {t_max})
gt = dde.geometry.GeometryXTime(geom, td)
def pde(x, y): return y[:, 0:1] * 0
data  = dde.data.TimePDE(gt, pde, [], num_domain=100, num_test=100)
net   = dde.nn.FNN({layers}, "{activation}", "Glorot uniform")
model = dde.Model(data, net)
if "{compile_opt}" == "lbfgs":
    dde.optimizers.set_LBFGS_options(maxiter=1)
    model.compile("L-BFGS", loss="{loss_type}")
else:
    model.compile("adam", lr=0.001, loss="{loss_type}")
model.restore(r"{model_path}", verbose=0)
print("Model restored for error analysis.")
os.makedirs(r"{save_dir}", exist_ok=True)
metrics_lines = []
"""
        if not is_2d:
            script += f"""
x_vals = np.linspace({x_min}, {x_max}, 200)
times  = {times}
n_t = len(times)
fig, axes = plt.subplots(n_t, 3, figsize=(15, 4*n_t))
if n_t == 1: axes = [axes]
fig.suptitle("PINN vs Reference — Error Analysis", fontsize=13, fontweight='bold')
for row, tv in enumerate(times):
    xt = np.column_stack([x_vals, np.full_like(x_vals, tv)])
    u_pred = model.predict(xt)[:, 0].flatten()
"""
            if use_expr:
                script += f"""
    x = x_vals; t = tv
    u_ref = {expr_converted}
    u_ref = np.atleast_1d(u_ref)
    if len(u_ref) != len(x_vals):
        u_ref = np.full_like(x_vals, float(u_ref[0]))
"""
            else:
                script += f"""
    data_csv = np.loadtxt(r"{csv_path}", delimiter=",", skiprows=1)
    t_col = data_csv[:, 1]
    t_unique = np.unique(t_col)
    t_closest = t_unique[np.argmin(np.abs(t_unique - tv))]
    mask = np.abs(t_col - t_closest) < 1e-10
    u_ref = np.interp(x_vals, data_csv[mask, 0], data_csv[mask, 2])
"""
            script += f"""
    abs_err = np.abs(u_pred - u_ref)
    l2   = np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-10)
    mse  = np.mean((u_pred - u_ref)**2)
    maxe = np.max(abs_err)
    metrics_lines.append(f"t={{tv:.4f}}: L2={{l2:.4e}}, MSE={{mse:.4e}}, Max={{maxe:.4e}}")
    print(f"  t={{tv:.3f}} — L2={{l2:.4e}}, MSE={{mse:.4e}}, Max={{maxe:.4e}}")
    ax0, ax1, ax2 = axes[row]
    ax0.plot(x_vals, u_pred, color='#4dabf7', lw=2, label='PINN')
    ax0.plot(x_vals, u_ref,  color='#ff8787', lw=2, ls='--', label='Reference')
    ax0.set_title(f"t={{tv:.3f}} — PINN vs Reference"); ax0.legend(); ax0.grid(True, alpha=0.3)
    ax1.plot(x_vals, u_ref, color='#ff8787', lw=2)
    ax1.set_title(f"t={{tv:.3f}} — Reference"); ax1.grid(True, alpha=0.3)
    ax2.plot(x_vals, abs_err, color='#69db7c', lw=2)
    ax2.fill_between(x_vals, abs_err, alpha=0.3, color='#69db7c')
    ax2.set_title(f"t={{tv:.3f}} — |Error| L2={{l2:.2e}}"); ax2.grid(True, alpha=0.3)
plt.tight_layout()
"""
        else:
            script += f"""
tv = {snap_time}
x_vals = np.linspace({x_min}, {x_max}, 80)
y_vals = np.linspace({y_min}, {y_max}, 80)
Xg, Yg = np.meshgrid(x_vals, y_vals)
XYT = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, tv)])
u_pred = model.predict(XYT)[:, 0].reshape(80, 80)
"""
            if use_expr:
                script += f"""
x = Xg; y = Yg; t = tv
u_ref = {expr_converted}
if not hasattr(u_ref, 'shape') or u_ref.shape != (80, 80):
    u_ref = np.full((80, 80), float(u_ref))
"""
            else:
                script += f"""
data_csv = np.loadtxt(r"{csv_path}", delimiter=",", skiprows=1)
t_col = data_csv[:, 2]
t_unique = np.unique(t_col)
t_closest = t_unique[np.argmin(np.abs(t_unique - tv))]
mask = np.abs(t_col - t_closest) < 1e-10
from scipy.interpolate import griddata
u_ref = griddata(data_csv[mask, :2], data_csv[mask, 3], (Xg, Yg), method='linear', fill_value=0.0)
"""
            script += f"""
abs_err = np.abs(u_pred - u_ref)
l2   = np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-10)
mse  = np.mean((u_pred - u_ref)**2)
maxe = np.max(abs_err)
metrics_lines.append(f"t={snap_time:.4f}: L2={{l2:.4e}}, MSE={{mse:.4e}}, Max={{maxe:.4e}}")
print(f"  t={snap_time:.3f} — L2={{l2:.4e}}, MSE={{mse:.4e}}, Max={{maxe:.4e}}")
vmin = min(u_pred.min(), u_ref.min()); vmax = max(u_pred.max(), u_ref.max())
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
im0 = axes[0].contourf(Xg, Yg, u_pred, levels=40, cmap='RdBu_r', vmin=vmin, vmax=vmax)
axes[0].set_title(f"PINN at t={snap_time}"); axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
fig.colorbar(im0, ax=axes[0])
im1 = axes[1].contourf(Xg, Yg, u_ref, levels=40, cmap='RdBu_r', vmin=vmin, vmax=vmax)
axes[1].set_title(f"Reference at t={snap_time}"); axes[1].set_xlabel("x")
fig.colorbar(im1, ax=axes[1])
im2 = axes[2].contourf(Xg, Yg, abs_err, levels=40, cmap='YlOrRd')
axes[2].set_title(f"|Error| L2={{l2:.2e}}"); axes[2].set_xlabel("x")
fig.colorbar(im2, ax=axes[2])
fig.suptitle("PINN vs Reference — Error Analysis", fontsize=13, fontweight='bold')
plt.tight_layout()
"""
        script += f"""
out_path = os.path.join(r"{save_dir}", "comparison_plot.png")
plt.savefig(out_path, dpi=100); plt.close()
print(f"Comparison plot saved: {{out_path}}")
metrics_path = os.path.join(r"{save_dir}", "error_metrics.txt")
with open(metrics_path, "w") as f:
    f.write("Error Analysis Results\\n" + "="*40 + "\\n")
    for line in metrics_lines:
        f.write(line + "\\n")
print(f"Metrics saved: {{metrics_path}}")
print("ERROR_ANALYSIS_DONE")
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
            tf.write(script); tmp = tf.name

        from PyQt6.QtCore import QThread, pyqtSignal as _sig
        class _EAThread(QThread):
            line_sig = _sig(str)
            done_sig = _sig(bool)
            def __init__(self, tmp):
                super().__init__(); self._tmp = tmp
            def run(self):
                import subprocess, sys
                proc = subprocess.Popen([sys.executable, self._tmp],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    self.line_sig.emit(line.rstrip())
                proc.wait()
                os.unlink(self._tmp)
                self.done_sig.emit(proc.returncode == 0)

        self._ea_thread = _EAThread(tmp)
        self._ea_thread.line_sig.connect(self.log_box.append)
        self._ea_thread.done_sig.connect(self._on_ea_done)
        self._ea_thread.start()

    def _on_ea_done(self, success):
        save_dir = self.save_dir_input.text().strip()
        if success:
            self.log_box.append("✅ Error analysis complete!")
            plot_path = os.path.join(save_dir, "comparison_plot.png")
            if os.path.exists(plot_path):
                self.solution_label.setPixmap(QPixmap(plot_path).scaled(
                    500, 420, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
        else:
            self.log_box.append("❌ Error analysis failed — check log.")
    
    def _on_restore_viz_changed(self, text):
        self._on_restore_viz_settings(text)

    def _on_restore_viz_settings(self, viz_type=None):
        if viz_type is None:
            viz_type = self.restore_viz_combo.currentText()

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Settings — {viz_type}")
        dialog.setMinimumWidth(320)
        layout = QVBoxLayout(dialog)

        current = self._restore_viz_settings

        # Colormap
        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colormap:"))
        cmap_combo = QComboBox()
        cmap_combo.addItems(["RdBu_r", "viridis", "plasma", "jet", "coolwarm", "inferno", "turbo", "seismic", "bwr"])
        cmap_combo.setCurrentText(current.get('colormap', 'RdBu_r'))
        cmap_combo.setFixedWidth(120)
        cmap_row.addStretch(); cmap_row.addWidget(cmap_combo)
        layout.addLayout(cmap_row)

        # Contour levels
        levels_row = QHBoxLayout()
        levels_row.addWidget(QLabel("Contour levels:"))
        levels_spin = QSpinBox()
        levels_spin.setRange(5, 200); levels_spin.setValue(current.get('levels', 40))
        levels_spin.setFixedWidth(80)
        levels_row.addStretch(); levels_row.addWidget(levels_spin)
        layout.addLayout(levels_row)

        # Resolution
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Grid resolution:"))
        res_combo = QComboBox()
        res_combo.addItems(["80", "100", "150", "200"])
        res_combo.setCurrentText(str(current.get('resolution', 100)))
        res_combo.setFixedWidth(80)
        res_row.addStretch(); res_row.addWidget(res_combo)
        layout.addLayout(res_row)

        # DPI
        dpi_row = QHBoxLayout()
        dpi_row.addWidget(QLabel("Output DPI:"))
        dpi_combo = QComboBox()
        dpi_combo.addItems(["100", "150", "200", "300"])
        dpi_combo.setCurrentText(str(current.get('dpi', 100)))
        dpi_combo.setFixedWidth(80)
        dpi_row.addStretch(); dpi_row.addWidget(dpi_combo)
        layout.addLayout(dpi_row)

        # Surface time — only for Surface
        surface_time_widget = QWidget()
        st_layout = QHBoxLayout(surface_time_widget)
        st_layout.setContentsMargins(0, 0, 0, 0)
        st_layout.addWidget(QLabel("Plot at time t ="))
        surface_time_spin = QDoubleSpinBox()
        surface_time_spin.setRange(0.0, 1e6)
        surface_time_spin.setValue(current.get('surface_time', self.t_max.value()))
        surface_time_spin.setFixedWidth(100)
        st_layout.addStretch(); st_layout.addWidget(surface_time_spin)
        surface_time_widget.setVisible(viz_type == "Surface")
        layout.addWidget(surface_time_widget)

        # Color range — for Surface and Animation Surface
        color_range_widget = QWidget()
        cr_layout = QVBoxLayout(color_range_widget)
        cr_layout.setContentsMargins(0, 0, 0, 0)
        cr_auto_cb = QCheckBox("Auto color range")
        cr_auto_cb.setChecked(current.get('auto_range', True))
        cr_layout.addWidget(cr_auto_cb)
        cr_manual_widget = QWidget()
        cr_manual_layout = QHBoxLayout(cr_manual_widget)
        cr_manual_layout.setContentsMargins(0, 0, 0, 0)
        cr_manual_layout.addWidget(QLabel("vmin:"))
        vmin_spin = QDoubleSpinBox(); vmin_spin.setRange(-1e6, 1e6); vmin_spin.setValue(current.get('vmin', -1.0)); vmin_spin.setFixedWidth(80)
        cr_manual_layout.addWidget(vmin_spin)
        cr_manual_layout.addWidget(QLabel("vmax:"))
        vmax_spin = QDoubleSpinBox(); vmax_spin.setRange(-1e6, 1e6); vmax_spin.setValue(current.get('vmax', 1.0)); vmax_spin.setFixedWidth(80)
        cr_manual_layout.addWidget(vmax_spin)
        cr_manual_widget.setVisible(not cr_auto_cb.isChecked())
        cr_layout.addWidget(cr_manual_widget)
        cr_auto_cb.stateChanged.connect(lambda s: cr_manual_widget.setVisible(s != 2))
        color_range_widget.setVisible("Surface" in viz_type)
        layout.addWidget(color_range_widget)

        # Steps/frames
        steps_widget = QWidget()
        steps_layout = QHBoxLayout(steps_widget)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        label_text = "Time steps:" if "Line" in viz_type else "Animation frames:"
        steps_layout.addWidget(QLabel(label_text))
        steps_spin = QSpinBox()
        steps_spin.setRange(2, 100); steps_spin.setValue(current.get('n_steps', 10))
        steps_spin.setFixedWidth(80)
        steps_layout.addStretch(); steps_layout.addWidget(steps_spin)
        steps_widget.setVisible(viz_type != "Surface")
        layout.addWidget(steps_widget)

        # Line width — only for Line plots
        lw_widget = QWidget()
        lw_layout = QHBoxLayout(lw_widget)
        lw_layout.setContentsMargins(0, 0, 0, 0)
        lw_layout.addWidget(QLabel("Line width:"))
        lw_combo = QComboBox()
        lw_combo.addItems(["1.0", "1.5", "2.0", "2.5", "3.0"])
        lw_combo.setCurrentText(str(current.get('linewidth', 2.0)))
        lw_combo.setFixedWidth(80)
        lw_layout.addStretch(); lw_layout.addWidget(lw_combo)
        lw_widget.setVisible("Line" in viz_type)
        layout.addWidget(lw_widget)

        # FPS — only for animations
        fps_widget = QWidget()
        fps_layout = QHBoxLayout(fps_widget)
        fps_layout.setContentsMargins(0, 0, 0, 0)
        fps_layout.addWidget(QLabel("Animation FPS:"))
        fps_combo = QComboBox()
        fps_combo.addItems(["5", "8", "10", "15", "20"])
        fps_combo.setCurrentText(str(current.get('fps', 10)))
        fps_combo.setFixedWidth(80)
        fps_layout.addStretch(); fps_layout.addWidget(fps_combo)
        fps_widget.setVisible("Animation" in viz_type)
        layout.addWidget(fps_widget)

        # Colorbar
        colorbar_cb = QCheckBox("Show colorbar")
        colorbar_cb.setChecked(current.get('colorbar', True))
        colorbar_cb.setVisible("Surface" in viz_type)
        layout.addWidget(colorbar_cb)

        info_texts = {
            "Surface": "Single heatmap/contour at specified time.",
            "Line (time steps)": "Solution lines at evenly spaced time steps.",
            "Animation Line (GIF)": "Animated GIF of line plots over time.",
            "Animation Surface (GIF)": "Animated GIF of surface plots with colorbar.",
        }
        info = QLabel(info_texts.get(viz_type, ""))
        info.setStyleSheet("color: #586e75; font-size: 11px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK"); cancel_btn = QPushButton("Cancel")
        btn_row.addStretch(); btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        cancel_btn.clicked.connect(dialog.reject)

        def _on_ok():
            self._restore_viz_settings = {
                'colormap': cmap_combo.currentText(),
                'surface_time': surface_time_spin.value(),
                'n_steps': steps_spin.value(),
                'colorbar': colorbar_cb.isChecked(),
                'levels': levels_spin.value(),
                'resolution': int(res_combo.currentText()),
                'dpi': int(dpi_combo.currentText()),
                'auto_range': cr_auto_cb.isChecked(),
                'vmin': vmin_spin.value(),
                'vmax': vmax_spin.value(),
                'linewidth': float(lw_combo.currentText()),
                'fps': int(fps_combo.currentText()),
            }
            self.restore_tsteps_spin.setValue(steps_spin.value())
            self.log_box.append(f"✅ Viz settings saved — {viz_type}, cmap={cmap_combo.currentText()}, levels={levels_spin.value()}")
            dialog.accept()
        ok_btn.clicked.connect(_on_ok)
        dialog.exec()
    
    def _on_template_selected(self, text):
        if text == "📋 Examples":
            return

        templates_2d = {
            "2D Heat (Dirichlet/Neumann)": {
                'pde': ["du_t - 0.4*(du_xx + du_yy)"],
                'ic': ["0.0"],
                'num_domain': 5000,
                'num_boundary': 400,
                'num_initial': 400,
                'layers': 4,
                'neurons': 64,
                'iterations': 10000,
                'optimizer2': 'lbfgs',
                'iterations2': 10000,
                'x_min': 0.0, 'x_max': 1.0,
                'y_min': 0.0, 'y_max': 1.0,
                'periodic_bc': False,
                'bc_config': 'heat2d',
                'num_outputs': 1,
                'output_names': ['u'],
                'ic_weight': 100.0,
                'ref_dir': '/home/asfandyarkhan/deepxde_gui/FEM_Results/2D_Examples/HeatEquation_2D',
            },
            "2D Allen-Cahn (Mattey)": {
                'pde': ["du_t - 0.0001*(du_xx + du_yy) + 5*(u**3 - u)"],
                'ic': ["sin(4*pi*x)*cos(4*pi*y)"],
                'num_domain': 10000,
                'num_boundary': 400,
                'num_initial': 512,
                'layers': 4,
                'neurons': 128,
                'iterations': 10000,
                'optimizer2': 'lbfgs',
                'iterations2': 10000,
                'x_min': 0.0, 'x_max': 1.0,
                'y_min': 0.0, 'y_max': 1.0,
                'periodic_bc': True,
                'bc_config': 'periodic_all',
                'num_outputs': 1,
                'output_names': ['u'],
                'ic_weight': 100.0,
                'ref_dir': '/home/asfandyarkhan/deepxde_gui/FEM_Results/2D_Examples/AllenChan_2D_Mattey',
            },
            "2D Allen-Cahn (Wight)": {
                'pde': ["du_t - 0.00625*(du_xx + du_yy) + 10*(u**3 - u)"],
                'ic': ["tanh((0.35 - sqrt((x-0.5)**2 + (y-0.5)**2)) / (2*0.025))"],
                'num_domain': 10000,
                'num_boundary': 400,
                'num_initial': 512,
                'layers': 4,
                'neurons': 128,
                'iterations': 10000,
                'optimizer2': 'lbfgs',
                'iterations2': 10000,
                'x_min': 0.0, 'x_max': 1.0,
                'y_min': 0.0, 'y_max': 1.0,
                't_max': 10.0,
                'periodic_bc': True,
                'bc_config': 'periodic_all',
                'num_outputs': 1,
                'output_names': ['u'],
                'ic_weight': 100.0,
                'ref_dir': '/home/asfandyarkhan/deepxde_gui/FEM_Results/2D_Examples/AllenChan_2D_Wight',
            },
            "2D Cahn-Hilliard (Wight)": {
                'pde': ["du_t - (dmu_xx + dmu_yy)",
                        "mu - (u**3 - u) + 0.05*2*(du_xx + du_yy)"],
                'ic': ["max(tanh((0.4-sqrt((x-0.7*0.4)**2+(y)**2))/(2*0.05)), tanh((0.4-sqrt((x+0.7*0.4)**2+(y)**2))/(2*0.05)))",
                       "0.0"],
                'num_domain': 10000,
                'num_boundary': 400,
                'num_initial': 512,
                'layers': 4,
                'neurons': 128,
                'iterations': 10000,
                'optimizer2': 'lbfgs',
                'iterations2': 10000,
                'x_min': -0.5, 'x_max': 0.5,
                'y_min': -0.5, 'y_max': 0.5,
                'periodic_bc': True,
                'bc_config': 'ch2d',
                'num_outputs': 2,
                'output_names': ['u', 'mu'],
                'ic_weight': 100.0,
                'ref_dir': '/home/asfandyarkhan/deepxde_gui/FEM_Results/2D_Examples/CahnHilliard_2D_Wight',
            },
            "FeCr PINN": {
                'pde': ["dc_t - (86400/10)*(M*(dmu_xx + dmu_yy) + dMdc*(dc_x*dmu_x + dc_y*dmu_y))",
                        "mu - dfdc + (8.125e-16 * (1.0/(25e-9)**2))*(dc_xx + dc_yy)"],
                'ic': ["", ""],
                'num_outputs': 2,
                'output_names': ['c', 'mu'],
                'num_domain': 10000,
                'num_boundary': 400,
                'num_initial': 0,
                'num_test': 10000,
                'layers': 6,
                'neurons': 128,
                'iterations': 50000,
                'optimizer2': 'lbfgs',
                'iterations2': 50000,
                'x_min': 0.0, 'x_max': 1.0,
                'y_min': 0.0, 'y_max': 1.0,
                't_max': 1.0,
                'bc_config': 'fecr',
                'template_type': 'FeCr_PINN',
                'forward_ic_from_file': True,
                'forward_ic_file': '/home/asfandyarkhan/deepxde_gui/FEM_Results/2D_Examples/FeCr_PINN_2D/t_0.txt',
                'ref_dir': '/home/asfandyarkhan/deepxde_gui/FEM_Results/2D_Examples/FeCr_PINN_2D',
            },
        }
        if text in templates_2d:
            t = templates_2d[text]
            # Set number of outputs first
            n_out = t.get('num_outputs', 1)
            if n_out != self.num_outputs_spin.value():
                self.num_outputs_spin.setValue(n_out)
            # Set output names
            for i, name in enumerate(t.get('output_names', ['u'])):
                if i < len(self.output_name_inputs):
                    self.output_name_inputs[i].setText(name)
            # Set PDEs
            for i, pde_text in enumerate(t['pde']):
                if i < len(self.pde_inputs):
                    self.pde_inputs[i].setText(pde_text)
            # Set ICs
            for i, ic_text in enumerate(t['ic']):
                if i < len(self.ic_inputs):
                    self.ic_inputs[i].setText(ic_text)
            # Set domain
            self.x_min.setValue(t['x_min']); self.x_max.setValue(t['x_max'])
            self.y_min.setValue(t['y_min']); self.y_max.setValue(t['y_max'])
            if 't_max' in t: self.t_max.setValue(t['t_max'])
            # Set collocation points
            self.num_domain.setValue(t['num_domain'])
            self.num_boundary.setValue(t['num_boundary'])
            self.num_initial.setValue(t['num_initial'])
            if 'num_test' in t:
                self.num_test.setValue(t['num_test'])
            # Set network
            self.layers_spin.setValue(t['layers'])
            self.neurons_spin.setValue(t['neurons'])
            # Set training
            self.iter1_spin.setValue(t['iterations'])
            self.opt2_combo.setCurrentText(t['optimizer2'])
            self.iter2_spin.setValue(t['iterations2'])
            # Set IC weight
            for i in range(n_out):
                key = f"ic_{i}"
                if key in self.weight_widgets:
                    self.weight_widgets[key].setValue(t.get('ic_weight', 100.0))
            # Set BCs
            bc_config = t.get('bc_config', 'periodic_all')
            if bc_config == 'periodic_all':
                # All boundaries periodic for all outputs
                for i in range(n_out):
                    if i < len(self.bc_left_types):
                        self.bc_left_types[i].setCurrentText("Periodic")
                    if i < len(self.bc_bottom_types):
                        self.bc_bottom_types[i].setCurrentText("Periodic")
            elif bc_config == 'ch2d':
                # u: periodic on all, mu: no BCs
                if len(self.bc_left_types) > 0:
                    self.bc_left_types[0].setCurrentText("Periodic")
                if len(self.bc_bottom_types) > 0:
                    self.bc_bottom_types[0].setCurrentText("Periodic")
                # Deactivate all BCs for mu (output 1)
                if len(self.bc_left_active) > 1:
                    self.bc_left_active[1].setChecked(False)
                if len(self.bc_right_active) > 1:
                    self.bc_right_active[1].setChecked(False)
                if len(self.bc_bottom_active) > 1:
                    self.bc_bottom_active[1].setChecked(False)
                if len(self.bc_top_active) > 1:
                    self.bc_top_active[1].setChecked(False)
                if len(self.ic_active) > 1:
                    self.ic_active[1].setChecked(False)
            elif bc_config == 'fecr':
                # c (index 0): periodic on all sides
                if len(self.bc_left_types) > 0:
                    self.bc_left_types[0].setCurrentText("Periodic")
                if len(self.bc_bottom_types) > 0:
                    self.bc_bottom_types[0].setCurrentText("Periodic")
                # mu (index 1): no BCs, no IC
                if len(self.bc_left_active) > 1:
                    self.bc_left_active[1].setChecked(False)
                if len(self.bc_right_active) > 1:
                    self.bc_right_active[1].setChecked(False)
                if len(self.bc_bottom_active) > 1:
                    self.bc_bottom_active[1].setChecked(False)
                if len(self.bc_top_active) > 1:
                    self.bc_top_active[1].setChecked(False)
                if len(self.ic_active) > 0:
                    self.ic_active[0].setChecked(False)  # IC from file, not expression
                if len(self.ic_active) > 1:
                    self.ic_active[1].setChecked(False)
                # Set IC from file for c (index 0)
                if (hasattr(self, 'ic_from_file') and len(self.ic_from_file) > 0
                        and self.ic_from_file[0] is not None):
                    self.ic_from_file[0].setChecked(True)
                if (hasattr(self, 'ic_file_paths') and len(self.ic_file_paths) > 0
                        and self.ic_file_paths[0] is not None):
                    self.ic_file_paths[0].setText(
                        '/home/asfandyarkhan/deepxde_gui/FEM_Results/2D_Examples/FeCr_PINN_2D/t_0.txt')
                # Set FeCr default weights
                self._build_weight_inputs(self.num_outputs_spin.value())
                if "pde_0" in self.weight_widgets:
                    self.weight_widgets["pde_0"].setValue(100.0)
                if "pde_1" in self.weight_widgets:
                    self.weight_widgets["pde_1"].setValue(1e-4)
                if "bc_left_0" in self.weight_widgets:
                    self.weight_widgets["bc_left_0"].setValue(1.0)
                if "bc_bottom_0" in self.weight_widgets:
                    self.weight_widgets["bc_bottom_0"].setValue(1.0)
                if "ic_0" in self.weight_widgets:
                    self.weight_widgets["ic_0"].setValue(1000.0)
            elif bc_config == 'heat2d':
                # Right: Dirichlet=1, Left/Bottom/Top: Neumann=0
                for i in range(n_out):
                    if i < len(self.bc_left_types):
                        self.bc_left_types[i].setCurrentText("Neumann")
                        self.bc_left_vals[i].setValue(0.0)
                    if i < len(self.bc_right_types):
                        self.bc_right_types[i].setCurrentText("Dirichlet")
                        self.bc_right_vals[i].setValue(1.0)
                    if i < len(self.bc_bottom_types):
                        self.bc_bottom_types[i].setCurrentText("Neumann")
                        self.bc_bottom_vals[i].setValue(0.0)
                    if i < len(self.bc_top_types):
                        self.bc_top_types[i].setCurrentText("Neumann")
                        self.bc_top_vals[i].setValue(0.0)
            self._template_ref_dir = t.get('ref_dir', '')
            self._current_template = text
            self._current_template_type = t.get('template_type', '')
            self._auto_configure_ea(self._template_ref_dir)
            self.log_box.append(f"✅ Template loaded: {text}")
            return

        templates = {
            "1D Heat": {
                'pde': ["du_t - 0.4 * du_xx"],
                'ic': ["sin(pi*x)"],
                'num_domain': 2000,
                'num_boundary': 200,
                'num_initial': 200,
                'layers': 3,
                'neurons': 64,
                'iterations': 10000,
                'optimizer2': 'none',
                'iterations2': 5000,
                'x_min': 0.0, 'x_max': 1.0,
                'periodic_bc': False,
                'ref_dir': '/home/asfandyarkhan/deepxde_gui/FEM_Results/1D_Examples/1D_HeatEquation',
            },
            "1D Allen-Cahn": {
                'pde': ["du_t - 0.0001*du_xx + 5*u**3 - 5*u"],
                'ic': ["x**2*cos(pi*x)"],
                'num_domain': 10000,
                'num_boundary': 200,
                'num_initial': 512,
                'layers': 4,
                'neurons': 128,
                'iterations': 20000,
                'optimizer2': 'lbfgs',
                'iterations2': 20000,
                'x_min': -1.0, 'x_max': 1.0,
                'periodic_bc': True,
                'ref_dir': '/home/asfandyarkhan/deepxde_gui/FEM_Results/1D_Examples/1D_AllenCahn/ut−0.0001uxx+5u -5u^3=0_(example_1)',
            },
            "1D Cahn-Hilliard": {
                'pde': ["du_t - dv_xx", "v - 0.01*(u**3 - u) + 1e-6*du_xx"],
                'ic': ["-cos(2*pi*x)", ""],
                'num_outputs': 2,
                'output_names': "u, v",
                'num_domain': 10000,
                'num_boundary': 200,
                'num_initial': 512,
                'layers': 4,
                'neurons': 128,
                'iterations': 20000,
                'optimizer2': 'lbfgs',
                'iterations2': 20000,
                'x_min': -1.0, 'x_max': 1.0,
                'periodic_bc_u_only': True,
                'ref_dir': '/home/asfandyarkhan/deepxde_gui/FEM_Results/1D_Examples/1D_CahnHilliard/ut−(0.01(u^3-u)-1e-6uxx)xx=0_example_3',
            },
        }

        t = templates.get(text)
        if not t:
            return
        
        # Set num outputs and names if specified
        if 'num_outputs' in t:
            self.num_outputs_spin.setValue(t['num_outputs'])
            QApplication.processEvents()
        if 'output_names' in t:
            names = [n.strip() for n in t['output_names'].split(',')]
            for i, name in enumerate(names):
                if i < len(self.output_name_inputs):
                    self.output_name_inputs[i].setText(name)

        # Set PDE
        for i, pde_text in enumerate(t['pde']):
            if i < len(self.pde_inputs):
                self.pde_inputs[i].setText(pde_text)

        # Set IC
        for i, ic_text in enumerate(t['ic']):
            if i < len(self.ic_inputs):
                self.ic_inputs[i].setText(ic_text)

        # Set collocation points
        self.num_domain.setValue(t['num_domain'])
        self.num_boundary.setValue(t['num_boundary'])
        self.num_initial.setValue(t['num_initial'])

        # Set network
        self.layers_spin.setValue(t['layers'])
        self.neurons_spin.setValue(t['neurons'])

        # Set training
        self.iter1_spin.setValue(t['iterations'])
        self.opt2_combo.setCurrentText(t['optimizer2'])
        self.iter2_spin.setValue(t['iterations2'])

        # Set IC weight to 100 for Allen-Cahn and Cahn-Hilliard
        if text in ["1D Allen-Cahn"]:
            for i in range(self.num_outputs_spin.value()):
                key = f"ic_{i}"
                if key in self.weight_widgets:
                    self.weight_widgets[key].setValue(100.0)
        elif text == "1D Cahn-Hilliard":
            self._build_weight_inputs(self.num_outputs_spin.value())
            if "ic_0" in self.weight_widgets:
                self.weight_widgets["ic_0"].setValue(100.0)

        # Set domain x range
        if 'x_min' in t:
            self.x_min.setValue(t['x_min'])
            self.x_max.setValue(t['x_max'])

        # Set periodic BC
        if t.get('periodic_bc', False):
            for i in range(self.num_outputs_spin.value()):
                if i < len(self.bc_left_types):
                    self.bc_left_types[i].setCurrentText("Periodic")
        elif t.get('periodic_bc_u_only', False):
            # Only u (index 0) gets Periodic, v (index 1) gets no BC
            if len(self.bc_left_types) > 0:
                self.bc_left_types[0].setCurrentText("Periodic")
            if len(self.bc_left_types) > 1:
                self.bc_left_types[1].setCurrentText("None")
            if len(self.bc_right_types) > 1:
                self.bc_right_types[1].setCurrentText("None")
        else:
            for i in range(self.num_outputs_spin.value()):
                if i < len(self.bc_left_types):
                    self.bc_left_types[i].setCurrentText("Dirichlet")
                if i < len(self.bc_right_types):
                    self.bc_right_types[i].setCurrentText("Dirichlet")

        # Store ref_dir for error analysis auto-population
        self._template_ref_dir = t.get('ref_dir', '')
        self._current_template = text
        self._current_template_type = t.get('template_type', '')
        self._auto_configure_ea(self._template_ref_dir)
        self.log_box.append(f"✅ Template loaded: {text}")
    def _on_plot_settings(self):
        viz_type = self.plot_type_combo.currentText()
        if viz_type == "📊 Error Analysis":
            self._on_error_analysis_settings()
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Plot Settings — {viz_type}")
        dialog.setMinimumWidth(320)
        layout = QVBoxLayout(dialog)

        current = self._plot_viz_settings

        # Colormap
        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colormap:"))
        cmap_combo = QComboBox()
        cmap_combo.addItems(["RdBu_r", "viridis", "plasma", "jet", "coolwarm", "inferno", "turbo", "seismic", "bwr"])
        cmap_combo.setCurrentText(current.get('colormap', 'RdBu_r'))
        cmap_combo.setFixedWidth(120)
        cmap_row.addStretch(); cmap_row.addWidget(cmap_combo)
        layout.addLayout(cmap_row)

        # Contour levels
        levels_row = QHBoxLayout()
        levels_row.addWidget(QLabel("Contour levels:"))
        levels_spin = QSpinBox()
        levels_spin.setRange(5, 200); levels_spin.setValue(current.get('levels', 50))
        levels_spin.setFixedWidth(80)
        levels_row.addStretch(); levels_row.addWidget(levels_spin)
        layout.addLayout(levels_row)

        # Resolution
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Grid resolution:"))
        res_combo = QComboBox()
        res_combo.addItems(["80", "100", "150", "200"])
        res_combo.setCurrentText(str(current.get('resolution', 100)))
        res_combo.setFixedWidth(80)
        res_row.addStretch(); res_row.addWidget(res_combo)
        layout.addLayout(res_row)

        # DPI
        dpi_row = QHBoxLayout()
        dpi_row.addWidget(QLabel("Output DPI:"))
        dpi_combo = QComboBox()
        dpi_combo.addItems(["100", "150", "200", "300"])
        dpi_combo.setCurrentText(str(current.get('dpi', 100)))
        dpi_combo.setFixedWidth(80)
        dpi_row.addStretch(); dpi_row.addWidget(dpi_combo)
        layout.addLayout(dpi_row)

        # Color range — Surface only
        color_range_widget = QWidget()
        cr_layout = QVBoxLayout(color_range_widget)
        cr_layout.setContentsMargins(0, 0, 0, 0)
        cr_auto_cb = QCheckBox("Auto color range")
        cr_auto_cb.setChecked(current.get('auto_range', True))
        cr_layout.addWidget(cr_auto_cb)
        cr_manual_widget = QWidget()
        cr_manual_layout = QHBoxLayout(cr_manual_widget)
        cr_manual_layout.setContentsMargins(0, 0, 0, 0)
        cr_manual_layout.addWidget(QLabel("vmin:"))
        vmin_spin = QDoubleSpinBox(); vmin_spin.setRange(-1e6, 1e6)
        vmin_spin.setValue(current.get('vmin', -1.0)); vmin_spin.setFixedWidth(80)
        cr_manual_layout.addWidget(vmin_spin)
        cr_manual_layout.addWidget(QLabel("vmax:"))
        vmax_spin = QDoubleSpinBox(); vmax_spin.setRange(-1e6, 1e6)
        vmax_spin.setValue(current.get('vmax', 1.0)); vmax_spin.setFixedWidth(80)
        cr_manual_layout.addWidget(vmax_spin)
        cr_manual_widget.setVisible(not cr_auto_cb.isChecked())
        cr_layout.addWidget(cr_manual_widget)
        cr_auto_cb.stateChanged.connect(lambda s: cr_manual_widget.setVisible(s != 2))
        color_range_widget.setVisible(viz_type == "Surface")
        layout.addWidget(color_range_widget)

        # 2D snapshots — Surface only, 2D mode
        snap_widget = QWidget()
        snap_layout = QHBoxLayout(snap_widget)
        snap_layout.setContentsMargins(0, 0, 0, 0)
        snap_layout.addWidget(QLabel("2D time snapshots:"))
        snap_spin = QSpinBox()
        snap_spin.setRange(1, 10); snap_spin.setValue(current.get('n_2d_snapshots', 2))
        snap_spin.setFixedWidth(80)
        snap_layout.addStretch(); snap_layout.addWidget(snap_spin)
        snap_widget.setVisible(viz_type == "Surface" and self.radio_2d.isChecked())
        layout.addWidget(snap_widget)

        # Colorbar — Surface only
        colorbar_cb = QCheckBox("Show colorbar")
        colorbar_cb.setChecked(current.get('colorbar', True))
        colorbar_cb.setVisible(viz_type == "Surface")
        layout.addWidget(colorbar_cb)

        # Line width — Line only
        lw_widget = QWidget()
        lw_layout = QHBoxLayout(lw_widget)
        lw_layout.setContentsMargins(0, 0, 0, 0)
        lw_layout.addWidget(QLabel("Line width:"))
        lw_combo = QComboBox()
        lw_combo.addItems(["1.0", "1.5", "2.0", "2.5", "3.0"])
        lw_combo.setCurrentText(str(current.get('linewidth', 2.0)))
        lw_combo.setFixedWidth(80)
        lw_layout.addStretch(); lw_layout.addWidget(lw_combo)
        lw_widget.setVisible(viz_type == "Line (time steps)")
        layout.addWidget(lw_widget)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK"); cancel_btn = QPushButton("Cancel")
        btn_row.addStretch(); btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        cancel_btn.clicked.connect(dialog.reject)

        def _on_ok():
            self._plot_viz_settings = {
                'colormap': cmap_combo.currentText(),
                'levels': levels_spin.value(),
                'resolution': int(res_combo.currentText()),
                'dpi': int(dpi_combo.currentText()),
                'auto_range': cr_auto_cb.isChecked(),
                'vmin': vmin_spin.value(),
                'vmax': vmax_spin.value(),
                'colorbar': colorbar_cb.isChecked(),
                'linewidth': float(lw_combo.currentText()),
                'n_steps': current.get('n_steps', 4),
                'n_2d_snapshots': snap_spin.value(),
                'surface_time': current.get('surface_time', 1.0),
                'fps': current.get('fps', 10),
            }
            self.log_box.append(f"✅ Plot settings saved — {viz_type}, cmap={cmap_combo.currentText()}, levels={levels_spin.value()}, dpi={dpi_combo.currentText()}")
            dialog.accept()

        ok_btn.clicked.connect(_on_ok)
        dialog.exec()
    
    def _on_error_analysis_btn(self):
        """Standalone error analysis button — opens dialog."""
        self._ea_ref_files = getattr(self, '_ea_ref_files', [])
        self._show_ea_dialog()

    def _show_ea_dialog(self):
        import os, glob

        dialog = QDialog(self)
        dialog.setWindowTitle("📊 Error Analysis")
        dialog.setMinimumWidth(520)
        layout = QVBoxLayout(dialog)

        # ── Plot type selection ────────────────────────────────
        plot_group = QGroupBox("Plot Types")
        plot_layout = QVBoxLayout(plot_group)
        self._ea_line_cb    = QCheckBox("Line comparison (PINN vs FEM at each time step)")
        self._ea_surface_cb = QCheckBox("Surface comparison (PINN | FEM | Error)")
        self._ea_line_cb.setChecked(True)
        self._ea_surface_cb.setChecked(True)
        plot_layout.addWidget(self._ea_line_cb)
        plot_layout.addWidget(self._ea_surface_cb)
        layout.addWidget(plot_group)

        # ── Error metrics ──────────────────────────────────────
        metrics_group = QGroupBox("Error Metrics to Compute")
        metrics_layout = QHBoxLayout(metrics_group)
        self._ea_l2_cb  = QCheckBox("L2 Relative"); self._ea_l2_cb.setChecked(True)
        self._ea_mse_cb = QCheckBox("MSE");          self._ea_mse_cb.setChecked(True)
        self._ea_max_cb = QCheckBox("Max Error");    self._ea_max_cb.setChecked(True)
        for w in [self._ea_l2_cb, self._ea_mse_cb, self._ea_max_cb]:
            metrics_layout.addWidget(w)
        layout.addWidget(metrics_group)

        # ── Reference files ────────────────────────────────────
        files_group = QGroupBox("Reference Files (FEM/Exact)")
        files_layout = QVBoxLayout(files_group)

        hint = QLabel("Format: space-separated, 3 columns: x  t  u  (no header)\nEach file = one time snapshot.")
        hint.setStyleSheet("color: #586e75; font-size: 11px;")
        files_layout.addWidget(hint)

        # File list widget
        self._ea_file_list_widget = QWidget()
        self._ea_file_list_layout = QVBoxLayout(self._ea_file_list_widget)
        self._ea_file_list_layout.setSpacing(4)
        self._ea_file_list_layout.setContentsMargins(0, 0, 0, 0)
        files_layout.addWidget(self._ea_file_list_widget)

        self._ea_file_rows = []  # list of (path_label, t_label, remove_btn)

        def _add_file_row(path='', t_val=None):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            path_edit = QLineEdit()
            path_edit.setPlaceholderText("Browse for reference file...")
            path_edit.setFixedHeight(26)
            if path:
                path_edit.setText(path)
            row_layout.addWidget(path_edit)

            browse_btn = QPushButton("Browse")
            browse_btn.setFixedHeight(26); browse_btn.setFixedWidth(60)
            row_layout.addWidget(browse_btn)

            t_label = QLabel("")
            t_label.setStyleSheet("color: #69db7c; font-size: 11px; min-width: 80px;")
            row_layout.addWidget(t_label)

            remove_btn = QPushButton("✕")
            remove_btn.setFixedHeight(26); remove_btn.setFixedWidth(26)
            remove_btn.setStyleSheet("QPushButton { color: #ff8787; background: transparent; border: none; }")
            row_layout.addWidget(remove_btn)

            self._ea_file_list_layout.addWidget(row_widget)
            row_data = {'widget': row_widget, 'path': path_edit, 't_label': t_label}
            self._ea_file_rows.append(row_data)

            def _on_browse():
                f, _ = QFileDialog.getOpenFileName(dialog, "Select reference file", "", "Text files (*.txt *.csv *.dat)")
                if f:
                    path_edit.setText(f)
                    _detect_time(f, t_label)

            def _detect_time(fpath, lbl):
                try:
                    import numpy as np
                    data = np.loadtxt(fpath)
                    if data.ndim == 1: data = data.reshape(1, -1)
                    # Auto-detect: 2D files have 4 cols (x,y,t,u), 1D have 3 cols (x,t,u)
                    t_col = 2 if data.shape[1] >= 4 else 1
                    t_detected = float(data[0, t_col])
                    lbl.setText(f"✅ t = {t_detected:.4f} ({len(data)} pts)")
                    row_data['t_val'] = t_detected
                except Exception as e:
                    lbl.setText(f"❌ {str(e)[:30]}")

            browse_btn.clicked.connect(_on_browse)

            def _on_remove():
                row_widget.deleteLater()
                if row_data in self._ea_file_rows:
                    self._ea_file_rows.remove(row_data)

            remove_btn.clicked.connect(_on_remove)

            if path and t_val is not None:
                t_label.setText(f"✅ t = {t_val:.4f}")
                row_data['t_val'] = t_val
            elif path:
                _detect_time(path, t_label)

            return row_data

        # Auto-populate from template if available
        ref_dir = getattr(self, '_template_ref_dir', '')
        if ref_dir and os.path.isdir(ref_dir):
            txt_files = sorted(glob.glob(os.path.join(ref_dir, 't_*.txt')))
            if txt_files:
                hint2 = QLabel(f"📂 Auto-loaded from template: {os.path.basename(ref_dir)}")
                hint2.setStyleSheet("color: #a0c4ff; font-size: 11px;")
                files_layout.addWidget(hint2)
                for f in txt_files:
                    _add_file_row(f)
        else:
            _add_file_row()  # start with one empty row

        add_btn = QPushButton("➕ Add another file")
        add_btn.setStyleSheet("QPushButton { color: #69db7c; background: transparent; border: 1px solid #2a6a4a; border-radius: 4px; padding: 2px 8px; }")
        add_btn.clicked.connect(lambda: _add_file_row())
        files_layout.addWidget(add_btn)
        layout.addWidget(files_group)

        # ── Buttons ────────────────────────────────────────────
        btn_row = QHBoxLayout()
        run_btn = QPushButton("▶ Run Error Analysis")
        run_btn.setStyleSheet("QPushButton { background: #1a4a6a; color: #74c0fc; font-weight: bold; border-radius: 4px; padding: 4px 14px; }")
        cancel_btn = QPushButton("Cancel")
        btn_row.addStretch()
        btn_row.addWidget(run_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        cancel_btn.clicked.connect(dialog.reject)

        def _on_run():
            save_dir = self.save_dir_input.text().strip()
            if not save_dir:
                self.log_box.append("❌ Error Analysis: please set a save directory first.")
                dialog.reject(); return

            # Collect valid file rows
            valid_files = []
            for row in self._ea_file_rows:
                p = row['path'].text().strip()
                t = row.get('t_val', None)
                if p and os.path.exists(p) and t is not None:
                    valid_files.append((t, p))

            if not valid_files:
                self.log_box.append("❌ Error Analysis: no valid reference files loaded.")
                return

            valid_files.sort(key=lambda x: x[0])

            self._ea_settings = {
                'files': valid_files,
                'do_line': self._ea_line_cb.isChecked(),
                'do_surface': self._ea_surface_cb.isChecked(),
                'do_l2': self._ea_l2_cb.isChecked(),
                'do_mse': self._ea_mse_cb.isChecked(),
                'do_max': self._ea_max_cb.isChecked(),
            }
            self.log_box.append(f"✅ Error Analysis configured — {len(valid_files)} reference files, will run after training.")
            dialog.accept()

        run_btn.clicked.connect(_on_run)
        dialog.exec()

    def _execute_error_analysis_v2(self, ea, config):
        import tempfile, glob, json
        save_dir = self.save_dir_input.text().strip()
        is_2d = config.problem_dim == "2D"

        # Find most recently modified model
        model_path = ""
        all_models = glob.glob(os.path.join(save_dir, "model_lbfgs-*.pt")) + \
                     glob.glob(os.path.join(save_dir, "model_adam-*.pt"))
        if all_models:
            model_path = max(all_models, key=os.path.getmtime)
            self.log_box.append(f"📂 Using model: {os.path.basename(model_path)}")

        if not model_path:
            self.log_box.append("❌ No saved model found."); return

        config_path = model_path.replace(".pt", ".json")
        if not os.path.exists(config_path):
            config_path = os.path.join(save_dir, "model_config.json")

        try:
            with open(config_path) as f:
                cfg = json.load(f)
        except Exception as e:
            self.log_box.append(f"❌ Could not read model config: {e}"); return

        layers     = cfg["layers"]
        activation = cfg["activation"]
        x_min = cfg["x_min"]; x_max = cfg["x_max"]
        t_min = cfg["t_min"]; t_max = cfg["t_max"]
        loss_type  = cfg.get("loss_type", "MSE")
        compile_opt = "lbfgs" if "lbfgs" in model_path else "adam"

        files_list = ea['files']   # list of (t_val, filepath)
        do_line    = ea['do_line']
        do_surface = ea['do_surface']
        do_l2      = ea['do_l2']
        do_mse     = ea['do_mse']
        do_max     = ea['do_max']

        # Serialise file list for script
        files_repr = repr(files_list)

        script = f"""
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ── Restore model ─────────────────────────────────────────────
if {is_2d}:
    geom = dde.geometry.Rectangle([{x_min}, {cfg.get('y_min', 0.0)}], [{x_max}, {cfg.get('y_max', 1.0)}])
else:
    geom  = dde.geometry.Interval({x_min}, {x_max})
td    = dde.geometry.TimeDomain({t_min}, {t_max})
gt    = dde.geometry.GeometryXTime(geom, td)
def pde(x, y): return y[:, 0:1] * 0
data  = dde.data.TimePDE(gt, pde, [], num_domain=100, num_test=100)
net   = dde.nn.FNN({layers}, "{activation}", "Glorot uniform")
model = dde.Model(data, net)
if "{compile_opt}" == "lbfgs":
    dde.optimizers.set_LBFGS_options(maxiter=1)
    model.compile("L-BFGS", loss="{loss_type}")
else:
    model.compile("adam", lr=0.001, loss="{loss_type}")

model.restore(r"{model_path}", verbose=0)
print("✅ Model restored for error analysis.")
print(f"   Model path: {model_path}")
print(f"   x range: [{x_min}, {x_max}]")
print(f"   t range: [{t_min}, {t_max}]")

# ── Output directory ──────────────────────────────────────────
_ea_dir = os.path.join(r"{save_dir}", "error_analysis")
os.makedirs(_ea_dir, exist_ok=True)

# ── Load reference files ──────────────────────────────────────
_files = {files_repr}
_times = []; _x_refs = []; _u_refs = []; _d_refs = []; _d_shape = 0
for _tv, _fp in _files:
    _d = np.loadtxt(_fp)
    if _d.ndim == 1: _d = _d.reshape(1, -1)
    _d_shape = _d.shape[1]
    _idx = np.argsort(_d[:, 0])
    _x_refs.append(_d[_idx, 0])
    _u_col = 3 if _d_shape >= 4 else 2
    _u_refs.append(_d[_idx, _u_col])
    _d_refs.append(_d[_idx])
    _times.append(float(_tv))
    print(f"  Loaded t={{_tv:.4f}}: {{len(_d)}} points from {{os.path.basename(_fp)}}")

_n_t = len(_times)

# ── Predict PINN at EXACT FEM x values for each time ─────────
_u_pinns = []
for _i, _tv in enumerate(_times):
    _x_fem = _x_refs[_i]
    if _d_shape >= 4:  # 2D file: has x,y,t,c columns
        _y_fem = _d_refs[_i][:, 1]
        _xt = np.column_stack([_x_fem, _y_fem, np.full_like(_x_fem, _tv)])
    else:
        _xt = np.column_stack([_x_fem, np.full_like(_x_fem, _tv)])
    _u_pinns.append(model.predict(_xt)[:, 0].flatten())
    print(f"  PINN predicted at t={{_tv:.4f}}: {{len(_x_fem)}} points")

# ── Error metrics at exact FEM points ────────────────────────
_metrics = []
for _i, _tv in enumerate(_times):
    _up = _u_pinns[_i]; _uf = _u_refs[_i]
    _abs_err = np.abs(_up - _uf)
    _l2      = np.linalg.norm(_up - _uf) / (np.linalg.norm(_uf) + 1e-10)
    _mse     = np.mean((_up - _uf)**2)
    _mx      = np.max(_abs_err)
    _ma      = np.mean(_abs_err)
    _metrics.append((_tv, _l2, _mse, _mx, _ma))
    print(f"  t={{_tv:.4f}} — L2={{_l2:.4e}}, MSE={{_mse:.4e}}, Max={{_mx:.4e}}, MeanAbs={{_ma:.4e}}")

# Save metrics
with open(os.path.join(_ea_dir, "error_metrics.txt"), "w") as _mf:
    _mf.write("t,L2_relative,MSE,Max_error,Mean_abs_error\\n")
    for _tv, _l2, _mse, _mx, _ma in _metrics:
        _mf.write(f"{{_tv:.6f}},{{_l2:.6e}},{{_mse:.6e}},{{_mx:.6e}},{{_ma:.6e}}\\n")
print(f"Metrics saved: {{os.path.join(_ea_dir, 'error_metrics.txt')}}")

# ── Line comparison — PINN vs FEM at exact x values ──────────
if {do_line}:
    _ncols = min(4, _n_t)
    _nrows = (_n_t + _ncols - 1) // _ncols
    fig, axes = plt.subplots(_nrows, _ncols, figsize=(4*_ncols, 3.5*_nrows), squeeze=False)
    fig.suptitle("PINN vs FEM — Line Comparison", fontsize=13, fontweight='bold')
    _ax_flat = axes.flatten()
    for _i in range(_n_t):
        ax = _ax_flat[_i]
        _xv = _x_refs[_i]
        _tv, _l2, _mse, _mx, _ma = _metrics[_i]
        ax.plot(_xv, _u_refs[_i],  color='#4dabf7', linewidth=2.0, label='FEM (Exact)')
        ax.plot(_xv, _u_pinns[_i], color='#ff6b6b', linewidth=1.8, linestyle='--', label='PINN')
        ax.set_title(f"t = {{_tv:.3f}}  |  L2 = {{_l2:.2e}}", fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("u(x,t)")
        ax.grid(True, alpha=0.3)
    for _j in range(_n_t, len(_ax_flat)):
        _ax_flat[_j].set_visible(False)
    handles, labels = _ax_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _lp = os.path.join(_ea_dir, "line_comparison.png")
    plt.savefig(_lp, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Line comparison saved: {{_lp}}")

# ── Absolute error lines ──────────────────────────────────────
if {do_line}:
    _ncols = min(4, _n_t)
    _nrows = (_n_t + _ncols - 1) // _ncols
    fig, axes = plt.subplots(_nrows, _ncols, figsize=(4*_ncols, 3.5*_nrows), squeeze=False)
    fig.suptitle("Absolute Error  |PINN - FEM|", fontsize=13, fontweight='bold')
    _ax_flat = axes.flatten()
    for _i in range(_n_t):
        ax = _ax_flat[_i]
        _xv = _x_refs[_i]
        _abs_err = np.abs(_u_pinns[_i] - _u_refs[_i])
        _tv, _l2, _mse, _mx, _ma = _metrics[_i]
        ax.plot(_xv, _abs_err, color='#69db7c', linewidth=2.0)
        ax.fill_between(_xv, _abs_err, alpha=0.25, color='#69db7c')
        ax.set_title(f"t = {{_tv:.3f}}  |  Max = {{_mx:.2e}}", fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("|error|")
        ax.grid(True, alpha=0.3)
    for _j in range(_n_t, len(_ax_flat)):
        _ax_flat[_j].set_visible(False)
    plt.tight_layout()
    _ep = os.path.join(_ea_dir, "absolute_error_lines.png")
    plt.savefig(_ep, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Absolute error saved: {{_ep}}")

# ── Surface comparison — interpolate FEM to common grid ──────
if {do_surface}:
    _x_common = np.linspace({x_min}, {x_max}, 300)
    _t_arr = np.array(_times)
    _U_pinn_surf = np.zeros((len(_t_arr), len(_x_common)))
    _U_fem_surf  = np.zeros((len(_t_arr), len(_x_common)))

    for _i, _tv in enumerate(_times):
        if _d_shape >= 4:
            _y_common = _d_refs[_i][:, 1].mean() * np.ones_like(_x_common)
            _xt_c = np.column_stack([_x_common, _y_common, np.full_like(_x_common, _tv)])
        else:
            _xt_c = np.column_stack([_x_common, np.full_like(_x_common, _tv)])
        _U_pinn_surf[_i, :] = model.predict(_xt_c)[:, 0].flatten()
        _fi = interp1d(_x_refs[_i], _u_refs[_i], kind='linear', fill_value='extrapolate')
        _U_fem_surf[_i, :] = _fi(_x_common)

    _Xg, _Tg = np.meshgrid(_x_common, _t_arr)
    _U_err_surf = np.abs(_U_pinn_surf - _U_fem_surf)
    _vmin = min(_U_pinn_surf.min(), _U_fem_surf.min())
    _vmax = max(_U_pinn_surf.max(), _U_fem_surf.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("PINN vs FEM — Surface Comparison", fontsize=13, fontweight='bold')

    im0 = axes[0].contourf(_Tg, _Xg, _U_pinn_surf, levels=50, cmap='viridis', vmin=_vmin, vmax=_vmax)
    axes[0].set_title("PINN  u(x,t)"); axes[0].set_xlabel("t"); axes[0].set_ylabel("x")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(_Tg, _Xg, _U_fem_surf, levels=50, cmap='viridis', vmin=_vmin, vmax=_vmax)
    axes[1].set_title("FEM  u(x,t)"); axes[1].set_xlabel("t"); axes[1].set_ylabel("x")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].contourf(_Tg, _Xg, _U_err_surf, levels=50, cmap='YlOrRd')
    axes[2].set_title("Error  |PINN - FEM|"); axes[2].set_xlabel("t"); axes[2].set_ylabel("x")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    _sp = os.path.join(_ea_dir, "surface_comparison.png")
    plt.savefig(_sp, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Surface comparison saved: {{_sp}}")

print("ERROR_ANALYSIS_V2_DONE")
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
            tf.write(script); tmp = tf.name

        from PyQt6.QtCore import QThread, pyqtSignal as _sig
        class _EAThread2(QThread):
            line_sig = _sig(str)
            done_sig = _sig(bool)
            def __init__(self, tmp):
                super().__init__(); self._tmp = tmp
            def run(self):
                import subprocess, sys
                proc = subprocess.Popen([sys.executable, self._tmp],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    self.line_sig.emit(line.rstrip())
                proc.wait()
                os.unlink(self._tmp)
                self.done_sig.emit(proc.returncode == 0)

        self._ea_thread2 = _EAThread2(tmp)
        self._ea_thread2.line_sig.connect(self.log_box.append)
        self._ea_thread2.done_sig.connect(self._on_ea_v2_done)
        self._ea_thread2.start()

    def _on_ea_v2_done(self, success):
        save_dir = self.save_dir_input.text().strip()
        if success:
            self.log_box.append("✅ Error analysis complete! Results in error_analysis/ folder.")
            # Show surface comparison if it exists
            for name in ["surface_comparison.png", "line_comparison.png"]:
                p = os.path.join(save_dir, "error_analysis", name)
                if os.path.exists(p):
                    self.solution_label.setPixmap(QPixmap(p).scaled(
                        500, 420, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation))
                    break
        else:
            self.log_box.append("❌ Error analysis failed — check log.")
    
    def _build_scheduler_phases_json(self):
        import json
        if not hasattr(self, 'sched_cb') or not self.sched_cb.isChecked():
            return ""
        phases = []
        same_w = self.sched_same_weights_cb.isChecked()
        for ph in self.sched_phase_list:
            pn = ph['phase_num']
            if same_w:
                # Use phase 1 weights
                w_str = ",".join([
                    str(self.weight_widgets.get(f"pde_{i}", SciLineEdit(1.0)).value())
                    for i in range(self.num_outputs_spin.value())
                ] + [
                    str(self.weight_widgets.get(k, SciLineEdit(1.0)).value())
                    for i in range(self.num_outputs_spin.value())
                    for k in [f"bc_left_{i}", f"bc_bottom_{i}", f"ic_{i}"]
                    if k in self.weight_widgets
                ])
            else:
                w_str = ",".join([
                    str(self.weight_widgets.get(f"pde_{i}_p{pn}", SciLineEdit(1.0)).value())
                    for i in range(self.num_outputs_spin.value())
                ] + [
                    str(self.weight_widgets.get(k, SciLineEdit(1.0)).value())
                    for i in range(self.num_outputs_spin.value())
                    for k in [f"bc_left_{i}_p{pn}", f"bc_bottom_{i}_p{pn}", f"ic_{i}_p{pn}"]
                    if k in self.weight_widgets
                ])
            phases.append({
                'optimizer': ph['opt'].currentText(),
                'iterations': ph['iters'].value(),
                'lr': ph['lr'].value(),
                'weights': w_str
            })
        import json
        return json.dumps(phases)
    
    def _on_scheduler_changed(self, state):
        self.sched_widget.setVisible(state == 2)
        self._build_weight_inputs(self.num_outputs_spin.value())

    def _add_scheduler_phase(self, optimizer='adam', iterations=50000, lr=0.001):
        phase_num = len(self.sched_phase_list) + 2  # phase 2, 3, 4...
        phase_widget = QWidget()
        phase_layout = QVBoxLayout(phase_widget)
        phase_layout.setSpacing(3)
        phase_layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header_row = QHBoxLayout()
        header_lbl = QLabel(f"── Phase {phase_num} ──")
        header_lbl.setStyleSheet("color: #a0c4ff; font-size: 11px;")
        header_row.addWidget(header_lbl)
        remove_btn = QPushButton("✕")
        remove_btn.setFixedHeight(22); remove_btn.setFixedWidth(24)
        remove_btn.setStyleSheet(
            "QPushButton { color: #ff8787; background: transparent; border: none; }")
        header_row.addStretch(); header_row.addWidget(remove_btn)
        phase_layout.addLayout(header_row)

        # Optimizer
        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("Optimizer:"))
        opt_combo = QComboBox()
        opt_combo.addItems(["adam", "lbfgs"])
        opt_combo.setCurrentText(optimizer)
        opt_combo.setFixedHeight(26); opt_combo.setFixedWidth(80)
        opt_row.addStretch(); opt_row.addWidget(opt_combo)
        phase_layout.addLayout(opt_row)

        # Iterations
        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Iterations:"))
        iter_spin = QSpinBox()
        iter_spin.setRange(0, 500000); iter_spin.setSingleStep(1000)
        iter_spin.setValue(iterations); iter_spin.setFixedHeight(26)
        iter_row.addStretch(); iter_row.addWidget(iter_spin)
        phase_layout.addLayout(iter_row)

        # Learning rate
        lr_row = QHBoxLayout()
        lr_row.addWidget(QLabel("Learning rate:"))
        lr_spin = QDoubleSpinBox()
        lr_spin.setRange(1e-6, 1.0); lr_spin.setDecimals(6)
        lr_spin.setSingleStep(0.0001); lr_spin.setValue(lr)
        lr_spin.setFixedHeight(26)
        lr_row.addStretch(); lr_row.addWidget(lr_spin)
        phase_layout.addLayout(lr_row)

        self.sched_phases_layout.addWidget(phase_widget)
        phase_data = {
            'widget': phase_widget,
            'opt': opt_combo,
            'iters': iter_spin,
            'lr': lr_spin,
            'phase_num': phase_num
        }
        self.sched_phase_list.append(phase_data)

        def _remove():
            phase_widget.deleteLater()
            if phase_data in self.sched_phase_list:
                self.sched_phase_list.remove(phase_data)
            self._build_weight_inputs(self.num_outputs_spin.value())

        remove_btn.clicked.connect(_remove)
        self._build_weight_inputs(self.num_outputs_spin.value())
    
    def _on_ic_pretrain_changed(self, state):
        self.ic_pretrain_widget.setVisible(state == 2)

    def _on_batch_changed(self, state):
        self.batch_widget.setVisible(state == 2)

    def _on_browse_restore_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select model file", "", "PyTorch model (*.pt)")
        if f:
            self.restore_model_path.setText(f)
            base = os.path.dirname(f)
            fname = os.path.basename(f).lower()
            # First try: same name as .pt but .json
            specific = f.replace(".pt", ".json")
            if os.path.exists(specific):
                self.restore_config_path.setText(specific)
                self.log_box.append(f"✅ Auto-detected config: {specific}")
            elif os.path.exists(os.path.join(base, "model_config.json")):
                self.restore_config_path.setText(os.path.join(base, "model_config.json"))
                self.log_box.append(f"✅ Auto-detected config (generic): {os.path.join(base, 'model_config.json')}")
            else:
                self.log_box.append("⚠️ No config found — please browse manually.")

    def _on_browse_restore_config(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select config file", "", "JSON (*.json)")
        if f:
            self.restore_config_path.setText(f)

    def _on_browse_restore_save(self):
        folder = QFileDialog.getExistingDirectory(self, "Select save directory")
        if folder:
            self.restore_save_path.setText(folder)

    def _on_restore(self):
        import json, tempfile, subprocess, sys
        model_path  = self.restore_model_path.text().strip()
        config_path = self.restore_config_path.text().strip()
        save_dir    = self.restore_save_path.text().strip()
        optimizer   = self.restore_optimizer_combo.currentText()
        viz_type    = self.restore_viz_combo.currentText()
        output_idx  = self.restore_output_combo.currentIndex()
        t_steps     = self.restore_tsteps_spin.value()

        if not model_path:
            self.log_box.append("❌ Please select a model file."); return
        if not config_path:
            self.log_box.append("❌ Please select a model_config.json file."); return
        if not save_dir:
            self.log_box.append("❌ Please select a save directory."); return

        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
        except Exception as e:
            self.log_box.append(f"❌ Could not read config: {e}"); return

        self.restore_btn.setEnabled(False)
        self.restore_btn.setText("⏳ Restoring...")
        self.log_box.append(f"🔄 Restoring model from: {model_path}")

        script = self._build_restore_script(model_path, cfg, optimizer, viz_type, output_idx, t_steps, save_dir)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
            tf.write(script)
            tmp = tf.name

        from PyQt6.QtCore import QThread, pyqtSignal as _sig

        class _RestoreThread(QThread):
            line_signal = _sig(str)
            done_signal = _sig(bool)
            def __init__(self, tmp):
                super().__init__()
                self._tmp = tmp
            def run(self):
                proc = subprocess.Popen(
                    [sys.executable, self._tmp],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                for line in proc.stdout:
                    line = line.rstrip()
                    if line: self.line_signal.emit(line)
                proc.wait()
                os.unlink(self._tmp)
                self.done_signal.emit(proc.returncode == 0)

        self._restore_thread = _RestoreThread(tmp)
        self._restore_thread.line_signal.connect(self.log_box.append)
        self._restore_thread.done_signal.connect(self._on_restore_done)
        self._restore_thread.start()

    def _on_restore_done(self, success):
        self.restore_btn.setEnabled(True)
        self.restore_btn.setText("🔄  Restore & Visualize")
        save_dir = self.restore_save_path.text().strip()
        if success:
            self.log_box.append("✅ Restore complete!")
            plot_path = os.path.join(save_dir, "restored_plot.png")
            if os.path.exists(plot_path):
                self.solution_label.setPixmap(QPixmap(plot_path).scaled(
                    500, 420, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
            gif_path = os.path.join(save_dir, "restored_animation.gif")
            if os.path.exists(gif_path):
                self.log_box.append(f"🎬 Animation saved: {gif_path}")
        else:
            self.log_box.append("❌ Restore failed — check architecture matches saved model.")

    def _build_restore_script(self, model_path, cfg, optimizer, viz_type, output_idx, t_steps, save_dir):
        viz_settings = getattr(self, '_restore_viz_settings', {})
        colormap = viz_settings.get('colormap', 'RdBu_r')
        surface_time = viz_settings.get('surface_time', cfg.get('t_max', 1.0))
        show_colorbar = viz_settings.get('colorbar', True)
        n_steps = viz_settings.get('n_steps', t_steps)
        levels = viz_settings.get('levels', 40)
        resolution = viz_settings.get('resolution', 100)
        dpi = viz_settings.get('dpi', 100)
        auto_range = viz_settings.get('auto_range', True)
        vmin_val = viz_settings.get('vmin', -1.0)
        vmax_val = viz_settings.get('vmax', 1.0)
        linewidth = viz_settings.get('linewidth', 2.0)
        fps = viz_settings.get('fps', 10)
        layers     = cfg["layers"]
        activation = cfg["activation"]
        x_min = cfg["x_min"]; x_max = cfg["x_max"]
        y_min = cfg.get("y_min", 0.0); y_max = cfg.get("y_max", 1.0)
        t_min = cfg["t_min"]; t_max = cfg["t_max"]
        is_2d = cfg.get("problem_dim", "1D") == "2D"
        loss_type = cfg.get("loss_type", "MSE")
        out_names = cfg.get("output_names", "u").split(",")
        out_name = out_names[output_idx].strip() if output_idx < len(out_names) else "u"

        script = f"""
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Build minimal geometry for model restore
if {str(is_2d)}:
    geom = dde.geometry.Rectangle([{x_min}, {y_min}], [{x_max}, {y_max}])
else:
    geom = dde.geometry.Interval({x_min}, {x_max})
timedomain = dde.geometry.TimeDomain({t_min}, {t_max})
geomtime   = dde.geometry.GeometryXTime(geom, timedomain)

def pde(x, y): return y[:, 0:1] * 0

data = dde.data.TimePDE(geomtime, pde, [], num_domain=100, num_test=100)
net  = dde.nn.FNN({layers}, "{activation}", "Glorot uniform")
model = dde.Model(data, net)

if "{optimizer}" == "lbfgs":
    dde.optimizers.set_LBFGS_options(maxiter=1)
    model.compile("L-BFGS", loss="{loss_type}")
else:
    model.compile("{optimizer}", lr=0.001, loss="{loss_type}")

try:
    model.restore(r"{model_path}", verbose=1)
    print("✅ Model restored successfully.")
except Exception as e:
    print(f"❌ Restore error: {{e}}")
    print("Architecture mismatch — make sure network matches saved model.")
    exit(1)

os.makedirs(r"{save_dir}", exist_ok=True)
x_vals = np.linspace({x_min}, {x_max}, 100)
y_vals = np.linspace({y_min}, {y_max}, 100)
is_2d  = {str(is_2d)}
"""

        if viz_type == "Surface":
            vrange = f"vmin={vmin_val}, vmax={vmax_val}" if not auto_range else ""
            script += f"""
res = {resolution}
x_vals = np.linspace({x_min}, {x_max}, res)
y_vals = np.linspace({y_min}, {y_max}, res)
if is_2d:
    Xg, Yg = np.meshgrid(x_vals, y_vals)
    XYT = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, {surface_time})])
    pred = model.predict(XYT)[:, {output_idx}].reshape(res, res)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.contourf(Xg, Yg, pred, levels={levels}, cmap="{colormap}", {vrange})
    if {show_colorbar}: fig.colorbar(im, ax=ax)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Restored Model — {out_name}(x,y) at t={surface_time}")
else:
    t_vals = np.linspace({t_min}, {t_max}, res)
    X, T = np.meshgrid(x_vals, t_vals)
    XT   = np.vstack([X.ravel(), T.ravel()]).T
    pred = model.predict(XT)[:, {output_idx}].reshape(res, res)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.contourf(X, T, pred, levels={levels}, cmap="{colormap}", {vrange})
    if {show_colorbar}: fig.colorbar(im, ax=ax)
    ax.set_xlabel("x"); ax.set_ylabel("t")
    ax.set_title("Restored Model — {out_name}(x,t) Surface")
plt.tight_layout()
out_path = os.path.join(r"{save_dir}", "restored_plot.png")
plt.savefig(out_path, dpi={dpi}, bbox_inches='tight'); plt.close()
print(f"Surface plot saved to: {{out_path}}")
"""
            
        elif viz_type == "Line (time steps)":
            script += f"""
x_vals = np.linspace({x_min}, {x_max}, {resolution})
t_steps_vals = np.linspace({t_min}, {t_max}, {n_steps})
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.get_cmap("{colormap}")(np.linspace(0, 1, {n_steps}))
y_mid = ({y_min} + {y_max}) / 2.0
for i, tv in enumerate(t_steps_vals):
    if is_2d:
        xt = np.column_stack([x_vals, np.full_like(x_vals, y_mid), np.full_like(x_vals, tv)])
    else:
        xt = np.column_stack([x_vals, np.full_like(x_vals, tv)])
    u_line = model.predict(xt)[:, {output_idx}].flatten()
    ax.plot(x_vals, u_line, color=colors[i], linewidth={linewidth}, label=f"t={{tv:.3f}}")
ax.set_xlabel("x"); ax.set_ylabel("{out_name}")
ax.set_title("Restored Model — {out_name}(x,t) Line Plot")
ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.2)
plt.tight_layout()
out_path = os.path.join(r"{save_dir}", "restored_plot.png")
plt.savefig(out_path, dpi={dpi}, bbox_inches='tight'); plt.close()
print(f"Line plot saved to: {{out_path}}")
"""
        elif viz_type == "Animation Line (GIF)":
            script += f"""
import matplotlib.animation as _anim
t_frames = np.linspace({t_min}, {t_max}, {n_steps})
y_mid = ({y_min} + {y_max}) / 2.0
all_u = []
for tv in t_frames:
    if is_2d:
        xt = np.column_stack([x_vals, np.full_like(x_vals, y_mid), np.full_like(x_vals, tv)])
    else:
        xt = np.column_stack([x_vals, np.full_like(x_vals, tv)])
    all_u.append(model.predict(xt)[:, {output_idx}].flatten())
u_min = min(u.min() for u in all_u) 
u_max = max(u.max() for u in all_u)
fig, ax = plt.subplots(figsize=(7, 4))
ax.set_xlim({x_min}, {x_max})
ax.set_ylim(u_min - 0.05*abs(u_min), u_max + 0.05*abs(u_max))
ax.set_xlabel("x"); ax.set_ylabel("{out_name}")
line, = ax.plot([], [], color="#4dabf7", linewidth=2)
time_txt = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='#ff8787')
ax.grid(True, alpha=0.2)
def init():
    line.set_data([], []); time_txt.set_text(''); return line, time_txt
def update(i):
    line.set_data(x_vals, all_u[i])
    time_txt.set_text(f"t = {{t_frames[i]:.3f}}")
    return line, time_txt
ani = _anim.FuncAnimation(fig, update, init_func=init, frames={n_steps}, interval=100, blit=True)
out_path = os.path.join(r"{save_dir}", "restored_animation.gif")
ani.save(out_path, writer='pillow', fps={fps})
plt.close()
print(f"Animation saved to: {{out_path}}")
"""
        
        elif viz_type == "Animation Surface (GIF)":
            script += f"""
import matplotlib.animation as _anim
t_frames = np.linspace({t_min}, {t_max}, {n_steps})
x_anim = np.linspace({x_min}, {x_max}, 80)
all_frames = []
if is_2d:
    y_anim = np.linspace({y_min}, {y_max}, 80)
    Xg, Yg = np.meshgrid(x_anim, y_anim)
    for tv in t_frames:
        XYT = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, tv)])
        pred = model.predict(XYT)[:, {output_idx}].reshape(80, 80)
        all_frames.append((Xg, Yg, pred))
else:
    t_anim = np.linspace({t_min}, {t_max}, 80)
    X_anim, T_anim = np.meshgrid(x_anim, t_anim)
    for tv in t_frames:
        XT = np.vstack([X_anim.ravel(), np.full(X_anim.size, tv)]).T
        pred = model.predict(XT)[:, {output_idx}].reshape(80, 80)
        all_frames.append((X_anim, T_anim, pred))
v_min = min(f[2].min() for f in all_frames)
v_max = max(f[2].max() for f in all_frames)
fig, ax = plt.subplots(figsize=(8, 6))
from mpl_toolkits.axes_grid1 import make_axes_locatable
_div = make_axes_locatable(ax)
_cax = _div.append_axes("right", size="5%", pad=0.1)
_sm = plt.cm.ScalarMappable(cmap="{colormap}", norm=plt.Normalize(vmin=v_min, vmax=v_max))
if {show_colorbar}: fig.colorbar(_sm, cax=_cax)
def update(i):
    ax.cla()
    Xp, Yp, Zp = all_frames[i]
    c = ax.contourf(Xp, Yp, Zp, levels={levels}, cmap="{colormap}", vmin=v_min, vmax=v_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y" if is_2d else "t")
    ax.set_title(f"t = {{t_frames[i]:.3f}}")
    return c.collections
ani = _anim.FuncAnimation(fig, update, frames={n_steps}, interval=150)
out_path = os.path.join(r"{save_dir}", "restored_animation.gif")
ani.save(out_path, writer='pillow', fps={fps})
ani.save(out_path, writer='pillow', fps={fps})
plt.close()
print(f"Surface animation saved to: {{out_path}}")
"""
        script += '\nprint("RESTORE_DONE")\n'
        return script