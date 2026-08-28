"""
One-time patch script:
  Fix 1 - NameError in codegen.py for 1D problems.
  Fix 2 - missing BC-right/top weight in main_window.py's scheduler
          phase builder.
  Fix 3 - default optimizer schedule for all forward (non-FeCr)
          templates: Adam(10000) then L-BFGS(10000), instead of a
          single L-BFGS(20000) phase.
  Fix 4 - default IC loss weight is 100 for all cases (was 1.0),
          fixing a bug where per-template "set IC weight to 100"
          code was silently discarded by a later weight-widget
          rebuild.
  Fix 5 - the "Learning rate" box in each scheduler phase is now
          hidden when that phase's optimizer is L-BFGS (which does
          not use a learning rate), and reappears if switched back
          to Adam.
  Fix 6 - default L-BFGS float precision is float32 instead of
          float64, for faster computation (still changeable to
          float64 from the window).
  Fix 7 - Inverse problems crashed ("tensor a (5) must match tensor
          b (4)") once the scheduler's phases kicked in, because the
          scheduler weight-string builder didn't know about the
          extra observation-data loss term inverse problems add.
  Fix 8 - "Inverse PINN Settings" now appears right after the PDE
          Definition box instead of near the bottom of the window.
  Fix 9 - the inverse trainable-variable field now defaults to
          "trainable_variable" instead of "beta" (still renameable).
  Fix 10 - selecting Inverse now auto-replaces the known diffusion-
          type constant in the current built-in template's PDE with
          the trainable-variable name (e.g. 1D Heat's 0.4 becomes
          trainable_variable); switching back to Forward restores
          the original constant. Works for all seven non-FeCr
          templates; renaming the variable while Inverse is active
          keeps the PDE in sync.
  Fix 11 - two Inverse + Optimizer-Scheduler bugs in codegen.py:
          (a) the trainable variable was silently frozen the moment
          scheduler phases began, because each phase's model.compile()
          never re-registered it as an external trainable variable —
          it looked like training continued normally, but the
          inferred parameter stopped updating after the first phase
          (confirmed by an end-to-end run: previously frozen at the
          pre-scheduler value, now correctly converges to the true
          value across all phases). (b) Inverse runs also always did
          an extra, untracked training pass for the old "Phase 1
          iterations" field BEFORE the scheduler phases even started,
          so total training time and iteration counts didn't match
          what was configured in the scheduler — that pass is now
          skipped whenever the scheduler is active, matching how
          Forward problems already behaved. The parameter print/plot/
          save history now also correctly covers every phase instead
          of stopping after the first.
  Fix 13 - the "Parametric Study" feature is removed from the GUI and
          from the generated training script's config (it was untested
          and unused; the underlying config fields are just always sent
          as disabled now, which the training script already handles as
          a normal single run).
  Fix 14 - the FeCr example is removed from the GUI entirely: gone from
          the Quick Examples list, its template definition, and its
          special-case scheduler defaults; codegen.py's FeCr-specific
          physics (mobility/free-energy model) is also removed rather
          than just made unreachable, so it isn't sitting in the source.
  Fix 15 - for Inverse problems: the observed-data loss weight now
          defaults to 100 (was 1) for every example, and selecting a
          built-in template now auto-fills the observed-data file with
          that template's end-time (largest t) reference file instead
          of requiring you to browse for it (Browse still works to pick
          a different file).
  Fix 16 - Inverse parameter values now print/save periodically during
          L-BFGS phases too, not just at the very end. Root cause:
          DeepXDE only calls the progress callback once per outer
          L-BFGS step, which can cover hundreds or thousands of real
          iterations in one call (unlike Adam, which calls it every
          iteration) — so counting callback calls (the old approach)
          almost never crossed the configured print/save period. Now
          both the console print and the "Every N iters" convergence
          file use the model's own real iteration count instead.
  Fix 17 - the Quick Examples box now shows the name of the example you
          selected (it used to always snap back to the placeholder even
          though the example loaded correctly). The placeholder is now
          "None" instead of "── Select ──".
Safe to re-run: already-applied fixes are skipped. Run once from
inside the repo root, then delete this file.
"""
import pathlib

def apply(path, old, new, label):
    p = pathlib.Path(path)
    text = p.read_text()
    if new in text and old not in text:
        print(f"[skip] {label}: already applied")
        return
    if old not in text:
        raise SystemExit(f"[FAIL] {label}: expected text not found in {path} — "
                          f"file may already differ from what this script expects. "
                          f"No changes made to this file.")
    text = text.replace(old, new, 1)
    p.write_text(text)
    print(f"[ok] {label}: patched {path}")


# --- Fix 1: codegen.py — NameError for 1D problems -------------------------
apply(
    "pinnstudio/core/codegen.py",
    old='''    if _is_2d:
        if _oi_w < len(_bc_ba) and _bc_ba[_oi_w].strip() == "True":
            _bcb_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
        else:
            _bcb_w.append(None); _wi += 1 if _wi < len(_wm_list) else 0
        _btt_check = "{config.bc_top_types}".split(",")
        _bbt_check = "{config.bc_bottom_types}".split(",")
        _btt_is_periodic = _oi_w < len(_btt_check) and _btt_check[_oi_w].strip() == "Periodic"
        _bbt_is_periodic = _oi_w < len(_bbt_check) and _bbt_check[_oi_w].strip() == "Periodic"''',
    new='''    _btt_check = "{config.bc_top_types}".split(",")
    _bbt_check = "{config.bc_bottom_types}".split(",")
    if _is_2d:
        if _oi_w < len(_bc_ba) and _bc_ba[_oi_w].strip() == "True":
            _bcb_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
        else:
            _bcb_w.append(None); _wi += 1 if _wi < len(_wm_list) else 0
        _btt_is_periodic = _oi_w < len(_btt_check) and _btt_check[_oi_w].strip() == "Periodic"
        _bbt_is_periodic = _oi_w < len(_bbt_check) and _bbt_check[_oi_w].strip() == "Periodic"''',
    label="codegen.py NameError fix",
)

# --- Fix 2: main_window.py — missing bc_right/bc_top in scheduler weights --
apply(
    "pinnstudio/ui/main_window.py",
    old='''                w_str = ",".join([
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
                ])''',
    new='''                w_str = ",".join([
                    str(self.weight_widgets.get(f"pde_{i}", SciLineEdit(1.0)).value())
                    for i in range(self.num_outputs_spin.value())
                ] + [
                    str(self.weight_widgets.get(k, SciLineEdit(1.0)).value())
                    for i in range(self.num_outputs_spin.value())
                    for k in [f"bc_left_{i}", f"bc_right_{i}", f"bc_bottom_{i}", f"bc_top_{i}", f"ic_{i}"]
                    if k in self.weight_widgets
                ])
            else:
                w_str = ",".join([
                    str(self.weight_widgets.get(f"pde_{i}_p{pn}", SciLineEdit(1.0)).value())
                    for i in range(self.num_outputs_spin.value())
                ] + [
                    str(self.weight_widgets.get(k, SciLineEdit(1.0)).value())
                    for i in range(self.num_outputs_spin.value())
                    for k in [f"bc_left_{i}_p{pn}", f"bc_right_{i}_p{pn}", f"bc_bottom_{i}_p{pn}", f"bc_top_{i}_p{pn}", f"ic_{i}_p{pn}"]
                    if k in self.weight_widgets
                ])''',
    label="main_window.py scheduler weight fix",
)

# --- Fix 3: main_window.py — default schedule = Adam(10000) + L-BFGS(10000)
# Fix 14 (below) later removes the FeCr special-case that used to wrap this
# same block in "if template_type == 'FeCr_PINN': ... else: ...", which
# re-indents it and drops the "else:". That means after Fix 14 has run,
# this fix's own marker text (still expecting the "else:"-wrapped form)
# would never match again — check both the pre-Fix-14 and post-Fix-14
# shapes up front so reruns skip cleanly either way.
_fix3_new_wrapped = "            # Default: Adam warm-up phase, then L-BFGS refinement, using same weights\n            self._add_scheduler_phase('adam', 10000, 0.001)"
_fix3_new_flat = "        # Default: Adam warm-up phase, then L-BFGS refinement, using same weights\n        self._add_scheduler_phase('adam', 10000, 0.001)"
_fix3_text = pathlib.Path("pinnstudio/ui/main_window.py").read_text()
if _fix3_new_wrapped in _fix3_text or _fix3_new_flat in _fix3_text:
    print("[skip] main_window.py default schedule (Adam 10000 + L-BFGS 10000): already applied")
else:
    apply(
        "pinnstudio/ui/main_window.py",
        old='''        else:
            # Default: one L-BFGS phase using same weights
            self._add_scheduler_phase('lbfgs', 20000, 0.001)
            self.sched_same_weights_cb.setChecked(True)''',
        new='''        else:
            # Default: Adam warm-up phase, then L-BFGS refinement, using same weights
            self._add_scheduler_phase('adam', 10000, 0.001)
            self._add_scheduler_phase('lbfgs', 10000, 0.001)
            self.sched_same_weights_cb.setChecked(True)''',
        label="main_window.py default schedule (Adam 10000 + L-BFGS 10000)",
    )

# --- Fix 4: main_window.py — default IC weight = 100 for all cases ---------
apply(
    "pinnstudio/ui/main_window.py",
    old='''        def _w_row(label, key):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            w = SciLineEdit(1.0); w.setFixedWidth(100)''',
    new='''        def _w_row(label, key):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            _default_w = 100.0 if key.startswith("ic_") else 1.0
            w = SciLineEdit(_default_w); w.setFixedWidth(100)''',
    label="main_window.py default IC weight = 100",
)

# --- Fix 5: main_window.py — hide Learning rate box for L-BFGS phases ------
apply(
    "pinnstudio/ui/main_window.py",
    old='''        # Learning rate
        lr_row = QHBoxLayout()
        lr_row.addWidget(QLabel("Learning rate:"))
        lr_spin = QDoubleSpinBox()
        lr_spin.setRange(1e-6, 1.0); lr_spin.setDecimals(6)
        lr_spin.setSingleStep(0.0001); lr_spin.setValue(lr)
        lr_spin.setFixedHeight(26)
        lr_row.addStretch(); lr_row.addWidget(lr_spin)
        phase_layout.addLayout(lr_row)''',
    new='''        # Learning rate (not used by L-BFGS — hidden for lbfgs phases)
        lr_widget = QWidget()
        lr_row = QHBoxLayout(lr_widget)
        lr_row.setContentsMargins(0, 0, 0, 0)
        lr_row.addWidget(QLabel("Learning rate:"))
        lr_spin = QDoubleSpinBox()
        lr_spin.setRange(1e-6, 1.0); lr_spin.setDecimals(6)
        lr_spin.setSingleStep(0.0001); lr_spin.setValue(lr)
        lr_spin.setFixedHeight(26)
        lr_row.addStretch(); lr_row.addWidget(lr_spin)
        phase_layout.addWidget(lr_widget)
        lr_widget.setVisible(optimizer != "lbfgs")
        opt_combo.currentTextChanged.connect(
            lambda t, w=lr_widget: w.setVisible(t != "lbfgs")
        )''',
    label="main_window.py hide LR box for L-BFGS phases",
)

# --- Fix 6: main_window.py + config.py — default L-BFGS float = float32 ----
apply(
    "pinnstudio/ui/main_window.py",
    old='''        self.lbfgs_float_combo = QComboBox()
        self.lbfgs_float_combo.addItems(["float64", "float32"])
        self.lbfgs_float_combo.setFixedWidth(90)''',
    new='''        self.lbfgs_float_combo = QComboBox()
        self.lbfgs_float_combo.addItems(["float64", "float32"])
        self.lbfgs_float_combo.setCurrentText("float32")
        self.lbfgs_float_combo.setFixedWidth(90)''',
    label="main_window.py default L-BFGS float = float32",
)
apply(
    "pinnstudio/core/config.py",
    old='''    lbfgs_float_type: str = "float64"''',
    new='''    lbfgs_float_type: str = "float32"''',
    label="config.py default lbfgs_float_type = float32",
)

# --- Fix 7: main_window.py — scheduler weight count for Inverse problems ---
apply(
    "pinnstudio/ui/main_window.py",
    old='''                    for k in [f"bc_left_{i}_p{pn}", f"bc_right_{i}_p{pn}", f"bc_bottom_{i}_p{pn}", f"bc_top_{i}_p{pn}", f"ic_{i}_p{pn}"]
                    if k in self.weight_widgets
                ])
            phases.append({''',
    new='''                    for k in [f"bc_left_{i}_p{pn}", f"bc_right_{i}_p{pn}", f"bc_bottom_{i}_p{pn}", f"bc_top_{i}_p{pn}", f"ic_{i}_p{pn}"]
                    if k in self.weight_widgets
                ])
            if hasattr(self, 'radio_inverse') and self.radio_inverse.isChecked() and hasattr(self, 'inv_obs_weight'):
                # Match codegen's _multi_weights: one extra observation-loss
                # weight appended at the end for inverse problems.
                w_str = w_str + "," + str(self.inv_obs_weight.value())
            phases.append({''',
    label="main_window.py scheduler weight count for Inverse",
)

# --- Fix 8: main_window.py — move Inverse PINN Settings after PDE box ------
apply(
    "pinnstudio/ui/main_window.py",
    old='''        self.inverse_group.setVisible(False)
        left_layout.addWidget(self.inverse_group)''',
    new='''        self.inverse_group.setVisible(False)
        left_layout.insertWidget(7, self.inverse_group)  # right after PDE Definition''',
    label="main_window.py move Inverse PINN Settings after PDE",
)

# --- Fix 9: main_window.py + config.py — default trainable-variable name ---
apply(
    "pinnstudio/ui/main_window.py",
    old='''        inv_layout.addWidget(QLabel("Unknown parameter name:"))
        self.inv_param_name = QLineEdit(); self.inv_param_name.setText("beta"); self.inv_param_name.setFixedHeight(28)
        inv_layout.addWidget(self.inv_param_name)''',
    new='''        inv_layout.addWidget(QLabel("Unknown parameter name:"))
        self.inv_param_name = QLineEdit(); self.inv_param_name.setText("trainable_variable"); self.inv_param_name.setFixedHeight(28)
        self.inv_param_name.editingFinished.connect(self._on_inv_param_name_changed)
        inv_layout.addWidget(self.inv_param_name)''',
    label="main_window.py default trainable-variable name",
)
apply(
    "pinnstudio/core/config.py",
    old='''    inverse_param_name: str = "beta"''',
    new='''    inverse_param_name: str = "trainable_variable"''',
    label="config.py default inverse_param_name",
)

# --- Fix 10: main_window.py — auto-substitute constant for Inverse ---------
# Fix 14 (below) later edits the comment inside this fix's own output (to
# drop a FeCr mention), which would otherwise make this fix's "new" marker
# stop matching on a rerun. Guard with a comment-independent structural
# marker instead, so it skips cleanly whether or not Fix 14 has also run.
if "def _sync_inverse_pde_substitution(self, is_inv):" in pathlib.Path("pinnstudio/ui/main_window.py").read_text():
    print("[skip] main_window.py auto-substitute constant for Inverse (core): already applied")
else:
    apply(
        "pinnstudio/ui/main_window.py",
        old='''    def _on_problem_type_changed(self, checked):
        is_inv = self.radio_inverse.isChecked()
        self.inverse_group.setVisible(is_inv)
        self.param_save_label.setVisible(is_inv)
        self.param_save_combo.setVisible(is_inv)''',
        new='''    # Per built-in template: (PDE row index, original constant substring)
    # of the "diffusion coefficient"-style constant that gets swapped for
    # the inverse trainable variable.
    INVERSE_AUTO_CONST = {
        "1D Heat": (0, "0.4"),
        "1D Allen-Cahn": (0, "0.0001"),
        "1D Cahn-Hilliard": (1, "1e-6"),
        "2D Heat (Dirichlet/Neumann)": (0, "0.4"),
        "2D Allen-Cahn (Mattey)": (0, "0.0001"),
        "2D Allen-Cahn (Wight)": (0, "0.00625"),
        "2D Cahn-Hilliard (Wight)": (1, "0.05"),
    }

    def _sync_inverse_pde_substitution(self, is_inv):
        """Inverse ON: replace the current template's known constant with
        the trainable-variable name in its PDE box. Inverse OFF: restore
        the original numeric constant so Forward mode stays valid."""
        template = getattr(self, '_current_template', '')
        entry = self.INVERSE_AUTO_CONST.get(template)
        if is_inv:
            if not entry:
                return
            idx, const_str = entry
            if idx >= len(self.pde_inputs):
                return
            var_name = self.inv_param_name.text().strip() or "trainable_variable"
            text = self.pde_inputs[idx].text()
            if const_str in text:
                self.pde_inputs[idx].setText(text.replace(const_str, var_name, 1))
                self._inverse_sub_active = (template, idx, const_str, var_name)
        else:
            state = getattr(self, '_inverse_sub_active', None)
            if state:
                s_template, s_idx, s_const, s_var = state
                if s_idx < len(self.pde_inputs):
                    cur = self.pde_inputs[s_idx].text()
                    if s_var in cur:
                        self.pde_inputs[s_idx].setText(cur.replace(s_var, s_const, 1))
                self._inverse_sub_active = None

    def _on_inv_param_name_changed(self):
        """Keep an already-substituted PDE in sync if the trainable
        variable name is renamed while Inverse mode is active."""
        state = getattr(self, '_inverse_sub_active', None)
        if not state:
            return
        template, idx, const_str, old_var = state
        new_var = self.inv_param_name.text().strip()
        if not new_var or new_var == old_var or idx >= len(self.pde_inputs):
            return
        text = self.pde_inputs[idx].text()
        if old_var in text:
            self.pde_inputs[idx].setText(text.replace(old_var, new_var, 1))
            self._inverse_sub_active = (template, idx, const_str, new_var)

    def _on_problem_type_changed(self, checked):
        is_inv = self.radio_inverse.isChecked()
        self._sync_inverse_pde_substitution(is_inv)
        self.inverse_group.setVisible(is_inv)
        self.param_save_label.setVisible(is_inv)
        self.param_save_combo.setVisible(is_inv)''',
        label="main_window.py auto-substitute constant for Inverse (core)",
    )
apply(
    "pinnstudio/ui/main_window.py",
    old='''            self._current_template_type = t.get('template_type', '')
            # Setup default scheduler phases''',
    new='''            self._current_template_type = t.get('template_type', '')
            self._sync_inverse_pde_substitution(self.radio_inverse.isChecked())
            # Setup default scheduler phases''',
    label="main_window.py auto-substitute constant for Inverse (2D template hook)",
)
apply(
    "pinnstudio/ui/main_window.py",
    old='''        self._current_template_type = t.get('template_type', '')
        if hasattr(self, 'sched_cb'):''',
    new='''        self._current_template_type = t.get('template_type', '')
        self._sync_inverse_pde_substitution(self.radio_inverse.isChecked())
        if hasattr(self, 'sched_cb'):''',
    label="main_window.py auto-substitute constant for Inverse (1D template hook)",
)

# --- Fix 11: codegen.py — Inverse variable frozen after scheduler starts,
#             and a redundant untracked training pass before the scheduler --
# Fix 11 replaces one large block. If an earlier (possibly buggy) copy of
# this script already applied *some* version of Fix 11, the block below
# won't match exactly (old = pristine pre-Fix-11 text; new = the fully
# correct text) and would otherwise abort the whole script. Detect that
# case up front via a marker unique to Fix 11's new code, and skip the
# big block replacement then — Fix 12 below handles narrow repairs.
_fix11_marker = '_sched_active = {config.optimizer_scheduler}'
if _fix11_marker in pathlib.Path("pinnstudio/core/codegen.py").read_text():
    print("[skip] codegen.py fix frozen Inverse variable + redundant pass in scheduler: already applied")
else:
    apply(
        "pinnstudio/core/codegen.py",
        old='''    if not {config.time_adaptive}:
        if _problem_type == "Inverse":
            with open("/tmp/param_history.txt", "w") as _f:
                pass
            _var_cb = dde.callbacks.VariableValue(
                [{config.inverse_param_name}], period=1000, filename="/tmp/param_history.txt",
                precision=6
            )
            loss_history, train_state = model.train(iterations=_iters, display_every=1000, callbacks=[_var_cb, _print_cb, _save_cb])
        else:
            if {config.batch_size} > 0:
                data.batch_size = {config.batch_size}
                print(f"Batch size set to: {config.batch_size}")
            if _use_save:
                pass  # model saved after scheduler phases
                _adam_cfg_path = _os.path.join(_sol_dir, f"model_adam-{{_iters}}.json")
                with open(_adam_cfg_path, "w") as _acf:
                    _json.dump(_model_config, _acf, indent=2)
                print(f"Adam config saved to: {{_adam_cfg_path}}")

        # Optimizer Scheduler or Phase 2
        if {config.optimizer_scheduler} and _os.path.exists('/tmp') and {len(config.scheduler_phases) > 0}:
            import json as _json_sched
            _sched_phases = _json_sched.loads({repr(config.scheduler_phases)})
            for _sp_i, _sp in enumerate(_sched_phases):
                print(f"  === Scheduler Phase {{_sp_i+1}}: {{_sp['optimizer']}} {{_sp['iterations']}} iters ===")
                _sp_weights = [float(w) for w in _sp['weights'].split(',') if w.strip()]
                if _sp['optimizer'] == 'lbfgs':
                    dde.optimizers.set_LBFGS_options(
                        maxcor={config.lbfgs_maxcor}, ftol={config.lbfgs_ftol},
                        gtol={config.lbfgs_gtol}, maxiter=_sp['iterations'],
                        maxfun=int(_sp['iterations']*1.25), maxls={config.lbfgs_maxls})
                    _lbfgs_float = "{config.lbfgs_float_type}"
                    model.compile("L-BFGS", loss=_sp.get('loss', '{config.loss_type}'),
                                  loss_weights=_sp_weights)
                    loss_history, train_state = model.train(display_every=200)
                    if _use_save:
                        _nta_save_path = _os.path.join(_sol_dir, f"model_lbfgs-phase{{_sp_i+1}}")
                        model.save(_nta_save_path)
                        print(f"  Phase {{_sp_i+1}} L-BFGS model saved: {{_nta_save_path}}.pt")
                else:
                    model.compile(_sp['optimizer'], lr=_sp['lr'],
                                  loss=_sp.get('loss', '{config.loss_type}'), loss_weights=_sp_weights)
                    if {config.batch_size} > 0:
                        data.batch_size = {config.batch_size}
                    loss_history, train_state = model.train(iterations=_sp['iterations'], display_every=1000)
                    if _use_save:
                        _nta_save_path = _os.path.join(_sol_dir, f"model_adam-phase{{_sp_i+1}}")
                        model.save(_nta_save_path)
                        print(f"  Phase {{_sp_i+1}} Adam model saved: {{_nta_save_path}}.pt")''',
    new='''    if not {config.time_adaptive}:
        _sched_active = {config.optimizer_scheduler} and _os.path.exists('/tmp') and {len(config.scheduler_phases) > 0}
        if _problem_type == "Inverse":
            with open("/tmp/param_history.txt", "w") as _f:
                pass
            if not _sched_active:
                _var_cb = dde.callbacks.VariableValue(
                    [{config.inverse_param_name}], period=1000, filename="/tmp/param_history.txt",
                    precision=6
                )
                loss_history, train_state = model.train(iterations=_iters, display_every=1000, callbacks=[_var_cb, _print_cb, _save_cb])
            else:
                # Scheduler phases below define all training — skip this
                # standalone _iters-iteration pass so training only runs
                # for the iteration counts configured in the scheduler.
                print(f"  (Scheduler enabled — skipping standalone {{_iters}}-iteration pass; "
                      f"training runs only for the phases below)")
        else:
            if {config.batch_size} > 0:
                data.batch_size = {config.batch_size}
                print(f"Batch size set to: {config.batch_size}")
            if _use_save:
                pass  # model saved after scheduler phases
                _adam_cfg_path = _os.path.join(_sol_dir, f"model_adam-{{_iters}}.json")
                with open(_adam_cfg_path, "w") as _acf:
                    _json.dump(_model_config, _acf, indent=2)
                print(f"Adam config saved to: {{_adam_cfg_path}}")

        # Optimizer Scheduler or Phase 2
        if _sched_active:
            import json as _json_sched
            _sched_phases = _json_sched.loads({repr(config.scheduler_phases)})
            _sched_cum_iters = 0
            for _sp_i, _sp in enumerate(_sched_phases):
                print(f"  === Scheduler Phase {{_sp_i+1}}: {{_sp['optimizer']}} {{_sp['iterations']}} iters ===")
                _sp_weights = [float(w) for w in _sp['weights'].split(',') if w.strip()]
                _sp_ext_vars = [{config.inverse_param_name}] if _problem_type == "Inverse" else None
                if _sp['optimizer'] == 'lbfgs':
                    dde.optimizers.set_LBFGS_options(
                        maxcor={config.lbfgs_maxcor}, ftol={config.lbfgs_ftol},
                        gtol={config.lbfgs_gtol}, maxiter=_sp['iterations'],
                        maxfun=int(_sp['iterations']*1.25), maxls={config.lbfgs_maxls})
                    _lbfgs_float = "{config.lbfgs_float_type}"
                    model.compile("L-BFGS", loss=_sp.get('loss', '{config.loss_type}'),
                                  loss_weights=_sp_weights, external_trainable_variables=_sp_ext_vars)
                    if _problem_type == "Inverse":
                        _print_cb.set_offset(_sched_cum_iters)
                        _save_cb.set_offset(_sched_cum_iters)
                        _sp_var_cb = dde.callbacks.VariableValue(
                            [{config.inverse_param_name}], period=200,
                            filename=f"/tmp/param_history_sp{{_sp_i}}.txt", precision=6)
                        loss_history, train_state = model.train(
                            display_every=200, callbacks=[_sp_var_cb, _print_cb, _save_cb])
                        try:
                            with open(f"/tmp/param_history_sp{{_sp_i}}.txt", "r") as _spf:
                                _sp_lines = [l.strip() for l in _spf if l.strip()]
                            with open("/tmp/param_history.txt", "a") as _fa:
                                for _spl in _sp_lines:
                                    _spparts = _spl.replace("[","").replace("]","").split()
                                    if len(_spparts) >= 2:
                                        _fa.write(f"{{_sched_cum_iters + int(_spparts[0])}} [{{_spparts[1]}}]\\\\n")
                        except Exception as _spe:
                            print(f"Could not merge phase {{_sp_i+1}} parameter history: {{_spe}}")
                        _sched_cum_iters = loss_history.steps[-1] if loss_history.steps else (_sched_cum_iters + _sp['iterations'])
                    else:
                        loss_history, train_state = model.train(display_every=200)
                    if _use_save:
                        _nta_save_path = _os.path.join(_sol_dir, f"model_lbfgs-phase{{_sp_i+1}}")
                        model.save(_nta_save_path)
                        print(f"  Phase {{_sp_i+1}} L-BFGS model saved: {{_nta_save_path}}.pt")
                else:
                    model.compile(_sp['optimizer'], lr=_sp['lr'],
                                  loss=_sp.get('loss', '{config.loss_type}'), loss_weights=_sp_weights,
                                  external_trainable_variables=_sp_ext_vars)
                    if {config.batch_size} > 0:
                        data.batch_size = {config.batch_size}
                    if _problem_type == "Inverse":
                        _print_cb.set_offset(_sched_cum_iters)
                        _save_cb.set_offset(_sched_cum_iters)
                        _sp_var_cb = dde.callbacks.VariableValue(
                            [{config.inverse_param_name}], period=1000,
                            filename=f"/tmp/param_history_sp{{_sp_i}}.txt", precision=6)
                        loss_history, train_state = model.train(
                            iterations=_sp['iterations'], display_every=1000,
                            callbacks=[_sp_var_cb, _print_cb, _save_cb])
                        try:
                            with open(f"/tmp/param_history_sp{{_sp_i}}.txt", "r") as _spf:
                                _sp_lines = [l.strip() for l in _spf if l.strip()]
                            with open("/tmp/param_history.txt", "a") as _fa:
                                for _spl in _sp_lines:
                                    _spparts = _spl.replace("[","").replace("]","").split()
                                    if len(_spparts) >= 2:
                                        _fa.write(f"{{_sched_cum_iters + int(_spparts[0])}} [{{_spparts[1]}}]\\\\n")
                        except Exception as _spe:
                            print(f"Could not merge phase {{_sp_i+1}} parameter history: {{_spe}}")
                        _sched_cum_iters = loss_history.steps[-1] if loss_history.steps else (_sched_cum_iters + _sp['iterations'])
                    else:
                        loss_history, train_state = model.train(iterations=_sp['iterations'], display_every=1000)
                    if _use_save:
                        _nta_save_path = _os.path.join(_sol_dir, f"model_adam-phase{{_sp_i+1}}")
                        model.save(_nta_save_path)
                        print(f"  Phase {{_sp_i+1}} Adam model saved: {{_nta_save_path}}.pt")''',
    label="codegen.py fix frozen Inverse variable + redundant pass in scheduler",
)

# --- Fix 12: codegen.py — repair a newline-escaping bug in a previous
#             version of Fix 11 above. If you are running this script for
#             the first time, Fix 11 already writes the correct text and
#             both calls below will just say "already applied". If you
#             already ran an earlier copy of this script and then hit
#             "SyntaxError: unterminated string literal" while training an
#             Inverse problem with the scheduler on, this repairs it.
_f12_old = '_fa.write(f"{{_sched_cum_iters + int(_spparts[0])}} [{{_spparts[1]}}]\\n")'
_f12_new = '_fa.write(f"{{_sched_cum_iters + int(_spparts[0])}} [{{_spparts[1]}}]\\\\n")'
apply("pinnstudio/core/codegen.py", _f12_old, _f12_new,
      label="codegen.py repair newline escaping (occurrence 1/2)")
apply("pinnstudio/core/codegen.py", _f12_old, _f12_new,
      label="codegen.py repair newline escaping (occurrence 2/2)")

# --- Fix 13: main_window.py — remove Parametric Study from the GUI ---------
apply(
    "pinnstudio/ui/main_window.py",
    old='''        left_layout.insertWidget(7, self.inverse_group)  # right after PDE Definition

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
        left_layout.addWidget(param_group)''',
    new='''        left_layout.insertWidget(7, self.inverse_group)  # right after PDE Definition

        # Parametric Study removed (untested, not exposed in the GUI).''',
    label="main_window.py remove Parametric Study UI",
)
apply(
    "pinnstudio/ui/main_window.py",
    old='''        return json.dumps(groups)

    def _on_param_changed(self, state):
        self.param_widget.setVisible(state == 2)
''',
    new='''        return json.dumps(groups)
''',
    label="main_window.py remove _on_param_changed",
)
apply(
    "pinnstudio/ui/main_window.py",
    old='''            parametric_study=self.param_check.isChecked(),
            parametric_param=self.param_combo.currentText(),
            parametric_values=self.param_values_input.text(),''',
    new='''            parametric_study=False,
            parametric_param="none",
            parametric_values="",''',
    label="main_window.py hardcode parametric study disabled in config",
)

# --- Fix 14: remove the FeCr example from the GUI entirely -----------------
apply(
    "pinnstudio/ui/main_window.py",
    old='''            "FeCr PINN": {
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
                't_max': 10.0,
                'bc_config': 'fecr',
                'template_type': 'FeCr_PINN',
                'forward_ic_from_file': True,
                'forward_ic_file': '/home/asfandyarkhan/deepxde_gui/FEM_Results/2D_Examples/FeCr_PINN_2D/t_0.txt',
                'ref_dir': '/home/asfandyarkhan/deepxde_gui/FEM_Results/2D_Examples/FeCr_PINN_2D',
            },
        }''',
    new='''        }''',
    label="main_window.py remove FeCr template definition",
)
apply(
    "pinnstudio/ui/main_window.py",
    old='''            elif bc_config == 'fecr':
                # c (index 0): periodic on all sides
                if len(self.bc_left_types) > 0:
                    self.bc_left_types[0].setCurrentText("Periodic")
                if len(self.bc_bottom_types) > 0:
                    self.bc_bottom_types[0].setCurrentText("Periodic")
                # Uncheck master BC toggle for mu (output 1)
                for _w in self.bc_group.findChildren(QCheckBox):
                    if _w.text() == "Enable boundary conditions for mu":
                        _w.setChecked(False)
                        break
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
                # Enable optimizer scheduler with FeCr default phases
                self.sched_cb.setChecked(True)
                self._setup_default_scheduler_phases('FeCr_PINN')
                # Set two step groups for FeCr: 0→1 (10 steps) and 1→10 (9 steps)
                for row in list(self.ta_group_rows):
                    row['widget'].deleteLater()
                self.ta_group_rows.clear()
                self._add_ta_step_group(0.0, 1.0, 10)
                self._add_ta_step_group(1.0, 10.0, 9)
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
            elif bc_config == 'heat2d':''',
    new='''            elif bc_config == 'heat2d':''',
    label="main_window.py remove FeCr BC-config branch",
)
apply(
    "pinnstudio/ui/main_window.py",
    old='''        if template_type == 'FeCr_PINN':
            # Phase 2: Adam [0,0,0,0,1000]
            self._add_scheduler_phase('adam', 50000, 0.001)
            # Phase 3: Adam [100,1e-4,1,1,1000]
            self._add_scheduler_phase('adam', 50000, 0.001)
            # Phase 4: L-BFGS [100,1e-4,1,1,1000]
            self._add_scheduler_phase('lbfgs', 20000, 0.001)
            # Uncheck same weights so per-phase weights show
            self.sched_same_weights_cb.setChecked(False)
        else:
            # Default: Adam warm-up phase, then L-BFGS refinement, using same weights
            self._add_scheduler_phase('adam', 10000, 0.001)
            self._add_scheduler_phase('lbfgs', 10000, 0.001)
            self.sched_same_weights_cb.setChecked(True)
        self._build_weight_inputs(self.num_outputs_spin.value())

        if template_type == 'FeCr_PINN' and len(self.sched_phase_list) >= 3:
            p2 = self.sched_phase_list[0]['phase_num']
            p3 = self.sched_phase_list[1]['phase_num']
            p4 = self.sched_phase_list[2]['phase_num']
            # Set L-BFGS phase iterations to 50000
            self.sched_phase_list[2]['iters'].setValue(50000)
            for key, val in [
                (f"pde_0_p{p2}", 100.0), (f"pde_1_p{p2}", 1e-6),
                (f"bc_left_0_p{p2}", 1.0), (f"bc_bottom_0_p{p2}", 1.0),
                (f"ic_0_p{p2}", 1000.0),
                (f"pde_0_p{p3}", 100.0), (f"pde_1_p{p3}", 1e-4),
                (f"bc_left_0_p{p3}", 1.0), (f"bc_bottom_0_p{p3}", 1.0),
                (f"ic_0_p{p3}", 1000.0),
                (f"pde_0_p{p4}", 100.0), (f"pde_1_p{p4}", 1e-4),
                (f"bc_left_0_p{p4}", 1.0), (f"bc_bottom_0_p{p4}", 1.0),
                (f"ic_0_p{p4}", 1000.0),
            ]:
                if key in self.weight_widgets:
                    self.weight_widgets[key].setValue(val)''',
    new='''        # Default: Adam warm-up phase, then L-BFGS refinement, using same weights
        self._add_scheduler_phase('adam', 10000, 0.001)
        self._add_scheduler_phase('lbfgs', 10000, 0.001)
        self.sched_same_weights_cb.setChecked(True)
        self._build_weight_inputs(self.num_outputs_spin.value())''',
    label="main_window.py remove FeCr scheduler special-case",
)
apply(
    "pinnstudio/ui/main_window.py",
    old='''    # Per built-in template: (PDE row index, original constant substring)
    # of the "diffusion coefficient"-style constant that gets swapped for
    # the inverse trainable variable. FeCr is excluded (bespoke, unpublished).''',
    new='''    # Per built-in template: (PDE row index, original constant substring)
    # of the "diffusion coefficient"-style constant that gets swapped for
    # the inverse trainable variable.''',
    label="main_window.py sanitize FeCr mention in comment",
)
apply(
    "pinnstudio/core/codegen.py",
    old='''    # ── FeCr PDE block (inserted verbatim for FeCr_PINN template) ──
    if config.forward_ic_from_file:
        _ta_ic_init = f"""_ic_ta_data = np.loadtxt(r"{config.forward_ic_file}")
    _ic_ta_mask = np.abs(_ic_ta_data[:, 2]) < 1e-10
    prev_u = _ic_ta_data[_ic_ta_mask, 3:4]"""
    else:
        _ta_ic_init = f"prev_u = np.reshape({ta_ic_expr}, (-1, 1))"
    _fecr_pde1 = config_pde_expressions.split("|")[0].strip() if config.template_type == "FeCr_PINN" else ""
    _fecr_pde2 = config_pde_expressions.split("|")[1].strip() if (config.template_type == "FeCr_PINN" and "|" in config_pde_expressions) else ""

    if config.template_type == "FeCr_PINN":
        _fecr_pde_block = f"""
import math as _math

_Lo        = 1.0 / (25e-9)**2
_LOGE10    = _math.log(10.0)
_EPS32     = 1e-6

def _dfdc_torch(c):
    eps = torch.tensor(1e-6, dtype=c.dtype, device=c.device)
    c = torch.clamp(c, eps, 1.0 - eps)
    return (8098.119000
            + 4167.994000 * torch.log(c)
            - 7052.907000 * torch.log(1.0 - c)
            + 14684.820000 * c
            - 71698.782000 * c**2
            + 37524.688000 * c**3)

def _M_torch(c):
    s  = torch.clamp(c, _EPS32, 1.0 - _EPS32)
    t  = 1.0 - s
    p1 = torch.clamp(t*t*s, min=1e-30)
    p2 = torch.clamp(s*s*t, min=1e-30)
    gCr = (-32.770969*s - 25.8186669*t
           - 3.29612744*s*torch.log(s)
           + 17.669757*t*torch.log(t)
           + 37.6197853*s*t
           + 20.6941796*s*t*(2.0*s-1.0)
           + 10.8095813*s*t*(2.0*s-1.0)**2)
    gFe = (-31.687117*s - 26.0291774*t
           + 0.2286581*s*torch.log(s)
           + 24.3633544*t*torch.log(t)
           + 44.3334237*s*t
           + 8.72990497*s*t*(2.0*s-1.0)
           + 20.956768*s*t*(2.0*s-1.0)**2)
    a = torch.log(p1) + gCr * _LOGE10
    b = torch.log(p2) + gFe * _LOGE10
    m = torch.maximum(a, b)
    ln_sum = m + torch.log(torch.exp(a - m) + torch.exp(b - m))
    return torch.exp(_math.log(_Lo) + ln_sum)

# ── PDE definition ──────────────────────────────────────────
def pde(x, y):
    c   = y[:, 0:1]
    mu  = y[:, 1:2]
    c_t   = dde.grad.jacobian(y, x, i=0, j=2)
    c_xx  = dde.grad.hessian(y, x, component=0, i=0, j=0)
    c_yy  = dde.grad.hessian(y, x, component=0, i=1, j=1)
    mu_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    mu_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    c_x   = dde.grad.jacobian(y, x, i=0, j=0)
    c_y   = dde.grad.jacobian(y, x, i=0, j=1)
    mu_x  = dde.grad.jacobian(y, x, i=1, j=0)
    mu_y  = dde.grad.jacobian(y, x, i=1, j=1)
    dfdc  = _dfdc_torch(c)
    Mloc  = _M_torch(c)
    dMdc  = torch.autograd.grad(
        Mloc, c, grad_outputs=torch.ones_like(Mloc), create_graph=True
    )[0]
    # Use values from GUI PDE expressions via eval
    _pde1 = "{_fecr_pde1}"
    _pde2 = "{_fecr_pde2}"
    _eval_fecr = {{
        "dc_t": c_t, "dmu_xx": mu_xx, "dmu_yy": mu_yy,
        "dMdc": dMdc, "dc_x": c_x, "dc_y": c_y,
        "dmu_x": mu_x, "dmu_y": mu_y,
        "M": Mloc, "dc_xx": c_xx, "dc_yy": c_yy,
        "mu": mu, "dfdc": dfdc,
        "np": np, "torch": torch
    }}
    eq1 = eval(_pde1, _eval_fecr)
    eq2 = eval(_pde2, _eval_fecr)
    return [eq1, eq2]
"""
    else:
        _fecr_pde_block = ""''',
    new='''    if config.forward_ic_from_file:
        _ta_ic_init = f"""_ic_ta_data = np.loadtxt(r"{config.forward_ic_file}")
    _ic_ta_mask = np.abs(_ic_ta_data[:, 2]) < 1e-10
    prev_u = _ic_ta_data[_ic_ta_mask, 3:4]"""
    else:
        _ta_ic_init = f"prev_u = np.reshape({ta_ic_expr}, (-1, 1))"
    _fecr_pde_block = ""''',
    label="codegen.py remove FeCr physics model",
)
apply(
    "pinnstudio/core/codegen.py",
    old='''{_fecr_pde_block}
_IS_FECR = "{config.template_type}" == "FeCr_PINN"
# ── PDE definition (standard) ───────────────────────────────''',
    new='''{_fecr_pde_block}
_IS_FECR = False
# ── PDE definition (standard) ───────────────────────────────''',
    label="codegen.py FeCr flag always false",
)

# --- Fix 15: Inverse observed-data weight default 100 + auto-load file -----
apply(
    "pinnstudio/ui/main_window.py",
    old='''        inv_layout.addWidget(QLabel("Observed data loss weight:"))
        self.inv_obs_weight = SciLineEdit(1.0)''',
    new='''        inv_layout.addWidget(QLabel("Observed data loss weight:"))
        self.inv_obs_weight = SciLineEdit(100.0)''',
    label="main_window.py default observed-data weight = 100",
)
_fix15b_marker = "Auto-select the end-time (largest t) reference file as the Inverse"
if _fix15b_marker in pathlib.Path("pinnstudio/ui/main_window.py").read_text():
    print("[skip] main_window.py auto-load end-time observed data file: already applied")
else:
    apply(
        "pinnstudio/ui/main_window.py",
        old='''        valid_files.sort(key=lambda x: x[0])
        self._ea_settings = {
            'files': valid_files,
            'do_line': True,
            'do_surface': True,
            'do_l2': True,
            'do_mse': True,
            'do_max': True,
        }
        self.log_box.append(f"✅ Error analysis auto-configured — {len(valid_files)} ground truth files from template")''',
        new='''        valid_files.sort(key=lambda x: x[0])
        self._ea_settings = {
            'files': valid_files,
            'do_line': True,
            'do_surface': True,
            'do_l2': True,
            'do_mse': True,
            'do_max': True,
        }
        self.log_box.append(f"✅ Error analysis auto-configured — {len(valid_files)} ground truth files from template")
        # Auto-select the end-time (largest t) reference file as the Inverse
        # observed-data file, so the user doesn't have to browse for it.
        if hasattr(self, 'inv_data_path'):
            _end_time_file = valid_files[-1][1]
            self.inv_data_path.setText(_end_time_file)
            self.log_box.append(
                f"📂 Inverse observed data auto-loaded: {os.path.basename(_end_time_file)} "
                f"(t={valid_files[-1][0]:.4g})")''',
        label="main_window.py auto-load end-time observed data file",
    )

# --- Fix 16: Inverse parameter print/save doesn't fire periodically during
#             L-BFGS phases (only at the very end) -------------------------
apply(
    "pinnstudio/core/codegen.py",
    old='''    class _PrintParamCallback(dde.callbacks.Callback):
        def __init__(self, var, name, period=1000):
            super().__init__()
            self.var = var
            self.name = name
            self.period = period
            self._step = 0
            self._offset = 0
        def set_offset(self, offset):
            self._offset = offset
            self._step = 0
        def on_batch_end(self):
            self._step += 1
            actual_iter = self._offset + self._step
            if actual_iter % self.period == 0:
                val = self.var.detach().cpu().numpy().item() if hasattr(self.var, 'detach') else float(self.var.numpy())
                print(f"  [{config.inverse_param_name}] Iter {{actual_iter}}: {{val:.6f}}", flush=True)''',
    new='''    class _PrintParamCallback(dde.callbacks.Callback):
        def __init__(self, var, name, period=1000):
            super().__init__()
            self.var = var
            self.name = name
            self.period = period
            self._last_bucket = 0
        def set_offset(self, offset):
            # No-op, kept so older scheduler-loop code can still call it.
            # Real progress now comes from the model's own train_state.step,
            # which is already cumulative across every phase — no manual
            # offset bookkeeping needed. This also fixes L-BFGS: DeepXDE only
            # calls on_batch_end once per outer L-BFGS step (which can cover
            # hundreds of real iterations at once), not once per real
            # iteration the way Adam does, so counting callback firings
            # (the old approach) almost never reached the print period.
            pass
        def on_batch_end(self):
            cur = self.model.train_state.step
            bucket = cur // self.period
            if bucket > self._last_bucket:
                self._last_bucket = bucket
                val = self.var.detach().cpu().numpy().item() if hasattr(self.var, 'detach') else float(self.var.numpy())
                print(f"  [{config.inverse_param_name}] Iter {{cur}}: {{val:.6f}}", flush=True)''',
    label="codegen.py fix L-BFGS periodic parameter print",
)
apply(
    "pinnstudio/core/codegen.py",
    old='''    class _SaveParamCallback(dde.callbacks.Callback):
        def __init__(self, var, name, period, path):
            super().__init__()
            self.var = var
            self.name = name
            self.period = period
            self.path = path
            self._step = 0
            self._offset = 0
        def set_offset(self, offset):
            self._offset = offset
            self._step = 0
        def on_batch_end(self):
            if self.period == 0: return
            self._step += 1
            actual_iter = self._offset + self._step
            if actual_iter % self.period == 0:
                val = self.var.detach().cpu().numpy().item() if hasattr(self.var, 'detach') else float(self.var.numpy())
                with open(self.path, "a") as _f:''',
    new='''    class _SaveParamCallback(dde.callbacks.Callback):
        def __init__(self, var, name, period, path):
            super().__init__()
            self.var = var
            self.name = name
            self.period = period
            self.path = path
            self._last_bucket = 0
        def set_offset(self, offset):
            # No-op — see _PrintParamCallback.set_offset above.
            pass
        def on_batch_end(self):
            if self.period == 0: return
            cur = self.model.train_state.step
            bucket = cur // self.period
            if bucket > self._last_bucket:
                self._last_bucket = bucket
                val = self.var.detach().cpu().numpy().item() if hasattr(self.var, 'detach') else float(self.var.numpy())
                with open(self.path, "a") as _f:''',
    label="codegen.py fix L-BFGS periodic parameter save (structure)",
)
# The final write line of _SaveParamCallback.on_batch_end needs its own
# careful, isolated patch: it both renames a variable (actual_iter -> cur,
# to match the structural change just above) and contains a "\n" escape
# that has to survive one extra level of parsing (codegen.py's own
# f-string), the same subtlety documented at Fix 12 above. Building it via
# plain string concatenation with an explicit backslash character — rather
# than typing backslashes directly into this script's source — sidesteps
# any risk of miscounting them by hand a second time.
_BS = chr(92)
_f16_suffix = _BS + _BS + 'n")'
_f16_old = '                    _f.write(f"{{actual_iter}},{{val:.8f}}' + _f16_suffix
_f16_new = '                    _f.write(f"{{cur}},{{val:.8f}}' + _f16_suffix
apply("pinnstudio/core/codegen.py", _f16_old, _f16_new,
      label="codegen.py fix L-BFGS periodic parameter save (write line)")

# --- Fix 17: Quick Examples box shows the selected example name ------------
apply(
    "pinnstudio/ui/main_window.py",
    old='''        self.quick_examples_combo.addItems(["── Select ──", "1D Heat", "1D Allen-Cahn", "1D Cahn-Hilliard"])''',
    new='''        self.quick_examples_combo.addItems(["None", "1D Heat", "1D Allen-Cahn", "1D Cahn-Hilliard"])''',
    label="main_window.py rename initial placeholder to None",
)
apply(
    "pinnstudio/ui/main_window.py",
    old='''        if is_2d:
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
        self.quick_examples_combo.blockSignals(False)''',
    new='''        if is_2d:
            self.quick_examples_combo.addItems([
                "None",
                "2D Heat (Dirichlet/Neumann)",
                "2D Allen-Cahn (Mattey)",
                "2D Allen-Cahn (Wight)",
                "2D Cahn-Hilliard (Wight)"
            ])
        else:
            self.quick_examples_combo.addItems([
                "None",
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
        if text == "None":
            return
        self._on_template_selected(text)''',
    label="main_window.py Quick Examples box shows selected example",
)

print("\nDone. Run: python -m pinnstudio.main  and re-test 1D Heat (Inverse).")
