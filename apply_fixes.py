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

print("\nDone. Run: python -m pinnstudio.main  and re-test 1D Heat.")
