#!/usr/bin/env python3
"""Verification runs and screenshots for tutorials 19-22 (website/tutorials/19-*.md ... 22-*.md).

    QT_QPA_PLATFORM=offscreen python tools/docs/run_new_tutorials.py <out_dir> [--only 19,20,21,22] [--shots]

Drives the real MainWindow offscreen through the same widgets a user edits (the Examples bar, the
equation boxes, the training phases, the loss weights, Error Analysis), presses Solve with shortened
training phases, waits for the run and prints the numbers the tutorials quote: the final training
loss terms and the relative L2 error from error_analysis/. With --shots it also saves the screenshots
the tutorials show into website/assets/screenshots/ and copies result figures into
website/assets/images/. Modelled on tools/docs/shots_tutorials.py, whose helpers it uses.
"""
import glob
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import shots_tutorials as st  # noqa: E402  (sets up the offscreen app and a neutral home folder)

SHOTS = os.path.join(REPO, "website", "assets", "screenshots")
IMAGES = os.path.join(REPO, "website", "assets", "images")


def last_run(save_dir):
    runs = [d for d in glob.glob(os.path.join(save_dir, "**", "solution_results"), recursive=True)]
    return os.path.dirname(max(runs, key=os.path.getmtime)) if runs else None


def metrics(run):
    out = {}
    for f in sorted(glob.glob(os.path.join(run, "error_analysis", "error_metrics*.txt"))):
        rows = [ln.strip().split(",") for ln in open(f, encoding="utf-8") if ln.strip()]
        head = [h.strip().lower() for h in rows[0]]
        i = head.index("l2_relative") if "l2_relative" in head else None
        out[os.path.basename(f)] = [(r[0], r[i]) for r in rows[1:]] if i is not None else rows
    return out


def last_loss(run):
    p = os.path.join(run, "solution_results", "training_log.txt")
    last = None
    try:
        for ln in open(p, encoding="utf-8", errors="replace"):
            if ln.strip().startswith("Step"):
                last = ln.strip()
    except OSError:
        pass
    return last


def solve(w, tag, save_dir, ea=None, shots=False, result_tabs=()):
    os.makedirs(save_dir, exist_ok=True)
    w.save_dir_input.setText(save_dir)
    if ea is not None:
        w._ea_settings = dict({'files': [], 'ref_mode': 'formula', 'ref_formula': '', 'ref_times': '',
                               'ref_grid': 0, 'do_line': True, 'do_surface': True, 'do_l2': True,
                               'do_mse': False, 'do_max': True}, **ea)
    w.log_box.clear()
    t0 = time.time()
    w._on_solve()
    st.wait_thread(w, limit=5400)
    took = time.time() - t0
    run = last_run(save_dir)
    print(f"\n=== {tag}: {took / 60:.1f} min, run folder {run}", flush=True)
    log = w.log_box.toPlainText()
    for ln in log.splitlines():
        if any(k in ln for k in ("❌", "Traceback", "Error", "L2", "Loss weights", "Final")):
            print("   log:", ln[:200])
    if run:
        print("   last step:", last_loss(run))
        for k, v in metrics(run).items():
            print("  ", k, v)
    if shots and run:
        for i in range(w.results_tabs.count()):
            name = w.results_tabs.tabText(i)
            if name in result_tabs:
                w.results_tabs.setCurrentIndex(i)
                st.settle(1.0)
                st.save(w, f"{tag}_results_{name.split()[0].lower()}.png")
    return run, took


def copy_fig(run, pattern, name):
    hits = sorted(glob.glob(os.path.join(run, "**", pattern), recursive=True))
    if hits:
        shutil.copy(hits[0], os.path.join(IMAGES, name))
        print("copied", hits[0], "->", name)
    else:
        print("no figure", pattern, "in", run)


# ── 19. Steady 2D flow (Kovasznay) ───────────────────────────────────────
KOV_EXACT = "1 - exp(l*x)*cos(2*pi*y)|l/(2*pi)*exp(l*x)*sin(2*pi*y)|0.5*(1 - exp(2*l*x))"


def tut19(out, shots):
    w = st.new_window()
    st.pick(w, "Forward", "Kovasznay flow")
    if shots:
        for key in ("equations", "geometry", "bcs"):
            st.page(w, key)
            st.save(w, f"tut19_{key}.png")
    st.set_phase(w, 1, iters=3000)
    st.set_phase(w, 2, iters=2000)
    if shots:
        st.page(w, "training")
        st.save(w, "tut19_training.png")
    run, _ = solve(w, "tut19", os.path.join(out, "tut19"), ea={'ref_formula': KOV_EXACT}, shots=shots,
                   result_tabs=("Overview", "Error analysis"))
    if shots and run:
        copy_fig(run, "*surface_comparison*u*.png", "tut19_surface_comparison_u.png")
        copy_fig(run, "*surface_comparison*p*.png", "tut19_surface_comparison_p.png")
    w.close()


# ── 20. Diagnosing a run that does not converge ──────────────────────────
DIFF_EXACT = "exp(-t)*sin(pi*x)"
DIFF_EA = {'ref_formula': DIFF_EXACT, 'ref_times': '0, 0.25, 0.5, 0.75, 1'}


def _diffusion(w):
    st.pick(w, "Forward", "Diffusion equation")
    st.set_phase(w, 1, iters=5000)


def tut20(out, shots):
    # A: the example as it is, shortened
    w = st.new_window()
    _diffusion(w)
    if shots:
        st.page(w, "equations")
        st.save(w, "tut20_equations.png")
    solve(w, "tut20_A_baseline", os.path.join(out, "tut20A"), ea=DIFF_EA, shots=shots,
          result_tabs=("Live loss", "Error analysis"))
    w.close()
    # B: a sign error in the residual (du_xx with the wrong sign: a backward heat equation)
    w = st.new_window()
    _diffusion(w)
    st.texts(w.pde_inputs, ["du_t + du_xx + exp(-t)*(sin(pi*x) - pi**2*sin(pi*x))"])
    run, _ = solve(w, "tut20_B_sign", os.path.join(out, "tut20B"), ea=DIFF_EA, shots=shots,
                   result_tabs=("Live loss", "Error analysis"))
    if shots and run:
        copy_fig(run, "*line_comparison*.png", "tut20_sign_line_comparison.png")
    w.close()
    # C: the conditions weighted far below the equation
    w = st.new_window()
    _diffusion(w)
    for k, v in (("bc_0", 0.001), ("ic_0", 0.001)):
        if k in w.weight_widgets:
            w.weight_widgets[k].setValue(v)
    w.sched_same_weights_cb.setChecked(True)
    st.settle(0.3)
    for k, v in (("bc_0", 0.001), ("ic_0", 0.001)):
        if k in w.weight_widgets:
            w.weight_widgets[k].setValue(v)
    if shots:
        st.page(w, "weights")
        st.save(w, "tut20_weights.png")
    run, _ = solve(w, "tut20_C_weights", os.path.join(out, "tut20C"), ea=DIFF_EA, shots=shots,
                   result_tabs=("Live loss", "Error analysis"))
    if shots and run:
        copy_fig(run, "*line_comparison*.png", "tut20_weights_line_comparison.png")
    w.close()
    # D: far too short a run, then the fix: an L-BFGS phase after it
    w = st.new_window()
    _diffusion(w)
    st.set_phase(w, 1, iters=300)
    solve(w, "tut20_D_short", os.path.join(out, "tut20D"), ea=DIFF_EA)
    w.close()
    w = st.new_window()
    _diffusion(w)
    st.set_phase(w, 1, iters=300)
    st.set_phase(w, 2, opt="lbfgs", iters=2000)
    if shots:
        st.page(w, "training")
        st.save(w, "tut20_training_lbfgs.png")
    solve(w, "tut20_E_short_lbfgs", os.path.join(out, "tut20E"), ea=DIFF_EA)
    w.close()


# ── 21. A 3D problem: Poisson in a sphere ────────────────────────────────
def tut21(out, shots):
    w = st.new_window()
    st.pick(w, "Forward", "3D Poisson (Sphere)")
    if shots:
        for key in ("equations", "geometry"):
            st.page(w, key)
            st.save(w, f"tut21_{key}.png")
    st.set_phase(w, 1, iters=3000)
    st.set_phase(w, 2, iters=1000)
    if shots:
        st.page(w, "results")
        st.save(w, "tut21_results_page.png")
    run, _ = solve(w, "tut21", os.path.join(out, "tut21"), shots=shots,
                   result_tabs=("Overview", "Error analysis"))
    if shots and run:
        copy_fig(run, "solution_plot.png", "tut21_solution_plot.png")
        copy_fig(run, "*comparison*.png", "tut21_comparison.png")
        print("files:", sorted(os.path.relpath(p, run) for p in glob.glob(os.path.join(run, "**", "*.png"),
                                                                      recursive=True)))
    w.close()


# ── 22. Designing a network as blocks ────────────────────────────────────
MSF = "Poisson 1D – multi-scale Fourier features"
MSF_ITERS = 5000


def _msf(w):
    st.pick(w, "Forward", MSF)
    st.set_phase(w, 1, iters=MSF_ITERS)


def _network_type(w, text):
    w.network_type_combo.setCurrentIndex(w.network_type_combo.findText(text))
    st.settle(0.4)


def _trained(run):
    """What the run really trained (model_config.json): the network type and the design's blocks."""
    import json
    p = glob.glob(os.path.join(run or "", "**", "model_config.json"), recursive=True)
    if not p:
        return "no model_config.json"
    c = json.load(open(p[0], encoding="utf-8"))
    d = c.get("network_design") or ""
    d = json.loads(d) if isinstance(d, str) and d else (d or {})
    return f"{c.get('network_type')} {[b['id'] for b in d.get('blocks', [])]}"


def tut22(out, shots):
    # A: the same problem with a plain fully connected network (the form's 3 x 100)
    w = st.new_window()
    _msf(w)
    _network_type(w, "FNN")
    if shots:
        st.page(w, "network")
        st.save(w, "tut22_fnn_form.png")
    run_a, _ = solve(w, "tut22_fnn", os.path.join(out, "tut22", "fnn"))
    print("   trained:", _trained(run_a))
    if shots and run_a:
        copy_fig(run_a, "*comparison*.png", "tut22_fnn_comparison.png")
    w.close()

    # B: Edit as blocks from the example's MsFFN: the same network, now as blocks
    w = st.new_window()
    _msf(w)
    st.page(w, "network")
    _network_type(w, "Designed")          # an empty design starts from the form's network (MsFFN), as Edit as blocks
    st.settle(0.6)
    lv = w.design_state.level()
    print("design:", [(b["id"], b["kind"], b.get("title")) for b in lv["blocks"]])
    if shots:
        w.designer_panel.canvas.refit()
        w.designer.select(next(b["id"] for b in lv["blocks"] if b["kind"] == "shared"))
        st.settle(0.6)
        st.save(w, "tut22_designer.png")
    run_b, _ = solve(w, "tut22_designed", os.path.join(out, "tut22", "designed"))
    print("   trained:", _trained(run_b))
    if shots and run_b:
        copy_fig(run_b, "*comparison*.png", "tut22_designed_comparison.png")

    # C: delete the sigma = 10 branch (its Fourier stack and the shared hidden layers it feeds)
    for b in list(lv["blocks"]):
        if b["kind"] == "stack" and any(ly.get("params", {}).get("sigma") == 10.0 for ly in b.get("layers", [])):
            w.designer.delete(b["id"])
    for b in list(w.design_state.level()["blocks"]):
        if b["kind"] == "shared":
            w.designer.delete(b["id"])
    st.settle(0.6)
    print("after deleting:", [(b["id"], b["kind"]) for b in w.design_state.level()["blocks"]],
          "|", w.designer_summary.text())
    if shots:
        w.right_tabs.setCurrentIndex(w._network_tab_index)   # (Solve left the Results tab open)
        st.settle(0.4)
        w.designer_panel.canvas.refit()
        w.designer.select(next(b["id"] for b in w.design_state.level()["blocks"] if b["kind"] == "merge"))
        st.settle(0.6)
        st.save(w, "tut22_one_branch.png")
    run_c, _ = solve(w, "tut22_one_branch", os.path.join(out, "tut22", "one_branch"))
    print("   trained:", _trained(run_c))
    if shots and run_c:
        copy_fig(run_c, "*comparison*.png", "tut22_one_branch_comparison.png")
    w.close()


if __name__ == "__main__":
    out = os.path.abspath(sys.argv[1])
    shots = "--shots" in sys.argv
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
    st.OUT = SHOTS
    for num, fn in (("19", tut19), ("20", tut20), ("21", tut21), ("22", tut22)):
        if only is None or num in only:
            fn(out, shots)
    print("done")
