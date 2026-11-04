"""Static figure for the website's "A first look" section (Navier-Stokes 2D, Kovasznay flow; a steady problem), made by PINNStudio itself: each example is loaded with its
Tuned settings (Settings: Tuned, as the window loads it; the example's own settings where it has no Tuned entry),
its plot type is set to one of PINNStudio's GIF animations, and Solve's script (generate_script) is run once on the
GPU with seed 2026. Nothing else is changed. Writes <out>/<slug>/ (run.py, log, the run's output folder) and
<out>/summary.json (example, settings message, plot type, final errors from the log, the GIF path)."""
import json, os, subprocess, sys, time, glob, re
sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tuning_study as ts
from pinnstudio.core import deepxde_examples as dx, example_settings as es
from pinnstudio.core.codegen import generate_script
OUT = os.path.abspath(sys.argv[1]); os.makedirs(OUT, exist_ok=True)
CASES = [  # (key, template?, inverse, plot type)
    ("Kovasznay flow", False, False, "Surface"),
]
# the same animation settings for every case (frames and resolution only; the training is the example's own)
ANIM = dict(num_timesteps_anim=30, plot_dpi_anim=100, param_anim_frames=60)
summary = []
for key, tmpl, inv, plot in CASES:
    if tmpl:
        base = ts.template_config(key, inv)
    else:
        ex = dx.BY_NAME[key]
        base = dx.build_config(ex, ts.REF_ROOT, inverse=inv and ex.inverse is not None)
    cfg, msg = es.settings_for(base, key, inv, es.TUNED)
    cfg.plot_type = plot
    for k, v in ANIM.items():
        setattr(cfg, k, v)
    d = os.path.join(OUT, re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_"))
    os.makedirs(d, exist_ok=True)
    cfg.save_dir = d
    cfg.random_seed = 2026
    open(os.path.join(d, "run.py"), "w", encoding="utf-8").write(generate_script(cfg))
    t0 = time.time()
    p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=3600)
    open(os.path.join(d, "log.txt"), "w", encoding="utf-8").write(p.stdout + "\n--- stderr ---\n" + p.stderr)
    gifs = sorted(glob.glob(os.path.join(d, "**", "*.gif"), recursive=True))
    l2 = re.findall(r"L(?:2|₂)=([\d.e+-]+)", p.stdout)
    rec = dict(key=key, inverse=inv, settings=msg or "own settings", plot_type=plot, rc=p.returncode,
               seconds=round(time.time() - t0, 1), gifs=[os.path.relpath(g, OUT) for g in gifs], last_l2=l2[-6:],
               params=re.findall(r"(C\d)\s*[=:]\s*([\d.]+)", p.stdout)[-3:])
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    summary.append(rec)
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, ensure_ascii=False)
