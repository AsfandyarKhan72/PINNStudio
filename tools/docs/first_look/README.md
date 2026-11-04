# Home page, "A first look": where the six pictures come from

The six pictures in `website/assets/images/home/first-look/` are representative results, each the solution figure
that PINNStudio's own Solve script wrote in one run of a built-in example. They illustrate what PINNStudio
produces; they are not the benchmark records, and the home page does not report numbers from them. The benchmark
results of each example are on its example page and in `benchmarks/examples.md`.

| Card | File | Example and settings loaded | Script |
|---|---|---|---|
| Lorenz system | `lorenz-inverse.gif` | Inverse: Lorenz system; its own settings (no Tuned preset) | `gallery_anim.py` |
| Burgers equation | `burgers-1d.gif` | 1D Burgers (Raissi et al.); Tuned = Raissi et al.'s settings | `gallery_anim.py` |
| Schrödinger equation | `schrodinger-1d.gif` | Schrödinger equation; its own settings (no Tuned preset) | `gallery_anim.py` |
| Navier–Stokes, Kovasznay flow | `navier-stokes-2d-kovasznay.png` | Kovasznay flow; its own settings (no Tuned preset) | `gallery_anim_kovasznay.py` |
| Allen–Cahn equation | `allen-cahn-2d-mattey-ghosh.gif` | 2D Allen-Cahn (Mattey & Ghosh); Tuned = exact IC and periodicity, whole interval, SA-PINN, 3× iterations | `gallery_anim.py` |
| Heat equation in 3D | `heat-3d.gif` | 3D Heat; its own settings (no Tuned preset) | `gallery_anim.py` |

**How they were made.** Each script loads the example with *Settings: Tuned* exactly as the window does (the
example's own settings where it has no Tuned entry), sets the plot type to the animation shown (a static surface
for the steady Kovasznay flow), sets the seed to 2026 and runs the script `generate_script` writes, once, on the GPU.
Animation settings: 30 time frames, 100 dpi, 60 parameter frames; nothing else is changed. The runs were made on
commit `b7f2108`, on an RTX 4090, in an environment with Python 3.11.5, PyTorch 2.5.1+cu121, DeepXDE 1.15.0 and
PySide6 6.11.3. The figures were then scaled to 480 px wide (360 px for the tall Lorenz figure) and saved with a
smaller colour palette. `summary_animations.json` and `summary_kovasznay.json` are the scripts' own summaries
(settings message, run time, the last errors printed and, for Lorenz, the recovered coefficients).

To make them again from a checkout (needs a GPU for the times above):

```bash
QT_QPA_PLATFORM=offscreen python tools/docs/first_look/gallery_anim.py /tmp/first_look_runs
QT_QPA_PLATFORM=offscreen python tools/docs/first_look/gallery_anim_kovasznay.py /tmp/first_look_runs_kovasznay
```

**Relation to the benchmark records.** The settings and the seed are those of the benchmark run of each example, but
the runs are separate: another commit and another PyTorch build, and training on a GPU is not bit-reproducible
across builds. Their results therefore differ slightly from the seed-2026 benchmark record of the same settings.
For the Lorenz card, the run recovered C1 = 10.0034, C2 = 14.9996, C3 = 2.66643 (true 10, 15, 8/3; worst error
0.034 %), while the seed-2026 benchmark record has C1 = 10, C2 = 14.9998, C3 = 2.66664; the record's seeds 1 and 2 give
C1 = 10.0038 and 9.99682, so the card's values are within the spread of the benchmark's own seeds
(`benchmarks/tables/inverse_per_seed.csv`, records in `benchmarks/records/corrected_2026-10-31/seeds3.jsonl`).
