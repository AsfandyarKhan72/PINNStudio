# Rerun 1: complete per-time output of the four windowed Figure 7(b) runs

These files are copied unchanged from the GPU workstation, from job `zzzzzp1_ch_f7_1d`. The job ran from 2026-10-31 19:38 to 20:31 local time.

**Job details:**
- **Code:** `79edd5d` on `audit/v1.6.0`, as printed by the job; see `job/output.log`.
- **Exit code:** 0. **Run time:** 3,161 s.
- **Hardware:** one RTX 4090.
- **Job script:** `job/run.sh`.

**Run settings:**
- **Problem:** "Cahn-Hilliard 1D – Wight & Zhao".
- **Seed:** 2026.
- **Runs:** `windows`, `bcpinn`, `windowsEq`, `bcpinnEq`, as defined in `benchmarks/plans/fig7_v2.json`.
- **Not retrained:** the matched `plain` run was not affected by the fix. The job wrote it only as a placeholder (`skipped`).

**Copied, per run:**

| File | Contents |
|---|---|
| `error_metrics_u.txt`, `error_metrics_mu.txt` | relative L2 and mean absolute error of u and μ at t = 0, 0.25, 0.5, 0.75, 1, written by the run's own error analysis |
| `error_spacetime_u.txt`, `error_spacetime_mu.txt` | the error over all reference times together |
| `result.json` | return code, training seconds, and the per-time L2 values from the log (5 significant figures) |
| `meta.json` | the run's full settings |
| `log.txt` | the training log |

**Workstation source paths:**
- `.gpu_runner/done/zzzzzp1_ch_f7_1d/out/f7v2_ch1d_<run>/` holds the run folders. That folder also keeps the model files and plots, which are not copied here.
- `.gpu_runner/done/zzzzzp1_ch_f7_1d.1793478678/` holds `run.sh` and `output.log`.

The summary file `../rerun1_fig7_ch1d_mg_per_time.json` holds only the all-times errors; it is also unchanged. Despite its name, these are the Wight & Zhao runs.
