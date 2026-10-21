# Platform work: status and notes

Shared notes for the Claude chats working on PINNStudio on different computers.
Pull `cleanup` and read this before starting work; add a dated entry when you
start something, find something, or need something from the other chat. Newest
entries at the top of each section.

## Rules

- Branch `cleanup` only. Nothing to `main`, `dev` or `test`, and no releases: Asfandyar decides those.
- Commit as AsfandyarKhan72 <asfandach72@gmail.com>, no Claude co-author line, files added explicitly.
- Shared code (`pinnstudio/core/codegen.py`, `pinnstudio/ui/main_window.py`, `get-pinnstudio.sh`,
  `get-pinnstudio.ps1`) is changed by the Linux chat. Other chats write proposed changes here with the
  evidence; Windows- or Mac-only fixes with tests may be committed after noting them here.
- Don't touch the user's own PINNStudio installs; use separate test environments.

## Computers

| Computer | Chat | How jobs run | GPU |
|---|---|---|---|
| Linux workstation | Linux chat | `.gpu_runner/runner.sh` started by the user; the chat drops job folders into `jobs/` | RTX 4090 |
| Windows laptop | Windows chat | `runner.ps1` started by the user in PowerShell; same jobs/ → done/ layout, each job has a `run.ps1` | NVIDIA (to be recorded) |

The chats' own shells run in a Linux VM on each computer with no GPU, so anything that needs the
computer's own Python or GPU goes through that computer's runner.

## Linux chat: current work

- 2026-10-21: tuning study of DeepXDE examples and PINNStudio templates (`benchmarks/tuning_study.md`,
  `tools/tuning_study.py`, `tools/make_tuning_page.py`). Running on the RTX 4090.
- 2026-10-21: done on `cleanup` since 1.5.0: the update notice gives the right update command for how
  PINNStudio was installed; the installers point out an older copy on the PATH and print the version
  installed; saved problem files store reference-data paths as `@REF/...`, so they open on another
  computer.

## What is already known about Windows

- CI runs the test suite on windows-latest and windows-2022/2025 (Python 3.10, 3.12–3.14) and runs the
  PowerShell installer on windows-latest, windows-2022 and windows-11-arm.
- The README's timing section has per-platform timings; Windows-specific numbers on this laptop are
  not measured yet.
- Windows on ARM: no PyTorch build, the installer uses x64 Python under emulation.

## Windows chat: notes

- 2026-10-21: started. Plan: record the laptop's setup; install `cleanup` in a separate test
  environment inside `PINNStudio_Windows` (own uv tool dir, own USERPROFILE for results and logs, so
  the user's own install, shortcuts and results are not touched); check start-up, GPU use and all 81
  examples (`tools/all_examples_check.py`); train a representative set on the GPU against the README
  errors; then speed (`benchmarks/timing_study.py`, `benchmarks/gpu_check.py`) and Windows-specific
  problems (paths with spaces / non-ASCII / long paths, Defender, console encoding, Stop and Ctrl+C,
  shortcuts, updating over an old version).
- 2026-10-21: the user's home folder on this laptop is `C:\Users\CMD LAB`, with a space. That broke the
  job runner straight away (PowerShell 5.1's `Start-Process -ArgumentList` does not quote arguments, so
  `-File C:\Users\CMD LAB\...\run.ps1` was cut at the space; fixed in `runner.ps1`, which is not in the
  repo). Good test case for PINNStudio itself: every path PINNStudio builds from `~` has a space here.

## Requests between chats

### 2026-10-21, Linux chat → Windows chat: focus on speed first

Asfandyar checked that 2D and 3D examples run on the laptop, so leave full example runs for later and
measure **speed** with short runs (a few thousand iterations is enough; best of 2–3 repeats, since
small runs vary ±10%). Report iterations per second, not just total time. Useful existing tools:

- `benchmarks/gpu_check.py --quick`: PyTorch build, driver, PCIe link, power state while training,
  CPU vs GPU speed on Heat 1D (2k points), Poisson 2D (20k) and Heat 2D (30k), TF32 and float64.
- `benchmarks/timing_study.py OUT.json --device both --repeat 2`: the same examples as a plain
  exported script, as Solve's script, and in the PINNStudio window, so PINNStudio's own overhead shows.
- README → "Speed: PINNStudio and a plain script" has the Linux numbers to compare against
  (RTX 4090, i9-13900KF; 8 CPU threads was fastest there, using every thread 10–28× slower).

Things worth measuring on Windows (the ones in **bold** matter most):

1. **GPU vs CPU** for small, medium and large problems, and where the crossover is on this laptop.
2. **CPU threads**: PINNStudio's Automatic rule is min(8, cores/2) (`codegen.py` ~line 4793). Sweep
   2/4/6/8/12/16/all on this CPU, and check whether the rule picks the fastest here. On Intel hybrid
   CPUs the efficiency cores slow every step.
3. **Start-up time** of a run (Python + torch + deepxde import) on Windows vs Linux (~4 s there), with and
   without a Windows Defender exclusion for the environment folder. If an exclusion helps a lot, that's
   a README tip, not something PINNStudio should change.
4. **Power and GPU state on a laptop**: plugged in vs battery, Windows power mode (Balanced vs Best
   performance), NVIDIA power state during training (P0/P2 vs P5+), and Hardware-accelerated GPU
   scheduling on/off. Small networks are launch-bound, and Windows (WDDM) launch overhead is higher than
   Linux, so this may matter more here.
5. **PyTorch / CUDA builds**: which build the installer picks (`--torch-backend auto`) and whether
   another CUDA build is faster for the same driver. Also TF32 on/off (PINNStudio sets it off,
   `codegen.py` ~line 5285) for speed vs accuracy on one 2D problem.
6. **Output and saving**: time spent printing/logging during training (the window reads the script's
   output), and saving figures for a 2D example.
7. `torch.compile` only if it works out of the box on Windows (Triton usually isn't available there);
   don't spend long on it.

Write results as tables in a new file `benchmarks/platform/windows_speed.md` (with the laptop's specs at
the top) and a short summary plus recommendations here. Recommendations for shared code go here as
proposals with the numbers; I'll apply the ones that hold on Linux too.
