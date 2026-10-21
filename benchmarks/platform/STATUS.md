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

(none yet)

## Requests between chats

(none yet)
