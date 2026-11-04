set -u
# Table 2 (Section 5.1) repeated on PINNStudio v1.6.0 (PySide6): the five problems of the 594d295 study
# (benchmarks/performance/), each in the three execution modes (exported script, Solve script, GUI), seeds 1, 2, 3,
# with the order rotated per seed, and exactly the settings that study saved (benchmarks/performance/configs,
# loaded with perf_study.py overhead --configs). Same machine and Python environment as that study (runner venv:
# Python 3.11.5, PyTorch 2.6.0+cu124, DeepXDE 1.15.0) with PySide6-Essentials added in front; the GUI runs use the
# logged-in desktop session. Writes only inside this job folder (out/).
JOB="$PWD"; OUT="$JOB/out"; mkdir -p "$OUT"
C=6dfab1c
git clone -q https://github.com/AsfandyarKhan72/PINNStudio.git tree && cd tree && git checkout -q "$C" || exit 2
git log -1 --format='%H %ad %s' --date=iso | tee "$OUT/commit.txt"
# environment: a venv on the runner's base Python whose path also holds the runner venv's packages (PyTorch 2.6.0
# etc.), with PySide6-Essentials installed in the venv itself
python_base=$("$PY" -c "import sys;print(sys.base_prefix)")/bin/python3.11
"$python_base" -m venv "$JOB/pv" || exit 3
RSP=$("$PY" -c "import site;print(site.getsitepackages()[0])")
PSP=$("$JOB/pv/bin/python" -c "import site;print(site.getsitepackages()[0])")
echo "$RSP" > "$PSP/zz_runner_venv.pth"
"$JOB/pv/bin/python" -m pip install -q "PySide6-Essentials>=6.5,<6.12" 2>&1 | tail -2
P="$JOB/pv/bin/python"
"$P" -c "import sys,platform,os,torch,deepxde,numpy,PySide6;sys.path.insert(0,os.getcwd());import pinnstudio;print('Python',sys.version.split()[0],'| PyTorch',torch.__version__,'| CUDA',torch.version.cuda,'| DeepXDE',deepxde.__version__,'| NumPy',numpy.__version__,'| PySide6',PySide6.__version__,'| PINNStudio',pinnstudio.__version__,os.path.dirname(pinnstudio.__file__),'|',platform.platform())" | tee "$OUT/env.txt"
grep -q "PyTorch 2.6.0" "$OUT/env.txt" || { echo "not the PyTorch 2.6.0 environment of the earlier study"; exit 4; }
nvidia-smi --query-gpu=name,driver_version,temperature.gpu --format=csv | tee "$OUT/gpu.txt"
lscpu | grep -E "Model name|^CPU\(s\)" | tee "$OUT/cpu.txt"
U=$(id -u)
if [ -S "/run/user/$U/wayland-0" ]; then
  export XDG_RUNTIME_DIR="/run/user/$U" WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland
elif [ -S /tmp/.X11-unix/X0 ] || [ -S /tmp/.X11-unix/X1 ]; then
  [ -S /tmp/.X11-unix/X0 ] && export DISPLAY=:0 || export DISPLAY=:1
  XA=$(ls /run/user/$U/.mutter-Xwaylandauth.* /run/user/$U/gdm/Xauthority "$HOME/.Xauthority" 2>/dev/null | head -1)
  [ -n "$XA" ] && export XAUTHORITY="$XA"
  export QT_QPA_PLATFORM=xcb
else
  echo "no desktop session found: the window runs offscreen"
fi
echo "Qt platform for the window runs: ${QT_QPA_PLATFORM:-default} DISPLAY=${DISPLAY:-} WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-}" | tee "$OUT/qt_platform.txt"
date -Is > "$OUT/started.txt"
"$P" benchmarks/perf_study.py overhead "$OUT/overhead.jsonl" --configs benchmarks/performance/configs --seeds 1,2,3 2>&1 | grep --line-buffered -vE "propagateSizeHints|backends|recommended"
"$P" benchmarks/perf_study.py tables "$OUT" --out-dir "$OUT/tables" 2>&1 | tail -2
{ date -Is; echo "exit 0"; } | tee "$OUT/finished.txt"
