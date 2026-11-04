set -u
# Section 5.2 on PINNStudio v1.6.0 (PySide6): Supplementary Table S3 (CPU, 8 threads, vs GPU Adam throughput, warm-up
# timed separately) and Table S4 (CPU-thread scaling, fresh process per thread count), repeated with the 2026-10-26
# study's settings (problems' configurations from benchmarks/performance/configs, the same defaults: --repeat 3,
# 200 warm-up and 2000 timed Adam steps, 300/500 for Allen-Cahn 3D/2D, 8 CPU threads; thread workloads from
# benchmarks/gpu_check.py), on the same machine and Python environment. No other job runs. Writes only into out/.
JOB="$PWD"; OUT="$JOB/out"; mkdir -p "$OUT"
C=239d925bcba1168c0fcd7f4ff6afb95475cbf1b7
git clone -q https://github.com/AsfandyarKhan72/PINNStudio.git tree && cd tree && git checkout -q "$C" || exit 2
git log -1 --format='%H %ad %s' --date=iso | tee "$OUT/commit.txt"
python_base=$("$PY" -c "import sys;print(sys.base_prefix)")/bin/python3.11
"$python_base" -m venv "$JOB/pv" || exit 3
RSP=$("$PY" -c "import site;print(site.getsitepackages()[0])")
PSP=$("$JOB/pv/bin/python" -c "import site;print(site.getsitepackages()[0])")
echo "$RSP" > "$PSP/zz_runner_venv.pth"
"$JOB/pv/bin/python" -m pip install -q "PySide6-Essentials>=6.5,<6.12" 2>&1 | tail -2
P="$JOB/pv/bin/python"
"$P" -c "import sys,platform,os,torch,deepxde,numpy,PySide6;sys.path.insert(0,os.getcwd());import pinnstudio;print('Python',sys.version.split()[0],'| PyTorch',torch.__version__,'| CUDA',torch.version.cuda,'| DeepXDE',deepxde.__version__,'| NumPy',numpy.__version__,'| PySide6',PySide6.__version__,'| PINNStudio',pinnstudio.__version__,os.path.dirname(pinnstudio.__file__),'|',platform.platform())" | tee "$OUT/env.txt"
grep -q "PyTorch 2.6.0" "$OUT/env.txt" || { echo "not the PyTorch 2.6.0 environment of the earlier study"; exit 4; }
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw --format=csv | tee "$OUT/gpu.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee "$OUT/gpu_processes_at_start.txt"
lscpu | grep -E "Model name|^CPU\(s\)" | tee "$OUT/cpu.txt"
uptime | tee "$OUT/load_at_start.txt"
date -Is > "$OUT/started.txt"
export QT_QPA_PLATFORM=offscreen
"$P" benchmarks/perf_study.py throughput "$OUT/throughput.jsonl" --repeat 3 --configs benchmarks/performance/configs 2>&1 | grep --line-buffered -vE "propagateSizeHints|backends|recommended"
uptime | tee "$OUT/load_between.txt"
"$P" benchmarks/perf_study.py threads "$OUT/threads.jsonl" --repeat 3 2>&1 | grep --line-buffered -vE "propagateSizeHints|backends|recommended"
"$P" benchmarks/perf_study.py tables "$OUT" --out-dir "$OUT/tables" 2>&1 | tail -2
{ date -Is; echo "exit 0"; } | tee "$OUT/finished.txt"
