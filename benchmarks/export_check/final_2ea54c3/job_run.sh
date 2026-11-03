set -u
# Export equivalence (Solve script vs exported standalone script) on the integrated v1.6.0 candidate, for the CPC
# manuscript. Same tool, settings and machine as the 89-check record (benchmarks/export_check, af321c1): every
# example in forward and, where it exists, inverse mode; Tuned settings; Adam phases 200 steps, L-BFGS 50; same
# seed; CPU. The commit to test is in ./commit. Writes only inside this job folder (out/).
# The earlier records stay untouched: done/a09_eqcheck/out (the 88-check run) and benchmarks/export_check (89).
JOB="$PWD"; OUT="$JOB/out"; mkdir -p "$OUT"
C="$(cat "$JOB/commit")"
git clone -q https://github.com/AsfandyarKhan72/PINNStudio.git tree && cd tree && git checkout -q "$C" || exit 2
git log -1 --format='%H %ad %s' --date=iso | tee "$OUT/commit.txt"
"$PY" -m venv --system-site-packages "$JOB/pv" && "$JOB/pv/bin/python" -m pip install -q "PySide6-Essentials>=6.5,<6.12" 2>&1 | tail -2
P="$JOB/pv/bin/python"
"$P" -c "import sys,platform,os,torch,deepxde,numpy,PySide6,pinnstudio; print('Python',sys.version.split()[0],'| PyTorch',torch.__version__,'| DeepXDE',deepxde.__version__,'| NumPy',numpy.__version__,'| PySide6',PySide6.__version__,'| PINNStudio',pinnstudio.__version__,os.path.dirname(pinnstudio.__file__),'|',platform.platform(),'| CPUs',os.cpu_count(),'| torch threads',torch.get_num_threads())" | tee "$OUT/env.txt"
date -Is > "$OUT/started.txt"
"$P" tools/check_export_equivalence.py --out "$OUT/run" --timeout 3600 --device cpu 2>&1 | grep --line-buffered -vE "propagateSizeHints" | tee "$OUT/equivalence.log"
rc=${PIPESTATUS[0]}
{ date -Is; echo "exit $rc"; } | tee "$OUT/finished.txt"
exit $rc
