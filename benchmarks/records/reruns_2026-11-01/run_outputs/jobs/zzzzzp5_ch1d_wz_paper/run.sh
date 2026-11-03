set -u
# Approved rerun: Cahn-Hilliard 1D (Wight & Zhao) with its paper's settings (paper_wz), seeds 2026, 1, 2.
# Code = the benchmark code of the recorded runs (5d7c821) + only the resampled-window-model fix (63de1b4).
git clone -q --depth 1 -b rerun/ch1d-wz-resampled-model https://github.com/AsfandyarKhan72/PINNStudio.git tree && cd tree && git log --oneline -1
OUT="$REPO/.gpu_runner/done/p5_ch1d_wz_paper/out"; mkdir -p "$OUT"
F='^(OK|ERR|--|!!) |Traceback|Error|unknown'
"$PY" tools/tuning_study.py --gui --timeout 21600 --seeds 2026,1,2 --out "$OUT" --variants paper_wz --templates '1D Cahn-Hilliard' 2>&1 | grep --line-buffered -E "$F"
