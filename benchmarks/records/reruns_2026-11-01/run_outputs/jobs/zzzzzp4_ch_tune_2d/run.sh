set -u
git clone -q --depth 1 -b audit/v1.6.0 https://github.com/AsfandyarKhan72/PINNStudio.git tree && cd tree && git log --oneline -1
# P0 rerun (audit/v1.6.0): 2D Cahn-Hilliard paper setups -- Wight & Zhao (Figure 4, seeds 2026, 1, 2) and Mattey & Ghosh.
OUT="$REPO/.gpu_runner/done/p4_ch_tune_2d/out"; mkdir -p "$OUT"
F='^(OK|ERR|--|!!) |Traceback|Error|unknown'
"$PY" tools/tuning_study.py --gui --out "$OUT" --timeout 64800 --seeds 2026,1,2 --variants paper_wz --templates '2D Cahn-Hilliard (Wight & Zhao)' 2>&1 | grep --line-buffered -E "$F"
"$PY" tools/tuning_study.py --gui --out "$OUT" --timeout 64800 --variants paper_mg --templates '2D Cahn-Hilliard (Mattey & Ghosh)' 2>&1 | grep --line-buffered -E "$F"
