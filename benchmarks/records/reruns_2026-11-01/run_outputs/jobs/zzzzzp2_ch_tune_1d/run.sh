set -u
git clone -q --depth 1 -b audit/v1.6.0 https://github.com/AsfandyarKhan72/PINNStudio.git tree && cd tree && git log --oneline -1
# P0 rerun (audit/v1.6.0): 1D Cahn-Hilliard (Mattey & Ghosh) -- paper setup (Table S2) and the best-setup variants.
OUT="$REPO/.gpu_runner/done/p2_ch_tune_1d/out"; mkdir -p "$OUT"
F='^(OK|ERR|--|!!) |Traceback|Error|unknown'
"$PY" tools/tuning_study.py --gui --out "$OUT" --timeout 21600 --variants paper_mg --templates '1D Cahn-Hilliard (Mattey & Ghosh)' 2>&1 | grep --line-buffered -E "$F"
"$PY" tools/tuning_study.py --gui --out "$OUT" --timeout 14400 --best --variants base,iters,net,windows2,all --templates '1D Cahn-Hilliard (Mattey & Ghosh)' 2>&1 | grep --line-buffered -E "$F"
