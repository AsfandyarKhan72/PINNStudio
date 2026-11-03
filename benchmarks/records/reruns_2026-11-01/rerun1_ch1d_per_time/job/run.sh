set -u
git clone -q --depth 1 -b audit/v1.6.0 https://github.com/AsfandyarKhan72/PINNStudio.git tree && cd tree && git log --oneline -1
# P0 rerun (audit/v1.6.0): Figure 7, 1D Cahn-Hilliard time-window runs; their weights were reset to 1 after window 1.
# The matched plain run is not affected: its scripts are built (the windows runs copy its settings) but it is not trained.
sub(){ "$PY" - "$1" "$2" <<'EOF'
import json, sys
p = json.load(open("benchmarks/plans/fig7_v2.json"))
pre = sys.argv[1]
keep = [e for e in p if e["name"].startswith(pre) and e["name"].split("_")[-1] in ("plain", "windows", "bcpinn", "windowsEq", "bcpinnEq")]
json.dump(keep, open(sys.argv[2], "w"), indent=1)
print([e["name"] for e in keep])
EOF
}
OUT="$REPO/.gpu_runner/done/zzzzzp1_ch_f7_1d/out"; mkdir -p "$OUT/f7v2_ch1d_plain"
echo '{"skipped": "not affected by the fix; the Figure 7 value is kept from a08_fig7v2"}' > "$OUT/f7v2_ch1d_plain/result.json"
sub f7v2_ch1d_ plan.json
"$PY" benchmarks/study_runner.py plan.json "$OUT" --jobs 1 2>&1 | grep --line-buffered -E "^(==|--) |Error|Traceback|Killed|NOT BUILT"
