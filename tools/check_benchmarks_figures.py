"""Check benchmarks/README.md: every example shows figures for its forward and inverse results, a
parameter plot for inverse problems, and a comparison for every output; prints what is missing."""
import re, os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
t = open("benchmarks/README.md").read()
probs = []
for sec in re.split(r'<details id="bench-', t)[1:]:
    name = re.search(r"<strong>(.*?)</strong>", sec).group(1)
    has_f, has_i = "**Forward problem**" in sec, "**Inverse problem**" in sec
    fwd = re.search(r"Forward results — figures</em></summary>(.*?)</details>", sec, re.S)
    inv = re.search(r"Inverse results — figures</em></summary>(.*?)</details>", sec, re.S)
    # outputs from the forward table header
    hdr = re.search(r"\| Training \| (.*?) \|\n", sec)
    outs = re.findall(r"L₂ (\S+?)(?:,| \|| $)", hdr.group(1)) if hdr else []
    def imgs(m):
        return re.findall(r"!\[[^\]]*\]\(([^)]+)\)", m.group(1)) if m else []
    if has_f:
        fi = imgs(fwd)
        if not fi:
            probs.append(f"{name}: forward — no figures")
        else:
            blocks = re.split(r"\n\*[^*\n]+:\*\n", "\n" + fwd.group(1))
            for b in blocks[1:]:
                if not re.findall(r"!\[", b):
                    probs.append(f"{name}: forward — a block without figures")
        for p in fi:
            if not os.path.exists(os.path.normpath(os.path.join("benchmarks", p.replace("%20", " ")))):
                probs.append(f"{name}: missing file {p}")
    if has_i:
        ii = imgs(inv)
        if not ii:
            probs.append(f"{name}: inverse — no figures")
        elif not any(re.search(r"param", os.path.basename(p)) for p in ii):
            probs.append(f"{name}: inverse — no parameter plot")
        if ii and not any(re.search(r"comparison|view3d", os.path.basename(p)) for p in ii):
            probs.append(f"{name}: inverse — no solution comparison")
        for p in ii:
            if not os.path.exists(os.path.normpath(os.path.join("benchmarks", p.replace("%20", " ")))):
                probs.append(f"{name}: missing file {p}")
print(len(probs)); print("\n".join(probs))
# every output's comparison in each forward block
SYM = {"μ": "mu", "φ": "phi", "σₓₓ": "sxx", "σᵧᵧ": "syy", "σₓᵧ": "sxy", "uₓ": "ux", "uᵧ": "uy"}
more = []
for sec in re.split(r'<details id="bench-', t)[1:]:
    name = re.search(r"<strong>(.*?)</strong>", sec).group(1)
    hdr = re.search(r"\*\*Forward problem\*\*\n\n\| Training \| (.*?) \|\n", sec)
    fwd = re.search(r"Forward results — figures</em></summary>(.*?)</details>", sec, re.S)
    if not hdr or not fwd:
        continue
    outs = [SYM.get(o, o) for o in re.findall(r"L₂ ([^\s,|]+)(?:, last time)? ", hdr.group(1) + " ") if o not in ("u",) or True]
    outs = [o for o in outs if not o.startswith("L")]
    if len(set(outs)) < 2:
        continue
    for b in re.split(r"\n\*[^*\n]+:\*\n", "\n" + fwd.group(1))[1:]:
        files = [os.path.basename(p) for p in re.findall(r"\]\(([^)]+)\)", b)]
        for o in set(outs):
            if not any(re.search(rf"_{o}\.(png|gif)$", f) for f in files) and not (o == outs[0] and any(not re.search(r"_[a-z]+\.(png|gif)$", f.split("comparison")[-1]) for f in files)):
                more.append(f"{name}: forward block without {o}: {files}")
print(len(more)); print("\n".join(more[:30]))
