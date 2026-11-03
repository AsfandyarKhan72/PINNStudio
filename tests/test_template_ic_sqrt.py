"""A template's initial condition with a distance sqrt(...) keeps a small number under the root, so its
derivatives stay finite at the centre (Hammersley sampling puts a training point exactly there; with
the IC built into the network the residual was NaN from the first step)."""
import os
import re

SRC = os.path.join(os.path.dirname(__file__), "..", "pinnstudio", "ui", "main_window.py")


def test_ic_square_roots_have_an_offset():
    text = open(SRC, encoding="utf-8").read()
    ics = re.findall(r"'ic': \[(.*?)\]", text, re.S)
    bad = []
    for ic in ics:
        for m in re.finditer(r"sqrt\(\(x", ic):
            depth, j = 0, m.start() + 4
            while j < len(ic):
                depth += {"(": 1, ")": -1}.get(ic[j], 0)
                if depth == 0:
                    break
                j += 1
            if not re.search(r"\+\s*1e-\d+\s*$", ic[m.start() + 5:j]):
                bad.append(ic[m.start():j + 1])
    assert not bad, bad
