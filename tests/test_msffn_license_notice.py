"""The multi-scale Fourier feature network class is under the LGPL-2.1 (pinnstudio/core/msffn_lgpl.py): its file
carries the notice, the license text is in LICENSES/, and every generated script that uses MsFFN or STMsFFN gets
the class with its notice, while scripts with other networks get neither."""
import os
import re

from pinnstudio.core import codegen
from pinnstudio.core.config import PINNConfig
from pinnstudio.core.msffn_lgpl import MSFFN_CODE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTICE = "# ---- LGPL-2.1 section: class _MsFFN (multi-scale Fourier feature networks) ----"
END = "# ---- end of the LGPL-2.1 section ----"


def _cfg(network_type, steady=True):
    c = PINNConfig()
    for k, v in dict(problem_dim="1D", steady_state=steady, geometry_type="Interval", x_min=0.0, x_max=1.0,
                     pde_expressions="du_xx + pi**2*sin(pi*x)" if steady else "du_t - 0.1*du_xx",
                     network_type=network_type, msffn_sigmas="1, 5", msffn_sigmas_t="1, 5", layers=[1, 32, 32, 1],
                     iterations=10).items():
        setattr(c, k, v)
    return c


def test_module_file_and_license_text():
    src = open(os.path.join(ROOT, "pinnstudio", "core", "msffn_lgpl.py"), encoding="utf-8").read()
    assert src.startswith("# SPDX-License-Identifier: LGPL-2.1-only")
    assert "LICENSES/LGPL-2.1.txt" in src and "deepxde/nn/pytorch/msffn.py" in src
    text = open(os.path.join(ROOT, "LICENSES", "LGPL-2.1.txt"), encoding="utf-8").read()
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in text and "Version 2.1, February 1999" in text
    assert codegen._MSFFN_CODE is MSFFN_CODE
    assert MSFFN_CODE.lstrip("\n").startswith(NOTICE) and MSFFN_CODE.rstrip().endswith(END)


def test_generated_scripts_carry_the_notice():
    for nt in ("MsFFN", "STMsFFN"):
        cfg = _cfg(nt, steady=(nt == "MsFFN"))
        for gen in (codegen.generate_script, codegen.generate_clean_script):
            script = gen(cfg)
            assert script.count("class _MsFFN(") == 1, (nt, gen.__name__)
            i, j, k = script.index(NOTICE), script.index("class _MsFFN("), script.index(END)
            assert i < j < k, (nt, gen.__name__)
            assert "lgpl-2.1.html" in script[i:j]


def test_other_networks_get_no_lgpl_code():
    for nt in ("FNN", "PFNN"):
        for gen in (codegen.generate_script, codegen.generate_clean_script):
            script = gen(_cfg(nt))
            assert "class _MsFFN(" not in script and not re.search("LGPL-2.1 section", script), (nt, gen.__name__)
