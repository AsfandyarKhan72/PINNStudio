"""The Qt binding must not lose references to None, True or False.

PySide6 6.12.0 drops one reference to None on every QTimer.singleShot(ms,
callable) and one reference to True on every Python-side signal emit(). From
Python 3.12 on these objects are immortal and nothing happens; on Python 3.10
and 3.11 their count reaches zero within a few thousand calls and the process
aborts ("Fatal Python error: none_dealloc: deallocating None" /
"bool_dealloc"). The Smoke test (Python 3.11) aborted this way in
test_training_methods.py. setup.py therefore keeps PySide6 below 6.12 on
Python < 3.12; this test fails if an installed binding has the defect again.

Run with pytest or as a script; the counting runs in a fresh interpreter.
"""
import os
import subprocess
import sys
import textwrap

CHILD = textwrap.dedent(r"""
    import os, sys, gc
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication, QObject, QTimer, Signal
    app = QCoreApplication.instance() or QCoreApplication([])

    class O(QObject):
        sig = Signal()

    o = O()
    o.sig.connect(lambda: None)
    N = 300
    out = {}
    for name, step in (("singleShot", lambda: QTimer.singleShot(0, lambda: None)), ("emit", o.sig.emit)):
        gc.collect()
        before = {k: sys.getrefcount(v) for k, v in (("None", None), ("True", True), ("False", False))}
        for _ in range(N):
            step()
        for _ in range(5):
            app.processEvents()
        gc.collect()
        out[name] = {k: sys.getrefcount(v) - before[k] for k, v in (("None", None), ("True", True), ("False", False))}
    print(repr(out), flush=True)
    os._exit(0)
""")


def test_singleshot_and_emit_keep_none_and_bool_references():
    if sys.version_info >= (3, 12):
        print("SKIPPED: None, True and False are immortal from Python 3.12 on (nothing to count)")
        return
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (root, os.environ.get("PYTHONPATH", "")) if p))
    r = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=300, env=env, cwd=root)
    assert r.returncode == 0, f"the counting process ended with {r.returncode}:\n{r.stdout}\n{r.stderr[-3000:]}"
    deltas = eval(r.stdout.strip().splitlines()[-1])
    # a few references come and go with the lambdas themselves; the defect is one per call (300)
    bad = {call: d for call, d in deltas.items() if min(d.values()) < -50}
    assert not bad, (f"the Qt binding lost references per call (300 calls): {bad} -- with this binding "
                     "Python 3.10/3.11 aborts (none_dealloc / bool_dealloc); PySide6 6.12.0 has this defect")


if __name__ == "__main__":
    test_singleshot_and_emit_keep_none_and_bool_references()
    print("ALL BINDING REFCOUNT TESTS PASSED")
