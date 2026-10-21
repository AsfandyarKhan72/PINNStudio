"""An edge can be taken out of any BC by clicking it, also when its Where was
typed or came with an example (Helmholtz 2D with a hole: "left or right or
bottom or top", np.isclose(x**2 + y**2, R**2)). Clicking used to only remove
text the sketch had written itself and otherwise said the Where "was written
by hand". Now the Where is written again from the edges it still picks; a
Where that picks only part of an edge is left alone; the last edge can be
taken out, and Solve then asks for edges."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

from pinnstudio.ui.main_window import MainWindow

app = QApplication.instance() or QApplication([])


def _picks(w, index):
    w._refresh_domain_view()
    app.processEvents()
    return sorted(k for k, lst in enumerate(w._gpanel().canvas.edge_bcs()) if index in lst)


def test_deselect_typed_where():
    w = MainWindow()
    w.quick_examples_combo.setCurrentText("Helmholtz 2D – square with hole")
    for _ in range(5):
        app.processEvents()
    w._refresh_domain_view()
    app.processEvents()
    edges = w._gpanel().canvas.edges()
    circle = [k for k, e in enumerate(edges) if e["kind"] == "arc"][0]
    sides = _picks(w, 1)
    assert len(sides) == 4 and _picks(w, 0) == [circle]

    w._on_sketch_toggle_edge(1, sides[2], add=False)          # take one side out of BC 2
    assert _picks(w, 1) == [k for k in sides if k != sides[2]]
    assert "rewritten from its edges" in w.log_box.toPlainText()
    w._on_sketch_toggle_edge(1, sides[2], add=True)           # and back in
    assert _picks(w, 1) == sides

    w._on_sketch_toggle_edge(0, circle, add=False)            # BC 1's only edge
    assert w.custom_bc_list[0]["location"].text() == "False" and _picks(w, 0) == []
    assert any("BC 1 picks no part" in e for e in w._build_config().validate())
    w._on_sketch_toggle_edge(0, circle, add=True)
    assert _picks(w, 0) == [circle]
    assert not any("picks no part" in e for e in w._build_config().validate())

    bottom = [k for k in sides if abs(edges[k]["mid"][1] + 0.5) < 1e-9][0]
    w.custom_bc_list[1]["location"].setText("np.isclose(y, -0.5) and x < 0")   # half of the bottom edge
    before = w.custom_bc_list[1]["location"].text()
    w._on_sketch_toggle_edge(1, bottom, add=False)
    assert w.custom_bc_list[1]["location"].text() == before
    assert "only part" in w._gpanel().hint.text()
    w.close()
