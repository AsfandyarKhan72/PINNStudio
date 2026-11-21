"""Copy and save what the Results area shows: a right-click menu on saved
figures (copy, save as, open in the viewer, show in folder), on live
matplotlib figures (copy, save as PNG/PDF/SVG) and on tables (copy as
tab-separated text that pastes into a spreadsheet, export CSV)."""
import csv
import io
import os
import shutil

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication, QImage
from PySide6.QtWidgets import QFileDialog, QMenu


# ── saved figures (files) ────────────────────────────────────────────────
def copy_image_file(path):
    img = QImage(path)
    if img.isNull():
        return False
    QGuiApplication.clipboard().setImage(img)
    return True


def save_image_file(path, parent=None, target=None):
    """Save a copy of a figure file; a different image extension converts it."""
    if not path or not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower() or ".png"
    if target is None:
        filters = "PNG image (*.png);;JPEG image (*.jpg)" if ext == ".png" else f"{ext[1:].upper()} (*{ext})"
        target, _ = QFileDialog.getSaveFileName(parent, "Save image as", os.path.basename(path), filters)
    if not target:
        return None
    if not os.path.splitext(target)[1]:
        target += ext
    if os.path.splitext(target)[1].lower() == ext:
        if os.path.abspath(target) != os.path.abspath(path):
            return write_replacing(target, lambda tmp: shutil.copyfile(path, tmp), parent)
        return target
    ok = []
    out = write_replacing(target, lambda tmp: ok.append(QImage(path).save(tmp)), parent)
    return out if (out and all(ok)) else None


def write_replacing(target, write, parent=None):
    """Write `target` through a temporary file beside it, then put it in place: a file open in another
    program (a PDF viewer, Excel) is then never left half-written. On Windows such a file cannot be
    replaced at all: the old one is kept and the user is told to close it there. `write(tmp_path)` makes
    the file. Returns the target, or None when it could not be written."""
    import tempfile
    import time
    folder = os.path.dirname(os.path.abspath(target))
    fd, tmp = tempfile.mkstemp(prefix=".pinnstudio-", suffix=os.path.splitext(target)[1], dir=folder)
    os.close(fd)
    try:
        write(tmp)
        for attempt in range(5):          # a scanner or the indexer may hold the new file for a moment
            try:
                os.replace(tmp, target)
                return target
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.2)
    except PermissionError:
        _remove_quietly(tmp)
        if parent is not None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(parent, "Not saved",
                                f"{os.path.basename(target)} could not be replaced: it is open in another "
                                "program. Close it there and save again; the old file is unchanged.")
        return None
    except BaseException:
        _remove_quietly(tmp)
        raise


def _remove_quietly(path):
    import time
    for _ in range(5):
        try:
            os.remove(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.2)


def show_in_folder(path):
    if path:
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(os.path.abspath(path))))


def image_menu(parent, path, open_viewer=None):
    """The right-click menu for a saved figure."""
    menu = QMenu(parent)
    ok = bool(path) and os.path.exists(path)
    a = menu.addAction("Copy image")
    a.setEnabled(ok)
    a.triggered.connect(lambda: copy_image_file(path))
    a = menu.addAction("Save image as…")
    a.setEnabled(ok)
    a.triggered.connect(lambda: save_image_file(path, parent))
    if open_viewer is not None:
        a = menu.addAction("Open in viewer")
        a.setEnabled(ok and not path.lower().endswith(".gif"))
        a.triggered.connect(open_viewer)
    a = menu.addAction("Show in folder")
    a.setEnabled(ok)
    a.triggered.connect(lambda: show_in_folder(path))
    return menu


def attach_image_menu(widget, path_fn, open_viewer=None):
    """Right-click on `widget` -> the figure menu for the file `path_fn()`."""
    widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def show(pos):
        menu = image_menu(widget, path_fn(), open_viewer)
        menu.exec(widget.mapToGlobal(pos))
    widget.customContextMenuRequested.connect(show)


# ── live matplotlib figures ──────────────────────────────────────────────
def copy_figure(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    img = QImage.fromData(buf.getvalue(), "PNG")
    if img.isNull():
        return False
    QGuiApplication.clipboard().setImage(img)
    return True


def save_figure(fig, parent=None, name="figure.png", target=None, dpi=200):
    if target is None:
        target, _ = QFileDialog.getSaveFileName(parent, "Save figure as", name,
                                                "PNG image (*.png);;PDF (*.pdf);;SVG (*.svg)")
    if not target:
        return None
    if not os.path.splitext(target)[1]:
        target += ".png"
    return write_replacing(target, lambda tmp: fig.savefig(tmp, dpi=dpi, facecolor=fig.get_facecolor()), parent)


def attach_figure_menu(canvas, fig_fn, name="figure.png"):
    canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def show(pos):
        fig = fig_fn()
        menu = QMenu(canvas)
        menu.addAction("Copy image").triggered.connect(lambda: copy_figure(fig))
        menu.addAction("Save image as…").triggered.connect(lambda: save_figure(fig, canvas, name))
        menu.exec(canvas.mapToGlobal(pos))
    canvas.customContextMenuRequested.connect(show)


# ── tables ───────────────────────────────────────────────────────────────
def table_rows(table, selected_only=True):
    """[header, row, row, ...] as text; the selected rows, or all if none is."""
    cols = table.columnCount()
    header = [(table.horizontalHeaderItem(c).text() if table.horizontalHeaderItem(c) else "") for c in range(cols)]
    rows = list(range(table.rowCount()))
    if selected_only and table.selectionModel() is not None:
        sel = sorted({i.row() for i in table.selectionModel().selectedIndexes()})
        if sel:
            rows = sel
    def cell(r, c):
        it = table.item(r, c)
        if it is None:
            return ""
        full = it.data(Qt.ItemDataRole.UserRole)   # the unrounded value, when the table keeps one
        return repr(full) if isinstance(full, float) else it.text()
    out = [header]
    for r in rows:
        out.append([cell(r, c) for c in range(cols)])
    return out


def copy_table(table, selected_only=True):
    text = "\n".join("\t".join(row) for row in table_rows(table, selected_only))
    QGuiApplication.clipboard().setText(text)
    return text


def export_table_csv(table, parent=None, name="table.csv", target=None):
    if target is None:
        target, _ = QFileDialog.getSaveFileName(parent, "Export table", name, "CSV (*.csv)")
    if not target:
        return None
    if not target.lower().endswith(".csv"):
        target += ".csv"
    rows = table_rows(table, selected_only=False)
    rows = [[c.replace("—", "") for c in row] for row in rows]

    def write(tmp):
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(rows)
    return write_replacing(target, write, parent)


def attach_table_menu(table, name="table.csv"):
    table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def show(pos):
        menu = QMenu(table)
        has_sel = bool(table.selectionModel() and table.selectionModel().selectedIndexes())
        menu.addAction("Copy selected rows" if has_sel else "Copy table").triggered.connect(
            lambda: copy_table(table))
        if has_sel:
            menu.addAction("Copy whole table").triggered.connect(lambda: copy_table(table, False))
        menu.addAction("Export CSV…").triggered.connect(lambda: export_table_csv(table, table, name))
        menu.exec(table.viewport().mapToGlobal(pos))
    table.customContextMenuRequested.connect(show)
