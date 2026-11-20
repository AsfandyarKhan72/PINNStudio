"""The network designer's controller: every edit of the design, and what it asks the user.

Each edit copies the design, changes the copy at the level on view, checks it (`netdesign.design.check`)
and hands it to the state, which keeps it as one step of Undo. A refused edit changes nothing and says
why through the state's `message`. Views call these methods; they never change the state themselves.
"""
from __future__ import annotations

import copy
import os

from PySide6.QtCore import QObject

from pinnstudio.core.netdesign import design as D
from pinnstudio.core.netdesign import export as X
from pinnstudio.core.netdesign import layout as L
from pinnstudio.core.netdesign import library as LIB
from pinnstudio.core.netdesign import presets as P
from pinnstudio.core.netdesign import problem as PR
from pinnstudio.core.netdesign import shapes as S
from pinnstudio.core.netdesign import trust as T
from pinnstudio.core.netdesign.layers import LAYERS
from pinnstudio.core.netdesign.params import ParamError
from pinnstudio.ui.controllers.prompter import Prompter
from pinnstudio.ui.state.design_state import DesignState

# A new block's settings by kind and palette preset.
_NEW_LAYER = {"dense": {"units": 64}, "fourier": {"features": 32, "sigma": 1.0},
              "periodic": {"harmonics": 1, "period": 1.0}}
EDITABLE = {"title", "merge", "inputs", "outputs", "direct", "trainable", "ref", "port", "width_in", "width_out",
            "fine_tune", "layers"}


class DesignerController(QObject):
    def __init__(self, state: DesignState, prompter: Prompter | None = None, parent=None):
        super().__init__(parent)
        self.state = state
        self.prompter = prompter or Prompter()
        self._baseline = []
        self._config = None

    # ── the problem and the stored design ──────────────────────────
    def set_problem(self, config) -> None:
        """The problem's inputs and outputs, against which widths and problems are worked out."""
        self._config = config
        try:
            ins, outs = PR.network_io(config)
        except Exception:                   # a problem mid-edit: keep the last names
            return
        self.state.set_io(ins, outs)

    def load_json(self, text: str) -> bool:
        """The design stored in a problem (PINNConfig.network_design); no undo across it."""
        try:
            d = D.check(text or "")
        except ParamError as exc:
            self.state.reset(D.empty())
            self.state.set_message(f"The saved network design could not be read: {exc}")
            return False
        self.state.reset(d)
        return True

    def design_json(self) -> str:
        """The design to store in the problem ("" while it is empty)."""
        d = self.state.design
        return D.to_json(d) if d["blocks"] else ""

    # ── edits ──────────────────────────────────────────────────────
    def _edit(self, fn, message_ok: str = "") -> bool:
        d = self.state.design
        try:
            level = D.level_of(d, self.state.path)
        except ParamError as exc:
            self.state.set_message(str(exc))
            return False
        try:
            result = fn(level, d)
        except (ParamError, ValueError) as exc:
            self.state.set_message(str(exc))
            return False
        if result is False:
            return False
        try:
            checked = D.check(d)
        except ParamError as exc:
            self.state.set_message(f"Not changed: {exc}")
            return False
        changed = self.state.set_design(checked)
        if changed and message_ok:
            self.state.set_message(message_ok)
        return changed

    def add(self, kind: str, preset: str = "", x: float = 0.0, y: float = 0.0) -> str:
        """Add a block from the palette at (x, y); its id, or "" when refused."""
        new = {}

        def fn(level, d):
            in_group = bool(self.state.path)
            if kind in D.TOP_ONLY and in_group:
                raise ParamError(f"{kind.capitalize()} blocks belong at the top level, not inside a group")
            if kind in D.INNER_ONLY and not in_group:
                raise ParamError("Ports belong inside a group: open a group (double-click it) first")
            if kind == "port_out" and any(b["kind"] == "port_out" for b in level["blocks"]):
                raise ParamError("a group has one Port out")
            b = {"id": D.new_id(d, kind), "kind": kind, "title": "", "x": float(x), "y": float(y)}
            if kind == "stack":
                layer = preset or "dense"
                if layer not in LAYERS:
                    raise ParamError(f"unknown layer {layer!r}")
                b["layers"] = [{"type": layer, "params": dict(_NEW_LAYER.get(layer, {}))}]
            elif kind == "merge":
                b["merge"] = preset if preset in D.MERGES else "concatenate"
            elif kind == "port_in":
                names = {x["port"] for x in level["blocks"] if x["kind"] == "port_in"}
                k = 1
                while f"in{k}" in names:
                    k += 1
                b["port"] = f"in{k}"
            elif kind == "group":
                used = D.all_ids(d) | {b["id"]}

                def nid(k):
                    n = 1
                    while f"{D.PREFIX[k]}{n}" in used:
                        n += 1
                    used.add(f"{D.PREFIX[k]}{n}")
                    return f"{D.PREFIX[k]}{n}"
                pi, st, po = nid("port_in"), nid("stack"), nid("port_out")
                b["inner"] = {"blocks": [
                    {"id": pi, "kind": "port_in", "port": "in", "x": 0.0, "y": 0.0},
                    {"id": st, "kind": "stack", "layers": [{"type": "dense", "params": {"units": 64}}],
                     "x": 240.0, "y": 0.0},
                    {"id": po, "kind": "port_out", "x": 480.0, "y": 0.0}],
                    "edges": [{"src": pi, "dst": st, "port": "in"}, {"src": st, "dst": po, "port": "in"}]}
            elif kind == "shared":
                ref = self.state.selection
                target = next((x for x in level["blocks"] if x["id"] == ref), None)
                if target is None or target["kind"] not in ("stack", "group", "saved"):
                    raise ParamError("select the stack, group or saved module to share first, then add Shared")
                b["ref"] = ref
                b["title"] = f"{D.label(target)} (again)"
            elif kind == "saved":
                path = self.prompter.ask_open_path("Saved module (TorchScript)", "TorchScript (*.pt *.pth)")
                if not path:
                    return False
                b["path"] = path
                b["sha256"] = T.sha256_file(path)
                b["title"] = os.path.splitext(os.path.basename(path))[0]
            level["blocks"].append(b)
            new["id"] = b["id"]
            return True

        if not self._edit(fn):
            return ""
        self.state.select(new.get("id", ""))
        return new.get("id", "")

    def connect(self, src: str, dst: str, port: str = "in") -> bool:
        ok, why = self.legal(src, dst, port)
        if not ok:
            self.state.set_message(f"Not connected: {why}")
            return False
        return self._edit(lambda level, d: level["edges"].append({"src": src, "dst": dst, "port": port}))

    def drag_started(self) -> None:
        """A connection drag began: the problems the design has now do not block it."""
        self._baseline = self.state.problems

    def legal(self, src: str, dst: str, port: str = "in"):
        """(legal, reason) of a connection src → dst.port at the level on view."""
        return S.can_connect(self.state.design, self.state.path, src, dst, port, self._baseline,
                             self.state.inputs, self.state.outputs)

    def move(self, bid: str, x: float, y: float) -> bool:
        def fn(level, d):
            b = next((b for b in level["blocks"] if b["id"] == bid), None)
            if b is None or (b["x"], b["y"]) == (float(x), float(y)):
                return False
            b["x"], b["y"] = float(x), float(y)
        return self._edit(fn)

    def delete(self, key: str) -> bool:
        """Delete a block (with its connections, and the shared blocks that call it) or a connection
        ("src>dst@port")."""
        if not key:
            return False

        def fn(level, d):
            if ">" in key:
                src, rest = key.split(">", 1)
                dst, _, port = rest.partition("@")
                before = len(level["edges"])
                level["edges"][:] = [e for e in level["edges"]
                                     if not (e["src"] == src and e["dst"] == dst and e["port"] == (port or "in"))]
                return len(level["edges"]) != before
            gone = {key} | {b["id"] for b in level["blocks"] if b["kind"] == "shared" and b["ref"] == key}
            if not any(b["id"] == key for b in level["blocks"]):
                return False
            removed = next(b for b in level["blocks"] if b["id"] == key)
            level["blocks"][:] = [b for b in level["blocks"] if b["id"] not in gone]
            level["edges"][:] = [e for e in level["edges"] if e["src"] not in gone and e["dst"] not in gone]
            if removed["kind"] == "port_in" and self.state.path:        # the group's port goes too
                outer = D.level_of(d, self.state.path[:-1])
                gid = self.state.path[-1]
                outer["edges"][:] = [e for e in outer["edges"]
                                     if not (e["dst"] == gid and e["port"] == removed["port"])]
            return True
        ok = self._edit(fn)
        if ok:
            self.state.select("")
        return ok

    def duplicate(self, bid: str) -> str:
        level = self.state.level()
        b = next((b for b in level["blocks"] if b["id"] == bid), None)
        if b is None or b["kind"] in ("input", "head", "port_out"):
            if b is not None:
                self.state.set_message(f"{D.label(b)}: a design has its {b['kind'].replace('_', ' ')} once; add "
                                       "another from the palette")
            return ""
        copy_b = copy.deepcopy(b)
        try:
            d = LIB.insert_block(self.state.design, self.state.path, copy_b, "", b["x"] + 40, b["y"] + 40)
        except ParamError as exc:
            self.state.set_message(str(exc))
            return ""
        new_id = D.level_of(d, self.state.path)["blocks"][-1]["id"]
        if self.state.set_design(d):
            self.state.select(new_id)
            return new_id
        return ""

    def auto_layout(self) -> bool:
        def fn(level, d):
            L.organize_level(level)
        return self._edit(fn)

    def update_block(self, bid: str, values: dict) -> bool:
        """Change a block's settings (title, merge, inputs, outputs, …, layers); a value the block does not
        take refuses the whole change."""
        bad = [k for k in values if k not in EDITABLE]
        if bad:
            self.state.set_message(f"Not changed: {bad[0]} cannot be set here")
            return False

        def fn(level, d):
            b = next((b for b in level["blocks"] if b["id"] == bid), None)
            if b is None:
                return False
            old_port = b.get("port")
            for k, v in values.items():
                if k not in b and k != "title":
                    raise ParamError(f"a {b['kind']} block has no setting {k!r}")
                b[k] = copy.deepcopy(v)
            if b["kind"] == "port_in" and old_port != b.get("port") and self.state.path:
                outer = D.level_of(d, self.state.path[:-1])
                for e in outer["edges"]:
                    if e["dst"] == self.state.path[-1] and e["port"] == old_port:
                        e["port"] = b["port"]
        return self._edit(fn)

    def set_code(self, code: str) -> bool:
        def fn(level, d):
            if d.get("code", "") == code:
                return False
            d["code"] = code
        return self._edit(fn)

    def select(self, key: str) -> None:
        self.state.select(key)

    def enter(self, bid: str) -> bool:
        """Open a group: its inside becomes the level on view."""
        level = self.state.level()
        b = next((b for b in level["blocks"] if b["id"] == bid), None)
        if b is None or b["kind"] != "group":
            return False
        self.state.set_path(self.state.path + (bid,))
        return True

    def up(self) -> bool:
        if not self.state.path:
            return False
        self.state.set_path(self.state.path[:-1])
        return True

    def undo(self) -> bool:
        return self.state.undo()

    def redo(self) -> bool:
        return self.state.redo()

    # ── whole designs ──────────────────────────────────────────────
    def apply_preset(self, name: str, config=None) -> bool:
        """Put a preset in place of the design (one step of Undo)."""
        cfg = config if config is not None else self._config
        try:
            d = P.make(name, cfg, self.state.inputs, self.state.outputs)
        except (ValueError, ParamError) as exc:
            self.state.set_message(str(exc))
            return False
        self.state.set_path(())
        return self.state.set_design(d)

    def from_settings(self, config=None) -> bool:
        """**Edit as blocks**: the design that computes what the Neural network page's form makes."""
        cfg = config if config is not None else self._config
        d = P.from_settings(cfg, self.state.inputs, self.state.outputs)
        self.state.set_path(())
        return self.state.set_design(d)

    # ── files ──────────────────────────────────────────────────────
    def library_blocks(self) -> list:
        """[(name, path)] of the block files in the library folder, for the palette's My blocks."""
        return LIB.list_blocks()

    def save_block(self, bid: str = "") -> bool:
        bid = bid or self.state.selection
        _p, b = D.find(self.state.design, bid)
        if b is None:
            self.state.set_message("Select the block to save first")
            return False
        default = os.path.join(LIB.blocks_dir(), f"{(b.get('title') or bid).replace(' ', '_')}{LIB.BLOCK_EXT}")
        path = self.prompter.ask_save_path("Save block", default, f"PINNStudio block (*{LIB.BLOCK_EXT})")
        if not path:
            return False
        try:
            LIB.save_block(self.state.design, bid, path)
        except (ParamError, OSError) as exc:
            self.state.set_message(f"Not saved: {exc}")
            return False
        self.state.set_message(f"Saved {os.path.basename(path)}")
        return True

    def save_design(self) -> bool:
        path = self.prompter.ask_save_path("Save design", f"network{LIB.DESIGN_EXT}",
                                           f"PINNStudio design (*{LIB.DESIGN_EXT})")
        if not path:
            return False
        try:
            LIB.save_design(self.state.design, path)
        except (ParamError, OSError) as exc:
            self.state.set_message(f"Not saved: {exc}")
            return False
        self.state.set_message(f"Saved {os.path.basename(path)}")
        return True

    def load_file(self, path: str = "", x: float = 0.0, y: float = 0.0) -> bool:
        """A block file (added where asked) or a design file (in place of this design)."""
        path = path or self.prompter.ask_open_path(
            "Load block or design", f"PINNStudio blocks and designs (*{LIB.BLOCK_EXT} *{LIB.DESIGN_EXT})")
        if not path:
            return False
        try:
            what, obj, code = LIB.read(path)
            if what == "design":
                self.state.set_path(())
                return self.state.set_design(obj)
            d = LIB.insert_block(self.state.design, self.state.path, obj, code, x, y)
        except ParamError as exc:
            self.state.set_message(f"Not loaded: {exc}")
            return False
        return self.state.set_design(d)

    def export_code(self, activation="tanh", kernel_initializer="Glorot uniform", weight_decay=0.0) -> bool:
        path = self.prompter.ask_save_path("Export network as Python code", "designed_net.py", "Python (*.py)")
        if not path:
            return False
        try:
            src = X.module_source(self.state.design, self.state.inputs, self.state.outputs, activation,
                                  kernel_initializer, weight_decay)
            folder = os.path.dirname(os.path.abspath(path))
            import tempfile
            fd, tmp = tempfile.mkstemp(prefix=".pinnstudio-", suffix=".py", dir=folder)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(src)
            try:
                os.replace(tmp, path)     # a file open elsewhere is kept, never left half-written
            except OSError:
                os.remove(tmp)
                raise
        except PermissionError:
            self.state.set_message(f"Not exported: {os.path.basename(path)} is open in another program; close it "
                                   "there and export again")
            return False
        except (ParamError, OSError) as exc:
            self.state.set_message(f"Not exported: {exc}")
            return False
        self.state.set_message(f"Exported {os.path.basename(path)}")
        return True

    # ── fixes and keyboard alternatives ─────────────────────────────
    def missing_outputs(self) -> list:
        """Outputs no head gives (after the number of outputs changed, say)."""
        d = self.state.design
        covered = set()
        for b in d["blocks"]:
            if b["kind"] == "head":
                covered |= set(self.state.outputs if b["outputs"] == [D.ALL] else b["outputs"])
        return [o for o in self.state.outputs if o not in covered] if d["blocks"] else []

    def add_missing_heads(self) -> bool:
        """A head for each output no head gives, fed by what feeds the existing heads (one step of Undo)."""
        missing = self.missing_outputs()
        if not missing:
            return False

        def fn(level, d):
            if self.state.path:
                raise ParamError("heads are at the top level: go Up first")
            heads = [b for b in d["blocks"] if b["kind"] == "head"]
            srcs = []
            for h in heads:
                srcs += [e["src"] for e in d["edges"] if e["dst"] == h["id"] and e["src"] not in srcs]
            y = max([b["y"] for b in heads] or [0.0])
            x = max([b["x"] for b in heads] or [480.0])
            for k, name in enumerate(missing):
                hid = D.new_id(d, "head")
                d["blocks"].append({"id": hid, "kind": "head", "title": f"Output {name}", "outputs": [name],
                                    "x": x, "y": y + 120.0 * (k + 1)})
                if len(srcs) == 1:
                    d["edges"].append({"src": srcs[0], "dst": hid, "port": "in"})
        return self._edit(fn, f"Added a head for {', '.join(missing)}")

    def legal_targets(self, src: str) -> list:
        """[(block id, port, label)] of the in-ports `src` may be connected to now (for the keyboard)."""
        self.drag_started()
        level = self.state.level()
        out = []
        for b in level["blocks"]:
            for port in D.ports_of(b):
                ok, _why = self.legal(src, b["id"], port)
                if ok:
                    out.append((b["id"], port, D.label(b) + ("" if port == "in" else f" (port {port})")))
        return out

    # ── problem files ──────────────────────────────────────────────
    @staticmethod
    def design_for_file(design_json: str, problem_path: str) -> str:
        """The design as a problem file keeps it: saved modules' paths relative to the file's folder."""
        return LIB.relocate_paths(design_json, os.path.dirname(os.path.abspath(problem_path)), relative=True)

    @staticmethod
    def design_from_file(design_json: str, problem_path: str) -> str:
        """The design read from a problem file, with saved modules' paths absolute again."""
        return LIB.relocate_paths(design_json, os.path.dirname(os.path.abspath(problem_path)), relative=False)

    # ── trust ──────────────────────────────────────────────────────
    def untrusted(self) -> list:
        return T.untrusted(self.state.design)

    def check_trust(self, design_json: str | None = None) -> bool:
        """True when everything the design runs or loads is trusted (asking once for what is not). With
        `design_json`, that design (a run being restored) rather than the one on the canvas."""
        if design_json is not None:
            try:
                d = D.check(design_json or "")
            except ParamError as exc:
                self.prompter.inform("Network design", f"The run's network design cannot be read: {exc}")
                return False
        else:
            d = self.state.design
        items = T.untrusted(d)
        if not items:
            return True
        broken = [it for it in items if it["problem"]]
        if broken:
            self.prompter.inform("Network design", "\n".join(f"{it['name']}: {it['problem']}" for it in broken))
            return False
        if not self.prompter.confirm_trust(items):
            self.state.set_message("The design's code and files stay inactive until you trust them")
            return False
        T.remember(items)
        self.state.set_message("Trusted")
        return True
