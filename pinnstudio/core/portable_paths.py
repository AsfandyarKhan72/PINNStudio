"""Reference-data paths in a saved problem file that work on any machine.

A problem saved from a DeepXDE example or a template points into the
installed reference_data folder (Error Analysis files, measured data for
an inverse problem, ...). Saved as absolute paths, those only exist on
the computer that saved the file. to_portable writes them as
"@REF/<path inside reference_data>", the form the example definitions
already use; from_portable turns them back into paths into this
installation's reference_data folder. It also repairs a file saved before
this existed: an absolute path that is not on this computer but has a
"reference_data" folder in it is pointed at the same file here, when that
file exists.

Both work on a config dict (field -> value). A path can be a whole string
value or sit inside a JSON or Python-literal string (ea_files is a repr'd
list of tuples, inverse_obs_files_json a JSON list of dicts); such a
string is parsed, its strings changed, and written back. A value with no
path to change is kept exactly as it was.
"""
import ast
import json
import ntpath
import os
import posixpath

REF = "@REF/"
_SEGMENT = "reference_data"


def _parts(path):
    """The folder names of a POSIX or Windows path, separators either way."""
    return [p for p in path.replace("\\", "/").split("/") if p not in ("", ".")]


def _norm(path):
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _looks_absolute(s):
    return posixpath.isabs(s) or ntpath.isabs(s) or (len(s) > 2 and s[1] == ":" and s[2] in "\\/")


def _walk(x, fn):
    if isinstance(x, str):
        return fn(x)
    if isinstance(x, list):
        return [_walk(i, fn) for i in x]
    if isinstance(x, tuple):
        return tuple(_walk(i, fn) for i in x)
    if isinstance(x, dict):
        return {k: _walk(i, fn) for k, i in x.items()}
    return x


def _apply(value, fn):
    """fn applied to `value`'s strings; `value` itself when nothing changed."""
    if not isinstance(value, str):
        if isinstance(value, (list, tuple, dict)):
            new = _walk(value, fn)
            return new if new != value else value
        return value
    s = value.lstrip()
    if s[:1] in "[{(":
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = None
        else:
            new = _walk(parsed, fn)
            return json.dumps(new) if new != parsed else value
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return value
        new = _walk(parsed, fn)
        return repr(new) if new != parsed else value
    new = fn(value)
    return new


def _to_ref(ref_dir):
    root = _norm(ref_dir)
    root_real = os.path.normpath(os.path.abspath(ref_dir))

    def fn(s):
        if not s or s.startswith(REF) or not _looks_absolute(s):
            return s
        try:
            p = _norm(s)        # case-folded on Windows: only for the comparison
            if os.path.commonpath([p, root]) != root or p == root:
                return s
            # the relative part keeps the folder names as written ("2D", not "2d")
            rel = os.path.relpath(os.path.normpath(os.path.abspath(s)), root_real)
        except ValueError:      # another drive on Windows
            return s
        if rel.startswith(".."):  # only differs from the root in letter case
            return s
        return REF + "/".join(_parts(rel))
    return fn


def _from_ref(ref_dir):
    def fn(s):
        if s.startswith(REF):
            return os.path.join(ref_dir, *_parts(s[len(REF):]))
        if not s or not _looks_absolute(s) or os.path.exists(s):
            return s
        parts = _parts(s)
        if _SEGMENT not in parts:
            return s
        tail = parts[len(parts) - 1 - parts[::-1].index(_SEGMENT) + 1:]
        if not tail:
            return s
        local = os.path.join(ref_dir, *tail)
        return local if os.path.exists(local) else s
    return fn


def to_portable(cfg_dict, ref_dir):
    """A copy of cfg_dict with paths inside ref_dir written as "@REF/..."."""
    fn = _to_ref(ref_dir)
    return {k: _apply(v, fn) for k, v in cfg_dict.items()}


def from_portable(cfg_dict, ref_dir):
    """A copy of cfg_dict with "@REF/..." (and reference_data paths saved on
    another computer) pointing into this installation's ref_dir."""
    fn = _from_ref(ref_dir)
    return {k: _apply(v, fn) for k, v in cfg_dict.items()}
