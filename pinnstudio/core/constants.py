"""Named constants (Round 70): "Re = 20", "nu = 1/Re", ... set once in the
PDE panel and usable by name in every expression of a problem -- PDE,
initial and boundary conditions, transforms, the exact-solution reference,
custom plot fields, training monitors.

PINNConfig.constants is a JSON list of {"name": ..., "value": ...}; each
value is a number or an expression in numbers, math functions (sin, sqrt,
exp, ... and pi) and the constants defined above it. The generators
substitute each constant's value into the expressions (see
codegen._effective_config), so every place an expression is evaluated --
including a saved model's restore script -- sees plain numbers, and the
exported script lists the constants at its top.
"""
import json
import keyword
import math
import re

_MATH = {n: getattr(math, n) for n in ("sin", "cos", "tan", "sinh", "cosh", "tanh", "exp", "log", "log10",
                                       "sqrt", "ceil", "floor", "asin", "acos", "atan")}
_MATH.update(arcsin=math.asin, arccos=math.acos, arctan=math.atan, abs=abs, pi=math.pi)

# Names a constant can't take: coordinates, side/limit names, math names.
RESERVED = ({"x", "y", "z", "t", "u", "np", "torch", "dde", "inputs", "outputs", "X", "True", "False", "None",
             "left", "right", "bottom", "top", "front", "back",
             "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "t_min", "t_max"} | set(_MATH))


def parse(text):
    """[(name, expression)] from PINNConfig.constants (JSON list)."""
    try:
        rows = json.loads(text or "[]")
    except ValueError:
        raise ValueError("Constants: not valid JSON.")
    out = []
    for r in rows if isinstance(rows, list) else []:
        name = str(r.get("name", "")).strip()
        value = str(r.get("value", "")).strip()
        if name or value:
            out.append((name, value))
    return out


def evaluate(text, other_names=()):
    """[(name, expression, value)] in order; raises ValueError with a clear
    message for a bad name, a clash, or an expression that can't be
    evaluated. `other_names` (output and inverse-variable names) are also
    off-limits."""
    blocked = RESERVED | {n.strip() for n in other_names if n and n.strip()}
    ns = dict(_MATH)
    ns["__builtins__"] = {}
    out, seen = [], set()
    for name, expr in parse(text):
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(f"Constants: '{name}' is not a valid name (letters, digits, _; not starting with a digit).")
        if name in blocked or re.match(r"^d[A-Za-z]\w*_[xyzt]+$", name):
            raise ValueError(f"Constants: '{name}' is already used for something else (a coordinate, output, "
                             f"math function or side name) -- pick another name.")
        if name in seen:
            raise ValueError(f"Constants: '{name}' is defined twice.")
        if not expr:
            raise ValueError(f"Constants: '{name}' has no value.")
        try:
            value = float(eval(compile(expr, "<constant>", "eval"), ns))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Constants: can't evaluate {name} = {expr} ({exc}). "
                             f"A constant can use numbers, math functions and constants defined above it.")
        if not math.isfinite(value):
            raise ValueError(f"Constants: {name} = {expr} is not a finite number.")
        ns[name] = value
        seen.add(name)
        out.append((name, expr, value))
    return out


def substitute(expr, values):
    """`expr` with every constant name replaced by its value (whole names
    only -- not inside du_Re, np.e or Re2)."""
    if not expr or not values:
        return expr
    for name, _e, value in sorted(values, key=lambda r: -len(r[0])):
        expr = re.sub(rf"(?<![\w.]){re.escape(name)}(?![\w])", f"({value!r})", expr)
    return expr


def substitute_json(text, values, keys):
    """substitute() in the string fields `keys` of a JSON list of dicts."""
    if not text or not values:
        return text
    try:
        rows = json.loads(text)
    except ValueError:
        return text
    if not isinstance(rows, list):
        return text
    changed = False
    for r in rows:
        if isinstance(r, dict):
            for k in keys:
                if isinstance(r.get(k), str):
                    new = substitute(r[k], values)
                    changed |= new != r[k]
                    r[k] = new
    return json.dumps(rows) if changed else text


def header_lines(values, comment="#"):
    """Comment lines listing the constants, for the top of an exported script."""
    if not values:
        return []
    lines = [f"{comment} Constants (their values are substituted into the expressions below):"]
    for name, expr, value in values:
        lines.append(f"{comment}   {name} = {expr}" + ("" if expr == repr(value) or expr == f"{value:g}" else f"  = {value!r}"))
    return lines
