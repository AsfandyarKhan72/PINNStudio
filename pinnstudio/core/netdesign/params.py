"""Typed parameters of the designer's blocks and layers.

Each spec knows its default, how to coerce a value typed or read from a file, and what is allowed; the
inspector builds its form from them. Pure and free of I/O, so views may use it (VIEW_SAFE).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass


class ParamError(ValueError):
    """A value a parameter does not accept, or a design that breaks a rule; the message says which."""


@dataclass(frozen=True)
class Param:
    name: str
    label: str = ""
    default: object = None
    help: str = ""

    def title(self) -> str:
        return self.label or self.name.replace("_", " ").capitalize()

    def coerce(self, value):
        return value


@dataclass(frozen=True)
class Int(Param):
    min: int = -(2 ** 31)
    max: int = 2 ** 31 - 1

    def coerce(self, value):
        if isinstance(value, bool):
            raise ParamError(f"{self.title()}: a whole number is needed, not {value!r}")
        try:
            f = float(value)
        except (TypeError, ValueError):
            raise ParamError(f"{self.title()}: a whole number is needed, not {value!r}")
        if not math.isfinite(f) or f != int(f):
            raise ParamError(f"{self.title()}: a whole number is needed, not {value!r}")
        v = int(f)
        if not self.min <= v <= self.max:
            raise ParamError(f"{self.title()}: {v} is outside {self.min} to {self.max}")
        return v


@dataclass(frozen=True)
class Float(Param):
    min: float = -math.inf
    max: float = math.inf
    positive: bool = False

    def coerce(self, value):
        if isinstance(value, bool):
            raise ParamError(f"{self.title()}: a number is needed, not {value!r}")
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ParamError(f"{self.title()}: a number is needed, not {value!r}")
        if not math.isfinite(v):
            raise ParamError(f"{self.title()}: a finite number is needed")
        if self.positive and v <= 0:
            raise ParamError(f"{self.title()}: must be greater than 0")
        if not self.min <= v <= self.max:
            raise ParamError(f"{self.title()}: {v:g} is outside {self.min:g} to {self.max:g}")
        return v


@dataclass(frozen=True)
class Bool(Param):
    def coerce(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0", "yes", "no"):
            return value.strip().lower() in ("true", "1", "yes")
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        raise ParamError(f"{self.title()}: on or off is needed, not {value!r}")


@dataclass(frozen=True)
class Choice(Param):
    options: tuple = ()

    def coerce(self, value):
        if value in self.options:
            return value
        low = {str(o).lower(): o for o in self.options}
        if isinstance(value, str) and value.lower() in low:
            return low[value.lower()]
        raise ParamError(f"{self.title()}: {value!r} is not one of {', '.join(map(str, self.options))}")


@dataclass(frozen=True)
class Text(Param):
    max_len: int = 100_000
    pattern: str = ""         # a regular expression the whole text must match ("" = any)
    pattern_help: str = ""

    def coerce(self, value):
        if not isinstance(value, str):
            raise ParamError(f"{self.title()}: text is needed")
        if len(value) > self.max_len:
            raise ParamError(f"{self.title()}: at most {self.max_len} characters")
        if self.pattern:
            import re
            if not re.fullmatch(self.pattern, value):
                raise ParamError(f"{self.title()}: {self.pattern_help or 'not allowed: ' + repr(value)}")
        return value


@dataclass(frozen=True)
class JsonDict(Param):
    """Keyword arguments as a JSON object (a custom layer's)."""

    def coerce(self, value):
        if isinstance(value, dict):
            obj = value
        else:
            try:
                obj = json.loads(value or "{}")
            except (TypeError, ValueError) as exc:
                raise ParamError(f"{self.title()}: not valid JSON ({exc})")
        if not isinstance(obj, dict) or not all(isinstance(k, str) for k in obj):
            raise ParamError(f"{self.title()}: a JSON object such as {{\"scale\": 2}} is needed")
        return json.dumps(obj, sort_keys=True)


@dataclass(frozen=True)
class Names(Param):
    """A list of names (inputs or outputs a block reads or predicts); ["@all"] means every one."""

    def coerce(self, value):
        if isinstance(value, str):
            value = [v.strip() for v in value.replace(";", ",").split(",") if v.strip()]
        if not isinstance(value, (list, tuple)) or not value or not all(isinstance(v, str) and v for v in value):
            raise ParamError(f"{self.title()}: at least one name is needed")
        if len(set(value)) != len(value):
            raise ParamError(f"{self.title()}: a name is listed twice")
        if "@all" in value and len(value) > 1:
            raise ParamError(f"{self.title()}: @all stands for every name on its own")
        return list(value)


def validate(specs, values, where: str = "") -> dict:
    """`values` completed with the specs' defaults and coerced; unknown names are refused."""
    known = {s.name: s for s in specs}
    unknown = [k for k in (values or {}) if k not in known]
    if unknown:
        raise ParamError(f"{where}unknown setting {unknown[0]!r}")
    out = {}
    for s in specs:
        raw = (values or {}).get(s.name, s.default)
        try:
            out[s.name] = s.coerce(raw)
        except ParamError as exc:
            raise ParamError(f"{where}{exc}")
    return out


__all__ = ["ParamError", "Param", "Int", "Float", "Bool", "Choice", "Text", "JsonDict", "Names", "validate"]
