"""A number box for values that can be very small or large."""
from PyQt6.QtWidgets import QDoubleSpinBox


class SciSpinBox(QDoubleSpinBox):
    """A number box for values that can be very small or large (learning
    rates, decay rates, tolerances, scales): shows up to 12 significant
    digits ("1e-06", "0.00025", "2.5e+07") and takes anything float()
    reads, instead of rounding to a fixed number of decimals (a learning
    rate of 5e-7 used to round to 0.000001, a tolerance of 5e-7 to 0)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        super().setDecimals(300)       # no rounding by Qt; the text shows what matters

    def setDecimals(self, _d):
        pass                           # callers that set decimals for a plain box: keep full precision

    def textFromValue(self, v):
        return "0" if v == 0 else f"{v:.12g}"

    def valueFromText(self, text):
        try:
            return float((text or "").strip())
        except ValueError:
            return self.value()

    def validate(self, text, pos):
        from PyQt6.QtGui import QValidator
        t = (text or "").strip()
        try:
            float(t)
            return (QValidator.State.Acceptable, text, pos)
        except ValueError:
            import re as _re
            if _re.fullmatch(r"[0-9.eE+\-]*", t):
                return (QValidator.State.Intermediate, text, pos)
            return (QValidator.State.Invalid, text, pos)

    def stepBy(self, steps):
        # arrows: one step of the value's own size (1e-3 -> 1.1e-3, not +0.001)
        v = self.value()
        step = self.singleStep() if v == 0 else abs(v) * 0.1
        self.setValue(v + steps * step)
