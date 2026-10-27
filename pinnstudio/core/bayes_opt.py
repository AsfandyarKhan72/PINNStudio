"""
Bayesian optimization for Parameter Sweep's "Bayesian search" mode
-- the hyperparameter search of DeepXDE's
Helmholtz_Dirichlet_2d_HPO.py (scikit-optimize's gp_minimize with
expected improvement), implemented here with NumPy only so PINNStudio
needs no extra package.

Ask/tell interface, because each evaluation is a full training run that
the sweep runner starts as a subprocess:

    opt = BayesOptimizer(dims, n_initial=5, seed=0)
    opt.tell(baseline_point, value)        # optional: points already run
    for _ in range(n):
        p = opt.ask()                      # [value per dimension]
        opt.tell(p, objective(p))
    opt.best()                             # (point, value)

Dimensions: Dim(kind, low, high, log, choices) -- kind "float", "int" or
"categorical". Every dimension is mapped to [0, 1] (log-scaled when
`log`); the surrogate is a Gaussian process with a Matern 5/2 kernel on
those coordinates, fitted to log10 of the objective (losses span decades)
with its length scale picked from a small set by marginal likelihood;
the next point maximizes expected improvement over random candidates
plus local perturbations of the best point so far.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Dim:
    kind: str                     # "float" | "int" | "categorical"
    low: float = 0.0
    high: float = 1.0
    log: bool = False
    choices: List[str] = field(default_factory=list)

    def to_unit(self, v):
        if self.kind == "categorical":
            n = max(1, len(self.choices))
            i = self.choices.index(str(v)) if str(v) in self.choices else 0
            return (i + 0.5) / n
        lo, hi, v = float(self.low), float(self.high), float(v)
        if self.log:
            lo, hi, v = math.log(lo), math.log(hi), math.log(max(v, 1e-300))
        return 0.0 if hi == lo else min(1.0, max(0.0, (v - lo) / (hi - lo)))

    def from_unit(self, u):
        u = min(1.0, max(0.0, float(u)))
        if self.kind == "categorical":
            n = max(1, len(self.choices))
            return self.choices[min(n - 1, int(u * n))]
        lo, hi = float(self.low), float(self.high)
        if self.log:
            v = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
        else:
            v = lo + u * (hi - lo)
        if self.kind == "int":
            return int(min(max(round(v), math.ceil(lo)), math.floor(hi)))
        return float(v)


def _matern52(a, b, ls):
    d = np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1), 0.0)) / ls
    s5 = math.sqrt(5.0) * d
    return (1.0 + s5 + 5.0 / 3.0 * d ** 2) * np.exp(-s5)


def _norm_pdf(z):
    return np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _norm_cdf(z):
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


class BayesOptimizer:
    def __init__(self, dims: List[Dim], n_initial: int = 5, seed: Optional[int] = 0, xi: float = 0.01):
        self.dims = list(dims)
        self.n_initial = max(1, int(n_initial))
        self.rng = np.random.default_rng(seed)
        self.xi = xi
        self.X: List[np.ndarray] = []      # unit coordinates
        self.points: List[list] = []       # actual values
        self.y: List[float] = []           # objective values

    # -- bookkeeping ------------------------------------------------------
    def _unit(self, point):
        return np.array([d.to_unit(v) for d, v in zip(self.dims, point)], dtype=float)

    def _point(self, u):
        return [d.from_unit(x) for d, x in zip(self.dims, u)]

    def tell(self, point, value):
        """Record one finished evaluation. A failed run (None/NaN/inf) is
        recorded as worse than anything seen, so the search moves away."""
        v = float(value) if value is not None else float("nan")
        if not math.isfinite(v):
            finite = [y for y in self.y if math.isfinite(y)]
            v = (max(finite) * 10 if finite and max(finite) > 0 else 1e5)
        self.points.append(list(point))
        self.X.append(self._unit(point))
        self.y.append(v)

    def best(self):
        if not self.y:
            return None, None
        i = int(np.argmin(self.y))
        return self.points[i], self.y[i]

    # -- suggestion -------------------------------------------------------
    def ask(self):
        d = len(self.dims)
        if len(self.y) < self.n_initial:
            return self._point(self.rng.random(d))
        X = np.array(self.X)
        yv = np.array(self.y, dtype=float)
        y = np.log10(np.maximum(yv, 1e-300)) if np.all(yv > 0) else yv
        mu0, sd0 = y.mean(), y.std() or 1.0
        yn = (y - mu0) / sd0
        best_ll, best = -np.inf, None
        for ls in (0.1, 0.2, 0.35, 0.6, 1.0):
            K = _matern52(X, X, ls) + 1e-6 * np.eye(len(X))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                continue
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, yn))
            ll = -0.5 * yn @ alpha - np.log(np.diag(L)).sum()
            if ll > best_ll:
                best_ll, best = ll, (ls, L, alpha)
        if best is None:
            return self._point(self.rng.random(d))
        ls, L, alpha = best
        xb = X[int(np.argmin(yn))]
        cand = np.vstack([self.rng.random((3000, d)),
                          np.clip(xb + 0.05 * self.rng.standard_normal((1000, d)), 0, 1)])
        Ks = _matern52(cand, X, ls)
        mu = Ks @ alpha
        v = np.linalg.solve(L, Ks.T)
        sigma = np.sqrt(np.maximum(1.0 - (v * v).sum(0), 1e-12))
        imp = yn.min() - mu - self.xi
        z = imp / sigma
        ei = imp * _norm_cdf(z) + sigma * _norm_pdf(z)
        # Don't re-propose a point already evaluated (after int/categorical rounding).
        seen = {tuple(map(str, p)) for p in self.points}
        for i in np.argsort(-ei):
            p = self._point(cand[i])
            if tuple(map(str, p)) not in seen:
                return p
        return self._point(self.rng.random(d))
