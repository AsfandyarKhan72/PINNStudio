#!/usr/bin/env python3
"""
Reference solution for the mixed-convection template (lid-driven triangular
cavity with a cold cylinder and a magnetic field, after Soomro et al., 2020,
J. Energy Storage 31, 101702):

    U U_X + V U_Y = -P_X + (U_XX + U_YY)/Re
    U V_X + V V_Y = -P_Y + (V_XX + V_YY)/Re + Ri T - (Ha^2/Re) V
    U T_X + V T_Y = (T_XX + T_YY)/(Pr Re),        U_X + V_Y = 0

on an equilateral triangle (side 1, lid on top at Y = 1, apex at
Y = 1 - sqrt(3)/2) with a circular hole of radius 0.1 centred at (0.5, 0.7).
Lid: U = -1 (moving to the left, as in the paper's Fig. 1 and
its flow patterns; its eq. 11c writes U = 1), V = 0, T = 1; inclined walls: U = V = 0, dT/dn = 0; cylinder:
U = V = 0, T = 0. The lid's two end points belong to the walls (U = 0).

Finite elements (scikit-fem): Taylor-Hood P2-P1 for velocity and pressure,
P2 for temperature, the coupled steady problem solved by Newton's method
with continuation in Re. Prints the average Nusselt number on the lid,
Nu = int_0^1 dT/dY dX (the paper's eq. 12), and writes "x y value" files of
u, v, p (mean zero) and T on a grid of points inside the cavity
(solution_<output>.txt, as the template's Error Analysis expects).

    pip install scikit-fem triangle
    python tools/make_mixed_convection_reference.py --re 300 --ri 0.1 --ha 10 \
        --write reference_data/2D/mixed_convection
"""
import argparse
import os

import numpy as np
import scipy.sparse as sp
import triangle as tr
from skfem import (Basis, BilinearForm, ElementTriP1, ElementTriP2, ElementVector, FacetBasis, Functional,
                   LinearForm, MeshTri, condense, solve)
from skfem.helpers import dot, grad, div

H = np.sqrt(3) / 2
Y0 = 1 - H                      # apex height
CX, CY, R = 0.5, 0.7, 0.1


def make_mesh(area, n_circle=256):
    th = np.linspace(0, 2 * np.pi, n_circle, endpoint=False)
    outer = np.array([[0.0, 1.0], [0.5, Y0], [1.0, 1.0]])
    circ = np.c_[CX + R * np.cos(th), CY + R * np.sin(th)]
    seg = [[0, 1], [1, 2], [2, 0]] + [[3 + i, 3 + (i + 1) % n_circle] for i in range(n_circle)]
    m = tr.triangulate(dict(vertices=np.r_[outer, circ], segments=seg, holes=[[CX, CY]]), f"pq32a{area:.10f}")
    return MeshTri(m["vertices"].T.copy(), m["triangles"].T.copy())


def _mv(g, b):
    """(b . grad) u for a vector field: sum_j g[i, j] b[j]."""
    return np.einsum("ij...,j...->i...", g, b)


class Solver:
    def __init__(self, area=6e-6, pr=6.2, lid=1.0):
        self.pr = pr
        self.mesh = m = make_mesh(area)
        self.bu = Basis(m, ElementVector(ElementTriP2()), intorder=4)
        self.bp = Basis(m, ElementTriP1(), intorder=4)
        self.bt = Basis(m, ElementTriP2(), intorder=4)
        self.nu, self.np_, self.nt = self.bu.N, self.bp.N, self.bt.N
        top = m.facets_satisfying(lambda X: np.isclose(X[1], 1.0, atol=1e-9))
        circ = m.facets_satisfying(lambda X: np.abs(np.hypot(X[0] - CX, X[1] - CY) - R) < 2e-3)
        walls = np.setdiff1d(m.boundary_facets(), np.r_[top, circ])
        self.top = top
        bu, bt = self.bu, self.bt
        du_top = bu.get_dofs(top).all()
        du_wall = np.r_[bu.get_dofs(walls).all(), bu.get_dofs(circ).all()]
        lid_x = np.setdiff1d(bu.get_dofs(top).all("u^1"), du_wall)
        o = self.nu + self.np_
        self.U = np.zeros(self.nu + self.np_ + self.nt)
        self.U[lid_x] = lid
        dt_top, dt_circ = bt.get_dofs(top).all(), bt.get_dofs(circ).all()
        self.U[o + dt_top] = 1.0
        fix_p = self.nu + int(np.argmin(np.hypot(self.bp.doflocs[0] - 0.5, self.bp.doflocs[1] - Y0)))
        self.D = np.unique(np.r_[du_top, du_wall, o + dt_top, o + dt_circ, fix_p])
        print(f"mesh: {m.t.shape[1]} triangles, {len(self.U)} unknowns", flush=True)

    def newton(self, re, ri, ha, tol=1e-10, maxit=25):
        bu, bp, bt, pr = self.bu, self.bp, self.bt, self.pr
        nu, np_ = self.nu, self.np_

        @BilinearForm
        def a_uu(u, v, w):
            b = w["b"]
            return ((1 / re) * np.einsum("ij...,ij...", grad(u), grad(v)) + dot(_mv(grad(u), b.value), v)
                    + dot(_mv(b.grad, u), v) + (ha ** 2 / re) * u[1] * v[1])

        @BilinearForm
        def a_up(p, v, w):
            return -p * div(v)

        @BilinearForm
        def a_ut(t, v, w):
            return -ri * t * v[1]

        @BilinearForm
        def a_tu(u, s, w):
            return dot(u, w["T"].grad) * s

        @BilinearForm
        def a_tt(t, s, w):
            return (1 / (pr * re)) * dot(grad(t), grad(s)) + dot(w["b"].value, grad(t)) * s

        @LinearForm
        def r_u(v, w):
            b = w["b"]
            return ((1 / re) * np.einsum("ij...,ij...", b.grad, grad(v)) + dot(_mv(b.grad, b.value), v)
                    - w["p"].value * div(v) - ri * w["T"].value * v[1] + (ha ** 2 / re) * b.value[1] * v[1])

        @LinearForm
        def r_p(q, w):
            return -np.einsum("ii...", w["b"].grad) * q

        @LinearForm
        def r_t(s, w):
            return (1 / (pr * re)) * dot(w["T"].grad, grad(s)) + dot(w["b"].value, w["T"].grad) * s

        for it in range(maxit):
            U = self.U
            wu, wp, wt = bu.interpolate(U[:nu]), bp.interpolate(U[nu:nu + np_]), bt.interpolate(U[nu + np_:])
            A = a_uu.assemble(bu, b=wu)
            B = a_up.assemble(bp, bu)
            C = a_ut.assemble(bt, bu)
            E = a_tu.assemble(bu, bt, T=wt)
            F = a_tt.assemble(bt, b=wu)
            J = sp.bmat([[A, B, C], [B.T, None, None], [E, None, F]], "csr")
            Fv = np.r_[r_u.assemble(bu, b=wu, p=wp, T=wt), r_p.assemble(bp, b=wu), r_t.assemble(bt, b=wu, T=wt)]
            rn = np.linalg.norm(np.delete(Fv, self.D))
            dU = solve(*condense(J, -Fv, x=np.zeros_like(U), D=self.D))
            self.U = U + dU
            print(f"  Re={re:g} Ri={ri:g} Ha={ha:g}  it {it}: |F| = {rn:.2e}, |dU| = {np.linalg.norm(dU):.2e}", flush=True)
            if rn < tol:
                break
        return rn

    def solve(self, re, ri, ha):
        for r in [r for r in (100, 200, 300, 400, 500) if r < re] + [re]:
            self.newton(r, ri, ha)
        return self

    def fields(self):
        nu, np_ = self.nu, self.np_
        return self.U[:nu], self.U[nu:nu + np_], self.U[nu + np_:]

    def nusselt(self):
        fb = FacetBasis(self.mesh, self.bt.elem, facets=self.top, intorder=6)

        @Functional
        def f(w):
            return w["T"].grad[1]
        return float(f.assemble(fb, T=fb.interpolate(self.fields()[2])))


def grid_points(h):
    xs = np.arange(0, 1 + 1e-9, h)
    ys = np.arange(Y0, 1 + 1e-9, h)
    X, Y = np.meshgrid(xs, ys)
    X, Y = X.ravel(), Y.ravel()
    inside = (Y >= 1 - 2 * H * np.minimum(X, 1 - X) - 1e-12) & (np.hypot(X - CX, Y - CY) >= R - 1e-12)
    return np.vstack([X[inside], Y[inside]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=300)
    ap.add_argument("--ri", type=float, default=0.1)
    ap.add_argument("--ha", type=float, default=10)
    ap.add_argument("--area", type=float, default=1e-5)
    ap.add_argument("--lid", type=float, default=-1.0, help="lid velocity (the paper's lid moves to the left)")
    ap.add_argument("--write", default=None, help="folder for the x y value files")
    ap.add_argument("--h", type=float, default=0.01, help="grid spacing of the written points")
    a = ap.parse_args()
    s = Solver(a.area, lid=a.lid).solve(a.re, a.ri, a.ha)
    print(f"Re={a.re:g} Ri={a.ri:g} Ha={a.ha:g}: average Nusselt number on the lid = {s.nusselt():.4f}", flush=True)
    if a.write:
        os.makedirs(a.write, exist_ok=True)
        xy = grid_points(a.h)
        u, p, t = s.fields()
        n = xy.shape[1]
        U, V, P, T = (np.empty(n) for _ in range(4))
        for i in range(0, n, 500):           # in chunks: the point search is memory-hungry
            sl = slice(i, min(i + 500, n))
            vel = s.bu.probes(xy[:, sl]) @ u
            m = sl.stop - sl.start
            # ElementVector probes return the x components first, then the y components
            U[sl], V[sl] = vel[:m], vel[m:]
            P[sl] = s.bp.probes(xy[:, sl]) @ p
            T[sl] = s.bt.probes(xy[:, sl]) @ t
        # one file per output, named like the other steady templates' files
        for name, val in (("u", U), ("v", V), ("p", P - P.mean()), ("T", T)):
            np.savetxt(os.path.join(a.write, f"solution_{name}.txt"), np.c_[xy.T, val], fmt="%.6f %.6f %.10e")
        print(f"wrote {n} points to {a.write}")


if __name__ == "__main__":
    main()
