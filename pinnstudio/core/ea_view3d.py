"""3D view of a 3D box problem for Error Analysis (view3d_comparison.png).

Two rows, one column per reference time, in the style of Wight & Zhao's
Fig. 3.9: the PINN's level surface (for a phase field, the interface) inside
the box, and the |error| on the box with the corner facing the viewer cut away,
so the cuts through the middle show the inside. The reference's level surface
is drawn on the cuts as a black line.

The generated scripts do not import PinnStudio, so the helpers below are
plain source that is pasted into each script (HELPERS) and also executed here,
so the tests can call them.
"""

HELPERS = r'''
def _v3_march(V, gx, gy, gz, level):
    """Triangles (M, 3, 3) of the surface V = level on the grid gx x gy x gz
    (V indexed [ix, iy, iz]), by marching tetrahedra; each triangle is turned
    so that its normal points from V > level to V < level."""
    nx, ny, nz = V.shape
    I, J, K = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij")
    I, J, K = I.ravel(), J.ravel(), K.ravel()
    _corner = [((b & 1), (b >> 1) & 1, (b >> 2) & 1) for b in range(8)]
    tris = []
    for tet in ((0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)):
        ii = np.stack([I + _corner[c][0] for c in tet], 1)
        jj = np.stack([J + _corner[c][1] for c in tet], 1)
        kk = np.stack([K + _corner[c][2] for c in tet], 1)
        v = V[ii, jj, kk] - level
        inside = v > 0
        cnt = inside.sum(1)
        keep = (cnt > 0) & (cnt < 4) & np.isfinite(v).all(1)
        if not keep.any():
            continue
        v, inside, cnt = v[keep], inside[keep], cnt[keep]
        p = np.stack([gx[ii[keep]], gy[jj[keep]], gz[kk[keep]]], 2)        # (m, 4, 3)
        order = np.argsort(~inside, axis=1, kind="stable")                 # inside vertices first
        v = np.take_along_axis(v, order, 1)
        p = np.take_along_axis(p, order[:, :, None], 1)

        def cut(a, b, sel):
            va, vb = v[sel, a][:, None], v[sel, b][:, None]
            w = va / np.where(va - vb == 0, 1e-30, va - vb)
            return p[sel, a] + w * (p[sel, b] - p[sel, a])
        out_dir = []
        s1 = cnt == 1                                  # one vertex inside: one triangle
        if s1.any():
            tris.append(np.stack([cut(0, 1, s1), cut(0, 2, s1), cut(0, 3, s1)], 1))
            out_dir.append(p[s1, 1:].mean(1) - p[s1, 0])
        s3 = cnt == 3                                  # one vertex outside: one triangle
        if s3.any():
            tris.append(np.stack([cut(0, 3, s3), cut(1, 3, s3), cut(2, 3, s3)], 1))
            out_dir.append(p[s3, 3] - p[s3, :3].mean(1))
        s2 = cnt == 2                                  # two and two: a quad, two triangles
        if s2.any():
            a, b, c, d = cut(0, 2, s2), cut(0, 3, s2), cut(1, 3, s2), cut(1, 2, s2)
            tris.append(np.stack([a, b, c], 1)); tris.append(np.stack([a, c, d], 1))
            o = p[s2, 2:].mean(1) - p[s2, :2].mean(1)
            out_dir += [o, o]
        for k in range(len(out_dir)):
            t = tris[-len(out_dir) + k]
            nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            flip = (nrm * out_dir[k]).sum(1) < 0
            t[flip] = t[flip][:, ::-1]
    if not tris:
        return np.zeros((0, 3, 3))
    tris = np.concatenate(tris)
    area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    return tris[area > 1e-12 * max(float(area.max()), 1e-300)]   # drop the zero-area slivers


def _v3_cut_quads(gx, gy, gz, E):
    """Quads and their values for the box's three visible faces (x = max,
    y = min, z = max) with the corner x > mid, y < mid, z > mid removed, and
    the three faces of the cut."""
    nx, ny, nz = E.shape
    mx, my, mz = nx // 2, ny // 2, nz // 2
    polys, vals = [], []

    def add(plane, f, ra, rb):
        for a in range(*ra):
            for b in range(*rb):
                if plane == "x":
                    q = [(gx[f], gy[a], gz[b]), (gx[f], gy[a + 1], gz[b]), (gx[f], gy[a + 1], gz[b + 1]), (gx[f], gy[a], gz[b + 1])]
                    w = E[f, a:a + 2, b:b + 2]
                elif plane == "y":
                    q = [(gx[a], gy[f], gz[b]), (gx[a + 1], gy[f], gz[b]), (gx[a + 1], gy[f], gz[b + 1]), (gx[a], gy[f], gz[b + 1])]
                    w = E[a:a + 2, f, b:b + 2]
                else:
                    q = [(gx[a], gy[b], gz[f]), (gx[a + 1], gy[b], gz[f]), (gx[a + 1], gy[b + 1], gz[f]), (gx[a], gy[b + 1], gz[f])]
                    w = E[a:a + 2, b:b + 2, f]
                polys.append(q); vals.append(float(np.nanmean(w)))
    lx, ly, lz = nx - 1, ny - 1, nz - 1
    add("x", lx, (0, ly), (0, mz)); add("x", lx, (my, ly), (mz, lz))
    add("y", 0, (0, mx), (0, lz)); add("y", 0, (mx, lx), (0, mz))
    add("z", lz, (0, mx), (0, ly)); add("z", lz, (mx, lx), (my, ly))
    add("x", mx, (0, my), (mz, lz))
    add("y", my, (mx, lx), (mz, lz))
    add("z", mz, (mx, lx), (0, my))
    return polys, np.array(vals)


def _v3_box(ax, gx, gy, gz, cut=False, zorder=2):
    """The box's edges; with cut=True, the edges of the removed corner."""
    x0, x1, y0, y1, z0, z1 = gx[0], gx[-1], gy[0], gy[-1], gz[0], gz[-1]
    if not cut:
        for a in (y0, y1):
            for b in (z0, z1):
                ax.plot([x0, x1], [a, a], [b, b], color="0.45", lw=0.6, zorder=zorder)
        for a in (x0, x1):
            for b in (z0, z1):
                ax.plot([a, a], [y0, y1], [b, b], color="0.45", lw=0.6, zorder=zorder)
        for a in (x0, x1):
            for b in (y0, y1):
                ax.plot([a, a], [b, b], [z0, z1], color="0.45", lw=0.6, zorder=zorder)
        return
    xm, ym, zm = gx[len(gx) // 2], gy[len(gy) // 2], gz[len(gz) // 2]
    for p, q in [((xm, y0, zm), (x1, y0, zm)), ((xm, y0, zm), (xm, ym, zm)), ((xm, y0, zm), (xm, y0, z1)),
                 ((x1, y0, zm), (x1, ym, zm)), ((xm, ym, zm), (x1, ym, zm)), ((xm, ym, zm), (xm, ym, z1)),
                 ((xm, y0, z1), (xm, ym, z1)), ((x1, ym, zm), (x1, ym, z1)), ((xm, ym, z1), (x1, ym, z1))]:
        ax.plot(*zip(p, q), color="0.25", lw=0.8, zorder=zorder)


def _v3_contours(ax, gx, gy, gz, R, level):
    """The reference's level line on the three cut faces (black)."""
    import matplotlib.pyplot as _plt_v3
    nx, ny, nz = R.shape
    mx, my, mz = nx // 2, ny // 2, nz // 2
    _f = _plt_v3.figure()
    _a = _f.add_subplot(111)
    for sl, A, B, put in (
            (R[mx, :my + 1, mz:], gy[:my + 1], gz[mz:], lambda a, b: (np.full(len(a), gx[mx]), a, b)),
            (R[mx:, my, mz:], gx[mx:], gz[mz:], lambda a, b: (a, np.full(len(a), gy[my]), b)),
            (R[mx:, :my + 1, mz], gx[mx:], gy[:my + 1], lambda a, b: (a, b, np.full(len(a), gz[mz])))):
        if not (np.nanmin(sl) < level < np.nanmax(sl)):
            continue
        cs = _a.contour(B, A, sl, levels=[level])
        for seg in cs.allsegs[0]:
            if len(seg) > 1:
                ax.plot(*put(seg[:, 1], seg[:, 0]), color="k", lw=1.0, zorder=4)
    _plt_v3.close(_f)
'''

exec(compile("import numpy as np\n" + HELPERS, "<view3d helpers>", "exec"))


def view3d_block(config, indent):
    """The Error Analysis code that draws view3d_comparison.png, to sit right
    after the slice comparison of a 3D box problem (names as in that block)."""
    from pinnstudio.core.codegen import err_cmap
    lvl = str(getattr(config, "plot_iso_level", "") or "").strip()
    bad = ""
    try:
        lvl_val = repr(float(lvl)) if lvl else "None"
    except ValueError:
        lvl_val, bad = "None", lvl
    code = f'''
if _ea_did_surface and _is_3d and _geom_type == "Cuboid":
    # 3D view: the level surface inside the box and the |error| on the box
    # with one corner cut away (Plot settings -> Surface -> 3D grid
    # resolution and 3D view surface level).
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection as _P3C_v3
    import matplotlib.colors as _mcol_v3
    _bb_v3 = np.asarray(geom.bbox)
    _n_v3 = {max(10, int(getattr(config, "plot_resolution_3d", 40) or 40))}
    _gx_v3, _gy_v3, _gz_v3 = (np.linspace(_bb_v3[0][_k], _bb_v3[1][_k], _n_v3) for _k in range(3))
    _X_v3, _Y_v3, _Z_v3 = np.meshgrid(_gx_v3, _gy_v3, _gz_v3, indexing="ij")
    _lvl_v3 = {lvl_val}
    _auto_v3 = _lvl_v3 is None
    if _auto_v3:   # automatic: the middle of the reference's range at the first time, to 1% of the range
        _lo_v3, _hi_v3 = float(np.min(_ea_u_refs[0])), float(np.max(_ea_u_refs[0]))
        _sp_v3 = max(_hi_v3 - _lo_v3, 1e-30)
        _lvl_v3 = round(0.5 * (_lo_v3 + _hi_v3) / _sp_v3, 2) * _sp_v3 + 0.0
    print(f"  3D view: surface at {{_ea_field_txt(_ea_sel)}} = {{_lvl_v3:.4g}}"
          f"{{' (middle of the reference range; set it in Plot settings -> Surface)' if _auto_v3 else ''}}")
    if {bad!r}:
        print("  3D view: the surface level " + {bad!r} + " in Plot settings is not a number, so the middle of the range is used")
    _nt_v3 = len(_ea_times)
    fig = plt.figure(figsize=_panel_figsize(3.2, 3.0, _nt_v3, 2))
    fig.suptitle(_ea_title_txt(f"3D view, surface {{_ea_field_txt(_ea_sel)}} = {{_fmt_num(_lvl_v3)}}", _ea_sel), fontsize={config.plot_title_fontsize}, fontweight='bold')
    _cm_v3 = plt.get_cmap('{err_cmap(config)}')
    for _ei, _ea_tv in enumerate(_ea_times):
        _cols_v3 = [_X_v3.ravel(), _Y_v3.ravel(), _Z_v3.ravel()]
        if not _is_steady:
            _cols_v3.append(np.full(_X_v3.size, _ea_tv))
        _P_v3 = np.asarray(_ea_extract(np.column_stack(_cols_v3), _ea_sel, model)).reshape(_X_v3.shape)
        _pts_v3 = np.column_stack([_ea_x_refs[_ei], _ea_y_refs[_ei], _ea_z_refs[_ei]])
        _ux_v3, _uy_v3, _uz_v3 = (np.unique(_pts_v3[:, _k]) for _k in range(3))
        if len(_ux_v3) * len(_uy_v3) * len(_uz_v3) == len(_pts_v3) and min(len(_ux_v3), len(_uy_v3), len(_uz_v3)) > 1:
            # reference on a full grid: interpolate it exactly onto the plot grid
            from scipy.interpolate import RegularGridInterpolator as _RGI_v3
            _G_v3 = np.full((len(_ux_v3), len(_uy_v3), len(_uz_v3)), np.nan)
            _G_v3[np.searchsorted(_ux_v3, _pts_v3[:, 0]), np.searchsorted(_uy_v3, _pts_v3[:, 1]),
                  np.searchsorted(_uz_v3, _pts_v3[:, 2])] = _ea_u_refs[_ei]
            _R_v3 = _RGI_v3((_ux_v3, _uy_v3, _uz_v3), _G_v3, bounds_error=False, fill_value=None)(
                np.column_stack([_X_v3.ravel(), _Y_v3.ravel(), _Z_v3.ravel()])).reshape(_X_v3.shape)
        else:
            from scipy.interpolate import griddata as _gd_v3
            _R_v3 = _gd_v3(_pts_v3, _ea_u_refs[_ei], (_X_v3, _Y_v3, _Z_v3), method='nearest')
        _E_v3 = np.abs(_P_v3 - _R_v3)
        _tt_v3 = "" if _is_steady else f"t={{_fmt_num(_ea_tv)}}"
        for _row_v3 in (0, 1):
            _ax_v3 = fig.add_subplot(2, _nt_v3, _row_v3 * _nt_v3 + _ei + 1, projection='3d')
            _ax_v3.computed_zorder = False
            if _row_v3 == 0:
                _tri_v3 = _v3_march(_P_v3, _gx_v3, _gy_v3, _gz_v3, _lvl_v3)
                if len(_tri_v3):
                    _ax_v3.add_collection3d(_P3C_v3(_tri_v3, facecolors="#d62728", edgecolors="#d62728", linewidths=0,
                                                    shade=True, zorder=1,
                                                    lightsource=_mcol_v3.LightSource(azdeg=315, altdeg=45)))
                _v3_box(_ax_v3, _gx_v3, _gy_v3, _gz_v3, zorder=2)
                _ax_v3.set_title(f"PINN  {{_tt_v3}}".strip(),
                                 fontsize={config.plot_subplot_title_fontsize})
            else:
                _q_v3, _qv_v3 = _v3_cut_quads(_gx_v3, _gy_v3, _gz_v3, _E_v3)
                _nm_v3 = _mcol_v3.Normalize(0.0, max(float(np.nanmax(_E_v3)), 1e-30))
                _fc_v3 = _cm_v3(_nm_v3(_qv_v3))
                _ax_v3.add_collection3d(_P3C_v3(_q_v3, facecolors=_fc_v3, edgecolors=_fc_v3, linewidths=0.3, zorder=1))
                _v3_box(_ax_v3, _gx_v3, _gy_v3, _gz_v3, zorder=0)
                _v3_box(_ax_v3, _gx_v3, _gy_v3, _gz_v3, cut=True, zorder=3)
                _v3_contours(_ax_v3, _gx_v3, _gy_v3, _gz_v3, _R_v3, _lvl_v3)
                _ax_v3.set_title(f"|Error|  {{_tt_v3}}  max = {{_sci_txt(float(np.nanmax(_E_v3)))}}".replace("    ", "  "),
                                 fontsize={config.plot_subplot_title_fontsize})
                if {bool(config.plot_colorbar)}:
                    _cb_v3 = fig.colorbar(plt.cm.ScalarMappable(norm=_nm_v3, cmap=_cm_v3), ax=_ax_v3,
                                          orientation="horizontal", shrink=0.7, pad=0.02, fraction=0.05, aspect=25)
                    _cb_v3.ax.tick_params(labelsize=8)
            _ax_v3.set_xlim(_gx_v3[0], _gx_v3[-1]); _ax_v3.set_ylim(_gy_v3[0], _gy_v3[-1]); _ax_v3.set_zlim(_gz_v3[0], _gz_v3[-1])
            _ax_v3.set_box_aspect((_gx_v3[-1] - _gx_v3[0], _gy_v3[-1] - _gy_v3[0], _gz_v3[-1] - _gz_v3[0]))
            _ax_v3.view_init(elev=22, azim=-58)
            _ax_v3.set_axis_off()
    fig.text(0.5, 0.005, "Bottom row: the box with the corner facing the viewer cut away; black line: the reference surface on the cuts",
             ha="center", fontsize=8, color="0.3")
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    _ea_v3p = _os.path.join(_ea_dir, f"view3d_comparison{{_ea_suffix}}.png")
    plt.savefig(_ea_v3p, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
    print(f"  3D view saved: {{_ea_v3p}}")
'''
    pad = " " * indent
    return "\n".join((pad + ln) if ln.strip() else ln for ln in code.strip("\n").splitlines())

