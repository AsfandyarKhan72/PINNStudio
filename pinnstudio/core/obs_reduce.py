"""Inverse -> Measured data -> Use: how a measured-data file is laid out
(a grid or scattered points, how many times) and how many points a
reduction keeps, for the window's summary line. The reduction itself runs
in the generated scripts (codegen.OBS_REDUCE_HELPER), which this mirrors."""
import os
import re

import numpy as np

_cache = {}


def coord_columns(problem_dim, steady):
    """(coordinate columns, space columns) of a measured-data file: x, y, z
    as the problem has them, plus t unless it is steady. An ODE's only
    input is t, read as its one axis."""
    n_space = {"3D": 3, "2D": 2}.get(problem_dim, 1)
    if problem_dim == "ODE" or steady:
        return n_space, n_space
    return n_space + 1, n_space


def read_points(path):
    """The numbers of a data file (comma or space separated, header and
    comment lines skipped), cached until the file changes. None if it
    can't be read."""
    try:
        key = (os.path.abspath(path), os.path.getmtime(path))
    except OSError:
        return None
    if key in _cache:
        return _cache[key]
    rows = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line[0] in "#%":
                    continue
                try:
                    rows.append([float(v) for v in re.split(r"[,\s;]+", line) if v])
                except ValueError:
                    continue
    except OSError:
        return None
    width = max((len(r) for r in rows), default=0)
    data = np.array([r for r in rows if len(r) == width]) if rows else np.zeros((0, 0))
    if len(_cache) > 32:
        _cache.clear()
    _cache[key] = data
    return data


def describe(data, n_coord, n_space):
    """{'points', 'times', 'grid': per-axis counts or None}. 'grid' is set
    when every time in the file is the same full grid of the space axes."""
    if data is None or data.ndim != 2 or len(data) == 0 or data.shape[1] < n_coord:
        return {"points": 0, "times": 0, "grid": None}
    if n_coord > n_space:
        groups = [data[data[:, n_coord - 1] == t] for t in np.unique(data[:, n_coord - 1])]
    else:
        groups = [data]
    shapes = []
    for g in groups:
        counts = tuple(len(np.unique(g[:, k])) for k in range(n_space))
        shapes.append(counts if int(np.prod(counts)) == len(g) else None)
    grid = shapes[0] if shapes and all(s == shapes[0] and s is not None for s in shapes) else None
    if grid is not None and int(np.prod(grid)) < 2:
        grid = None          # one point per time: scattered (x, t) data, not a grid
    return {"points": len(data), "times": len(groups), "grid": grid}


def kept(info, mode, n, n_space, n_coord=None):
    """How many points a reduction keeps (as the script will). A file that
    is not a grid keeps n ** (axes, time counted) random points in grid mode."""
    if not info["points"]:
        return 0
    if mode == "grid":
        if info["grid"] is None:
            return min(info["points"], max(2, n) ** (n_coord or n_space))
        return int(np.prod([min(c, max(2, n)) for c in info["grid"]])) * info["times"]
    if mode == "random":
        return min(info["points"], max(1, n))
    return info["points"]


def summary(paths, problem_dim, steady, mode, n):
    """One line for the window: what the files hold, and what is kept."""
    n_coord, n_space = coord_columns(problem_dim, steady)
    total = kept_total = 0
    grids, scattered, readable = [], 0, 0
    for p in paths:
        info = describe(read_points(p), n_coord, n_space)
        if not info["points"]:
            continue
        readable += 1
        total += info["points"]
        kept_total += kept(info, mode, n, n_space, n_coord)
        if info["grid"]:
            grids.append((info["grid"], info["times"]))
        else:
            scattered += 1
    if not readable:
        return ""
    if grids and not scattered and len({g for g in grids}) == 1:
        shape, times = grids[0]
        what = (" × ".join(str(c) for c in shape) + " grid") if len(shape) > 1 else f"{shape[0]:,} points along one axis"
        if times > 1:
            what += f" at {times} times"
        if readable > 1:
            what += f", {readable} files"
    elif not grids:
        what = "scattered points" + (f", {readable} files" if readable > 1 else "")
    else:
        what = f"{readable} files"
    line = f"{what}, {total:,} points"
    if mode in ("grid", "random") and kept_total != total:
        if mode == "grid" and grids and not scattered and len({g for g in grids}) == 1:
            shape = grids[0][0]
            if len(shape) > 1:
                line += " → " + " × ".join(str(min(c, max(2, n))) for c in shape) + f" = {kept_total:,} points"
            else:
                line += f" → {min(shape[0], max(2, n)):,} along it = {kept_total:,} points"
        else:
            line += f" → {kept_total:,} points"
        if mode == "grid" and scattered:
            line += " (not a grid: random points instead)"
    return line
