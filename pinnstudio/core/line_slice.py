"""Where the line plots of a 2D/3D problem cut the domain: y (and z, in
3D) from Plot settings -> Line, the middle of the domain when "middle of
the domain" is ticked. Nothing is chosen behind the user's back -- an
example whose middle line shows nothing (Helmholtz 2D: u = 0 on it) sets
its own value there, where it can be seen and changed."""


def auto_line_slice(config):
    """(y, z) for the line plots."""
    y = (config.y_min + config.y_max) / 2.0 if getattr(config, "line_slice_y_auto", True) else config.line_slice_y
    z = (config.z_min + config.z_max) / 2.0 if getattr(config, "line_slice_z_auto", True) else config.line_slice_z
    return y, z
