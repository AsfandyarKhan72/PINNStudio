<!-- Replaces the PyQt6 rows of THIRD_PARTY_LICENSES.md, section 2 "Installed separately". -->

| Library | Used for | License | Notes |
|---|---|---|---|
| [Qt for Python](https://doc.qt.io/qtforpython-6/) (`PySide6-Essentials`, `shiboken6`) | the graphical interface (Python binding of Qt) | LGPL-3.0-only (also offered under GPL-2.0/GPL-3.0 and a commercial license) | Installed from PyPI as separate packages, which carry their license texts. PINNStudio depends on `PySide6-Essentials` only; the `PySide6-Addons` package, which contains modules available only under the GPL, is not needed. |
| [Qt](https://www.qt.io) 6: Qt Core, Qt GUI, Qt Widgets, Qt SVG | the interface's widgets, drawing and icons | LGPL-3.0-only (open-source edition) | Shipped inside the PySide6 wheels as shared libraries; Copyright (C) The Qt Company Ltd. and other contributors. Source: https://code.qt.io. Users can replace them with their own build. |

PINNStudio does not modify, statically link or redistribute PySide6 or Qt: `pip`/`uv` install them from PyPI.
Under the LGPLv3 PINNStudio can therefore be licensed on its own terms; the LGPL applies to PySide6 and Qt
themselves. Anyone who distributes a bundle that contains them (for example a frozen installer) must meet the
LGPLv3 for those libraries: include the license texts and Qt's notices, keep them replaceable, and provide or point
to their source.
