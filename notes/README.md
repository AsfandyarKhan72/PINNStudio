# Development notes

Records of investigations behind the v1.6.0 release; not user documentation and not shipped in the package.

- `pyside6-migration/`: the move from PyQt6 to PySide6. `PYSIDE6_READINESS.md` (the release-readiness report:
  the defects found, how often they occurred in the stress loop, the fixes and the checks) and `MEMORY.md` (memory
  use over long sessions, PySide6 against PyQt6).
- `licensing-proposal/`: the licensing analysis for a PySide6 release and its drafts, kept as the record of the
  decision (adopted 2026-11-02: MIT, with PySide6-Essentials, shiboken6 and Qt used under the LGPL-3.0). The LGPL
  notice template for a bundled distribution, `drafts/LGPL-NOTICE.txt`, is referred to by `THIRD_PARTY_LICENSES.md`.
