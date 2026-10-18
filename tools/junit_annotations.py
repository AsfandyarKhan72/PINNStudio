"""Print each failed or errored test in a JUnit XML file as a GitHub Actions
error annotation (used by .github/workflows/windows-tests.yml)."""
import sys
import xml.etree.ElementTree as ET


def esc(s):
    return s.replace("%", "%25").replace("\r", "").replace("\n", "%0A")


def main(path, label=""):
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as e:
        # no report: the test process itself crashed -- show where, from
        # the end of its output (pytest.log, written by the workflow)
        tail = ""
        try:
            with open("pytest.log", encoding="utf-8", errors="replace") as f:
                tail = f.read()[-3000:]
        except OSError:
            pass
        print(f"::error title={esc((label + ': ' if label else '') + 'No test report (crash?)')}::"
              f"{esc(str(e) + chr(10) + tail)}")
        return
    n = 0
    for case in root.iter("testcase"):
        for kind in ("failure", "error"):
            el = case.find(kind)
            if el is None:
                continue
            n += 1
            name = f"{label + ': ' if label else ''}{case.get('classname', '')}::{case.get('name', '')}"
            text = ((el.get("message") or "") + "\n" + (el.text or ""))[-3500:]
            print(f"::error title={esc(name)}::{esc(text)}")
    print(f"{n} failing tests")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):    # test messages with non-ASCII text on a cp1252 console
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
