"""Print each failed or errored test in a JUnit XML file as a GitHub Actions
error annotation (used by .github/workflows/windows-tests.yml)."""
import sys
import xml.etree.ElementTree as ET


def esc(s):
    return s.replace("%", "%25").replace("\r", "").replace("\n", "%0A")


def main(path):
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as e:
        print(f"::error title=No test report::{esc(str(e))}")
        return
    n = 0
    for case in root.iter("testcase"):
        for kind in ("failure", "error"):
            el = case.find(kind)
            if el is None:
                continue
            n += 1
            name = f"{case.get('classname', '')}::{case.get('name', '')}"
            text = ((el.get("message") or "") + "\n" + (el.text or ""))[-3500:]
            print(f"::error title={esc(name)}::{esc(text)}")
    print(f"{n} failing tests")


if __name__ == "__main__":
    main(sys.argv[1])
