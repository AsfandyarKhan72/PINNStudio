path = "README.md"
with open(path, encoding="utf-8") as f:
    content = f.read()
old = "<a href=\"https://pypi.org/project/pinnstudio/\"><img src=\"https://img.shields.io/pypi/dm/pinnstudio.svg\" alt=\"PyPI downloads\"></a>"
count = content.count(old)
assert count == 1, f"expected 1 match, found {count}"
new = "<a href=\"https://pepy.tech/project/pinnstudio\"><img src=\"https://static.pepy.tech/badge/pinnstudio\" alt=\"PyPI downloads\"></a>"
content = content.replace(old, new, 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("README.md updated: downloads badge switched from shields.io/pypistats to pepy.tech (shows total downloads, unaffected by the shields.io rate-limit issue).")
