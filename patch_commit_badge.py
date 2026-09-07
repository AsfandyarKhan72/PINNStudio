path = "README.md"
with open(path, encoding="utf-8") as f:
    content = f.read()
anchor = "  <a href=\"https://github.com/AsfandyarKhan72/PINNStudio/stargazers\"><img src=\"https://img.shields.io/github/stars/AsfandyarKhan72/PINNStudio?style=social\" alt=\"GitHub stars\"></a>\n"
count = content.count(anchor)
assert count == 1, f"anchor: expected 1 match, found {count}"
new_badge = "  <a href=\"https://github.com/AsfandyarKhan72/PINNStudio/commits/main\"><img src=\"https://img.shields.io/github/commit-activity/m/AsfandyarKhan72/PINNStudio\" alt=\"Commit activity\"></a>\n"
content = content.replace(anchor, anchor + new_badge, 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("README.md updated: added monthly commit-activity badge.")
