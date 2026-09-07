path = "pinnstudio/ui/main_window.py"
with open(path, encoding="utf-8") as f:
    content = f.read()
old1 = "0.0001*(du_xx + du_yy) + 5*(u**3 - u)"
assert content.count(old1) == 1, f"anchor1: expected 1 match, found {content.count(old1)}"
content = content.replace(old1, "0.0001*(du_xx + du_yy) + (u**3 - u)", 1)
old2 = "mu - (u**3 - u) + 0.05*2*(du_xx + du_yy)"
assert content.count(old2) == 1, f"anchor2: expected 1 match, found {content.count(old2)}"
content = content.replace(old2, "mu - (u**3 - u) + 0.05**2*(du_xx + du_yy)", 1)
old3 = "'x_min': -0.5, 'x_max': 0.5,"
assert content.count(old3) == 1, f"anchor3: expected 1 match, found {content.count(old3)}"
content = content.replace(old3, "'x_min': -1.0, 'x_max': 1.0,", 1)
old4 = "'y_min': -0.5, 'y_max': 0.5,"
assert content.count(old4) == 1, f"anchor4: expected 1 match, found {content.count(old4)}"
content = content.replace(old4, "'y_min': -1.0, 'y_max': 1.0,", 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("main_window.py updated: 2D Allen-Cahn (Mattey) coefficient fixed 5->1, 2D Cahn-Hilliard (Wight) domain fixed to [-1,1]^2 and mu coefficient fixed 0.1->0.0025 (epsilon^2).")
