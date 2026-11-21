"""The application's state that views observe (MVC's model, state tier).

Imports only PySide6.QtCore and this package (tests/test_mvc.py, rule R2). Views read it and connect to
its signals; only controllers call its mutators (rule R5).
"""
