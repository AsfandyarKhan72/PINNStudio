"""The port through which controllers ask the user something.

A controller never opens a dialog itself: it calls these methods, which the window implements with
dialogs (`pinnstudio/ui/designer/window_prompter.py`) and tests implement with fixed answers. A new
question is added here, in WindowPrompter and in a test's fake together.
"""
from __future__ import annotations


class Prompter:
    """Answers that need the user. Every method here has a safe default (no, nothing), so a controller
    built without a window refuses rather than acts."""

    def ask_open_path(self, title: str, file_filter: str) -> str:
        """A file to read, or "" when cancelled."""
        return ""

    def ask_save_path(self, title: str, default_name: str, file_filter: str) -> str:
        """A file to write, or "" when cancelled."""
        return ""

    def confirm_trust(self, items: list) -> bool:
        """Trust these items (code and files, see netdesign.trust.needed)? True to run them."""
        return False

    def ask_text(self, title: str, label: str, text: str = "") -> str | None:
        """A line of text, or None when cancelled."""
        return None

    def inform(self, title: str, text: str) -> None:
        """Tell the user something that needs no answer."""
