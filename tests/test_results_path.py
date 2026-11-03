"""Windows without long paths: a results folder too long for the files a run
saves in it is caught before Solve, instead of failing while saving after
training ("[WinError 206] The filename or extension is too long")."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinnstudio.core import pyexe  # noqa: E402


def _folder(n):
    f = "C:\\Users\\A B\\PINNStudio_Results\\"
    return f + "x" * (n - len(f))


def test_short_folder_is_fine():
    assert pyexe.results_path_problem(_folder(120), platform="win32", long_paths=False) is None


def test_long_folder_is_caught():
    msg = pyexe.results_path_problem(_folder(227), platform="win32", long_paths=False)
    assert msg and "227 characters" in msg and "shorter" in msg


def test_sweeps_need_more_room():
    assert pyexe.results_path_problem(_folder(180), platform="win32", long_paths=False) is None
    assert pyexe.results_path_problem(_folder(180), sweep=True, platform="win32", long_paths=False)


def test_only_windows_without_long_paths():
    assert pyexe.results_path_problem(_folder(240), platform="win32", long_paths=True) is None
    assert pyexe.results_path_problem("/home/a/" + "x" * 300, platform="linux") is None
    assert pyexe.results_path_problem("", platform="win32", long_paths=False) is None
