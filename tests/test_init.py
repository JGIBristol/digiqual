import sys
from importlib.resources import files
from unittest.mock import patch

from digiqual import dq_ui


@patch("subprocess.Popen")
def test_dq_ui_launches_gui_module(mock_popen):
    dq_ui()
    mock_popen.assert_called_once_with([sys.executable, "-m", "digiqual.gui"])


def test_gui_static_assets_are_packaged():
    www = files("digiqual.gui") / "www"
    assert (www / "RCNDE-Logo-100.png").is_file()
    assert (www / "ukri-epsrc-square-logo.png").is_file()
