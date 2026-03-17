from unittest.mock import patch
from digiqual import dq_ui

@patch("subprocess.Popen")
@patch("pathlib.Path.exists")
def test_dq_ui_found_location(mock_exists, mock_popen):
    # Mock exists to return True on the first path
    mock_exists.side_effect = [True]

    dq_ui()
    mock_popen.assert_called_once()

@patch("subprocess.Popen")
@patch("pathlib.Path.exists")
def test_dq_ui_not_found(mock_exists, mock_popen, capsys):
    # Force exists to False for all paths checked loop
    mock_exists.return_value = False

    dq_ui()

    mock_popen.assert_not_called()

    captured = capsys.readouterr()
    assert "Critical Error" in captured.out
