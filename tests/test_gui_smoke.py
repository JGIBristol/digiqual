"""Smoke tests for the Shiny GUI: the app imports, builds its page and serves it."""

import pytest

pytest.importorskip("shiny")
pytest.importorskip("faicons")

TAB_NAMES = ["Experimental Design", "Simulation Diagnostics", "Model Fit", "PoD Explorer", "UQ Analysis"]


def test_app_is_a_shiny_app():
    import shiny

    from digiqual.gui import app as gui_app

    assert isinstance(gui_app.app, shiny.App)
    # Names other code relies on
    assert gui_app.logger.name == "digiqual.gui"
    assert isinstance(gui_app.MATHJAX_SRC, str) and gui_app.MATHJAX_SRC.endswith(".js")


def test_page_contains_all_tabs_and_styles():
    from digiqual.gui.app import app_ui

    html = str(app_ui)
    for name in ["Home", *TAB_NAMES]:
        assert name in html, name
    # The stylesheet from www/app.css is inlined in the page head
    assert "--bs-primary: #006abc" in html
    # Input IDs the docs refer to
    for input_id in ["upload_csv", "input_cols", "outcome_col", "btn_run_diagnostics", "generate_btn", "btn_restart"]:
        assert f'id="{input_id}"' in html, input_id


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_server_serves_the_page():
    pytest.importorskip("httpx")  # needed by starlette's TestClient
    from starlette.testclient import TestClient

    from digiqual.gui.app import app

    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        for name in TAB_NAMES:
            assert name in response.text, name
        assert "one standard error" in response.text  # Home page model-selection note
        assert client.get("/favicon.png").status_code == 200
