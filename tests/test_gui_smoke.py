"""Smoke tests for the Shiny GUI: the app imports, builds its page and serves it."""

import pytest

pytest.importorskip("shiny")
pytest.importorskip("faicons")

TAB_NAMES = ["Simulation Design", "Data Diagnostics", "Model Explorer", "PoD Analysis"]


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
        for old in ["Experimental Design", "Simulation Diagnostics", "Model Diagnostics", "PoD Explorer", "UQ Analysis"]:
            assert old not in response.text, old
        assert client.get("/favicon.png").status_code == 200


def test_tab_values_are_unique_and_in_the_page():
    from digiqual.gui.app import app_ui
    from digiqual.gui.components import TABS

    values = [t.value for t in TABS]
    assert len(values) == len(set(values))
    html = str(app_ui)
    for value in values:
        assert f'data-value="{value}"' in html, value


def test_diagnostic_failures_lists_failed_tests_in_order():
    import pandas as pd

    from digiqual.gui.components import diagnostic_failures

    diag = pd.DataFrame({
        "Test": ["Collinearity Check", "Input Coverage", "Input Coverage", "Model Fit (CV)"],
        "Variable": ["A", "A", "B", "Signal"],
        "Pass": [False, True, False, True],
    })
    failures = diagnostic_failures(diag)
    assert [f[0] for f in failures] == ["Input Coverage", "Collinearity Check"]
    assert "B" in failures[0][2] and "A" not in failures[0][2]
    assert diagnostic_failures(None) == []


def test_curve_export_df_handles_1d_2d_and_missing_bounds():
    import numpy as np

    from digiqual.gui.components import curve_export_df

    one = {"poi_cols": ["L"], "X_eval": np.linspace(0, 1, 5).reshape(-1, 1),
           "curves": {"pod": np.ones(5), "ci_lower": None, "ci_upper": None}}
    df = curve_export_df(one)
    assert list(df.columns) == ["L", "pod_mean", "ci_lower", "ci_upper"]
    assert df["ci_lower"].isna().all()

    two = {"poi_cols": ["L", "A"], "X_eval": np.zeros((4, 2)),
           "curves": {"pod": np.ones(4), "ci_lower": np.zeros(4), "ci_upper": np.ones(4)}}
    assert list(curve_export_df(two).columns) == ["L", "A", "pod_mean", "ci_lower", "ci_upper"]


def test_config_errors():
    from digiqual.gui.components import config_errors, setup_rows

    good = {"model": ("Polynomial", 2), "plot": ["L"], "integrate": {"A": {"dist": "Uniform", "bounds": (0.0, 1.0)}},
            "slice": {"B": 2.0}, "threshold": 0.5}
    assert config_errors(good) == []
    assert "Integrated: A" in setup_rows(good, "Kriging")

    assert config_errors({**good, "plot": []})
    assert config_errors({**good, "plot": ["L", "B", "C"]})
    assert config_errors({**good, "integrate": {c: {"dist": "Uniform", "bounds": (0, 1)} for c in "ABC"}})
    assert config_errors({**good, "integrate": {"A": {"dist": "Uniform", "bounds": (1.0, 0.0)}}})
    assert config_errors({**good, "slice": {"B": None}})
    assert config_errors({**good, "threshold": None})
    assert config_errors(None)


def test_fmt_value_uses_3_decimal_places():
    from digiqual.gui.components import fmt_value

    assert fmt_value(3.390576318415282) == "3.391"
    assert fmt_value(-16.6) == "-16.600"
    assert fmt_value(0.000123456) == "0.0001235"  # tiny values keep 4 significant figures
