"""Smoke tests for the Shiny GUI: the app imports, builds its page and serves it."""

import pytest

pytest.importorskip("shiny")
pytest.importorskip("faicons")

STAGE_TITLES = ["Simulation Design", "Data Diagnostics", "Model Diagnostics", "Model Explorer", "PoD Analysis"]


def test_app_is_a_shiny_app():
    import shiny

    from digiqual.gui import app as gui_app

    assert isinstance(gui_app.app, shiny.App)
    # Names other code relies on
    assert isinstance(gui_app.MATHJAX_SRC, str) and gui_app.MATHJAX_SRC.endswith(".js")


def test_page_contains_all_stages_and_styles():
    from digiqual.gui.app import app_ui

    html = str(app_ui)
    for name in STAGE_TITLES:
        assert name in html, name
    # One hidden navset holds the start page and the stages; there is no navbar
    assert 'id="stage"' in html and 'id="navbar"' not in html
    assert 'id="stepper_ui"' in html
    # The stylesheet from www/app.css is inlined in the page head
    assert "--bs-primary: #006abc" in html
    # Input IDs the docs refer to
    for input_id in ["upload_csv", "input_cols", "outcome_col", "btn_run_diagnostics", "generate_btn", "btn_restart",
                     "pod_n_boot", "btn_run_uq"]:
        assert f'id="{input_id}"' in html, input_id


def test_start_page_and_back_next_navigation():
    from html import unescape

    from digiqual.gui.app import app_ui
    from digiqual.gui.components import STAGES

    html = unescape(str(app_ui))
    # The two entry points on the start page; Home and About in the header
    assert "Plan simulations" in html and "Analyse results" in html
    assert 'data-value="about"' in html and "Supported by" in html and "Methodology" in html
    # Every stage's Back / Next goes to its neighbours
    for i, stage in enumerate(STAGES):
        assert f'data-value="{stage.value}"' in html, stage.value
        if i + 1 < len(STAGES):
            assert f"Next: {STAGES[i + 1].title}" in html
    for value in ["start", "about", *[s.value for s in STAGES]]:
        assert f'"goto_stage", "{value}"' in html, value


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_server_serves_the_page():
    pytest.importorskip("httpx")  # needed by starlette's TestClient
    from starlette.testclient import TestClient

    from digiqual.gui.app import app

    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        for name in STAGE_TITLES:
            assert name in response.text, name
        for old in ["Experimental Design", "Simulation Diagnostics", "PoD Explorer", "UQ Analysis"]:
            assert old not in response.text, old
        assert client.get("/favicon.png").status_code == 200


def test_stage_values_are_unique():
    from digiqual.gui.components import STAGES

    values = [s.value for s in STAGES]
    assert len(values) == len(set(values)) and "start" not in values
    assert [s.number for s in STAGES] == list(range(1, len(STAGES) + 1))


def _snapshot(**changes):
    base = {"generated_rows": None, "n_rows": None, "n_inputs": None, "diagnostics": None,
            "diagnostics_stale": False, "model_name": None, "cv_r2": None, "min_r2": 0.5,
            "applied": False, "dirty": False, "plot": [], "slice": {}, "uq_value": None, "uq_stale": False}
    return {**base, **changes}


def test_stepper_statuses_at_the_start():
    from digiqual.gui.stepper import stage_statuses

    st = stage_statuses(_snapshot())
    assert st["design"] == ("todo", "optional")
    assert st["data"] == ("todo", "upload results")
    assert st["model"][0] == st["explorer"][0] == st["pod"][0] == "locked"


def test_stepper_statuses_through_the_workflow():
    from digiqual.gui.stepper import stage_statuses

    data = dict(generated_rows=100, n_rows=250, n_inputs=2)
    st = stage_statuses(_snapshot(**data))
    assert st["design"] == ("done", "100 runs")
    assert st["data"] == ("todo", "250 rows · 2 inputs · run checks")

    assert stage_statuses(_snapshot(**data, diagnostics="passed", fitting=True))["model"] == ("running", "fitting models…")
    st = stage_statuses(_snapshot(**data, diagnostics="failed"))
    assert st["data"][0] == "warning" and st["model"] == ("todo", "run the models")
    assert stage_statuses(_snapshot(**data, diagnostics="passed", diagnostics_stale=True))["data"][0] == "stale"

    fitted = dict(data, diagnostics="passed", model_name="Poly 1", cv_r2=0.83)
    st = stage_statuses(_snapshot(**fitted))
    assert st["data"] == ("done", "250 rows · 2 inputs")
    assert st["model"] == ("done", "Poly 1 · R² 0.83")
    assert st["explorer"] == ("todo", "set up the PoD") and st["pod"][0] == "locked"
    assert stage_statuses(_snapshot(**{**fitted, "cv_r2": 0.3}))["model"][0] == "warning"

    applied = dict(fitted, applied=True, plot=["Length"], slice={"Angle": 3.390576})
    st = stage_statuses(_snapshot(**applied))
    assert st["explorer"] == ("done", "Length · Angle = 3.391")
    assert st["pod"] == ("todo", "run the bootstrap")
    st = stage_statuses(_snapshot(**applied, dirty=True))
    assert st["explorer"] == ("stale", "update needed") and st["pod"][0] == "locked"

    assert stage_statuses(_snapshot(**applied, pod_running=True))["explorer"] == ("running", "computing PoD…")
    assert stage_statuses(_snapshot(**applied, uq_running=True))["pod"] == ("running", "bootstrap running…")
    st = stage_statuses(_snapshot(**applied, uq_value="a90/95 = 7.244"))
    assert st["pod"] == ("done", "a90/95 = 7.244")
    assert stage_statuses(_snapshot(**applied, uq_value="a90/95 = 7.244", uq_stale=True))["pod"][0] == "stale"


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
