"""Tests of the GUI's pure helpers (no running Shiny session needed)."""

import pytest

pytest.importorskip("shiny")
pytest.importorskip("faicons")


def test_parse_n_boot_rejects_small_and_non_numeric_values():
    from digiqual.gui.stage_pod import MIN_N_BOOT, parse_n_boot

    assert parse_n_boot(1000) == 1000
    assert parse_n_boot(10.0) == 10
    assert parse_n_boot("200") == 200
    for bad in (None, "", "abc", MIN_N_BOOT - 1, 0, -5, 12.5, float("nan"), float("inf")):
        assert parse_n_boot(bad) is None, bad


def test_uq_metrics_uses_the_given_target_and_confidence():
    from digiqual.gui.stage_pod import uq_metrics

    res = {"threshold": 1.5, "n_boot": 100, "poi_cols": ["Length"],
           "reliability_table": {(90, 95): 2.5, (50, 90): 1.25}}
    rows = uq_metrics(res, 90, 95)
    assert rows["a90/95 Reliability Index"] == "2.500" and rows["Confidence Level (%)"] == "95%"
    assert uq_metrics(res, 50, 90)["a50/90 Reliability Index"] == "1.250"
    assert uq_metrics(res, 99, 99)["a99/99 Reliability Index"] == "Not Reached"
    # A PoD surface has no aX/Y value
    surface = uq_metrics({**res, "poi_cols": ["Length", "Angle"]}, 90, 95)
    assert "a90/95 Reliability Index" not in surface and surface["Confidence Level (%)"] == "95%"


def test_visualise_closes_the_figures_the_app_never_shows():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from digiqual.gui.state import visualise

    class FakeStudy:
        plots = {}

        def visualise(self, show):
            assert show is False
            self.plots = {
                "signal_model": plt.subplots()[1],
                "pod_curve": plt.subplots()[1],
                "model_selection": plt.figure(),
                "kriging_diagnostics": plt.subplots()[1],
            }

    plt.close("all")
    study = FakeStudy()
    visualise(study)
    assert set(study.plots) == {"signal_model", "pod_curve"}
    # Only the two figures the explorer draws are still open in pyplot
    open_figs = {plt.figure(n) for n in plt.get_fignums()}
    assert open_figs == {study.plots["signal_model"].get_figure(), study.plots["pod_curve"].get_figure()}
    plt.close("all")


def test_explorer_input_ids_change_with_each_study():
    from digiqual.gui.state import AppState

    state = AppState(input=None)
    first = state.col_id("int_lo", "Depth")
    state.study_version += 1
    # A new dataset gets fresh controls, not the previous one's (kept) input values
    assert state.col_id("int_lo", "Depth") != first
    assert state.col_id("role", "Depth") != state.col_id("dist", "Depth")
