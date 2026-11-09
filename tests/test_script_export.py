"""The 'Python Script' export repeats the app's analysis exactly, for any data and choices.

Each scenario runs the analysis the way the app does (same CSV cleaning, diagnostics,
candidate fit, then `analysis_kwargs` for the `pod()` arguments), generates the script
from the same settings, runs it on the same file and compares the results.
"""

import runpy
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shiny")

THRESHOLDS = {"ui_max_gap": 0.2, "ui_min_r2": 0.5, "ui_avg_cv": 0.1, "ui_max_cv": 0.2, "ui_max_vif": 5.0}


def _data(n=60, seed=1):
    rng = np.random.default_rng(seed)
    length = rng.uniform(0, 10, n)
    angle = rng.uniform(-45, 45, n)
    rough = rng.uniform(0.1, 0.4, n)
    depth = rng.uniform(0.5, 2.0, n)
    signal = 2 * length + 0.05 * angle - 4 * rough + depth + rng.normal(0, 0.4, n)
    return pd.DataFrame({"Length": length, "Angle": angle, "Roughness": rough, "Depth": depth, "Signal": signal})


def _app_and_script(tmp_path, monkeypatch, raw, *, csv_name, inputs, outcome, plot, integrate=None,
                    slices=None, choice="auto", n_boot=10, n_jobs=1, bom=False, target=90, conf=95):
    """Run the analysis as the app does and as the exported script does; return both results."""
    import matplotlib
    import matplotlib.pyplot
    matplotlib.use("Agg")

    from digiqual import SimulationStudy
    from digiqual.gui.script_export import analysis_script
    from digiqual.gui.stage_model import _parse_choice
    from digiqual.gui.state import analysis_kwargs
    from digiqual.io import clean_columns

    path = tmp_path / (csv_name or "results.csv")
    raw.to_csv(path, index=False, encoding="utf-8-sig" if bom else "utf-8")

    # --- The app ---
    df, _, _ = clean_columns(pd.read_csv(path))
    inputs = inputs(df) if callable(inputs) else inputs
    study = SimulationStudy()
    study.add_data(df, outcome_col=outcome, input_cols=inputs, overwrite=True)
    study.diagnose(**{"max_gap_ratio": 0.2, "min_r2_score": 0.5, "max_avg_cv": 0.1, "max_max_cv": 0.2,
                      "max_allowed_vif": 5.0})
    study.fit_model()  # Run models: all candidates, Auto
    override, degree = _parse_choice(choice)
    fit = study.fit_model(model_override=override, force_degree=degree)

    integrate = integrate or {}
    slice_values = {c: round(float(study.clean_data[c].median()), 3)
                    for c in inputs if c not in plot and c not in integrate}
    slice_values.update(slices or {})
    threshold = round(float(study.clean_data[outcome].median()), 3)
    config = {"model": fit["model_key"], "plot": list(plot), "integrate": integrate,
              "slice": slice_values, "threshold": threshold}
    kwargs = analysis_kwargs(study, config)
    if len(plot) == 1:
        study.compute_pod_spectrum(n_threshold_points=100, **kwargs)
    expected = study.pod(threshold=threshold, n_boot=n_boot, n_jobs=n_jobs, **kwargs)

    # --- The exported script ---
    settings = {"n_boot": n_boot, "n_jobs": n_jobs, "target": target, "conf": conf, "csv_name": csv_name,
                "inputs": inputs, "outcome": outcome, "thresholds": THRESHOLDS}
    text = analysis_script(config, choice, settings, used_dists=expected["nuisance_dists"])
    script = tmp_path / "digiqual_analysis.py"
    script.write_text(text)
    monkeypatch.chdir(tmp_path)
    got = runpy.run_path(str(script), run_name="__main__")["results"]
    matplotlib.pyplot.close("all")  # the script draws its plots
    return expected, got, text


def _assert_same(expected, got):
    for key in ("pod", "ci_lower", "ci_upper", "mean_response"):
        np.testing.assert_allclose(np.asarray(got["curves"][key], dtype=float),
                                   np.asarray(expected["curves"][key], dtype=float), equal_nan=True, err_msg=key)
    assert got["reliability_table"].keys() == expected["reliability_table"].keys()
    np.testing.assert_allclose(list(got["reliability_table"].values()),
                               list(expected["reliability_table"].values()), equal_nan=True)
    for key in ("poi_cols", "nuisance_cols", "slice_values", "nuisance_bounds", "threshold", "n_boot"):
        assert got[key] == expected[key], key
    assert (got["nuisance_dists"] or {}) == (expected["nuisance_dists"] or {})
    assert got["mean_model"].model_type_ == expected["mean_model"].model_type_
    if expected["mean_model"].model_type_ == "Polynomial":
        assert got["mean_model"].model_params_ == expected["mean_model"].model_params_


SCENARIOS = {
    # One plotted variable, everything else sliced at the median, Auto model
    "basic-auto": dict(plot=["Length"]),
    # Kriging, a narrowed Normal integration and an exact slice value
    "kriging-normal-integration": dict(plot=["Length"], choice="kriging",
                                       integrate={"Angle": {"dist": "Normal", "bounds": (-20.0, 30.0)}},
                                       slices={"Roughness": 0.234}),
    # Two plotted variables (a PoD surface), Uniform integration, a forced polynomial
    "two-plotted": dict(plot=["Length", "Angle"], choice="poly-2",
                        integrate={"Roughness": {"dist": "Uniform", "bounds": (0.1, 0.4)}}),
    # Two integrated variables with Lognormal and Weibull, run in parallel
    "lognormal-weibull": dict(plot=["Length"], choice="poly-3", n_jobs=-1,
                              integrate={"Roughness": {"dist": "Lognormal", "bounds": (0.15, 0.35)},
                                         "Depth": {"dist": "Weibull", "bounds": (0.5, 2.0)}}),
    # Lognormal can't be fitted to negative angles: the app uses Uniform, so must the script
    "distribution-fallback": dict(plot=["Length"],
                                  integrate={"Angle": {"dist": "Lognormal", "bounds": (-45.0, 45.0)}}),
    # Integration limits beyond the data are clamped to it
    "bounds-clamped": dict(plot=["Length"], integrate={"Angle": {"dist": "Uniform", "bounds": (-1000.0, 1000.0)}}),
    # Only some numeric columns used as inputs; other target PoD and confidence
    "input-subset": dict(plot=["Angle"], inputs=["Length", "Angle"], target=50, conf=99),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_script_reproduces_the_app(name, tmp_path, monkeypatch):
    options = dict(SCENARIOS[name])
    inputs = options.pop("inputs", ["Length", "Angle", "Roughness", "Depth"])
    expected, got, text = _app_and_script(tmp_path, monkeypatch, _data(), csv_name="results.csv",
                                          inputs=inputs, outcome="Signal", **options)
    _assert_same(expected, got)
    if name == "distribution-fallback":
        assert "estimate_nuisance_distribution" not in text and "integrated uniformly" in text
    if name == "two-plotted":
        assert "reliability_matrix" not in text and expected["reliability_table"] == {}


def test_script_with_messy_file_and_column_names(tmp_path, monkeypatch):
    """Spaces, brackets, a BOM, duplicate and text columns, and an awkward file name."""
    raw = _data()
    raw.columns = [" Crack Length (mm)", "Angle", "Roughness", "Angle", "Signal value"]
    raw["Run label"] = [f"run-{i}" for i in range(len(raw))]
    expected, got, text = _app_and_script(
        tmp_path, monkeypatch, raw, csv_name="my results (v2)'s.csv", bom=True,
        inputs=lambda df: [c for c in df.columns if c != "Signal_value"], outcome="Signal_value",
        plot=["Crack_Length_mm"],
    )
    _assert_same(expected, got)
    assert "Run_label" not in text  # text columns are dropped by the cleaning


def test_script_without_a_file_name(tmp_path, monkeypatch):
    expected, got, text = _app_and_script(tmp_path, monkeypatch, _data(), csv_name=None,
                                          inputs=["Length", "Angle"], outcome="Signal", plot=["Length"])
    _assert_same(expected, got)
    assert "DATA_FILE = 'results.csv'" in text


def test_script_with_more_than_1000_runs(tmp_path, monkeypatch):
    """Kriging isn't fitted above 1000 runs; Auto then picks among the polynomials."""
    expected, got, _ = _app_and_script(tmp_path, monkeypatch, _data(n=1001), csv_name="big.csv",
                                       inputs=["Length", "Angle"], outcome="Signal", plot=["Length"], n_boot=5)
    _assert_same(expected, got)
    assert expected["mean_model"].model_type_ == "Polynomial"


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed")
@pytest.mark.parametrize("name", ["kriging-normal-integration", "two-plotted", "distribution-fallback"])
def test_exported_script_is_lint_clean(name, tmp_path):
    from digiqual.gui.script_export import analysis_script

    options = SCENARIOS[name]
    integrate = options.get("integrate", {})
    used = {c: ("norm", (0.0, 1.0)) for c, s in integrate.items() if s["dist"] not in ("Uniform", "Lognormal")}
    config = {"model": ("Polynomial", 2), "plot": options["plot"], "integrate": integrate,
              "slice": {"Depth": 1.0}, "threshold": 12.0}
    settings = {"n_boot": 10, "n_jobs": 1, "target": 90, "conf": 95, "csv_name": "results.csv",
                "inputs": ["Length", "Angle", "Roughness", "Depth"], "outcome": "Signal", "thresholds": THRESHOLDS}
    script = tmp_path / "digiqual_analysis.py"
    script.write_text(analysis_script(config, "auto", settings, used_dists=used))
    result = subprocess.run(["ruff", "check", "--select", "E4,E7,E9,F,W,B,I", "--no-cache", str(script)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout
