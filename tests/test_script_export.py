"""The 'Python Script' export repeats the app's analysis and reproduces its results."""

import runpy
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shiny")


@pytest.fixture
def raw_csv(tmp_path):
    """Results as a user might export them: spaces in names and a text column."""
    rng = np.random.default_rng(1)
    n = 80
    length = rng.uniform(0, 10, n)
    angle = rng.uniform(-45, 45, n)
    rough = rng.uniform(0.1, 0.4, n)
    df = pd.DataFrame({
        "Length": length,
        "Angle": angle,
        "Roughness": rough,
        "Signal value": 2 * length + 0.05 * angle - 4 * rough + rng.normal(0, 0.4, n),
        "Run label": [f"run-{i}" for i in range(n)],
    })
    path = tmp_path / "results.csv"
    df.to_csv(path, index=False)
    return path


CONFIG = {
    "model": ("Polynomial", 2),
    "plot": ["Length"],
    "integrate": {"Angle": {"dist": "Normal", "bounds": (-30.0, 40.0)}},
    "slice": {"Roughness": 0.25},
    "threshold": 12.0,
}
SETTINGS = {
    "n_boot": 20, "n_jobs": 1, "target": 90, "conf": 95, "csv_name": "results.csv",
    "inputs": ["Length", "Angle", "Roughness"], "outcome": "Signal_value",
    "thresholds": {"ui_max_gap": 0.2, "ui_min_r2": 0.5, "ui_avg_cv": 0.1, "ui_max_cv": 0.2, "ui_max_vif": 5.0},
}


def _app_results(csv_path):
    """The analysis as the app runs it (see AppState.pod_kwargs)."""
    from digiqual import SimulationStudy
    from digiqual.integration import estimate_nuisance_distribution
    from digiqual.io import clean_columns

    df, dropped, _ = clean_columns(pd.read_csv(csv_path))
    assert dropped == ["Run_label"] and "Signal_value" in df
    study = SimulationStudy()
    study.add_data(df, outcome_col="Signal_value", input_cols=SETTINGS["inputs"])
    study.diagnose()
    study.fit_model(model_override="polynomial", force_degree=2)
    return study.pod(
        poi_col=["Length"], nuisance_col=["Angle"], slice_values={"Roughness": 0.25},
        nuisance_dists={"Angle": estimate_nuisance_distribution(study.clean_data["Angle"], "Normal")},
        nuisance_bounds={"Angle": (-30.0, 40.0)}, model_override="polynomial", force_degree=2,
        threshold=12.0, n_boot=20, n_jobs=1,
    )


def test_exported_script_reproduces_the_app(raw_csv, tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")

    from digiqual.gui.script_export import analysis_script

    script = tmp_path / "digiqual_analysis.py"
    script.write_text(analysis_script(CONFIG, "poly-2", SETTINGS))

    monkeypatch.chdir(tmp_path)  # DATA_FILE is relative to the script's folder
    namespace = runpy.run_path(str(script), run_name="__main__")
    expected = _app_results(raw_csv)

    results = namespace["results"]
    for key in ("pod", "ci_lower", "ci_upper", "mean_response"):
        np.testing.assert_allclose(results["curves"][key], expected["curves"][key])
    assert results["reliability_table"].keys() == expected["reliability_table"].keys()
    np.testing.assert_allclose(list(results["reliability_table"].values()),
                               list(expected["reliability_table"].values()))
    assert results["nuisance_bounds"] == {"Angle": (-30.0, 40.0)}


def test_exported_script_without_integration_or_with_two_plotted():
    from digiqual.gui.script_export import analysis_script

    config = {"model": ("Kriging", None), "plot": ["Length", "Angle"], "integrate": {},
              "slice": {"Roughness": 0.25}, "threshold": 12.0}
    text = analysis_script(config, "auto", SETTINGS)
    assert "nuisance_dists = None" in text
    assert "estimate_nuisance_distribution" not in text
    assert "reliability_matrix" not in text and "plot_pod_vs_threshold" not in text  # need 1 plotted variable
    assert "model_override': 'kriging'" in text and "Auto (the cross-validation pick)" in text
    compile(text, "digiqual_analysis.py", "exec")


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed")
def test_exported_script_is_lint_clean(tmp_path):
    from digiqual.gui.script_export import analysis_script

    script = tmp_path / "digiqual_analysis.py"
    script.write_text(analysis_script(CONFIG, "poly-2", SETTINGS))
    result = subprocess.run(["ruff", "check", "--select", "E4,E7,E9,F,W,B,I", "--no-cache", str(script)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout
