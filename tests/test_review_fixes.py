"""Regression tests for library fixes from the gui-restructure review: spectrum
side effects and range, infinite values, failed-point filtering, confidence labels,
None distributions, constant inputs, blank columns, CLI paths and model names."""

import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

from digiqual.adaptive import _filter_by_graveyard
from digiqual.core import SimulationStudy
from digiqual.diagnostics import check_input_coverage, validate_simulation
from digiqual.executors import CLIExecutor
from digiqual.integration import estimate_nuisance_distribution
from digiqual.io import clean_columns
from digiqual.pod import calculate_sobol_indices


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def _frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"Length": rng.uniform(0.5, 5, n), "Angle": rng.uniform(-15, 15, n)})
    df["Signal"] = 2 * df.Length - 0.05 * np.abs(df.Angle) + rng.normal(0, 0.2, n)
    return df


@pytest.fixture
def study():
    st = SimulationStudy()
    with _quiet():
        st.add_data(_frame(), outcome_col="Signal")
    return st


# --- compute_pod_spectrum ---

def test_spectrum_on_fresh_data_validates_first(study):
    spec = study.compute_pod_spectrum(poi_col="Length", n_threshold_points=10)
    assert spec["pod_matrix"].shape == (100, 10)


def test_spectrum_keeps_existing_results(study):
    res = study.pod(poi_col="Length", threshold=3.0, n_boot=10)
    study.compute_pod_spectrum(poi_col="Length", n_threshold_points=10)
    assert study.pod_results is res
    assert study.pod_results["n_boot"] == 10 and study.pod_results["threshold"] == 3.0


def test_threshold_outside_spectrum_is_computed_not_clamped(study):
    study.compute_pod_spectrum(poi_col="Length", n_threshold_points=20)
    above = study.clean_data["Signal"].max() + 5
    interpolated = study.pod(poi_col="Length", threshold=above, n_boot=0)["curves"]["pod"].copy()

    fresh = SimulationStudy()
    with _quiet():
        fresh.add_data(_frame(), outcome_col="Signal")
    direct = fresh.pod(poi_col="Length", threshold=above, n_boot=0)["curves"]["pod"]
    np.testing.assert_allclose(interpolated, direct)


# --- Validation and refinement ---

def test_infinite_values_are_invalid():
    df = _frame(20)
    df.loc[3, "Signal"] = np.inf
    df.loc[5, "Length"] = -np.inf
    clean, removed = validate_simulation(df, ["Length", "Angle"], "Signal")
    assert len(clean) == 18 and set(removed.index) == {3, 5}
    assert np.isfinite(clean.to_numpy()).all()


def test_graveyard_with_missing_or_text_inputs_is_ignored():
    candidates = pd.DataFrame({"Length": [1.0, 2.0, 3.0], "Angle": [0.0, 5.0, 10.0]})
    graveyard = pd.DataFrame({"Length": [np.nan, "bad"], "Angle": [1.0, 2.0]}, dtype=object)
    kept = _filter_by_graveyard(candidates, graveyard, ["Length", "Angle"])
    assert len(kept) == 3

    # A real failed point still filters its neighbour
    graveyard = pd.DataFrame({"Length": [np.nan, 2.0], "Angle": [1.0, 5.0]})
    kept = _filter_by_graveyard(candidates, graveyard, ["Length", "Angle"])
    assert list(kept.index) == [0, 2]


def test_input_coverage_ignores_missing_values():
    df = pd.DataFrame({"x": [0.0, 0.1, np.nan, 0.2, 1.0]})
    out = check_input_coverage(df, ["x"])
    assert out["x"]["max_gap_ratio"] == pytest.approx(0.8)
    assert set(out["x"]) == {"min", "max", "max_gap_ratio", "sufficient_coverage"}


# --- Plot labels ---

def test_visualise_rejects_unknown_confidence_level(study):
    study.pod(poi_col="Length", threshold=3.0, n_boot=10)
    with pytest.raises(ValueError, match="80% confidence"):
        study.visualise(show=False, confidence_level=80)
    study.visualise(show=False, confidence_level=90)  # a computed level still works


# --- Nuisance distributions ---

def test_uniform_none_distribution_is_accepted(study):
    uniform = estimate_nuisance_distribution(study.data["Angle"], "Uniform")
    assert uniform is None
    a = study.pod(poi_col="Length", threshold=3.0, nuisance_col="Angle", n_boot=0,
                  nuisance_dists={"Angle": uniform})["curves"]["pod"].copy()
    b = study.pod(poi_col="Length", threshold=3.0, nuisance_col="Angle", n_boot=0)["curves"]["pod"]
    np.testing.assert_allclose(a, b)


def test_array_distribution_parameters_are_hashable(study):
    dist = estimate_nuisance_distribution(study.data["Angle"], "Normal")
    as_array = {"Angle": (dist[0], np.array(dist[1]))}
    a = study.pod(poi_col="Length", threshold=3.0, nuisance_col="Angle", n_boot=0,
                  nuisance_dists=as_array)["curves"]["pod"].copy()
    b = study.pod(poi_col="Length", threshold=3.0, nuisance_col="Angle", n_boot=0,
                  nuisance_dists={"Angle": dist})["curves"]["pod"]
    np.testing.assert_allclose(a, b)


# --- Model fitting ---

def test_constant_input_gets_zero_sobol_index():
    df = _frame()
    df["Gain"] = 3.0
    st = SimulationStudy()
    with _quiet():
        st.add_data(df, outcome_col="Signal", input_cols=["Length", "Angle", "Gain"])
    fit = st.fit_model(model_override="polynomial", force_degree=2)
    assert fit["sobol_indices"]["Gain"] == 0.0
    assert fit["sobol_indices"]["Length"] > 0.5


def test_sobol_with_only_constant_inputs():
    class Flat:
        def predict(self, X):
            return np.zeros(len(X))
    assert calculate_sobol_indices(Flat(), ["a"], pd.DataFrame({"a": [1.0, 1.0]})) == {"a": 0.0}


def test_unknown_model_override_raises(study):
    with pytest.raises(ValueError, match="model_override"):
        study.fit_model(model_override="poly")
    assert study.fit_model(model_override="Polynomial", force_degree=1)["model_key"] == ("Polynomial", 1)


def test_cv_attributes_exist_before_fitting():
    st = SimulationStudy()
    assert st.cv_scores_cache == {} and st.cv_winner_key is None


# --- I/O ---

def test_blank_columns_are_dropped():
    raw = pd.DataFrame({"Length": [1.0, 2.0], "Signal": [0.1, 0.2], "Unnamed: 2": [np.nan, np.nan]})
    df, text_cols, n_bad = clean_columns(raw)
    assert list(df.columns) == ["Length", "Signal"]
    assert text_cols == [] and n_bad == 0


# --- CLI executor ---

@pytest.mark.skipif(os.name == "nt", reason="POSIX argument splitting")
def test_cli_paths_with_spaces_stay_one_argument(tmp_path):
    folder = tmp_path / "My Sims"
    folder.mkdir()
    ex = CLIExecutor("solver --in {input} --out={output}", str(folder / "in.csv"), str(folder / "out.csv"))
    assert ex._command() == ["solver", "--in", str(folder / "in.csv"), f"--out={folder / 'out.csv'}"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX argument splitting")
def test_cli_executor_runs_with_spaces_in_paths(tmp_path):
    folder = tmp_path / "My Sims"
    folder.mkdir()
    script = tmp_path / "solver.py"
    script.write_text(
        "import sys, pandas as pd\n"
        "df = pd.read_csv(sys.argv[1]); df['Signal'] = df['x'] * 2; df.to_csv(sys.argv[2], index=False)\n"
    )
    ex = CLIExecutor(f"{sys.executable} {script} {{input}} {{output}}", str(folder / "in.csv"), str(folder / "out.csv"))
    out = ex.run(pd.DataFrame({"x": [1.0, 2.0]}))
    assert list(out["Signal"]) == [2.0, 4.0]
