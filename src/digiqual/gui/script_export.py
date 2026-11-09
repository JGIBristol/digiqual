"""Write a bootstrapped PoD analysis from the app as a standalone Python script.

The script repeats the analysis with the digiqual library: the same data cleaning,
diagnostic thresholds, model, plotted/integrated/sliced variables, distributions,
integration ranges, threshold and bootstrap. All randomness in digiqual is seeded
(bootstrap iteration i uses seed i; the Monte Carlo integration uses seed 42), so it
reproduces the app's numbers.
"""

import datetime
import pprint

from digiqual import __version__


def _override_args(model_key) -> tuple[str, int | None]:
    """`model_override`, `force_degree` for a fitted model key like ('Polynomial', 3).

    The key form of `digiqual.core.model_override_args`, which takes the model itself.
    """
    return ("polynomial", model_key[1]) if model_key[0] == "Polynomial" else ("kriging", None)


def _dict_literal(d: dict) -> str:
    """A dict as Python source with one key per line."""
    items = [f"    {k!r}: {pprint.pformat(v, width=100, sort_dicts=False)}," for k, v in d.items()]
    return "{\n" + "\n".join(items) + "\n}"


_CHOICE_TEXT = {"auto": "Auto (the cross-validation pick)", "kriging": "Kriging"}


def analysis_script(config: dict, model_choice: str, settings: dict) -> str:
    """
    The Python source of a script that repeats a bootstrapped analysis from the app.

    Args:
        config (dict): The Model Explorer configuration the bootstrap used
            (`AppState.explorer_config`): model, plot, integrate, slice, threshold.
        model_choice (str): The model dropdown value ('auto', 'poly-3' or 'kriging').
        settings (dict): The run settings stored with the bootstrap result: n_boot,
            n_jobs, target, conf, csv_name, inputs, outcome and the diagnostic thresholds.
    """
    override, degree = _override_args(config["model"])
    choice_text = _CHOICE_TEXT.get(model_choice, f"Polynomial of degree {model_choice.split('-')[-1]}")
    thresholds = settings["thresholds"]
    one_plotted = len(config["plot"]) == 1

    pod_settings = {
        "poi_col": list(config["plot"]),
        "nuisance_col": list(config["integrate"]),
        "slice_values": dict(config["slice"]),
        "nuisance_bounds": {c: tuple(s["bounds"]) for c, s in config["integrate"].items()},
        "model_override": override,
        "force_degree": degree,
    }
    non_uniform = {c: s["dist"] for c, s in config["integrate"].items() if s["dist"] != "Uniform"}

    lines = [
        f'"""DigiQual analysis exported from the DigiQual app on {datetime.date.today():%d %B %Y}.',
        "",
        "Repeats the app's PoD analysis with the same data, model and settings, so it",
        "gives the same results (all randomness in digiqual is seeded). Run it next to",
        "the data file, or change DATA_FILE below.",
        "",
        f"Made with digiqual {__version__}.",
        '"""',
        "",
        "from digiqual import SimulationStudy",
    ]
    if non_uniform:
        lines.append("from digiqual.integration import estimate_nuisance_distribution")
    lines += [
        "from digiqual.io import read_results_csv",
    ]
    if one_plotted:
        lines.append("from digiqual.pod import reliability_matrix")
    lines += [
        "",
        f"DATA_FILE = {settings.get('csv_name') or 'results.csv'!r}",
        "",
        "# 1. Data, with the column names cleaned as in the app",
        "df = read_results_csv(DATA_FILE)",
        "study = SimulationStudy()",
        f"study.add_data(df, outcome_col={settings['outcome']!r}, input_cols={settings['inputs']!r})",
        "",
        "# 2. Data diagnostics, with the thresholds used in the app",
        "diagnostics = study.diagnose(",
        f"    max_gap_ratio={thresholds['ui_max_gap']!r},",
        f"    min_r2_score={thresholds['ui_min_r2']!r},",
        f"    max_avg_cv={thresholds['ui_avg_cv']!r},",
        f"    max_max_cv={thresholds['ui_max_cv']!r},",
        f"    max_allowed_vif={thresholds['ui_max_vif']!r},",
        ")",
        "print(diagnostics.to_string(index=False))",
        "",
        f"# 3. Surrogate model. Chosen in the app as: {choice_text}",
        f"fit = study.fit_model(model_override={override!r}, force_degree={degree!r})",
        'print("Model equation:", fit["equation"])',
        "",
    ]
    if non_uniform:
        lines.append("# 4. Distributions of the integrated variables, fitted to the data as in the app")
        lines.append("nuisance_dists = {")
        for col, dist in non_uniform.items():
            lines.append(f"    {col!r}: estimate_nuisance_distribution(study.clean_data[{col!r}], {dist!r}),")
        lines.append("}")
    else:
        lines.append("# 4. Integrated variables (if any) are sampled uniformly over their range")
        lines.append("nuisance_dists = None")
    lines += [
        "",
        "# 5. The PoD setup from the Model Explorer: plotted (poi_col), integrated",
        "#    (nuisance_col, over nuisance_bounds) and sliced (slice_values) variables",
        f"settings = {_dict_literal(pod_settings)}",
        "",
    ]
    if one_plotted:
        lines += [
            "# Threshold spectrum, for the PoD vs threshold plot",
            "study.compute_pod_spectrum(**settings, nuisance_dists=nuisance_dists)",
            "",
        ]
    lines += [
        "# 6. PoD with bootstrap confidence bounds",
        "results = study.pod(",
        "    **settings,",
        "    nuisance_dists=nuisance_dists,",
        f"    threshold={config['threshold']!r},",
        f"    n_boot={settings['n_boot']!r},",
        f"    n_jobs={settings['n_jobs']!r},  # -1: all cores but two; 1: one core (same results)",
        ")",
        "",
    ]
    if one_plotted:
        lines += [
            "# Reliability indices aX/Y: flaw size reaching X% PoD at Y% confidence",
            'print(reliability_matrix(results["reliability_table"], as_text=True).to_string(index=False))',
            "",
        ]
    lines += [
        "# 7. Plots: signal model and PoD curve (with the app's target PoD and confidence level)",
        f"study.visualise(target_pod={settings['target'] / 100!r}, confidence_level={settings['conf']!r})",
    ]
    if one_plotted:
        lines.append("study.plot_pod_vs_threshold()")
    return "\n".join(lines) + "\n"
