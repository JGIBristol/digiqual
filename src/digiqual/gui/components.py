"""Small UI builders and pure helpers shared by the app's tabs."""

from typing import NamedTuple

import numpy as np
import pandas as pd
from faicons import icon_svg
from shiny import reactive, ui


class Tab(NamedTuple):
    value: str
    title: str
    icon: str
    colour: str  # Bootstrap colour name used on the Home page
    lead: str
    bullets: tuple


# The workflow tabs, left to right. Panels, the Home page and the "go to" buttons all
# use this list, so a tab is only ever named in one place.
TABS = (
    Tab("design", "Simulation Design", "table", "primary",
        "Plan the simulation runs with a space-filling Latin Hypercube design.",
        ("Set the range of each input variable.",
         "Download the design as a CSV to run in your solver.")),
    Tab("data", "Data Diagnostics", "check-double", "warning",
        "Check that the simulation results are sufficient before modelling them.",
        ("Coverage, model-fit, bootstrap-stability and collinearity checks.",
         "Generate targeted extra samples where the data falls short.")),
    Tab("explorer", "Model Explorer", "magnifying-glass-chart", "info",
        "Fit the surrogate model, then choose what to plot, integrate over and slice, and see the PoD instantly.",
        ("Candidate models are fitted on opening; cross-validation picks the simplest model within one "
         "standard error of the best, or choose one yourself.",
         "Plot 1 or 2 variables; integrate over others with a chosen distribution and range.",
         "Fix the rest at exact values and move the detection threshold live.")),
    Tab("pod", "PoD Analysis", "chart-line", "danger",
        "Add bootstrap confidence bounds to the PoD and report aX/Y values.",
        ("Parallel bootstrap with bounds at 50% to 99% confidence.",
         "Reliability matrix and a full Excel report.")),
)
TAB = {t.value: t for t in TABS}


def notice(message, kind: str = "muted", title: str | None = None, action=None):
    """A full-width message box. ``kind`` is 'muted', 'info', 'warning' or 'danger'."""
    icons = {"info": "circle-info", "warning": "triangle-exclamation", "danger": "circle-xmark"}
    if kind == "muted":
        body = ui.div(ui.p(message, class_="mb-0"), action or "",
                      class_="text-center p-4 text-muted bg-light rounded border")
    else:
        body = ui.div(
            ui.h5(icon_svg(icons[kind]), f" {title}") if title else "",
            ui.div(message, class_="mb-0 small") if not isinstance(message, str) else ui.p(message, class_="mb-0"),
            action or "",
            class_=f"alert alert-{kind} shadow-sm"
        )
    return ui.layout_columns(body, col_widths=[-1, 10, -1])


def goto_button(input_id: str, tab_value: str, label: str | None = None, primary: bool = False):
    """Button that switches to another tab (wired up with `register_goto`)."""
    tab = TAB[tab_value]
    return ui.input_action_button(
        input_id, label or f"Go to {tab.title}", icon=icon_svg(tab.icon),
        class_=("btn-primary" if primary else "btn-outline-primary") + " btn-sm mt-2"
    )


def register_goto(input, input_id: str, tab_value: str):
    """Switch the navbar to `tab_value` when the button `input_id` is clicked."""
    @reactive.effect
    @reactive.event(input[input_id])
    def _goto():
        ui.update_navset("navbar", selected=tab_value)


def format_duration(seconds: float) -> str:
    return f"~{max(1, int(seconds))} seconds" if seconds < 90 else f"~{int(seconds / 60)} minutes"


def largest_gap(values) -> tuple[float, float] | None:
    """The ends of the largest gap between sorted values, or None for < 2 values."""
    vals = np.sort(np.asarray(values, dtype=float))
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return None
    i = int(np.argmax(np.diff(vals)))
    return float(vals[i]), float(vals[i + 1])


# Message for each failing diagnostic test: (title, text). {vars} lists the variables.
_FAILURE_TEXT = {
    "Input Coverage": ("Input Coverage Gaps",
                       "Large gaps in parameter coverage detected for: {vars}."),
    "Model Fit (CV)": ("Weak Model Fit",
                       "The surrogate model R² score is below the required threshold. The relationship "
                       "might be too noisy or non-linear for the current sample size."),
    "Bootstrap Convergence": ("Bootstrap Instability",
                              "Model predictions are highly sensitive to sample variance (insufficient "
                              "convergence). Confidence intervals may be too wide."),
    "Collinearity Check": ("High Multicollinearity",
                           "Strong correlation detected among inputs for: {vars}. This can make the "
                           "surrogate model fit unstable and predictions unreliable."),
}


def diagnostic_failures(diag: pd.DataFrame | None) -> list[tuple[str, str, str]]:
    """Failed tests in a diagnostics table, as (test, title, message), in display order."""
    if diag is None or diag.empty:
        return []
    failures = []
    for test, (title, text) in _FAILURE_TEXT.items():
        rows = diag[diag["Test"] == test]
        failed = rows[~rows["Pass"].astype(bool)]
        if not failed.empty:
            variables = ", ".join(str(v) for v in failed["Variable"].unique())
            failures.append((test, title, text.format(vars=variables)))
    return failures


def failure_list(failures):
    """Red bullet lines for the output of `diagnostic_failures`."""
    return ui.div(*[
        ui.p(
            ui.span(icon_svg("triangle-exclamation"), class_="text-danger me-1"),
            ui.tags.strong(f"{title}: "), message,
            class_="small mb-1 text-danger"
        )
        for _test, title, message in failures
    ])


def curve_export_df(results: dict) -> pd.DataFrame:
    """PoD curve (and bounds, when bootstrapped) on the evaluation grid, one column per PoI."""
    poi_cols = results["poi_cols"]
    X_eval = np.asarray(results["X_eval"]).reshape(len(results["X_eval"]), -1)
    data = {col: X_eval[:, i] for i, col in enumerate(poi_cols)}
    curves = results["curves"]
    data["pod_mean"] = curves["pod"]
    data["ci_lower"] = curves["ci_lower"] if curves.get("ci_lower") is not None else np.nan
    data["ci_upper"] = curves["ci_upper"] if curves.get("ci_upper") is not None else np.nan
    return pd.DataFrame(data)


MAX_PLOTTED = 2
MAX_INTEGRATED = 2


def config_errors(config: dict | None) -> list[str]:
    """Problems with a Model Explorer configuration that stop the PoD being computed."""
    if config is None:
        return ["The variable table has not loaded yet."]
    errors = []
    n_plot = len(config["plot"])
    if n_plot == 0:
        errors.append("Choose 1 or 2 variables to Plot.")
    elif n_plot > MAX_PLOTTED:
        errors.append(f"At most {MAX_PLOTTED} variables can be plotted ({n_plot} chosen). "
                      "Integrate over or slice the others.")
    if len(config["integrate"]) > MAX_INTEGRATED:
        errors.append(f"At most {MAX_INTEGRATED} variables can be integrated over "
                      f"({len(config['integrate'])} chosen).")
    for col, spec in config["integrate"].items():
        lo, hi = spec["bounds"]
        if lo is None or hi is None:
            errors.append(f"Enter both integration limits for '{col}'.")
        elif lo >= hi:
            errors.append(f"Integration range for '{col}': the lower limit must be below the upper limit.")
    for col, val in config["slice"].items():
        if val is None:
            errors.append(f"Enter a slice value for '{col}'.")
    if config["threshold"] is None:
        errors.append("Enter a detection threshold.")
    return errors


def fmt_value(v: float) -> str:
    """A variable value as shown in the app: 3 decimal places, or 4 significant figures if tiny."""
    return f"{v:.3f}" if v == 0 or abs(v) >= 0.01 else f"{v:.4g}"


def setup_rows(config: dict, model_name: str) -> dict:
    """Readable summary of a Model Explorer configuration, for tables and the report."""
    rows = {"Model": model_name, "Plotted": ", ".join(config["plot"])}
    for col, spec in config["integrate"].items():
        lo, hi = spec["bounds"]
        rows[f"Integrated: {col}"] = f"{spec['dist']} on [{fmt_value(lo)}, {fmt_value(hi)}]"
    for col, val in config["slice"].items():
        rows[f"Sliced: {col}"] = fmt_value(val)
    rows["Detection Threshold"] = fmt_value(config["threshold"])
    return rows


def slider_precision(lo: float, hi: float) -> tuple[float, int]:
    """Slider step (range / 100) and the decimals for exact values: 3, or more for tiny ranges."""
    step = (hi - lo) / 100.0
    if step <= 0:
        step = abs(lo) / 100.0 if lo != 0 else 0.001
    decimals = max(3, int(np.ceil(-np.log10(step))) + 1)
    return round(step, decimals), decimals


def metrics_table(rows: dict):
    """A two-column Metric / Value table."""
    return pd.DataFrame(list(rows.items()), columns=["Metric", "Value"])
