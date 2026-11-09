"""Small UI builders and pure helpers shared by the app's stages."""

import json
from typing import NamedTuple

import numpy as np
import pandas as pd
from faicons import icon_svg
from shiny import ui


class Stage(NamedTuple):
    value: str
    number: int
    title: str
    icon: str
    purpose: str
    colour: str  # accent colour on the start page (the app's Fluent palette)


# The workflow stages, in order. The page, the stepper, the start page and every
# navigation button use this list, so a stage is only ever named in one place.
STAGES = (
    Stage("design", 1, "Simulation Design", "table",
          "Plan the simulation runs with a space-filling Latin Hypercube design.", "#006abc"),
    Stage("data", 2, "Data Diagnostics", "check-double",
          "Upload the simulation results and check they are sufficient to model.", "#e3a100"),
    Stage("model", 3, "Model Diagnostics", "chart-area",
          "Choose the surrogate model of the signal response and check how well it fits.", "#107c10"),
    Stage("explorer", 4, "Model Explorer", "magnifying-glass-chart",
          "Choose what to plot, integrate over and slice, and explore the PoD live.", "#038387"),
    Stage("pod", 5, "PoD Analysis", "chart-line",
          "Add bootstrap confidence bounds, read off aX/Y values and export the results.", "#d13438"),
)
STAGE = {s.value: s for s in STAGES}

# Navigation: buttons set this input from the browser; one effect switches stage
GOTO_INPUT = "goto_stage"


def _goto_js(value: str) -> str:
    return f"Shiny.setInputValue({json.dumps(GOTO_INPUT)}, {json.dumps(value)}, {{priority: 'event'}})"


def nav_button(value: str, label: str | None = None, primary: bool = False, icon: str | None = None,
               class_: str = "btn-sm mt-2"):
    """A button that goes to stage `value` ('start' for the start page)."""
    stage = STAGE.get(value)
    return ui.tags.button(
        icon_svg(icon or (stage.icon if stage else "house")), " ",
        label or f"Go to {stage.title if stage else 'Start'}",
        type="button", onclick=_goto_js(value),
        class_=f"btn {'btn-primary' if primary else 'btn-outline-primary'} {class_}",
    )


def accent_style(colour: str) -> str:
    """CSS variables for an accent colour, with dark text on light colours (e.g. amber)."""
    r, g, b = (int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return f"--dq-accent: {colour}; --dq-accent-text: {'#1b1a19' if luminance > 0.5 else '#ffffff'};"


def hero(title: str, subtitle: str):
    """The banner at the top of the start and About pages, with a strip of the stage colours."""
    return ui.div(
        ui.h1(title, class_="fw-bold mb-1"),
        ui.p(subtitle, class_="lead mb-0"),
        ui.div(*[ui.span(style=f"background: {s.colour};") for s in STAGES], class_="dq-colour-strip"),
        class_="dq-hero text-center"
    )


def icon_tile(icon: str):
    """An icon centred on a tile tinted with the surrounding --dq-accent colour."""
    return ui.div(icon_svg(icon, margin_left="0", margin_right="0"), class_="dq-entry-icon")


def stage_page(value: str, sidebar: list, *main, sidebar_width: int = 380):
    """The page every stage uses: title, settings sidebar, results, and Back / Next."""
    stage = STAGE[value]
    i = STAGES.index(stage)
    back = STAGES[i - 1].value if i > 0 else "start"
    nxt = STAGES[i + 1] if i + 1 < len(STAGES) else None
    footer = ui.div(
        nav_button(back, "Back", icon="arrow-left", class_=""),
        nav_button(nxt.value, f"Next: {nxt.title}", primary=True, icon="arrow-right", class_="")
        if nxt else ui.span(),
        class_="dq-stage-footer d-flex justify-content-between mt-3"
    )
    return ui.nav_panel(
        stage.title,
        ui.div(
            ui.div(
                ui.h3(ui.span(f"{stage.number}", class_="dq-stage-number"), stage.title, class_="mb-0"),
                ui.p(stage.purpose, class_="text-muted mb-0"),
                class_="dq-stage-header"
            ),
            ui.layout_sidebar(
                ui.sidebar(*sidebar, width=sidebar_width, open="always", bg="#ffffff"),
                *main,
                fillable=False, class_="dq-stage-body"
            ),
            footer,
            class_="dq-stage"
        ),
        value=value,
    )


def notice(message, kind: str = "muted", title: str | None = None, action=None):
    """A message box. ``kind`` is 'muted', 'info', 'warning' or 'danger'."""
    icons = {"info": "circle-info", "warning": "triangle-exclamation", "danger": "circle-xmark"}
    if kind == "muted":
        return ui.div(ui.p(message, class_="mb-0"), action or "",
                      class_="text-center p-4 text-muted bg-light rounded border mb-3")
    return ui.div(
        ui.h5(icon_svg(icons[kind]), f" {title}") if title else "",
        ui.div(message, class_="mb-0 small") if not isinstance(message, str) else ui.p(message, class_="mb-0"),
        action or "",
        class_=f"alert alert-{kind} shadow-sm"
    )


def view_tabs(id: str, *views):
    """Result views of a stage, as (title, icon, content) tuples, in underline tabs."""
    return ui.navset_card_underline(
        *[ui.nav_panel(title, content, icon=icon_svg(icon), value=title) for title, icon, content in views],
        id=id,
    )


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
