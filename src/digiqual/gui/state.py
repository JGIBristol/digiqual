"""Per-session reactive state and helpers shared by the app's tabs.

Results are layered, and each layer is only valid while the ones above it are:

    data (study)  >  diagnostics
                  >  model fit  >  explorer PoD view  >  bootstrap (UQ)

`AppState.invalidate` clears a layer and the ones that depend on it. The Model Explorer's
settings (`explorer_config`) are the single source the explorer and the PoD Analysis
tab read from.
"""

import logging

import pandas as pd
from shiny import reactive

from digiqual import SimulationStudy
from digiqual.core import model_override_args
from digiqual.integration import estimate_nuisance_distribution

logger = logging.getLogger("digiqual.gui")


def read_input(input, input_id: str, default=None):
    """An input's value, or `default` if it has not been rendered yet."""
    try:
        value = input[input_id]()
    except Exception:  # SilentException until the control exists
        return default
    return default if value is None else value


class AppState:
    """Reactive values shared between tabs, created once per session in `server()`."""

    def __init__(self, input):
        self.input = input

        # Simulation Design
        self.final_generated_df = reactive.value(None)

        # Data Diagnostics
        self.uploaded_data = reactive.value(None)
        self.study_instance = reactive.value(None)
        self.diagnostic_table = reactive.value(None)
        self.validation_passed = reactive.value(False)
        self.diagnostics_stale = reactive.value(False)  # thresholds changed since the run
        self.new_samples = reactive.value(None)

        # Model fit (in the Model Explorer): all candidate models are cross-validated
        # once per dataset; model_fit is `study.fit_model()` for the chosen one
        self.candidates_ready = reactive.value(False)
        self.fit_error = reactive.value(None)
        self.model_fit = reactive.value(None)

        # Model Explorer: exact slice values and threshold (sliders and number boxes
        # both write here), and the plot/integrate settings of the PoD on screen
        self.slice_values = reactive.value({})
        self.threshold = reactive.value(None)
        self.applied_structure = reactive.value(None)

        # PoD Analysis: {"config", "results", "metrics"} from the last bootstrap
        self.uq_result = reactive.value(None)

        # Bumped whenever study.plots is regenerated
        self.plot_trigger = reactive.value(0)

        self.current_study = reactive.calc(lambda: self.study_instance())
        self.explorer_config = reactive.calc(self._explorer_config)

    # ---- Invalidation ------------------------------------------------------
    def invalidate(self, level: str):
        """Clear stale results.

        'data' clears everything; 'diagnostics' only the diagnostics (the model doesn't
        use them); 'pod' the explorer PoD and the bootstrap.
        """
        with reactive.isolate():
            if level in ("data", "diagnostics"):
                self.diagnostic_table.set(None)
                self.validation_passed.set(False)
                self.diagnostics_stale.set(False)
                self.new_samples.set(None)
            if level == "data":
                self.candidates_ready.set(False)
                self.fit_error.set(None)
                self.model_fit.set(None)
                self.slice_values.set({})
                self.threshold.set(None)
            if level in ("data", "pod"):
                self.applied_structure.set(None)
                self.uq_result.set(None)

    def bump_plots(self):
        with reactive.isolate():
            self.plot_trigger.set(self.plot_trigger() + 1)

    def study_plot(self, plot_key: str):
        _ = self.plot_trigger()
        study = self.current_study()
        return study.plots.get(plot_key) if study else None

    # ---- Columns -----------------------------------------------------------
    def selected_columns(self) -> list[str]:
        """The chosen input columns and outcome that exist in the uploaded data."""
        df = self.uploaded_data()
        if df is None:
            return []
        cols = list(self.input.input_cols())
        outcome = self.input.outcome_col()
        if outcome and outcome not in cols:
            cols.append(outcome)
        cols = [c for c in cols if c in df.columns]
        return cols or list(df.columns)

    def data_values(self, col: str) -> pd.Series:
        study = self.current_study()
        df = study.clean_data if not study.clean_data.empty else study.data
        return pd.to_numeric(df[col], errors="coerce").dropna()

    # ---- Model Explorer configuration --------------------------------------
    def _explorer_config(self):
        """{model, plot, integrate: {col: {dist, bounds}}, slice: {col: value}, threshold}, or None."""
        study = self.current_study()
        fit = self.model_fit()
        if study is None or fit is None:
            return None
        input = self.input
        slice_values = self.slice_values()
        plot, integrate, slices = [], {}, {}
        for col in study.inputs:
            role = read_input(input, f"role_{col}")
            if role is None:
                return None  # role table not rendered yet
            if role == "plot":
                plot.append(col)
            elif role == "integrate":
                integrate[col] = {
                    "dist": read_input(input, f"dist_{col}", "Uniform"),
                    "bounds": (read_input(input, f"int_lo_{col}"), read_input(input, f"int_hi_{col}")),
                }
            else:
                slices[col] = slice_values.get(col)
        return {"model": fit["model_key"], "plot": plot, "integrate": integrate, "slice": slices,
                "threshold": self.threshold()}

    @staticmethod
    def structure(config: dict | None):
        """The parts of a config that need the Update PoD button (not slices or threshold)."""
        if config is None:
            return None
        return {"model": config["model"], "plot": config["plot"], "integrate": config["integrate"]}

    def nuisance_dists(self, config: dict) -> dict:
        """Fitted non-uniform distributions for the integrated variables."""
        dists = {}
        for col, spec in config["integrate"].items():
            if spec["dist"] == "Uniform":
                continue
            try:
                dist = estimate_nuisance_distribution(self.data_values(col), spec["dist"])
            except ValueError as e:
                # The explorer's integration notes already show this to the user
                logger.warning("Nuisance '%s': %s Using Uniform instead.", col, e)
                continue
            if dist is not None:
                dists[col] = dist
        return dists

    def pod_kwargs(self, config: dict) -> dict:
        """`study.pod()` arguments for a config, locked to the fitted model."""
        override, degree = model_override_args(self.model_fit()["mean_model"])
        return dict(
            poi_col=list(config["plot"]),
            nuisance_col=list(config["integrate"]),
            slice_values=dict(config["slice"]),
            nuisance_dists=self.nuisance_dists(config),
            nuisance_bounds={c: tuple(s["bounds"]) for c, s in config["integrate"].items()},
            model_override=override,
            force_degree=degree,
        )


def register_shared(input, output, session, state: AppState):
    """Effects that keep the study in sync with the data and clear stale results."""

    @reactive.effect
    @reactive.event(state.uploaded_data, input.input_cols, input.outcome_col)
    def update_study():
        df = state.uploaded_data()
        selected_inputs = list(input.input_cols())
        selected_outcome = input.outcome_col()
        state.invalidate("data")

        # Missing selections, a conflict, or selections from the previous file
        # (the selectors haven't caught up with a new upload yet)
        if (df is None or not selected_inputs or not selected_outcome
                or selected_outcome in selected_inputs
                or not all(c in df.columns for c in selected_inputs + [selected_outcome])):
            state.study_instance.set(None)
            return

        study = SimulationStudy()
        study.add_data(df, outcome_col=selected_outcome, input_cols=selected_inputs, overwrite=True)
        state.study_instance.set(study)

    @reactive.effect
    @reactive.event(input.ui_max_gap, input.ui_min_r2, input.ui_avg_cv, input.ui_max_cv, input.ui_max_vif,
                    ignore_init=True)
    def mark_diagnostics_stale():
        """New thresholds only make the diagnostics out of date, not the fitted model."""
        if state.diagnostic_table() is not None:
            state.diagnostics_stale.set(True)
