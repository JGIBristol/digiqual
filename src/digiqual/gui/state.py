"""Per-session reactive state and helpers shared by the app's tabs."""

import logging

import pandas as pd
from shiny import reactive, ui

from digiqual import SimulationStudy
from digiqual.integration import estimate_nuisance_distribution

logger = logging.getLogger("digiqual.gui")


def create_warning_card(message: str):
    return ui.layout_columns(
        ui.div(
            ui.p(message, class_="text-center p-4 text-muted bg-light rounded border")
        ),
        col_widths=[-1,10,-1]
    )


class AppState:
    """Reactive values shared between tabs, created once per session in `server()`."""

    def __init__(self, input):
        self.input = input

        # Experimental Design
        self.final_generated_df = reactive.Value(None)

        # Simulation Diagnostics
        self.uploaded_data = reactive.Value(None)
        self.validation_passed = reactive.Value(False)
        self.new_samples = reactive.Value(None)
        self.diagnostic_table = reactive.Value(None)

        # Model Fit, PoD Explorer and UQ Analysis
        self.study_instance = reactive.value(None)
        self.locked_model_type = reactive.value(None)
        self.locked_model_degree = reactive.value(None)
        self.fit_metrics = reactive.value(None)
        self.uq_metrics = reactive.value(None)
        self.pod_export_data = reactive.value(None)
        self.global_plot_trigger = reactive.value(0)

        @reactive.calc
        def current_study():
            return self.study_instance()

        self.current_study = current_study

    def get_nuisance_dists(self, nuisance_cols):
        study = self.current_study()
        if not study or not nuisance_cols:
            return {}
        df_target = study.clean_data if not study.clean_data.empty else study.data

        nuisance_dists = {}
        for col in nuisance_cols:
            if col not in df_target.columns:
                continue
            try:
                dist_type = self.input[f"nuis_dist_{col}"]()
            except Exception:
                continue  # selector not rendered yet: Uniform
            vals = pd.to_numeric(df_target[col], errors="coerce").dropna()
            try:
                dist = estimate_nuisance_distribution(vals, dist_type)
            except ValueError as e:
                # The distribution card already shows this message to the user
                logger.warning("Nuisance '%s': %s Using Uniform instead.", col, e)
                continue
            if dist is not None:
                nuisance_dists[col] = dist
        return nuisance_dists

    def render_study_plot(self, plot_key: str):
        _ = self.global_plot_trigger()  # Listen to the single global trigger
        study = self.current_study()
        return study.plots[plot_key] if study and plot_key in study.plots else None


def register_shared(input, output, session, state: AppState):
    """Effects that keep the study in sync with the data and clear stale results."""

    @reactive.effect
    @reactive.event(state.uploaded_data, input.input_cols, input.outcome_col)
    def update_study():
        df = state.uploaded_data()
        selected_inputs = list(input.input_cols())
        selected_outcome = input.outcome_col()

        # Guard 1: Missing basic selections
        if df is None or not selected_inputs or not selected_outcome or (selected_outcome in selected_inputs):
            state.study_instance.set(None)
            return

        # Guard 2: Reactive Race Condition Guard
        # Ensure the UI selections actually exist in the CURRENT dataset before processing.
        required_cols = selected_inputs + [selected_outcome]
        if not all(col in df.columns for col in required_cols):
            # The UI hasn't caught up to the newly uploaded data yet. Safely abort.
            state.study_instance.set(None)
            return

        # Initialize empty, then add_data with overwrite=True
        study = SimulationStudy()
        study.add_data(df, outcome_col=selected_outcome, input_cols=selected_inputs, overwrite=True)
        state.study_instance.set(study)

    # -----------------------------------------------------------------
    # TWO-TIER RESET LOGIC
    # -----------------------------------------------------------------
    @reactive.effect
    @reactive.event(
        state.uploaded_data, input.input_cols, input.outcome_col,
        input.ui_max_gap, input.ui_min_r2, input.ui_avg_cv, input.ui_max_cv, input.ui_max_vif
    )
    def reset_core_data_state():
        """TIER 1: Wipes EVERYTHING when the physical data or core definitions change."""
        state.diagnostic_table.set(None)
        state.validation_passed.set(False)
        state.new_samples.set(None)

        state.fit_metrics.set(None)
        state.uq_metrics.set(None)
        state.locked_model_type.set(None)
        state.locked_model_degree.set(None)
        state.pod_export_data.set(None)

    @reactive.effect
    @reactive.event(input.pod_pois, input.pod_nuisance)
    def reset_analysis_results():
        """TIER 2: Clears just the downstream math when plot targets change, keeping the core study intact."""
        state.fit_metrics.set(None)
        state.uq_metrics.set(None)
        state.locked_model_type.set(None)
        state.locked_model_degree.set(None)
        state.pod_export_data.set(None)

    @reactive.effect
    def reset_on_distribution_change():
        """Changing a nuisance distribution after fitting makes the results stale."""
        study = state.current_study()
        if state.fit_metrics() is None or study is None or not study.pod_results:
            return
        nuisance_cols = list(input.pod_nuisance()) if input.pod_nuisance() else []
        chosen = state.get_nuisance_dists(nuisance_cols)
        fitted = study.pod_results.get("nuisance_dists") or {}
        if chosen != fitted:
            with reactive.isolate():
                state.fit_metrics.set(None)
                state.uq_metrics.set(None)
                state.pod_export_data.set(None)
            ui.notification_show("Nuisance distribution changed: please re-fit the model.", type="warning")
