"""PoD Explorer tab: move the detection threshold and slice values without refitting."""

import numpy as np
from faicons import icon_svg
from shiny import reactive, render, ui

from .state import AppState, create_warning_card

panel = ui.nav_panel(
    "PoD Explorer",
    ui.div(
        ui.h3("Reliability Explorer", class_="mb-4 text-center"),
        ui.output_ui("explorer_warnings_ui"),
        ui.output_ui("explorer_main_ui"),
        ui.output_ui("explorer_results_ui"),
        class_="container-fluid py-3"
    ),
    icon=icon_svg("magnifying-glass-chart")
)


def register(input, output, session, state: AppState):
    current_study = state.current_study
    fit_metrics = state.fit_metrics
    uq_metrics = state.uq_metrics
    global_plot_trigger = state.global_plot_trigger
    get_nuisance_dists = state.get_nuisance_dists
    _render_study_plot = state.render_study_plot

    @render.ui
    def explorer_main_ui():
        # Hide entirely if model hasn't been fit
        if fit_metrics() is None:
            return ui.div()

        study = current_study()
        if study and study.outcome:
            summary = study.get_data_summary(study.outcome)
            s_min, s_max, s_med = summary["min"], summary["max"], summary["median"]
            step_size = (s_max - s_min) / 100.0 if s_max != s_min else 0.001
        else:
            s_min, s_max, s_med, step_size = 0, 100, 50, 0.1

        return ui.layout_columns(
            ui.div(
                ui.card(
                    ui.card_header("Real-Time Reliability Configuration"),
                    ui.p("Adjust the detection threshold to see the impact on reliability across the parameter space.", class_="small text-muted mb-3"),
                    ui.input_slider("pod_threshold_slider", f"Detection Threshold ({study.outcome if study else ''})", min=round(s_min, 4), max=round(s_max, 4), value=round(s_med, 4), step=round(step_size, 4)),
                ),
                ui.output_ui("dynamic_slice_sliders"),
            ),
            col_widths=[-1,10,-1]
        )

    @render.ui
    def dynamic_slice_sliders():
        study = current_study()

        # LAZY RENDER: Only show if a model has been successfully fitted!
        if study is None or fit_metrics() is None:
            return ui.div()

        selected_pois = list(input.pod_pois()) if input.pod_pois() else []
        selected_nuis = list(input.pod_nuisance()) if input.pod_nuisance() else []

        leftovers = study.get_unassigned_parameters(selected_pois, selected_nuis)

        if not leftovers:
            return ui.div()

        sliders = []
        for col in leftovers:
            # Get clean bounds from the package
            summary = study.get_data_summary(col)
            if summary["min"] is None:
                continue

            col_min = float(summary["min"])
            col_max = float(summary["max"])
            col_med = float(summary["median"])

            step_size = (col_max - col_min) / 100.0
            if step_size <= 0:
                step_size = abs(col_min) / 100.0 if col_min != 0 else 0.001

            # Dynamic precision formatting
            # Calculate how many decimal places we actually need based on the step size
            try:
                # Use np.log10 to find the magnitude.
                # e.g., if step is 0.0005 -> -log10 is ~3.3 -> ceil is 4 -> +1 means 5 decimals
                required_decimals = max(3, int(np.ceil(-np.log10(step_size))) + 1)
            except Exception:
                required_decimals = 3 # Safe fallback

            # Round specifically to the required precision for this specific variable
            clean_min = round(col_min, required_decimals)
            clean_max = round(col_max, required_decimals)
            clean_med = round(col_med, required_decimals)
            clean_step = round(step_size, required_decimals)

            sliders.append(
                ui.input_slider(
                    f"slice_{col}", col,
                    min=clean_min, max=clean_max,
                    value=clean_med, step=clean_step
                )
            )

        return ui.card(
            ui.card_header(
                ui.span(icon_svg("sliders"), class_="text-primary me-2"),
                "Real-Time Slice Explorer"
            ),
            ui.p(
                "Adjust these constant parameters to instantly update the surface plot below. "
                "There is no need to re-fit the model.",
                class_="small text-muted mb-3"
            ),
            ui.layout_columns(*sliders, col_widths=6),
            class_="mt-3 shadow-sm"
        )

    @render.ui
    def explorer_warnings_ui():
        if fit_metrics() is None:
            return create_warning_card("Please fit a model in the 'Model Fit' tab first.")
        return ui.div()

    @render.ui
    def explorer_results_ui():
        if fit_metrics() is None:
            return ui.div()

        study = current_study()
        is_multi_dim = len(study.pod_results.get("poi_cols", [])) > 1

        return ui.layout_columns(
            # Signal model plot on the left
            ui.card(
                ui.card_header(f"{input.outcome_col()} Surface" if is_multi_dim else "Model Fit"),
                ui.output_plot("plot_signal_explorer", height="450px"),
                full_screen=True, class_="mt-3"
            ),
            # PoD plot on the right
            ui.card(
                ui.card_header("PoD Surface Heatmap" if is_multi_dim else "PoD Reliability Curve"),
                ui.output_plot("plot_explorer", height="450px"),
                full_screen=True, class_="mt-3"
            ),
            col_widths=[-1, 5, 5, -1]
        )

    @render.plot
    def plot_explorer():
        # The PoD Explorer's PoD curve is the "pod_curve" plot
        return _render_study_plot("pod_curve")

    @reactive.effect
    def realtime_slice_update():
        """Listens to the dynamic sliders and instantly updates the plots without refitting."""
        study = current_study()
        if study is None or fit_metrics() is None:
            return

        poi_cols = study.pod_results.get("poi_cols", [])
        nuis_cols = study.pod_results.get("nuisance_cols", [])

        leftovers = study.get_unassigned_parameters(poi_cols, nuis_cols)
        if not leftovers:
            return

        slice_values = {}
        for col in leftovers:
            try:
                val = input[f"slice_{col}"]()
                slice_values[col] = val if val is not None else study.get_data_summary(col)["median"]
            except Exception:
                slice_values[col] = study.get_data_summary(col)["median"]

        if not slice_values:
            return

        # This effect also fires right after a fit and when the sliders first render
        # (with a rounded median). Skip it when every slider is within half a step
        # (sliders use range/100 steps) of the slice the current results already use.
        current = study.pod_results.get("slice_values", {})
        unchanged = True
        for col, val in slice_values.items():
            summary = study.get_data_summary(col)
            if summary["max"] is None or col not in current:
                unchanged = False
                break
            half_step = 0.5 * max(summary["max"] - summary["min"], 1e-12) / 100.0
            if abs(float(val) - float(current[col])) > half_step:
                unchanged = False
                break
        if unchanged:
            return

        # 1. Update the math instantly using the Layer 3 Cache
        study.update_slice(slice_values)

        # 2. Re-generate the visualisations in memory
        study.visualise(show=False)

        # 3. Trigger the UI to redraw (Isolated to prevent infinite loops!)
        with reactive.isolate():
            global_plot_trigger.set(global_plot_trigger() + 1)

        # 4. Clear UQ bounds
        uq_metrics.set(None)


    @reactive.effect
    @reactive.event(input.pod_threshold_slider)
    def realtime_threshold_update():
        """Instantly updates plots using Layer 4 Spectrum Interpolation."""
        study = current_study()
        if study is None or fit_metrics() is None or not study.pod_results:
            return

        # 1. Grab the currently active slice values so they don't reset!
        current_slices = study.pod_results.get("slice_values", {})

        # 2. Lock the model so it doesn't attempt to run auto-CV again
        current_model = study.pod_results["mean_model"]
        override = "polynomial" if current_model.model_type_ == "Polynomial" else "kriging"
        degree = current_model.model_params_ if current_model.model_type_ == "Polynomial" else None

        # 3. Re-run .pod() with the new threshold
        # Since Layer 4 is primed, this takes microseconds.
        nuisance_cols = list(input.pod_nuisance()) if input.pod_nuisance() else []
        study.pod(
            poi_col=list(input.pod_pois()),
            threshold=input.pod_threshold_slider(),
            nuisance_col=nuisance_cols,
            slice_values=current_slices,
            model_override=override,
            force_degree=degree,
            n_boot=0,
            nuisance_dists=get_nuisance_dists(nuisance_cols)
        )

        study.visualise(show=False)

        with reactive.isolate():
            global_plot_trigger.set(global_plot_trigger() + 1)

        # The new results have no bootstrap bounds, so the UQ tables are now stale
        uq_metrics.set(None)


    @render.plot
    def plot_signal_explorer():
        return _render_study_plot("signal_model")

