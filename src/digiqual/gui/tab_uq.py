"""UQ Analysis tab: bootstrap confidence bounds, reliability indices and the Excel report."""

import asyncio

import numpy as np
import pandas as pd
from faicons import icon_svg
from shiny import reactive, render, ui

from digiqual.defaults import CONFIDENCE_LEVELS, DEFAULT_N_BOOT, TARGET_PODS
from digiqual.pod import reliability_matrix

from .state import AppState, create_warning_card, logger

panel = ui.nav_panel(
    "UQ Analysis",
    ui.div(
        ui.h3("Uncertainty Quantification (PoD)", class_="mb-4 text-center"),
        ui.output_ui("uq_warnings_ui"),
        ui.output_ui("uq_main_ui"),
        ui.output_ui("uq_results_ui"),
        class_="container-fluid py-3"
    ),
    icon=icon_svg("chart-line")
)


def register(input, output, session, state: AppState):
    current_study = state.current_study
    fit_metrics = state.fit_metrics
    uq_metrics = state.uq_metrics
    locked_model_type = state.locked_model_type
    locked_model_degree = state.locked_model_degree
    pod_export_data = state.pod_export_data
    global_plot_trigger = state.global_plot_trigger
    get_nuisance_dists = state.get_nuisance_dists
    _render_study_plot = state.render_study_plot

    @render.ui
    def uq_main_ui():
        # Hide entirely if model hasn't been fit
        if locked_model_type() is None:
            return ui.div()

        return ui.layout_columns(
            ui.card(
                ui.card_header("Bootstrap & Target Configuration"),
                ui.layout_columns(
                    ui.input_numeric("pod_n_boot", "Bootstrap Iterations", value=DEFAULT_N_BOOT, min=10, step=100),
                    ui.div(
                        ui.input_checkbox("pod_parallel", "Enable Parallel Compute (Faster)", value=True),
                        class_="pt-4"
                    ),
                    col_widths=[6, 6]
                ),
                ui.layout_columns(
                    ui.input_select("pod_target_val", "Target PoD (%)", choices=[str(p) for p in TARGET_PODS], selected="90"),
                    ui.input_select("pod_conf_val", "Confidence Level (%)", choices=[str(c) for c in CONFIDENCE_LEVELS], selected="95"),
                    col_widths=[6, 6]
                ),
                ui.input_task_button("btn_run_uq", "Run Uncertainty Quantification", class_="btn-danger w-100", icon=icon_svg("layer-group")),
            ),
            col_widths=[-1,10,-1]
        )

    @render.ui
    def uq_warnings_ui():
        if locked_model_type() is None:
            return create_warning_card("Please fit a model in the 'Model Fit' tab first.")

        model_str = f"Polynomial (Degree {locked_model_degree()})" if locked_model_type() == 'Polynomial' else "Kriging"
        return ui.layout_columns(
            ui.div(
                ui.h5(icon_svg("lock"), f" Structural Shape Locked: {model_str}"),
                ui.p("Note: The confidence bounds generated here reflect parameter uncertainty assuming the mathematical shape chosen in the Model Fit tab is correct. If you add more data later, the 'Auto' selector may choose a different shape, causing new curves to fall outside these bounds.", class_="small mb-0"),
                class_="alert alert-info shadow-sm"
            ),
            col_widths=[-1,10,-1]
        )

    @render.ui
    def uq_results_ui():
        if uq_metrics() is None:
            return ui.div()

        study = current_study()
        is_multi_dim = len(study.pod_results.get("poi_cols", [])) > 1

        # We conditionally layout the Sensitivity plot and the Matrix table
        if not is_multi_dim:
            row2 = ui.layout_columns(
                ui.card(
                    ui.card_header("Probability of Detection vs. Threshold"),
                    ui.output_plot("plot_uq_sensitivity", height="400px"),
                    ui.p("Probability of detection as a function of the detection threshold for representative defect sizes.", class_="small text-muted px-3 pt-2 mb-0"),
                    full_screen=True, class_="mt-3"
                ),
                ui.card(
                    ui.card_header("Reliability Index Matrix (aX/Y)"),
                    ui.output_data_frame("uq_reliability_matrix_table"),
                    ui.p("Flaw sizes for combinations of Target Probability of Detection (rows) and Confidence Levels (columns).", class_="small text-muted mt-2 mb-0"),
                    class_="mt-3"
                ),
                col_widths=[7, 5]
            )
        else:
            row2 = ui.layout_columns(
                ui.card(
                    ui.card_header("Reliability Index Matrix (aX/Y)"),
                    ui.output_data_frame("uq_reliability_matrix_table"),
                    ui.p("Note: Reliability matrices and threshold sensitivity plots are only supported for a single Parameter of Interest.", class_="small text-muted mt-2 mb-0"),
                    class_="mt-3"
                ),
                col_widths=[12]
            )

        return ui.layout_columns(
            ui.div(
                # --- PLOTS ---
                ui.layout_columns(
                    ui.card(
                        ui.card_header(f"{input.outcome_col()} Surface" if is_multi_dim else "Model Fit"),
                        ui.output_plot("plot_signal_uq", height="400px"),
                        full_screen=True, class_="mt-3"
                    ),
                    ui.card(
                        ui.card_header("PoD Surface Heatmap" if is_multi_dim else "PoD Curve with Confidence Bounds"),
                        ui.output_plot("plot_curve", height="400px"),
                        full_screen=True, class_="mt-3"
                    ),
                    col_widths=[6,6]
                ),
                # --- SENSITIVITY AND MATRIX ---
                row2,
                # --- METRICS & EXPORT ---
                ui.layout_columns(
                    ui.card(ui.card_header("Selected Reliability Metric"), ui.output_data_frame("uq_stats_table"), class_="mt-3"),
                    ui.card(
                        ui.card_header("Export Results"),
                        ui.p("Download a comprehensive Excel workbook containing all configuration metrics and full curve data across separate tabs.", class_="small text-muted mb-3"),
                        ui.download_button("download_excel_report", "Download Excel Report", class_="btn-success w-100", icon=icon_svg("file-excel")),
                        class_="text-center mt-3"
                    ),
                    col_widths=[8, 4]
                )
            ),
            col_widths=[-1,10,-1]
        )

    @reactive.effect
    @reactive.event(input.btn_run_uq)
    async def compute_uq():
        uq_metrics.set(None)
        study = current_study()
        if study is None:
            return

        poi_cols, nuisance_cols = list(input.pod_pois()), list(input.pod_nuisance())
        from digiqual._parallel import resolve_n_jobs
        actual_cores = resolve_n_jobs(-1 if input.pod_parallel() else 1)

        # The PoD Explorer sliders only exist once that tab has been opened. If they
        # haven't rendered, use the threshold and slices from the current fit.
        fitted = study.pod_results or {}
        slice_values = {}
        leftovers = [c for c in study.inputs if c not in poi_cols and c not in nuisance_cols]
        for col in leftovers:
            try:
                slice_values[col] = input[f"slice_{col}"]()
            except Exception:
                if col in fitted.get("slice_values", {}):
                    slice_values[col] = fitted["slice_values"][col]
        try:
            threshold = input.pod_threshold_slider()
        except Exception:
            threshold = fitted.get("threshold")
        if threshold is None:
            ui.notification_show("Fit a model in the Model Fit tab first.", type="warning")
            return

        # --- UQ TIME ESTIMATION HEURISTIC ---
        # Get the currently locked model type
        override = "polynomial" if locked_model_type() == "Polynomial" else "kriging"

        # Call the package!
        est_seconds = study.estimate_compute_time(
            model_type=override,
            n_boot=input.pod_n_boot(),
            n_nuisances=len(nuisance_cols),
            n_jobs=-1 if input.pod_parallel() else 1
        )

        time_str = f"~{max(1, int(est_seconds))} seconds" if est_seconds < 90 else f"~{int(est_seconds / 60)} minutes"

        n_boot = input.pod_n_boot()
        ui.notification_show(f"Running Bootstrap on {actual_cores} core(s). Estimated time: {time_str}...", id="uq_toast", duration=None, type="message")
        await asyncio.sleep(0.1)

        try:
            override = "polynomial" if locked_model_type() == "Polynomial" else "kriging"

            with ui.Progress(min=0, max=n_boot) as p:
                p.set(message="Running Parallel Bootstrap...", detail=f"0/{n_boot} iterations (0%)")

                def uq_progress_cb(current, total):
                    pct = int((current / total) * 100)
                    p.set(value=current, message="Bootstrapping Confidence Bounds...", detail=f"{current}/{total} iterations ({pct}%)")

                results = study.pod(
                    poi_col=poi_cols,
                    threshold=threshold,
                    nuisance_col=nuisance_cols,
                    slice_values=slice_values,
                    model_override=override, force_degree=locked_model_degree(),
                    n_boot=n_boot, n_jobs=-1 if input.pod_parallel() else 1,
                    nuisance_dists=get_nuisance_dists(nuisance_cols),
                    progress_callback=uq_progress_cb
                )

            target_pod = float(input.pod_target_val()) / 100.0
            conf_level = int(input.pod_conf_val())

            val = results["reliability_table"].get((int(target_pod * 100), conf_level), np.nan)
            aX_Y_str = "N/A (Surface)" if len(poi_cols) > 1 else (f"{val:.3f}" if not np.isnan(val) else "Not Reached")
            selected_label = f"a{int(target_pod*100)}/{conf_level} Reliability Index"

            # --- EXPANDED UQ METRICS ---
            metrics = {
                "Detection Threshold": results["threshold"],
                "Bootstrap Iterations": results["n_boot"],
                "Target PoD (%)": f"{target_pod*100:.0f}%",
                "Confidence Level (%)": f"{conf_level}%",
                selected_label: aX_Y_str
            }
            uq_metrics.set(metrics)

            export_data = {}
            if len(poi_cols) == 1:
                export_data[poi_cols[0]] = results["X_eval"].flatten()
            else:
                for i, col in enumerate(poi_cols):
                    export_data[col] = results["X_eval"][:, i]
            export_data["pod_mean"] = results["curves"]["pod"]
            export_data["ci_lower"] = results["curves"]["ci_lower"]
            export_data["ci_upper"] = results["curves"]["ci_upper"]
            pod_export_data.set(pd.DataFrame(export_data))

            study.visualise(show=False, target_pod=target_pod, confidence_level=conf_level)
            global_plot_trigger.set(global_plot_trigger() + 1)
            ui.notification_show("Uncertainty Quantification Complete!", type="success")

        except Exception as e:
            logger.exception("UQ failed")
            ui.notification_show(f"UQ Failed: {str(e) or type(e).__name__}", type="error")
        finally:
            ui.notification_remove("uq_toast")


    @render.plot
    def plot_signal_uq():
        return _render_study_plot("signal_model")

    @render.plot
    def plot_curve():
        # The UQ Analysis PoD curve is also the "pod_curve" plot
        return _render_study_plot("pod_curve")

    @reactive.calc
    def uq_selected_metrics():
        data = uq_metrics()
        if data is None:
            return None

        study = current_study()
        if study is None or not study.pod_results:
            return data

        try:
            target_pod = float(input.pod_target_val()) / 100.0
            conf_level = int(input.pod_conf_val())

            poi_cols = study.pod_results.get("poi_cols", [])

            table = study.pod_results.get("reliability_table", {})
            val = table.get((int(target_pod * 100), conf_level), np.nan)
            aX_Y_str = "N/A (Surface)" if len(poi_cols) > 1 else (f"{val:.3f}" if not np.isnan(val) else "Not Reached")

            selected_label = f"a{int(target_pod*100)}/{conf_level} Reliability Index"

            return {
                "Detection Threshold": study.pod_results["threshold"],
                "Bootstrap Iterations": study.pod_results["n_boot"],
                "Target PoD (%)": f"{target_pod*100:.0f}%",
                "Confidence Level (%)": f"{conf_level}%",
                selected_label: aX_Y_str
            }
        except Exception:
            return data

    @reactive.effect
    @reactive.event(input.pod_target_val, input.pod_conf_val)
    def update_uq_plots_target_conf():
        study = current_study()
        if study is None or not study.pod_results:
            return

        try:
            target_pod = float(input.pod_target_val()) / 100.0
            conf_level = int(input.pod_conf_val())

            study.visualise(
                show=False,
                target_pod=target_pod,
                confidence_level=conf_level
            )

            with reactive.isolate():
                global_plot_trigger.set(global_plot_trigger() + 1)
        except Exception as e:
            logger.exception("Updating the UQ plots failed")
            ui.notification_show(f"Could not update the plots: {e}", type="error")

    @render.data_frame
    def uq_reliability_matrix_table():
        study = current_study()
        if study is None or not study.pod_results:
            return None

        table = study.pod_results.get("reliability_table", {})
        if not table:
            return None

        df = reliability_matrix(table, as_text=True)
        return render.DataGrid(df, width="100%", filters=False)

    @render.plot
    def plot_uq_sensitivity():
        _ = global_plot_trigger()
        study = current_study()
        if study is None or not study.pod_results:
            return None
        if len(study.pod_results.get("poi_cols", [])) > 1:
            return None
        # The library picks the spectrum matching the current results
        return study.plot_pod_vs_threshold(show=False)

    @render.data_frame
    def uq_stats_table():
        data = uq_selected_metrics()
        return render.DataGrid(pd.DataFrame(list(data.items()), columns=["Metric", "Value"]), width="100%", filters=False) if data else None

    @render.download(filename="digiqual_pod_report.xlsx")
    def download_excel_report():
        """
        Generates an in-memory Excel workbook containing multiple tabs
        for both the configuration metrics and the raw curve data.
        """
        import io

        output = io.BytesIO()

        # Use pandas ExcelWriter to create multiple sheets
        with pd.ExcelWriter(output, engine='openpyxl') as writer:

            # --- Sheet 1: Summary Metrics ---
            combined_metrics = {}
            fit_data = fit_metrics()
            if fit_data is not None:
                # Strip out the LaTeX, but keep "Model Equation" (the plain text)
                excel_data = {k: v for k, v in fit_data.items() if k != "LaTeX Equation"}
                combined_metrics.update(excel_data)
            uq_data = uq_selected_metrics()
            if uq_data is not None:
                combined_metrics.update(uq_data)

            if combined_metrics:
                df_metrics = pd.DataFrame(list(combined_metrics.items()), columns=["Metric", "Value"])
                df_metrics.to_excel(writer, sheet_name="Summary Metrics", index=False)

            # --- Sheet 2: Full Curve Data ---
            df_curve = pod_export_data()
            if df_curve is not None:
                df_curve.to_excel(writer, sheet_name="PoD Curve Data", index=False)

            # --- Sheet 3: Reliability Matrix ---
            study = current_study()
            if study is not None and study.pod_results:
                table = study.pod_results.get("reliability_table", {})
                if table:
                    # Numbers stay numeric in Excel; missing values read "Not Reached"
                    df_matrix = reliability_matrix(table).astype(object)
                    df_matrix = df_matrix.where(df_matrix.notna(), "Not Reached")
                    df_matrix.to_excel(writer, sheet_name="Reliability Matrix", index=False)

        # Return the bytes to trigger the download
        yield output.getvalue()

