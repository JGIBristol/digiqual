"""PoD Analysis tab: bootstrap confidence bounds, reliability indices and the Excel report.

The analysis uses the Model Explorer's settings (what is plotted, integrated and
sliced, and the threshold); this tab only adds the bootstrap on top.
"""

import asyncio

import numpy as np
import pandas as pd
from faicons import icon_svg
from shiny import reactive, render, ui

from digiqual.defaults import CONFIDENCE_LEVELS, DEFAULT_N_BOOT, N_THRESHOLD_POINTS, TARGET_PODS
from digiqual.pod import reliability_matrix

from .components import (
    TAB,
    config_errors,
    curve_export_df,
    format_duration,
    goto_button,
    metrics_table,
    notice,
    register_goto,
    setup_rows,
)
from .state import AppState, logger

TITLE = TAB["pod"].title

panel = ui.nav_panel(
    TITLE,
    ui.div(
        ui.h3(TITLE, class_="mb-4 text-center"),
        ui.output_ui("uq_warnings_ui"),
        ui.output_ui("uq_main_ui"),
        ui.output_ui("uq_results_ui"),
        class_="container-fluid py-3"
    ),
    value="pod",
    icon=icon_svg(TAB["pod"].icon)
)


def register(input, output, session, state: AppState):
    current_study = state.current_study
    model_fit = state.model_fit
    uq_result = state.uq_result
    explorer_config = state.explorer_config

    for button, tab in [("goto_explorer_from_pod", "explorer"), ("edit_setup_from_pod", "explorer")]:
        register_goto(input, button, tab)

    @reactive.calc
    def ready_config():
        """The explorer settings, if they are valid and the explorer PoD is up to date."""
        config = explorer_config()
        if (config is None or config_errors(config) or state.applied_structure() is None
                or state.structure(config) != state.applied_structure()):
            return None
        return config

    @render.ui
    def uq_warnings_ui():
        if state.applied_structure() is None:
            return notice(f"Choose the variables to plot, integrate and slice in the {TAB['explorer'].title} "
                          "and press Update PoD first.",
                          action=goto_button("goto_explorer_from_pod", "explorer"))
        if ready_config() is None:
            return notice(f"The {TAB['explorer'].title} settings have changed or are incomplete. "
                          "Press Update PoD there before running the bootstrap.",
                          kind="warning", title="Explorer not up to date",
                          action=goto_button("goto_explorer_from_pod", "explorer"))
        return ui.div()

    @render.ui
    def uq_main_ui():
        if state.applied_structure() is None:
            return ui.div()

        return ui.layout_columns(
            ui.card(
                ui.card_header("Analysis Setup"),
                ui.output_data_frame("setup_table"),
                ui.p("The bootstrap refits the chosen model's shape to resampled data; the bounds reflect "
                     "parameter uncertainty assuming that shape is correct.", class_="small text-muted mt-2 mb-2"),
                goto_button("edit_setup_from_pod", "explorer", "Edit in " + TAB["explorer"].title),
            ),
            ui.card(
                ui.card_header("Bootstrap & Targets"),
                ui.layout_columns(
                    ui.input_numeric("pod_n_boot", "Bootstrap Iterations", value=DEFAULT_N_BOOT, min=10, step=100),
                    ui.div(ui.input_checkbox("pod_parallel", "Parallel compute (faster)", value=True), class_="pt-4"),
                    col_widths=[6, 6]
                ),
                ui.layout_columns(
                    ui.input_select("pod_target_val", "Target PoD (%)", choices=[str(p) for p in TARGET_PODS], selected="90"),
                    ui.input_select("pod_conf_val", "Confidence Level (%)", choices=[str(c) for c in CONFIDENCE_LEVELS], selected="95"),
                    col_widths=[6, 6]
                ),
                ui.input_task_button("btn_run_uq", "Run Bootstrap", class_="btn-danger w-100", icon=icon_svg("layer-group")),
            ),
            col_widths=[-1, 5, 5, -1]
        )

    @render.data_frame
    def setup_table():
        config = explorer_config()
        if config is None or config_errors(config) or model_fit() is None:
            return None
        return render.DataGrid(metrics_table(setup_rows(config, model_fit()["model_name"])), width="100%", filters=False)

    @reactive.effect
    @reactive.event(input.btn_run_uq)
    async def compute_uq():
        study = current_study()
        config = ready_config()
        if study is None or config is None:
            ui.notification_show(f"Update the PoD in the {TAB['explorer'].title} first.", type="warning")
            return

        kwargs = state.pod_kwargs(config)
        n_boot = int(input.pod_n_boot() or DEFAULT_N_BOOT)
        n_jobs = -1 if input.pod_parallel() else 1
        from digiqual._parallel import resolve_n_jobs
        est = study.estimate_compute_time(model_type=kwargs["model_override"], n_boot=n_boot,
                                          n_nuisances=len(config["integrate"]), n_jobs=n_jobs)
        ui.notification_show(f"Running bootstrap on {resolve_n_jobs(n_jobs)} core(s). "
                             f"Estimated time: {format_duration(est)}...", id="uq_toast", duration=None, type="message")
        await asyncio.sleep(0.1)

        try:
            spectrum = None
            if len(config["plot"]) == 1:
                # For the PoD-vs-threshold plot; a cache hit unless the slice moved
                spectrum = study.compute_pod_spectrum(n_threshold_points=N_THRESHOLD_POINTS, **kwargs)

            with ui.Progress(min=0, max=n_boot) as p:
                p.set(message="Bootstrapping confidence bounds...", detail=f"0/{n_boot} iterations (0%)")

                def progress(current, total):
                    p.set(value=current, detail=f"{current}/{total} iterations ({int(current / total * 100)}%)")

                results = study.pod(threshold=config["threshold"], n_boot=n_boot, n_jobs=n_jobs,
                                    progress_callback=progress, **kwargs)

            # Keep the model details with the results, in case the explorer's model changes later
            uq_result.set({"config": config, "results": results, "spectrum": spectrum,
                           "fit": {k: model_fit()[k] for k in ("metrics", "equation", "model_name")}})
            # The explorer shows these results (with bounds) until a setting changes
            study.visualise(show=False)
            state.bump_plots()
            ui.notification_show("Bootstrap complete.", type="message")

        except Exception as e:
            logger.exception("UQ failed")
            ui.notification_show(f"Bootstrap failed: {str(e) or type(e).__name__}", type="error")
        finally:
            ui.notification_remove("uq_toast")

    def _targets():
        return int(input.pod_target_val()), int(input.pod_conf_val())

    @reactive.calc
    def uq_metrics():
        uq = uq_result()
        if uq is None:
            return None
        res = uq["results"]
        target, conf = _targets()
        if len(res["poi_cols"]) > 1:
            value = "N/A (Surface)"
        else:
            val = res["reliability_table"].get((target, conf), np.nan)
            value = f"{val:.3f}" if not np.isnan(val) else "Not Reached"
        return {
            "Detection Threshold": res["threshold"],
            "Bootstrap Iterations": res["n_boot"],
            "Target PoD (%)": f"{target}%",
            "Confidence Level (%)": f"{conf}%",
            f"a{target}/{conf} Reliability Index": value,
        }

    @render.ui
    def uq_results_ui():
        uq = uq_result()
        if uq is None:
            return ui.div()
        is_multi_dim = len(uq["results"]["poi_cols"]) > 1
        stale = notice("These results are for the settings below. The explorer settings have changed since; "
                       "run the bootstrap again to update them.", kind="warning", title="Results out of date") \
            if explorer_config() != uq["config"] else ui.div()

        matrix_card = ui.card(
            ui.card_header("Reliability Index Matrix (aX/Y)"),
            ui.output_data_frame("uq_reliability_matrix_table"),
            ui.p("Flaw sizes for combinations of Target Probability of Detection (rows) and Confidence Levels (columns)."
                 if not is_multi_dim else
                 "Reliability matrices and threshold sensitivity plots need a single plotted variable.",
                 class_="small text-muted mt-2 mb-0"),
            class_="mt-3"
        )
        if is_multi_dim:
            row2 = ui.layout_columns(matrix_card, col_widths=[12])
        else:
            row2 = ui.layout_columns(
                ui.card(
                    ui.card_header("Probability of Detection vs. Threshold"),
                    ui.output_plot("plot_uq_sensitivity", height="400px"),
                    ui.p("PoD as a function of the detection threshold for representative defect sizes.",
                         class_="small text-muted px-3 pt-2 mb-0"),
                    full_screen=True, class_="mt-3"
                ),
                matrix_card,
                col_widths=[7, 5]
            )

        return ui.div(
            stale,
            ui.layout_columns(
                ui.div(
                    ui.card(
                        ui.card_header("PoD Surface Heatmap" if is_multi_dim else "PoD Curve with Confidence Bounds"),
                        ui.output_plot("plot_uq_curve", height="450px"),
                        full_screen=True, class_="mt-3"
                    ),
                    row2,
                    ui.layout_columns(
                        ui.card(ui.card_header("Selected Reliability Metric"), ui.output_data_frame("uq_stats_table"), class_="mt-3"),
                        ui.card(
                            ui.card_header("Export Results"),
                            ui.p("Excel workbook with the model, analysis setup, metrics, curve data and reliability matrix.",
                                 class_="small text-muted mb-3"),
                            ui.download_button("download_excel_report", "Download Excel Report", class_="btn-success w-100",
                                               icon=icon_svg("file-excel")),
                            class_="text-center mt-3"
                        ),
                        col_widths=[8, 4]
                    )
                ),
                col_widths=[-1, 10, -1]
            )
        )

    @render.plot
    def plot_uq_curve():
        uq = uq_result()
        if uq is None:
            return None
        from digiqual.plotting import plot_pod_curve, plot_pod_surface

        res = uq["results"]
        target, conf = _targets()
        ci_lower, ci_upper = res["ci_bounds"].get(conf, (res["curves"]["ci_lower"], res["curves"]["ci_upper"]))
        if len(res["poi_cols"]) == 1:
            ax = plot_pod_curve(X_eval=np.asarray(res["X_eval"]).flatten(), pod_curve=res["curves"]["pod"],
                                ci_lower=ci_lower, ci_upper=ci_upper, target_pod=target / 100,
                                confidence_level=conf, poi_name=res["poi_cols"][0])
        else:
            ax = plot_pod_surface(poi_grids=res["poi_grids"], pod_curve=res["curves"]["pod"],
                                  poi_names=res["poi_cols"], ci_lower=ci_lower)
        return ax.get_figure()

    @render.plot
    def plot_uq_sensitivity():
        uq = uq_result()
        if uq is None or uq["spectrum"] is None:
            return None
        from digiqual.plotting import plot_pod_vs_threshold

        res, spec = uq["results"], uq["spectrum"]
        ax = plot_pod_vs_threshold(X_eval=np.asarray(res["X_eval"]).flatten(), thresholds=spec["thresholds"],
                                   pod_matrix=spec["pod_matrix"], poi_name=res["poi_cols"][0])
        fig = ax.get_figure()
        fig.tight_layout()
        return fig

    @render.data_frame
    def uq_reliability_matrix_table():
        uq = uq_result()
        table = uq["results"].get("reliability_table") if uq else None
        if not table:
            return None
        return render.DataGrid(reliability_matrix(table, as_text=True), width="100%", filters=False)

    @render.data_frame
    def uq_stats_table():
        data = uq_metrics()
        return render.DataGrid(metrics_table(data), width="100%", filters=False) if data else None

    @render.download(filename="digiqual_pod_report.xlsx")
    def download_excel_report():
        """Excel workbook: model, analysis setup, UQ metrics, curve data and reliability matrix."""
        import io

        uq = uq_result()
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            if uq is not None:
                fit = uq["fit"]
                model_rows = {**fit["metrics"], "Model Equation": fit.get("equation") or "Gaussian Process (Non-parametric)"}
                metrics_table(model_rows).to_excel(writer, sheet_name="Model", index=False)
                metrics_table(setup_rows(uq["config"], fit["model_name"])).to_excel(
                    writer, sheet_name="Analysis Setup", index=False)
                metrics_table(uq_metrics()).to_excel(writer, sheet_name="UQ Metrics", index=False)
                curve_export_df(uq["results"]).to_excel(writer, sheet_name="PoD Curve Data", index=False)
                table = uq["results"].get("reliability_table")
                if table:
                    # Numbers stay numeric in Excel; missing values read "Not Reached"
                    df_matrix = reliability_matrix(table).astype(object)
                    df_matrix = df_matrix.where(df_matrix.notna(), "Not Reached")
                    df_matrix.to_excel(writer, sheet_name="Reliability Matrix", index=False)
        yield output.getvalue()
