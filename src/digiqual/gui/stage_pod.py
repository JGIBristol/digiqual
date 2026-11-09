"""PoD Analysis stage: bootstrap confidence bounds, reliability indices and the exports.

The analysis uses the Model Explorer's settings (the model, what is plotted, integrated
and sliced, and the threshold); this stage only adds the bootstrap on top. The bootstrap
result keeps a copy of everything it used, so the exports describe what was actually run.
"""

import numpy as np
import pandas as pd
from faicons import icon_svg
from shiny import reactive, render, ui
from shiny.types import SilentException

from digiqual.defaults import CONFIDENCE_LEVELS, DEFAULT_N_BOOT, N_THRESHOLD_POINTS, TARGET_PODS
from digiqual.pod import reliability_matrix

from .background import TaskProgress, progress_panel, push_progress
from .components import (
    _goto_js,
    config_errors,
    curve_export_df,
    metrics_table,
    nav_button,
    notice,
    setup_rows,
    stage_page,
)
from .state import AppState, logger, read_input, visualise

MIN_N_BOOT = 10


def parse_n_boot(value) -> int | None:
    """The bootstrap iterations typed in, or None if not a whole number >= MIN_N_BOOT."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(n) or n != int(n) or n < MIN_N_BOOT:
        return None
    return int(n)


def uq_metrics(res: dict, target: int, conf: int) -> dict:
    """The metrics table of bootstrap results `res`, for a target PoD and confidence (%)."""
    rows = {"Detection Threshold": res["threshold"], "Bootstrap Iterations": res["n_boot"]}
    if len(res["poi_cols"]) > 1:
        # A PoD surface has no single flaw size, so no aX/Y value
        rows["Confidence Level (%)"] = f"{conf}%"
        return rows
    val = res["reliability_table"].get((target, conf), np.nan)
    return {
        **rows,
        "Target PoD (%)": f"{target}%",
        "Confidence Level (%)": f"{conf}%",
        f"a{target}/{conf} Reliability Index": f"{val:.3f}" if not np.isnan(val) else "Not Reached",
    }


panel = stage_page(
    "pod",
    [
        ui.tags.h6("Analysis setup", class_="text-secondary small text-uppercase fw-bold mb-2"),
        ui.output_ui("setup_summary_ui"),
        ui.tags.h6("Bootstrap", class_="text-secondary small text-uppercase fw-bold mt-3 mb-2"),
        ui.input_numeric("pod_n_boot", "Iterations", value=DEFAULT_N_BOOT, min=MIN_N_BOOT, step=100),
        ui.input_checkbox("pod_parallel", "Parallel compute (faster)", value=True),
        ui.layout_columns(
            ui.input_select("pod_target_val", "Target PoD (%)", choices=[str(p) for p in TARGET_PODS], selected="90"),
            ui.input_select("pod_conf_val", "Confidence (%)", choices=[str(c) for c in CONFIDENCE_LEVELS], selected="95"),
            col_widths=[6, 6], gap="8px"
        ),
        ui.input_task_button("btn_run_uq", "Run Bootstrap", class_="btn-danger w-100", icon=icon_svg("layer-group")),
        ui.p("The bootstrap refits the chosen model's shape to resampled data; the bounds reflect parameter "
             "uncertainty assuming that shape is correct.", class_="small text-muted mt-2 mb-0"),
    ],
    ui.output_ui("uq_warnings_ui"),
    ui.output_ui("uq_results_ui"),
)


def register(input, output, session, state: AppState):
    current_study = state.current_study
    model_fit = state.model_fit
    uq_result = state.uq_result
    explorer_config = state.explorer_config

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
        if state.uq_running():
            return _uq_panel()
        if state.applied_structure() is None:
            return notice("Set up the PoD in the Model Explorer (what to plot, integrate over and slice) "
                          "and press Update PoD first.", action=nav_button("explorer"))
        if ready_config() is None:
            return notice("The Model Explorer settings have changed or are incomplete. Press Update PoD there "
                          "before running the bootstrap.", kind="warning", title="Explorer not up to date",
                          action=nav_button("explorer"))
        if uq_result() is None:
            return notice("Choose the bootstrap settings in the sidebar and press Run Bootstrap.", kind="info")
        return ui.div()

    @render.ui
    def setup_summary_ui():
        config = explorer_config()
        if config is None or config_errors(config) or model_fit() is None:
            return ui.p("Set up the PoD in the Model Explorer first.", class_="small text-muted")
        rows = setup_rows(config, model_fit()["model_name"])
        return ui.div(
            ui.tags.dl(*[item for k, v in rows.items()
                         for item in (ui.tags.dt(k, class_="col-6 fw-normal text-muted"), ui.tags.dd(v, class_="col-6 mb-1"))],
                       class_="row small mb-1"),
            ui.tags.button("Edit in Model Explorer", type="button", class_="btn btn-link btn-sm p-0",
                           onclick=_goto_js("explorer")),
        )

    # The bootstrap runs in a worker thread (an extended task), so the page stays live
    # and shows its progress; the bootstrap's own workers still use the chosen cores.
    progress = TaskProgress()

    @ui.bind_task_button(button_id="btn_run_uq")
    @reactive.extended_task
    async def uq_task(study, config, kwargs, n_boot, n_jobs, record):
        def work():
            spectrum = None
            if len(config["plot"]) == 1:
                # For the PoD-vs-threshold plot; a cache hit unless the slice moved
                spectrum = study.compute_pod_spectrum(n_threshold_points=N_THRESHOLD_POINTS, **kwargs)
            results = study.pod(threshold=config["threshold"], n_boot=n_boot, n_jobs=n_jobs,
                                progress_callback=progress.on_count, **kwargs)
            return spectrum, results
        spectrum, results = await progress.run(work)
        return study, {**record, "results": results, "spectrum": spectrum}

    @reactive.effect
    @reactive.event(input.btn_run_uq)
    def compute_uq():
        study = current_study()
        config = ready_config()
        if study is None or config is None:
            ui.notification_show("Update the PoD in the Model Explorer first.", type="warning")
            return
        if state.uq_running():
            return
        if state.pod_running() or state.fit_running():
            ui.notification_show("Wait for the model fit or PoD computation to finish first.", type="warning")
            return
        n_boot = parse_n_boot(input.pod_n_boot())
        if n_boot is None:
            ui.notification_show(f"Bootstrap iterations must be a whole number of at least {MIN_N_BOOT}.",
                                 type="warning")
            return

        n_jobs = -1 if input.pod_parallel() else 1
        # Keep everything the run used, so the exports describe it even if settings change later
        record = {
            "config": config,
            "fit": {k: model_fit()[k] for k in ("metrics", "equation", "model_name", "choice")},
            "settings": {
                "n_boot": n_boot, "n_jobs": n_jobs, "target": int(input.pod_target_val()),
                "conf": int(input.pod_conf_val()), "csv_name": state.uploaded_name(),
                "inputs": list(study.inputs), "outcome": study.outcome,
                "thresholds": {k: read_input(input, k) for k in
                               ("ui_max_gap", "ui_min_r2", "ui_avg_cv", "ui_max_cv", "ui_max_vif")},
            },
        }
        from digiqual._parallel import resolve_n_jobs
        progress.reset(one_plotted=len(config["plot"]) == 1, n_boot=n_boot, cores=resolve_n_jobs(n_jobs))
        state.uq_running.set(True)
        uq_task.invoke(study, config, state.pod_kwargs(config), n_boot, n_jobs, record)

    @reactive.effect
    def finish_uq():
        status = uq_task.status()
        if status not in ("success", "error"):
            return
        with reactive.isolate():
            state.uq_running.set(False)
            if status == "error":
                try:
                    uq_task.result()
                except Exception as e:
                    logger.exception("UQ failed")
                    ui.notification_show(f"Bootstrap failed: {str(e) or type(e).__name__}", type="error")
                return
            study, result = uq_task.result()
            if study is not current_study():
                return  # the data changed meanwhile
            uq_result.set(result)
            # The explorer shows these results (with bounds) until a setting changes
            visualise(study)
            state.bump_plots()

    def _uq_view():
        """(steps, fraction) of the running bootstrap."""
        info = progress.info
        cores = info["cores"]
        done, total, _label = progress.count or (0, info["n_boot"], "")
        bootstrapping = progress.seen("4. Running Bootstrap")
        steps = []
        if info["one_plotted"]:
            steps.append(("Threshold spectrum, for the PoD vs threshold plot", "", "done" if bootstrapping else "active"))
        steps.append(("Bootstrap: refitting the model to resampled data",
                      f"{done} of {total} iterations, on {cores} core{'s' if cores > 1 else ''}",
                      "todo" if not bootstrapping else "done" if done >= total else "active"))
        steps.append(("Confidence bounds and reliability indices", "", "active" if done >= total else "todo"))
        return steps, done / max(total, 1) if bootstrapping else 0.0

    def _uq_panel():
        steps, fraction = _uq_view()
        return progress_panel("uq-progress", "Running the bootstrap", progress, steps, fraction)

    @reactive.effect
    async def tick_uq_progress():
        """Update the panel in place twice a second (redrawing it would restart its spinners)."""
        if not state.uq_running():
            return
        reactive.invalidate_later(0.5)
        steps, fraction = _uq_view()
        await push_progress(session, "uq-progress", progress, steps, fraction)

    def _targets():
        return int(input.pod_target_val()), int(input.pod_conf_val())

    @render.ui
    def uq_results_ui():
        uq = uq_result()
        if uq is None or state.uq_running():
            return ui.div()
        is_multi_dim = len(uq["results"]["poi_cols"]) > 1
        stale = notice("These results are for the earlier settings. The model or explorer settings have changed "
                       "since; run the bootstrap again to update them.", kind="warning", title="Results out of date") \
            if explorer_config() != uq["config"] else ui.div()

        if is_multi_dim:
            # aX/Y values and PoD vs threshold need one plotted variable (a flaw size)
            row2 = ui.p(ui.span(icon_svg("circle-info"), class_="me-1"),
                        "With two plotted variables the PoD is a surface, so there are no aX/Y flaw sizes or "
                        "PoD vs threshold plot. Plot a single variable in the Model Explorer to get them.",
                        class_="small text-muted mb-3")
        else:
            matrix_card = ui.card(
                ui.card_header("Reliability Index Matrix (aX/Y)"),
                ui.output_data_frame("uq_reliability_matrix_table"),
                ui.p("Flaw sizes for combinations of Target PoD (rows) and Confidence Level (columns).",
                     class_="small text-muted mt-2 mb-0"),
            )
            row2 = ui.layout_columns(
                ui.card(
                    ui.card_header("PoD vs Detection Threshold"),
                    ui.output_plot("plot_uq_sensitivity", height="400px"),
                    ui.p("PoD as a function of the detection threshold for representative defect sizes.",
                         class_="small text-muted pt-2 mb-0"),
                    full_screen=True,
                ),
                matrix_card,
                col_widths=[7, 5]
            )

        return ui.div(
            stale,
            ui.card(
                ui.card_header("PoD Surface Heatmap" if is_multi_dim else "PoD Curve with Confidence Bounds"),
                ui.output_plot("plot_uq_curve", height="460px"),
                full_screen=True,
            ),
            row2,
            ui.layout_columns(
                ui.card(ui.card_header("Run Summary" if is_multi_dim else "Selected Reliability Metric"),
                        ui.output_data_frame("uq_stats_table")),
                ui.card(
                    ui.card_header("Export"),
                    ui.download_button("download_excel_report", "Excel Report", class_="btn-success w-100",
                                       icon=icon_svg("file-excel")),
                    ui.p("The report holds the model, setup, metrics, curve data and reliability matrix.",
                         class_="small text-muted mt-2 mb-0"),
                ),
                col_widths=[7, 5]
            ),
        )

    @render.plot
    def plot_uq_curve():
        uq = uq_result()
        if uq is None:
            raise SilentException()
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
            raise SilentException()
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
        uq = uq_result()
        data = uq_metrics(uq["results"], *_targets()) if uq else None
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
                # The bootstrap's own target and confidence, not the current selection
                metrics = uq_metrics(uq["results"], uq["settings"]["target"], uq["settings"]["conf"])
                metrics_table(metrics).to_excel(writer, sheet_name="UQ Metrics", index=False)
                curve_export_df(uq["results"]).to_excel(writer, sheet_name="PoD Curve Data", index=False)
                table = uq["results"].get("reliability_table")
                if table:
                    # Numbers stay numeric in Excel; missing values read "Not Reached"
                    df_matrix = reliability_matrix(table).astype(object)
                    df_matrix = df_matrix.where(df_matrix.notna(), "Not Reached")
                    df_matrix.to_excel(writer, sheet_name="Reliability Matrix", index=False)
        yield output.getvalue()
