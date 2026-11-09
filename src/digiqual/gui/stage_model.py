"""Model Diagnostics stage: choose the surrogate model and check how well it fits.

All candidate models (polynomials of degree 1-10 and Kriging) are cross-validated once
per dataset, when the user presses Run models; this runs in a worker thread so the page
can show its progress. Choosing a model afterwards only picks one of them (plus its
variance model), so it is quick. Nothing
here depends on which variables are later plotted, integrated or sliced.
"""

import numpy as np
from faicons import icon_svg
from shiny import reactive, render, ui

from digiqual.defaults import KRIGING_MAX_SAMPLES

from .background import TaskProgress, progress_panel, push_progress, step_states
from .components import (
    diagnostic_failures,
    failure_list,
    icon_tile,
    metrics_table,
    nav_button,
    notice,
    stage_page,
    view_tabs,
)
from .state import AppState, logger


def _used_key(model):
    return ('Polynomial', model.model_params_) if model.model_type_ == 'Polynomial' else ('Kriging', None)


def _model_name(model):
    return f"Polynomial (Degree {model.model_params_})" if model.model_type_ == 'Polynomial' else "Kriging (Gaussian Process)"


def _short_name(model):
    return f"Poly {model.model_params_}" if model.model_type_ == 'Polynomial' else "Kriging"


def _parse_choice(choice: str):
    """'auto', 'poly-3' or 'kriging' -> (model_override, force_degree)."""
    if choice.startswith("poly-"):
        return "polynomial", int(choice.split("-")[1])
    return choice, None


def _summarise(study, result, choice) -> dict:
    """The fit plus what the stage shows: CV R², display metrics and names."""
    model = result["mean_model"]
    # Judge polynomials and Kriging on the same out-of-sample basis: the 10-fold
    # CV error the model selection already computed, as an R²
    cv_mse = model.cv_scores_.get(_used_key(model), np.nan)
    var_y = float(np.var(result["y"]))
    cv_r2 = 1.0 - cv_mse / var_y if var_y > 0 else np.nan

    dist_name, dist_params = result["dist_info"]
    metrics = {
        "Total Samples (N)": len(study.clean_data),
        "Selected Model": _model_name(model),
        "Model Fit (CV MSE)": f"{cv_mse:.2e}",
        "Model Fit (CV R²)": f"{cv_r2:.3f}",
        "Smoothing Bandwidth (std devs)": f"{result['bandwidth']:.4f}",
        "Error Distribution": f"{dist_name.capitalize()} (Params: {tuple(round(float(p), 4) for p in dist_params)})",
    }
    for var, val in (result.get("sobol_indices") or {}).items():
        metrics[f"Sensitivity ({var})"] = f"{val * 100:.2f}%"
    return {**result, "choice": choice, "cv_r2": cv_r2, "metrics": metrics,
            "model_name": _model_name(model), "short_name": _short_name(model)}


def _fit_steps(progress: TaskProgress) -> list[tuple[str, str, str]]:
    """The three steps of `SimulationStudy.fit_model`, from its progress and log messages."""
    reached = 2 if progress.seen("-> Calculating Total-Order Sobol") else 1 if progress.seen("2.") else 0
    done, total, label = progress.count or (0, 1, "Starting")
    details = (f"{label} ({done + 1} of {total})" if done < total else "All candidates scored",
               "How the scatter changes across the inputs, and its distribution",
               "How much each input drives the signal")
    labels = ("Cross-validating the candidate models", "Fitting the scatter (variance) model",
              "Computing parameter sensitivity (Sobol indices)")
    return list(zip(labels, details, step_states(reached, 3), strict=True))


def _fit_fraction(progress: TaskProgress, steps) -> float:
    """Rough overall progress: cross-validation is most of the work."""
    states = [state for _l, _d, state in steps]
    if states[2] == "active":
        return 0.95
    if states[1] == "active":
        return 0.9
    done, total, _label = progress.count or (0, 1, "")
    return 0.85 * done / max(total, 1)


def model_choices(study) -> dict:
    """Dropdown choices for the candidate models fitted for `study`."""
    winner = study.cv_winner_key
    winner_name = f"Polynomial {winner[1]}" if winner[0] == "Polynomial" else "Kriging"
    choices = {"auto": f"Auto (cross-validation pick: {winner_name})"}
    for degree in sorted(k[1] for k in study.cv_scores_cache if k[0] == "Polynomial"):
        choices[f"poly-{degree}"] = f"Polynomial, degree {degree}"
    if ("Kriging", None) in study.cv_scores_cache:
        choices["kriging"] = "Kriging (Gaussian Process)"
    return choices


panel = stage_page(
    "model",
    [
        ui.output_ui("model_choice_ui"),
        ui.output_ui("model_summary_ui"),
    ],
    ui.output_ui("model_warnings_ui"),
    ui.output_ui("fit_caution_ui"),
    ui.output_ui("run_models_ui"),
    ui.output_ui("fit_results_ui"),
)


def register(input, output, session, state: AppState):
    current_study = state.current_study
    model_fit = state.model_fit

    # The fit runs in a worker thread (an extended task), so the page stays live and
    # shows its progress.
    progress = TaskProgress()

    @ui.bind_task_button(button_id="btn_run_models")
    @reactive.extended_task
    async def fit_task(study, choice, override, degree):
        result = await progress.run(study.fit_model, model_override=override, force_degree=degree,
                                    progress_callback=progress.on_count)
        return study, choice, result

    def start_fit(choice: str):
        study = current_study()
        if study is None or state.fit_running():
            return
        override, degree = _parse_choice(choice)
        # first: the slow fit of all candidates, rather than a quick model change
        progress.reset(first=not state.candidates_ready())
        state.fit_error.set(None)
        state.fit_running.set(True)
        fit_task.invoke(study, choice, override, degree)

    @reactive.effect
    def finish_fit():
        status = fit_task.status()
        if status not in ("success", "error"):
            return
        with reactive.isolate():
            state.fit_running.set(False)
            if status == "error":
                try:
                    fit_task.result()
                except Exception as e:
                    logger.exception("Model fit failed")
                    state.fit_error.set(str(e) or type(e).__name__)
                return
            study, choice, result = fit_task.result()
            if study is not current_study():
                return  # the data or columns changed while fitting
            model_fit.set(_summarise(study, result, choice))
            state.candidates_ready.set(True)

    @reactive.effect
    @reactive.event(input.btn_run_models)
    def run_models():
        start_fit("auto")

    @reactive.effect
    @reactive.event(input.model_choice, ignore_init=True)
    def change_model():
        current = model_fit()
        if not state.candidates_ready() or current is None or input.model_choice() == current["choice"]:
            return
        start_fit(input.model_choice())

    @render.ui
    def run_models_ui():
        """Before the first fit: what Run models does. While fitting: its progress."""
        study = current_study()
        if study is None or state.diagnostic_table() is None and not state.candidates_ready():
            return ui.div()
        if state.fit_running():
            return _progress_panel()
        if state.candidates_ready():
            return ui.div()
        n = len(study.clean_data) if not study.clean_data.empty else len(study.data)
        kriging = n <= KRIGING_MAX_SAMPLES
        return ui.div(
            ui.div(
                icon_tile("gears"),
                ui.div(
                    ui.h4("Fit the candidate models", class_="mb-1"),
                    ui.p(f"Trained on all {len(study.inputs)} inputs and {n} runs. This is done once per "
                         "dataset: afterwards you can switch between models instantly.", class_="text-muted mb-0"),
                ),
                class_="d-flex gap-3 align-items-start mb-3"
            ),
            ui.tags.ol(
                ui.tags.li(ui.tags.strong("Cross-validation: "),
                           "polynomials of degree 1 to 10" + (" and Kriging (a Gaussian process)" if kriging else "")
                           + " are each scored by 10-fold cross-validation, and the simplest model within one "
                           "standard error of the best is picked."
                           + ("" if kriging else f" Kriging is skipped above {KRIGING_MAX_SAMPLES} runs.")),
                ui.tags.li(ui.tags.strong("Scatter model: "),
                           "how the spread of the signal around the model changes across the inputs, "
                           "and which distribution describes it."),
                ui.tags.li(ui.tags.strong("Sensitivity: "), "how much each input drives the signal (Sobol indices)."),
                class_="text-muted mb-4"
            ),
            ui.input_task_button("btn_run_models", "Run models", icon=icon_svg("play"),
                                 class_="btn-primary btn-lg px-5"),
            ui.p("Usually takes from a few seconds to a minute; Kriging makes larger datasets slower.",
                 class_="small text-muted mt-2 mb-0"),
            class_="dq-run-models card p-4 mb-3",
        )

    def _fit_view():
        """(title, steps, fraction) of the running fit."""
        steps = _fit_steps(progress)
        if progress.info["first"]:
            return "Fitting the candidate models", steps, _fit_fraction(progress, steps)
        return "Fitting the chosen model", steps[1:], None

    def _progress_panel():
        title, steps, fraction = _fit_view()
        return progress_panel("fit-progress", title, progress, steps, fraction)

    @reactive.effect
    async def tick_fit_progress():
        """Update the panel in place twice a second (redrawing it would restart its spinners)."""
        if not state.fit_running():
            return
        reactive.invalidate_later(0.5)
        _title, steps, fraction = _fit_view()
        await push_progress(session, "fit-progress", progress, steps, fraction)

    @reactive.effect
    @reactive.event(input.btn_retry_fit)
    def retry_fit():
        state.fit_error.set(None)

    # ── Sidebar ─────────────────────────────────────────────────────────────
    @render.ui
    def model_choice_ui():
        study = current_study()
        if study is None or not state.candidates_ready():
            text = ("Fitting the candidate models..." if state.fit_running()
                    else "Run the models to choose between them here.")
            return ui.p(text, class_="small text-muted")
        with reactive.isolate():
            choice = model_fit()["choice"] if model_fit() else "auto"
        return ui.div(
            ui.input_select("model_choice", "Model", choices=model_choices(study), selected=choice),
            ui.p(f"Trained on all {len(study.inputs)} inputs. Auto picks the simplest model within one "
                 "standard error of the lowest cross-validation error.", class_="small text-muted"),
        )

    @render.ui
    def model_summary_ui():
        fit_ = model_fit()
        if fit_ is None:
            return ui.div()
        passed = fit_["cv_r2"] >= input.ui_min_r2()
        return ui.div(
            ui.div(
                ui.span("Cross-validated R²", class_="small text-muted"),
                ui.span(f"{fit_['cv_r2']:.3f} ", "✓" if passed else "✗",
                        class_=f"badge {'text-bg-success' if passed else 'text-bg-danger'} fs-6"),
                class_="d-flex justify-content-between align-items-center border-top pt-3 mb-3"
            ),
            ui.tags.h6("Parameter sensitivity", class_="text-secondary small text-uppercase fw-bold mb-1"),
            ui.output_ui("sobol_indices_ui"),
        )

    @render.ui
    def model_warnings_ui():
        if current_study() is None:
            return notice("Upload data and choose the input and outcome columns first.",
                          action=nav_button("data"))
        if state.diagnostic_table() is None and not state.candidates_ready():
            return notice("Run the data diagnostics first.", action=nav_button("data"))
        if state.fit_error() is not None:
            return notice(f"Fitting the model failed: {state.fit_error()}", kind="danger", title="Model fit failed",
                          action=ui.input_action_button("btn_retry_fit", "Try again",
                                                        class_="btn-outline-danger btn-sm mt-2"))
        return ui.div()

    @render.ui
    def fit_caution_ui():
        diag = state.diagnostic_table()
        if state.validation_passed() or diag is None:
            return ui.div()
        return ui.div(
            ui.h6(icon_svg("triangle-exclamation"), " Data Diagnostics found issues: results may be unreliable",
                  class_="mb-1"),
            failure_list(diagnostic_failures(diag)),
            class_="alert alert-warning py-2 mb-3"
        )

    @render.ui
    def fit_results_ui():
        fit_ = model_fit()
        if fit_ is None:
            return ui.div()

        views = [
            ("Model selection", "ranking-star", ui.div(
                ui.output_plot("plot_model_selection", height="520px"),
                ui.p("Cross-validated error of every candidate model. Auto picks the simplest model within "
                     "one standard error of the lowest error.", class_="small text-muted pt-2 mb-0"),
            )),
            ("Fit quality", "bullseye", ui.div(
                ui.output_plot("plot_fit_diagnostics", height="480px"),
                ui.p("Left: actual vs predicted (in-sample for polynomials, leave-one-out for Kriging); "
                     "pass/fail uses the 10-fold cross-validated R². Right: bootstrap convergence, the running "
                     "relative std dev across iterations. Thresholds are set in Data Diagnostics.",
                     class_="small text-muted pt-2 mb-0"),
            )),
        ]
        if fit_["mean_model"].model_type_ == "Kriging":
            views.append(("Kriging residuals", "chart-simple", ui.div(
                ui.output_plot("plot_kriging_diagnostics", height="420px"),
                ui.p("Standardised leave-one-out residuals against a standard normal. Points outside ±3 "
                     "suggest outliers or a kernel that is too smooth.", class_="small text-muted pt-2 mb-0"),
            )))
        views.append(("Equation & statistics", "square-root-variable", ui.div(
            ui.output_ui("mathjax_equation_ui"),
            ui.output_data_frame("fit_stats_table"),
        )))
        return view_tabs("model_view", *views)

    @render.ui
    def sobol_indices_ui():
        fit_ = model_fit()
        sobol = fit_.get("sobol_indices") if fit_ else None
        if not sobol:
            return ui.div()
        items = [
            ui.div(
                ui.tags.strong(f"{var}: "),
                ui.span(f"{val * 100:.1f}%", class_="badge bg-primary rounded-pill fs-6"),
                class_="d-flex justify-content-between align-items-center border-bottom py-2 text-muted small"
            )
            for var, val in sorted(sobol.items(), key=lambda kv: -kv[1])
        ]
        return ui.div(
            ui.div(*items),
            ui.p("Total-effect Sobol indices, including interactions; the sum may exceed 100%.",
                 class_="small text-muted mt-2 mb-0 fst-italic"),
        )

    @render.ui
    def mathjax_equation_ui():
        fit_ = model_fit()
        if fit_ is None:
            return ui.div()
        model = fit_["mean_model"]
        latex = fit_.get("equation") if model.model_type_ == "Polynomial" else "\\text{Gaussian Process (Non-parametric)}"
        return ui.div(
            # Inject the equation and a tiny script that asks MathJax to typeset it
            ui.HTML(f"""
                <div style="font-size: 1.15em; padding: 10px 0;">$$ {latex or 'Equation Not Available'} $$</div>
                <script>if (window.MathJax) {{ MathJax.typesetPromise(); }}</script>
            """),
            class_="text-center mb-3 px-2 py-2 bg-light rounded border",
            style="overflow-x: auto;"
        )

    @render.data_frame
    def fit_stats_table():
        fit_ = model_fit()
        if fit_ is None:
            return None
        return render.DataGrid(metrics_table(fit_["metrics"]), width="100%", filters=False)

    @render.plot
    def plot_model_selection():
        fit_ = model_fit()
        if fit_ is None:
            return None
        from digiqual.pod import plot_model_selection
        model = fit_["mean_model"]
        return plot_model_selection(model.cv_scores_, used_key=_used_key(model),
                                    cv_winner_key=model.cv_winner_, cv_se=getattr(model, "cv_se_", None))

    @render.plot
    def plot_kriging_diagnostics():
        fit_ = model_fit()
        model = fit_["mean_model"] if fit_ else None
        if model is None or model.model_type_ != "Kriging" or getattr(model, "loo_residuals_", None) is None:
            return None
        from digiqual.plotting import plot_kriging_diagnostics as plot_func
        return plot_func(
            std_residuals=model.loo_residuals_,
            outlier_scale_factor=getattr(model, "outlier_scale_factor_", 1.0),
            best_kernel_name=getattr(model, "best_kernel_name_", "Matérn 5/2"),
        )

    @reactive.calc
    def fit_check_data():
        """Predictions and the bootstrap trace (100 refits): only recomputed for a new model."""
        fit_ = model_fit()
        if fit_ is None:
            return None
        from digiqual.diagnostics import bootstrap_convergence_trace
        from digiqual.pod import build_refit_model, compute_kriging_loo_residuals

        model, X, y = fit_["mean_model"], fit_["X"], fit_["y"]
        if model.model_type_ == "Kriging":
            y_pred = compute_kriging_loo_residuals(model, X, y)[0]
        else:
            y_pred = model.predict(X)
        trace = bootstrap_convergence_trace(
            lambda: build_refit_model(model.model_type_, model.model_params_), X, y, n_boot=100
        )
        return {"y": y, "y_pred": y_pred, "trace": trace}

    @render.plot
    def plot_fit_diagnostics():
        fit_ = model_fit()
        data = fit_check_data()
        if fit_ is None or data is None:
            return None

        import matplotlib.pyplot as plt

        y, y_pred, trace = data["y"], data["y_pred"], data["trace"]
        r2_val = fit_["cv_r2"]
        thresh_r2, thresh_avg, thresh_max = input.ui_min_r2(), input.ui_avg_cv(), input.ui_max_cv()
        outcome = current_study().outcome

        fig, (ax_fit, ax_boot) = plt.subplots(1, 2, figsize=(12, 5))

        # ── LEFT: Actual vs Predicted ─────────────────────────────────────────
        ax_fit.scatter(y, y_pred, alpha=0.55, s=25, color="#1f77b4", label="Simulations", zorder=3)
        y_lo, y_hi = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
        pad = (y_hi - y_lo) * 0.05
        ax_fit.plot([y_lo - pad, y_hi + pad], [y_lo - pad, y_hi + pad], color="#d13438", linewidth=1.5,
                    linestyle="--", label="Perfect Fit", zorder=2)

        fit_passed = r2_val >= thresh_r2
        fit_colour = "#107c10" if fit_passed else "#d13438"
        ax_fit.set_title(f"Model Fit ({fit_['model_name']})  —  {'✓ Pass' if fit_passed else '✗ Fail'}",
                         color=fit_colour, fontweight="bold")
        ax_fit.set_xlabel(f"Actual  '{outcome}'", fontsize=9)
        ax_fit.set_ylabel(f"Predicted  '{outcome}'", fontsize=9)
        ax_fit.annotate(f"CV R² = {r2_val:.3f}\nThreshold > {thresh_r2:.2f}",
                        xy=(0.05, 0.95), xycoords="axes fraction", va="top", ha="left", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=fit_colour, alpha=0.9))
        ax_fit.legend(fontsize=8)
        ax_fit.grid(True, alpha=0.3)

        # ── RIGHT: Bootstrap Convergence Trace ────────────────────────────────
        running_avg, running_max, iters = trace["running_avg"], trace["running_max"], trace["iterations"]
        ax_boot.plot(iters, running_avg, color="#006abc", linewidth=2, label="Running Avg CV")
        ax_boot.plot(iters, running_max, color="#1f77b4", linewidth=1.5, linestyle=":", alpha=0.75,
                     label="Running Max CV")
        ax_boot.axhline(thresh_avg, color="#107c10", linewidth=1.2, linestyle="--", alpha=0.85,
                        label=f"Avg Threshold ({thresh_avg:.2f})")
        ax_boot.axhline(thresh_max, color="#ffb900", linewidth=1.2, linestyle="--", alpha=0.85,
                        label=f"Max Threshold ({thresh_max:.2f})")
        ax_boot.fill_between(iters, 0, thresh_avg, color="#107c10", alpha=0.04)

        final_avg_cv, final_max_cv = running_avg[-1], running_max[-1]
        boot_passed = (final_avg_cv < thresh_avg) and (final_max_cv < thresh_max)
        boot_colour = "#107c10" if boot_passed else "#d13438"
        ax_boot.set_title(f"Bootstrap Convergence  —  {'✓ Pass' if boot_passed else '✗ Fail'}",
                          color=boot_colour, fontweight="bold")
        ax_boot.set_xlabel("Bootstrap Iteration", fontsize=9)
        ax_boot.set_ylabel("Relative Std Dev (CV)", fontsize=9)
        ax_boot.set_xlim(1, len(iters))
        ax_boot.set_ylim(bottom=0)
        ax_boot.annotate(
            f"Final Avg CV = {final_avg_cv:.3f}  (< {thresh_avg:.2f})\n"
            f"Final Max CV = {final_max_cv:.3f}  (< {thresh_max:.2f})",
            xy=(0.97, 0.97), xycoords="axes fraction", va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=boot_colour, alpha=0.9)
        )
        ax_boot.legend(fontsize=8, loc="upper right", bbox_to_anchor=(0.97, 0.70))
        ax_boot.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig
