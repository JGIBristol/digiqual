"""Model fitting inside the Model Explorer: candidate models, the model choice and fit checks.

All candidate models (polynomials of degree 1-10 and Kriging) are cross-validated the
first time the Model Explorer opens for a dataset. Choosing a model afterwards only
picks one of them (plus its variance model), so it is quick. Nothing here depends on
which variables are plotted, integrated or sliced.
"""

import asyncio

import numpy as np
from faicons import icon_svg
from shiny import reactive, render, ui

from .components import (
    diagnostic_failures,
    failure_list,
    format_duration,
    metrics_table,
)
from .state import AppState, logger


def _used_key(model):
    return ('Polynomial', model.model_params_) if model.model_type_ == 'Polynomial' else ('Kriging', None)


def _model_name(model):
    return f"Polynomial (Degree {model.model_params_})" if model.model_type_ == 'Polynomial' else "Kriging (Gaussian Process)"


def _parse_choice(choice: str):
    """'auto', 'poly-3' or 'kriging' -> (model_override, force_degree)."""
    if choice.startswith("poly-"):
        return "polynomial", int(choice.split("-")[1])
    return choice, None


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


fit_view = ui.div(
    ui.output_ui("fit_caution_ui"),
    ui.output_ui("fit_results_ui"),
)


def register(input, output, session, state: AppState):
    current_study = state.current_study
    model_fit = state.model_fit

    async def fit(choice: str):
        study = current_study()
        if study is None:
            return
        override, degree = _parse_choice(choice)
        first = not state.candidates_ready()
        est = study.estimate_compute_time(model_type=override, n_boot=0, n_nuisances=0, n_jobs=1)
        message = ("Fitting the candidate models (cross-validation)..." if first
                   else "Fitting the variance model for the chosen model...")
        ui.notification_show(f"{message} Estimated time: {format_duration(est)}",
                             id="fit_toast", duration=None, type="message")
        await asyncio.sleep(0.1)
        try:
            result = study.fit_model(model_override=override, force_degree=degree)
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

            model_fit.set({**result, "choice": choice, "cv_r2": cv_r2, "metrics": metrics,
                           "model_name": _model_name(model)})
            state.fit_error.set(None)
            state.candidates_ready.set(True)
        except Exception as e:
            logger.exception("Model fit failed")
            state.fit_error.set(str(e) or type(e).__name__)
        finally:
            ui.notification_remove("fit_toast")

    @reactive.effect
    async def auto_fit():
        """Fit the candidate models the first time the Model Explorer is opened for a dataset."""
        if input.navbar() != "explorer" or current_study() is None or state.diagnostic_table() is None:
            return
        if state.candidates_ready() or state.fit_error() is not None:
            return
        await fit("auto")

    @reactive.effect
    @reactive.event(input.model_choice, ignore_init=True)
    async def change_model():
        current = model_fit()
        if not state.candidates_ready() or current is None or input.model_choice() == current["choice"]:
            return
        await fit(input.model_choice())

    @reactive.effect
    @reactive.event(input.btn_retry_fit)
    def retry_fit():
        state.fit_error.set(None)

    # ── Model fit sub-tab ───────────────────────────────────────────────────
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
            return ui.p("The model fit appears here once the candidate models are fitted.",
                        class_="text-muted text-center p-4")

        kriging_card = ui.card(
            ui.card_header("Kriging Leave-One-Out Residuals"),
            ui.output_plot("plot_kriging_diagnostics", height="380px"),
            ui.p("Standardised leave-one-out residuals against a standard normal. Points outside ±3 "
                 "suggest outliers or a kernel that is too smooth.", class_="small text-muted px-3 pt-2 mb-0"),
            full_screen=True,
        ) if fit_["mean_model"].model_type_ == "Kriging" else None

        return ui.div(
            ui.card(
                ui.card_header("Model Selection"),
                ui.output_plot("plot_model_selection", height="480px"),
                ui.p("Cross-validated error of every candidate model. Auto picks the simplest model within "
                     "one standard error of the lowest error.", class_="small text-muted px-3 pt-2 mb-0"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header("Fit Quality"),
                ui.output_plot("plot_fit_diagnostics", height="460px"),
                ui.p(
                    "Left: actual vs predicted for the selected model (in-sample for polynomials, "
                    "leave-one-out for Kriging); pass/fail uses the 10-fold cross-validated R². "
                    "Right: bootstrap convergence trace, the running relative std dev across iterations. "
                    "Lines flattening below the thresholds (set in Data Diagnostics) indicate convergence.",
                    class_="small text-muted px-3 pt-2 mb-0"
                ),
                full_screen=True,
            ),
            kriging_card,
            ui.layout_columns(
                ui.card(ui.card_header("Parameter Sensitivity"), ui.output_ui("sobol_indices_ui")),
                ui.card(
                    ui.card_header("Fit Summary"),
                    ui.output_ui("mathjax_equation_ui"),
                    ui.output_data_frame("fit_stats_table"),
                ),
                col_widths=[5, 7]
            ),
        )

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
            ui.p("Total-effect Sobol indices: the impact of each parameter, including interactions. "
                 "The sum may exceed 100%.", class_="small text-muted mb-2 fst-italic"),
            ui.div(*items),
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
