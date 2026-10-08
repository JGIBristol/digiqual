"""Model Fit tab: choose parameters, fit the mean model and inspect the fit."""

import asyncio

import numpy as np
import pandas as pd
from faicons import icon_svg
from shiny import reactive, render, ui

from digiqual.defaults import KRIGING_MAX_SAMPLES, MAX_POLY_DEGREE, N_THRESHOLD_POINTS
from digiqual.integration import estimate_nuisance_distribution

from .state import AppState, create_warning_card, logger

panel = ui.nav_panel(
    "Model Fit",
    ui.div(
        ui.h3("Model Fit & Response", class_="mb-4 text-center"),
        ui.output_ui("fit_warnings_ui"),
        ui.output_ui("fit_main_ui"),
        ui.output_ui("fit_results_ui"),
        class_="container-fluid py-3"
    ),
    icon=icon_svg("chart-area")
)


def register(input, output, session, state: AppState):
    uploaded_data = state.uploaded_data
    validation_passed = state.validation_passed
    diagnostic_table = state.diagnostic_table
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
    def fit_main_ui():
        # Hide entirely if diagnostics haven't run
        if diagnostic_table() is None:
            return ui.div()

        study = current_study()
        valid_inputs = list(input.input_cols()) if study else []
        n_samples = len(study.data) if study else 0
        model_choices = ["Auto (Best Fit)", "Polynomial"] if n_samples > KRIGING_MAX_SAMPLES else ["Auto (Best Fit)", "Polynomial", "Kriging"]

        return ui.layout_columns(
            ui.card(
                ui.card_header("Model Configuration"),
                ui.output_ui("model_context_note"),
                ui.layout_columns(
                    ui.div(
                        ui.input_selectize("pod_pois", "Parameters to plot (Select 1 or 2)", choices=valid_inputs, multiple=True, options={"maxItems": 2}),
                        ui.input_selectize("pod_nuisance", "Parameters to integrate over (Max 2)", choices=valid_inputs, multiple=True, options={"maxItems": 2}),
                        ui.output_ui("nuisance_distribution_ui"),
                    ),
                    ui.div(
                        ui.input_select("pod_model_override", "Model Override", choices=model_choices, selected="Auto (Best Fit)"),
                        ui.p("Auto compares polynomials and Kriging by 10-fold cross-validation and picks the "
                             "simplest model within one standard error of the lowest error.",
                             class_="small text-muted mt-n2"),
                        ui.panel_conditional("input.pod_model_override === 'Polynomial'",
                            ui.input_slider("pod_poly_degree", "Polynomial Degree", min=1, max=MAX_POLY_DEGREE, value=3, step=1),
                        ),
                    ),
                    col_widths=[6, 6]
                ),
                ui.input_task_button("btn_run_fit", "Step 1: Fit Physics Model", class_="btn-primary w-100", icon=icon_svg("bolt")),
            ),
            col_widths=[-1,10,-1]
        )

    @render.ui
    def nuisance_distribution_ui():
        study = current_study()
        if study is None:
            return ui.div()

        selected_nuis = list(input.pod_nuisance()) if input.pod_nuisance() else []
        if not selected_nuis:
            return ui.div()

        df_target = study.clean_data if not study.clean_data.empty else study.data

        configs = []
        import scipy.stats as stats

        from digiqual.integration import NUISANCE_DISTRIBUTIONS
        for col in selected_nuis:
            if col not in df_target.columns:
                continue
            vals = pd.to_numeric(df_target[col], errors="coerce").dropna()
            if vals.empty:
                continue

            c_min = float(vals.min())
            c_max = float(vals.max())

            dist_id = f"nuis_dist_{col}"
            try:
                selected_dist = input[dist_id]()
            except Exception:
                selected_dist = "Uniform"

            preview_text = ""

            if selected_dist == "Uniform":
                preview_text = f"Uniformly distributed on [{c_min:.2f}, {c_max:.2f}] (matching observed data limits)."
            else:
                try:
                    name, params = estimate_nuisance_distribution(vals, selected_dist)
                    dist = getattr(stats, name)(*params)
                    lo, hi = dist.ppf(0.025), dist.ppf(0.975)
                    labels = {"norm": ("Mean (μ)", "Std Dev (σ)"), "lognorm": ("Shape (s)", "Scale"),
                              "weibull_min": ("Shape (c)", "Scale")}[name]
                    p1, p2 = (params[0], params[1]) if name == "norm" else (params[0], params[2])
                    preview_text = (f"{selected_dist}, fitted to the data. {labels[0]} = {p1:.4f}, "
                                    f"{labels[1]} = {p2:.4f}. 95% of values lie in [{lo:.2f}, {hi:.2f}]; "
                                    f"samples are kept within the observed range [{c_min:.2f}, {c_max:.2f}].")
                except ValueError as e:
                    preview_text = f"{e} Uniform will be used instead."

            card = ui.div(
                ui.div(
                    ui.tags.strong(f"Distribution: '{col}'", class_="text-primary text-uppercase small tracking-wide"),
                    class_="border-bottom pb-1 mb-2 d-flex justify-content-between align-items-center"
                ),
                ui.input_select(
                    dist_id,
                    "Distribution Type",
                    choices=list(NUISANCE_DISTRIBUTIONS),
                    selected=selected_dist
                ),
                ui.div(
                    ui.span(icon_svg("circle-info"), class_="me-1"),
                    ui.span(preview_text, class_="small text-muted"),
                    class_="mt-2 p-2 bg-light border-start border-primary border-3 rounded-end d-flex align-items-center"
                ),
                class_="p-3 mb-3 bg-white border border-light-subtle rounded-3 shadow-sm"
            )
            configs.append(card)

        return ui.div(
            ui.tags.h6("Nuisance Parameters Integration Profile", class_="mt-3 mb-2 text-secondary small text-uppercase fw-bold"),
            ui.div(*configs),
            class_="mt-2"
        )

    # ─────────────────────────────────────────────────────────────────
    # DYNAMIC UI UPDATERS (Prevents resetting)
    # ─────────────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.pod_pois, input.pod_nuisance)
    def handle_mutual_exclusivity():
        """Prevents the same column from being selected as both PoI and Nuisance."""
        study = current_study()
        if study is None:
            return

        all_inputs = study.inputs
        selected_pois = list(input.pod_pois())
        selected_nuis = list(input.pod_nuisance())

        # Update Nuisance choices: All inputs MINUS current POIs
        nuisance_choices = [c for c in all_inputs if c not in selected_pois]
        ui.update_selectize("pod_nuisance", choices=nuisance_choices, selected=selected_nuis)

        # Update POI choices: All inputs MINUS current Nuisances
        poi_choices = [c for c in all_inputs if c not in selected_nuis]
        ui.update_selectize("pod_pois", choices=poi_choices, selected=selected_pois)

    @render.ui
    def model_context_note():
        study = current_study()
        if study is None:
            return ui.div()

        # Get all variables the model will be trained on
        all_vars = study.inputs
        var_list_str = ", ".join(all_vars)
        n_vars = len(all_vars)

        # Build a clean, styled banner
        return ui.div(
            ui.p(
                ui.span(icon_svg("circle-info"), class_="text-primary me-2"),
                ui.tags.strong("Global Model Fit: "),
                f"The underlying surrogate model is always trained on all {n_vars} initialized parameters ({var_list_str}). "
                "Use the controls below to dictate how this multi-dimensional surface is sliced and projected for visualisation.",
                class_="small text-muted mb-0"
            ),
            class_="bg-light border rounded p-2 mb-4"
        )

    @render.ui
    def fit_warnings_ui():
        if uploaded_data() is None:
            return create_warning_card("Please upload data in the 'Simulation Diagnostics' tab.")

        # Check that diagnostics have been run
        if diagnostic_table() is None:
            return ui.layout_columns(ui.div(ui.p("Please run diagnostics in the 'Simulation Diagnostics' tab before configuring your model.", class_="text-center p-4 text-muted bg-light rounded border")), col_widths=[-1,10,-1])

        if not validation_passed():
            diag = diagnostic_table()
            collinearity_failed = False
            if diag is not None and not diag.empty:
                collinearity_failed = not diag[diag["Test"] == "Collinearity Check"]["Pass"].all()

            msg = "The diagnostic tests found potential issues. Results may be unreliable."
            if collinearity_failed:
                msg = "High multicollinearity detected among input variables. Your model fit and reliability predictions might be unstable or unreliable."

            return ui.layout_columns(
                ui.div(
                    ui.h5(icon_svg("triangle-exclamation"), " Caution: Validation Issues"),
                    ui.p(msg, class_="mb-0"),
                    class_="alert alert-warning shadow-sm"
                ),
                col_widths=[-1,10,-1]
            )
        return ui.div()


    @render.ui
    def sobol_indices_ui():
        data = fit_metrics()
        if data and "Sobol Indices" in data and data["Sobol Indices"] is not None:
            sobol_data = data["Sobol Indices"]

            items = []
            for var, st_val in sobol_data.items():
                st_pct = st_val * 100

                items.append(
                    ui.div(
                        ui.tags.strong(f"{var}: "),
                        ui.span(f"{st_pct:.1f}%", class_="badge bg-primary rounded-pill fs-6"),
                        class_="d-flex justify-content-between align-items-center border-bottom py-2 text-muted small"
                    )
                )

            return ui.div(
                ui.h6(icon_svg("chart-pie"), " Parameter Sensitivity (Total Effect)", class_="text-primary mb-1 fw-bold"),
                ui.p("Impact of each parameter (including interactions). Sum may exceed 100%.", class_="small text-muted mb-2 fst-italic"),
                ui.div(*items),
                class_="mb-3 p-3 bg-light rounded border shadow-sm"
            )
        return ui.div()


    @render.ui
    def fit_results_ui():
        if fit_metrics() is None:
            return ui.div()

        study = current_study()
        is_multi_dim = len(study.pod_results.get("poi_cols", [])) > 1

        return ui.layout_columns(
            ui.div(
                ui.layout_columns(
                    ui.card(ui.card_header("Model Selection"), ui.output_plot("plot_model_selection"), full_screen=True),
                    ui.card(ui.card_header(f"{input.outcome_col()} Surface" if is_multi_dim else "Model Fit"), ui.output_plot("plot_signal"), full_screen=True),
                    col_widths=[6, 6]
                ),
                ui.card(
                    ui.card_header("Model Fit Diagnostics"),
                    ui.output_plot("plot_fit_diagnostics", height="500px"),
                    ui.p(
                        "Left: Actual vs Predicted scatter for the selected model. "
                        "Points hugging the diagonal indicate a good fit. "
                        "Right: Bootstrap convergence trace — running relative std dev across "
                        "iterations. Lines flattening below the thresholds indicate convergence.",
                        class_="small text-muted px-3 pt-2 mb-0"
                    ),
                    full_screen=True,
                    class_="mb-3"
                ),
                ui.card(
                    ui.card_header("Fit Statistics"),
                    ui.output_ui("sobol_indices_ui"),
                    ui.output_ui("mathjax_equation_ui"),
                    ui.output_data_frame("fit_stats_table")
                )
            ),
            col_widths=[-1,10,-1]
        )

    @reactive.effect
    @reactive.event(input.btn_run_fit)
    async def compute_model_fit():
        fit_metrics.set(None)
        uq_metrics.set(None)
        locked_model_type.set(None)

        study = current_study()
        if study is None:
            return

        poi_cols, nuisance_cols = list(input.pod_pois()), list(input.pod_nuisance())
        if not poi_cols or len(poi_cols) > 2:
            ui.notification_show("Select 1 or 2 Parameters to visualise.", type="error")
            return

        override_map = {"Auto (Best Fit)": "auto", "Polynomial": "polynomial", "Kriging": "kriging"}
        model_override = override_map.get(input.pod_model_override(), "auto")
        force_degree = int(input.pod_poly_degree()) if model_override == "polynomial" else None

        slice_values = {}
        leftovers = [c for c in study.inputs if c not in poi_cols and c not in nuisance_cols]
        for col in leftovers:
            try:
                # Attempt to get the dynamic slider value
                slice_values[col] = input[f"slice_{col}"]()
            except Exception:
                pass # If it hasn't rendered yet, core.py will safely default to the median!

        # --- FIT TIME ESTIMATION HEURISTIC ---
        # Call the package to get the exact estimate!
        # n_boot=0 because we are just fitting, n_jobs=1 because fitting is sequential
        est_sec = study.estimate_compute_time(
            model_type=model_override,
            n_boot=0,
            n_nuisances=len(nuisance_cols),
            n_jobs=1
        )

        time_str = f"~{max(1, int(est_sec))} seconds" if est_sec < 90 else f"~{int(est_sec / 60)} minutes"
        ui.notification_show(f"Fitting Models (Cross-Validation)... Estimated time: {time_str}", id="fit_toast", duration=None, type="message")
        await asyncio.sleep(0.1)

        try:
            # Gather non-uniform nuisance distributions
            nuisance_dists = get_nuisance_dists(nuisance_cols)

            # 1. Run the standard fit (n_boot=0) to establish Layer 1, 2, 3
            results = study.pod(
                poi_col=poi_cols, threshold=float(study.get_data_summary(study.outcome)["median"]),
                nuisance_col=nuisance_cols, slice_values=slice_values,
                model_override=model_override, force_degree=force_degree, n_boot=0,
                nuisance_dists=nuisance_dists
            )

            # 2. Trigger the Threshold Spectrum calculation (Layer 4)
            ui.notification_show("Generating Instant Threshold Spectrum...", id="spec_toast", duration=None)
            study.compute_pod_spectrum(
                poi_col=poi_cols, nuisance_col=nuisance_cols,
                slice_values=slice_values, n_threshold_points=N_THRESHOLD_POINTS,
                model_override=model_override, force_degree=force_degree,
                nuisance_dists=nuisance_dists
            )
            ui.notification_remove("spec_toast")

            mean_model = results["mean_model"]
            locked_model_type.set(mean_model.model_type_)

            # --- EXTRACT POLYNOMIAL EQUATION ---
            if mean_model.model_type_ == 'Polynomial':
                locked_model_degree.set(mean_model.model_params_)
                model_str = f"Polynomial (Degree {mean_model.model_params_})"
                # Core package now provides the formatted string directly!
                equation_latex = f"$$ {results.get('equation', 'Equation Not Available')} $$"
                equation_plain = results.get('equation', 'Equation Not Available')
            else:
                locked_model_degree.set(None)
                model_str = "Kriging (Gaussian Process)"
                equation_latex = "$$ \\text{Gaussian Process (Non-parametric)} $$"
                equation_plain = "Gaussian Process (Non-parametric)"

            cv_scores = mean_model.cv_scores_
            used_key = ('Polynomial', mean_model.model_params_) if mean_model.model_type_ == 'Polynomial' else ('Kriging', None)
            best_mse_str = f"{cv_scores.get(used_key, np.nan):.2e}"

            dist_name = results['dist_info'][0].capitalize()
            dist_params = [round(float(p), 4) for p in results['dist_info'][1]]

            slice_display = ", ".join(leftovers) if leftovers else "None"

            metrics = {
                "Parameter(s) of Interest": ", ".join(poi_cols),
                "Nuisance Parameter(s)": ", ".join(nuisance_cols) if nuisance_cols else "None",
                "Sliced Parameter(s)": slice_display,
                "Total Samples (N)": len(study.clean_data),
                "Selected Model": model_str,
                "Model Equation": equation_plain,
                "LaTeX Equation": equation_latex,
                "Model Fit (CV MSE)": best_mse_str,
                "Smoothing Bandwidth (std devs)": f"{results['bandwidth']:.4f}",
                "Error Distribution": f"{dist_name} (Params: {tuple(dist_params)})",
                "Sobol Indices": results.get("sobol_indices")
            }

            # 2. Add the nicely formatted, rounded rows specifically for the table
            sobol_raw = results.get("sobol_indices")
            if sobol_raw:
                for var, val in sobol_raw.items():
                    metrics[f"Sensitivity ({var})"] = f"{val * 100:.2f}%"

            fit_metrics.set(metrics)

            # --- EXPORT PRELIMINARY DATA ---
            export_data = {}
            if len(poi_cols) == 1:
                export_data[poi_cols[0]] = results["X_eval"].flatten()
            else:
                for i, col in enumerate(poi_cols):
                    export_data[col] = results["X_eval"][:, i]
            export_data["pod_mean"] = results["curves"]["pod"]
            export_data["ci_lower"] = np.nan
            export_data["ci_upper"] = np.nan
            pod_export_data.set(pd.DataFrame(export_data))

            study.visualise(show=False)
            global_plot_trigger.set(global_plot_trigger() + 1)
            ui.notification_show("Model Fit Complete. Proceed to Uncertainty Quantification.", type="success")

        except Exception as e:
            logger.exception("Model fit failed")
            ui.notification_show(f"Fit Failed: {str(e) or type(e).__name__}", type="error")
        finally:
            ui.notification_remove("fit_toast")
            ui.notification_remove("spec_toast")


    @render.ui
    def mathjax_equation_ui():
        data = fit_metrics()
        if data and "LaTeX Equation" in data:
            return ui.div(
                # Use HTML to inject the equation AND a tiny script to trigger MathJax
                ui.HTML(f"""
                    <div style="font-size: 1.15em; padding: 10px 0;">
                        {data['LaTeX Equation']}
                    </div>
                    <script>
                        if (window.MathJax) {{
                            MathJax.typesetPromise();
                        }}
                    </script>
                """),
                class_="text-center mb-3 px-2 py-2 bg-light rounded border",
                style="overflow-x: auto;"
            )
        return ui.div()

    @render.plot
    def plot_model_selection():
        return _render_study_plot("model_selection")

    @render.plot
    def plot_signal():
        return _render_study_plot("signal_model")

    @render.plot
    def plot_fit_diagnostics():
        study = current_study()
        if study is None or fit_metrics() is None:
            return None

        # Extract variables from study.pod_results
        res = study.pod_results
        X = res["X"]
        y = res["y"]
        model_type = locked_model_type()
        model_degree = locked_model_degree()

        import matplotlib.pyplot as plt
        from sklearn.metrics import r2_score

        from digiqual.diagnostics import bootstrap_convergence_trace
        from digiqual.pod import build_refit_model

        # Reuse the actual fitted production model (trained on all input columns,
        # same object driving the Signal Model pane) instead of refitting a throwaway
        # model in-sample, which always looked artificially perfect regardless of
        # which PoIs were selected.
        model = res["mean_model"]
        if model_type == 'Polynomial':
            y_pred = model.predict(X)
        elif model_type == 'Kriging':
            from digiqual.pod import compute_kriging_loo_residuals
            loo_means, _loo_stds, _std_residuals, _gamma = compute_kriging_loo_residuals(model, X, y)
            y_pred = loo_means
        else:
            return None

        # Get thresholds from UI (Simulation Diagnostics Tab)
        thresh_r2 = input.ui_min_r2()
        thresh_avg = input.ui_avg_cv()
        thresh_max = input.ui_max_cv()

        # Pass/fail uses the model's 10-fold cross-validated R² (from the CV error the
        # model selection already computed), so polynomials and Kriging are judged on
        # the same out-of-sample basis. The scatter shows in-sample predictions for
        # polynomials and leave-one-out predictions for Kriging.
        used_key = ('Polynomial', model.model_params_) if model_type == 'Polynomial' else ('Kriging', None)
        cv_mse = getattr(model, "cv_scores_", {}).get(used_key)
        var_y = float(np.var(y))
        r2_val = 1.0 - cv_mse / var_y if (cv_mse is not None and var_y > 0) else r2_score(y, y_pred)

        fig, (ax_fit, ax_boot) = plt.subplots(1, 2, figsize=(12, 5))

        # ── LEFT: Actual vs Predicted ─────────────────────────────────────────
        ax_fit.scatter(y, y_pred, alpha=0.55, s=25, color="#1f77b4",
                        label="Simulations", zorder=3)

        # Perfect-fit diagonal
        y_lo = min(y.min(), y_pred.min())
        y_hi = max(y.max(), y_pred.max())
        pad  = (y_hi - y_lo) * 0.05
        diag_range = [y_lo - pad, y_hi + pad]
        ax_fit.plot(diag_range, diag_range, color="#d13438", linewidth=1.5,
                    linestyle="--", label="Perfect Fit", zorder=2)

        fit_passed = r2_val >= thresh_r2
        fit_colour = "#107c10" if fit_passed else "#d13438"
        fit_status = "✓ Pass" if fit_passed else "✗ Fail"

        model_name_str = f"Poly Degree {model_degree}" if model_type == 'Polynomial' else "Kriging"
        ax_fit.set_title(f"Model Fit ({model_name_str})  —  {fit_status}",
                            color=fit_colour, fontweight="bold")
        ax_fit.set_xlabel(f"Actual  '{study.outcome}'", fontsize=9)
        ax_fit.set_ylabel(f"Predicted  '{study.outcome}'", fontsize=9)

        ax_fit.annotate(
            f"CV R² = {r2_val:.3f}\nThreshold > {thresh_r2:.2f}",
            xy=(0.05, 0.95), xycoords="axes fraction",
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                        edgecolor=fit_colour, alpha=0.9)
        )

        ax_fit.legend(fontsize=8)
        ax_fit.grid(True, alpha=0.3)

        # ── RIGHT: Bootstrap Convergence Trace ────────────────────────────────
        # Refits the selected model structure (same pipeline / fixed Kriging kernel
        # as the PoD bootstrap) to resampled data.
        n_boot = 100
        trace = bootstrap_convergence_trace(
            lambda: build_refit_model(model_type, model.model_params_), X, y, n_boot=n_boot
        )
        running_avg = trace["running_avg"]
        running_max = trace["running_max"]
        iters = trace["iterations"]

        ax_boot.plot(iters, running_avg, color="#006abc", linewidth=2,
                        label="Running Avg CV")
        ax_boot.plot(iters, running_max, color="#1f77b4", linewidth=1.5,
                        linestyle=":", alpha=0.75, label="Running Max CV")

        # Dynamic Threshold lines
        ax_boot.axhline(thresh_avg, color="#107c10", linewidth=1.2, linestyle="--",
                        alpha=0.85, label=f"Avg Threshold ({thresh_avg:.2f})")
        ax_boot.axhline(thresh_max, color="#ffb900", linewidth=1.2, linestyle="--",
                        alpha=0.85, label=f"Max Threshold ({thresh_max:.2f})")

        # Dynamic Shading
        ax_boot.fill_between(iters, 0, thresh_avg, color="#107c10", alpha=0.04)

        final_avg_cv = running_avg[-1]
        final_max_cv = running_max[-1]
        boot_passed = (final_avg_cv < thresh_avg) and (final_max_cv < thresh_max)
        boot_colour = "#107c10" if boot_passed else "#d13438"
        boot_status = "✓ Pass" if boot_passed else "✗ Fail"

        ax_boot.set_title(f"Bootstrap Convergence  —  {boot_status}",
                            color=boot_colour, fontweight="bold")
        ax_boot.set_xlabel("Bootstrap Iteration", fontsize=9)
        ax_boot.set_ylabel("Relative Std Dev (CV)", fontsize=9)
        ax_boot.set_xlim(1, n_boot)
        ax_boot.set_ylim(bottom=0)

        ax_boot.annotate(
            f"Final Avg CV = {final_avg_cv:.3f}  (< {thresh_avg:.2f})\n"
            f"Final Max CV = {final_max_cv:.3f}  (< {thresh_max:.2f})",
            xy=(0.97, 0.97), xycoords="axes fraction",
            va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                        edgecolor=boot_colour, alpha=0.9)
        )

        ax_boot.legend(fontsize=8, loc="upper right",
                        bbox_to_anchor=(0.97, 0.70))
        ax_boot.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    @render.data_frame
    def fit_stats_table():
        data = fit_metrics()
        if data is None:
            return None
        display_data = {k: v for k, v in data.items() if k not in ["LaTeX Equation", "Model Equation", "Sobol Indices"]}
        return render.DataGrid(pd.DataFrame(list(display_data.items()), columns=["Metric", "Value"]), width="100%", filters=False)

