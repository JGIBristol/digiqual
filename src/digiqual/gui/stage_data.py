"""Data Diagnostics stage: upload results, run the sufficiency checks and inspect the data."""

import numpy as np
import pandas as pd
from faicons import icon_svg
from shiny import reactive, render, ui

from digiqual.defaults import MAX_ALLOWED_VIF, MAX_AVG_CV, MAX_GAP_RATIO, MAX_MAX_CV, MIN_R2_SCORE
from digiqual.io import clean_columns

from .components import diagnostic_failures, failure_list, largest_gap, notice, stage_page, view_tabs
from .state import AppState, logger

panel = stage_page(
    "data",
    [
        ui.input_file("upload_csv", "Simulation results (CSV)", accept=[".csv"], multiple=False),
        ui.input_selectize("input_cols", "Input variables", choices=[], multiple=True),
        ui.input_selectize("outcome_col", "Outcome (signal)", choices=[], multiple=False),
        ui.output_ui("selection_error_display"),
        ui.accordion(
            ui.accordion_panel(
                "Diagnostic thresholds",
                ui.input_numeric("ui_max_gap", "Max Gap Ratio", value=MAX_GAP_RATIO, step=0.01),
                ui.input_numeric("ui_min_r2", "Min R² Score", value=MIN_R2_SCORE, step=0.01),
                ui.input_numeric("ui_avg_cv", "Max Allowed Avg CV", value=MAX_AVG_CV, step=0.01),
                ui.input_numeric("ui_max_cv", "Max Allowed Peak CV", value=MAX_MAX_CV, step=0.01),
                ui.input_numeric("ui_max_vif", "Max Allowed VIF", value=MAX_ALLOWED_VIF, step=0.1),
            ),
            open=False
        ),
        ui.input_task_button("btn_run_diagnostics", "Run Diagnostics", class_="btn-primary w-100",
                             icon=icon_svg("stethoscope")),
    ],
    ui.output_ui("validation_status"),
    ui.output_ui("viz_content"),
)


def register(input, output, session, state: AppState):
    uploaded_data = state.uploaded_data
    validation_passed = state.validation_passed
    new_samples = state.new_samples
    diagnostic_table = state.diagnostic_table
    diagnostics_stale = state.diagnostics_stale
    current_study = state.current_study

    @reactive.effect
    def read_uploaded_csv():
        file_info = input.upload_csv()
        if file_info is None:
            uploaded_data.set(None)
            return
        try:
            numeric, text_cols, n_bad_cells = clean_columns(pd.read_csv(file_info[0]["datapath"]))
            if text_cols:
                ui.notification_show(
                    f"Ignored non-numeric column(s): {', '.join(text_cols)}.", type="warning", duration=8
                )
            if n_bad_cells:
                ui.notification_show(
                    f"{n_bad_cells} non-numeric value(s) were found and will be treated as missing.",
                    type="warning", duration=8
                )
            if numeric.shape[1] < 2:
                raise ValueError("the file needs at least two numeric columns (inputs and an outcome)")

            uploaded_data.set(numeric)
            state.uploaded_name.set(file_info[0]["name"])
        except Exception as e:
            logger.exception("CSV upload failed")
            ui.notification_show(f"Could not read the CSV file: {e}", type="error", duration=10)
            uploaded_data.set(None)

    @reactive.effect
    @reactive.event(uploaded_data)
    def initialize_column_selectors():
        df = uploaded_data()
        if df is None:
            ui.update_selectize("input_cols", choices=[])
            ui.update_selectize("outcome_col", choices=[])
            return

        cols = list(df.columns)
        if len(cols) > 0:
            # Default to final column for outcome, everything else for inputs
            default_outcome = cols[-1]
            default_inputs = cols[:-1]

            ui.update_selectize("input_cols", choices=cols, selected=default_inputs)
            ui.update_selectize("outcome_col", choices=cols, selected=default_outcome)

    @reactive.effect
    @reactive.event(input.btn_refine)
    def handle_refinement():
        study = current_study()
        if study is None:
            return

        try:
            n_to_gen = input.n_new_samples()
            refined_df = study.refine(
                n_points=n_to_gen,
                max_gap_ratio=input.ui_max_gap(),
                min_r2_score=input.ui_min_r2(),
                max_avg_cv=input.ui_avg_cv(),
                max_max_cv=input.ui_max_cv()
            )

            new_samples.set(refined_df)
            ui.notification_show(f"Generated {n_to_gen} targeted samples.", type="message")
        except Exception as e:
            ui.notification_show(f"Refinement failed: {e}", type="error")

    @render.ui
    def selection_error_display():
        """Displays a permanent red error if selections conflict."""
        selected_inputs = list(input.input_cols())
        selected_outcome = input.outcome_col()

        if selected_outcome and selected_outcome in selected_inputs:
            return ui.div(
                ui.span(icon_svg("circle-xmark"), f" Error: '{selected_outcome}' cannot be both an input and an outcome."),
                class_="text-danger small fw-bold mt-2"
            )
        return None

    @render.ui
    def validation_status():
        """The result of the checks, with the remediation tools when they fail."""
        if uploaded_data() is None:
            return notice("Upload a CSV of simulation results: one row per run, one column per input "
                          "variable and one for the measured signal.")
        if current_study() is None:
            return notice("Choose the input variables and the outcome in the sidebar.")
        if diagnostic_table() is None:
            return notice("Ready: press Run Diagnostics to check the data is sufficient to model.", kind="info")

        stale = ui.p(
            ui.span(icon_svg("rotate"), class_="me-1"),
            "Thresholds changed since this run: run the diagnostics again to update.",
            class_="small text-warning-emphasis fw-semibold mt-2 mb-0"
        ) if diagnostics_stale() else ""

        if validation_passed():
            return ui.div(
                ui.h5(icon_svg("circle-check"), " All checks passed"),
                ui.p("The data is sufficient to model. Continue to Model Diagnostics.", class_="mb-0"),
                stale,
                class_="alert alert-success"
            )
        return ui.div(
            ui.h5(icon_svg("triangle-exclamation"), " Issues detected"),
            failure_list(diagnostic_failures(diagnostic_table())),
            ui.output_ui("remediation_ui"),
            stale,
            class_="alert alert-danger"
        )

    @reactive.effect
    @reactive.event(input.btn_run_diagnostics)
    def run_validation_diagnostics():
        new_samples.set(None)
        study = current_study()
        if study is None:
            state.invalidate("diagnostics")
            return

        try:
            diag_df = study.diagnose(
                max_gap_ratio=input.ui_max_gap(),
                min_r2_score=input.ui_min_r2(),
                max_avg_cv=input.ui_avg_cv(),
                max_max_cv=input.ui_max_cv(),
                max_allowed_vif=input.ui_max_vif()
            )
            if diag_df is None or diag_df.empty:
                state.invalidate("diagnostics")
                ui.notification_show("Diagnostics could not run: no valid rows in the data.", type="error")
                return

            diagnostic_table.set(diag_df)
            validation_passed.set(bool(diag_df["Pass"].astype(bool).all()))
            diagnostics_stale.set(False)

        except Exception as e:
            state.invalidate("diagnostics")
            logger.exception("Diagnostics failed")
            ui.notification_show(f"Diagnostics failed: {e}", type="error", duration=10)


    # --- OUTPUTS ---

    @render.data_frame
    def preview_uploaded_table():
        df = uploaded_data()
        if df is not None:
            return render.DataGrid(df.round(3).head(5), selection_mode="none", filters=False)
        return None

    @render.data_frame
    def validation_results_table():
        df = diagnostic_table()
        if df is not None:
            return render.DataGrid(df)
        return None


    @render.ui
    def remediation_ui():
        """What to do about the failed checks: targeted new samples, or decoupling inputs."""
        if current_study() is None or diagnostic_table() is None or validation_passed():
            return None

        failed = {test for test, _title, _msg in diagnostic_failures(diagnostic_table())}
        parts = []
        if failed & {"Input Coverage", "Model Fit (CV)", "Bootstrap Convergence"}:
            parts.append(ui.div(
                ui.p(ui.tags.strong("Fix it with more runs: "),
                     "generate new samples targeted at the gaps and unstable regions, run them in your "
                     "solver, and upload the combined results.", class_="small mb-2"),
                ui.layout_columns(
                    ui.input_numeric("n_new_samples", "Samples per issue", value=10, min=1),
                    ui.input_task_button("btn_refine", "Generate New Samples",
                                         icon=icon_svg("wand-magic-sparkles"), class_="btn-warning w-100"),
                    col_widths=[5, 7], gap="8px", class_="align-items-end"
                ),
                ui.output_ui("download_new_samples_ui"),
            ))
        if "Collinearity Check" in failed:
            parts.append(ui.p(
                ui.tags.strong("Correlated inputs: "),
                "more runs won't fix this. Remove or decouple the highly correlated variables in your design.",
                class_="small mb-0 mt-2"
            ))
        return ui.div(*parts, class_="bg-white text-body rounded border p-3 mt-3")

    @render.ui
    def download_new_samples_ui():
        # Only show the button if new_samples has been populated
        if new_samples() is None:
            return None

        return ui.download_button("download_new_samples", f"Download {len(new_samples())} New Samples (CSV)",
                                  class_="btn-success w-100 mt-2", icon=icon_svg("download"))

    @render.download(filename="remediation_samples.csv")
    def download_new_samples():
        df = new_samples()
        if df is not None:
            yield df.to_csv(index=False).encode('utf-8')


    # --- VISUALISATION ---

    @render.ui
    def viz_content():
        """The data views: report, per-variable inspector, coverage and collinearity."""
        if current_study() is None:
            return ui.div()
        selected_cols = state.selected_columns()
        report = ui.div(
            ui.card(ui.card_header("Diagnostic Results"), ui.output_data_frame("validation_results_table"))
            if diagnostic_table() is not None else "",
            ui.card(ui.card_header("Summary Statistics"), ui.output_data_frame("viz_summary_table")),
            ui.card(ui.card_header("Data Preview (first rows)"), ui.output_data_frame("preview_uploaded_table")),
        )
        inspector = ui.layout_columns(
            ui.div(
                ui.input_select("viz_variable", "Variable", choices=selected_cols,
                                selected=selected_cols[0] if selected_cols else None),
                ui.input_radio_buttons("viz_plot_type", "Plot", choices=["Distribution", "vs Outcome"],
                                       selected="Distribution"),
                ui.output_ui("viz_diagnostic_badge"),
            ),
            ui.output_plot("viz_variable_plot", height="380px"),
            col_widths=[3, 9]
        )
        coverage = ui.div(
            ui.p("Distribution of each input. Green title: coverage passed. Red title and orange shading: "
                 "the largest gap, where coverage failed.", class_="small text-muted"),
            ui.output_plot("viz_coverage_overview", height="320px"),
        )
        collinearity = ui.div(
            ui.p("Pearson correlation between inputs. High collinearity (VIF above the threshold) can make "
                 "model fits unstable.", class_="small text-muted text-center"),
            ui.div(ui.output_plot("viz_collinearity_matrix", width="500px", height="400px"),
                   style="display: flex; justify-content: center; width: 100%;"),
        )
        return view_tabs(
            "data_view",
            ("Report", "table-list", report),
            ("Variables", "magnifying-glass", inspector),
            ("Coverage", "chart-column", coverage),
            ("Collinearity", "border-all", collinearity),
        )

    @render.data_frame
    def viz_summary_table():
        df = uploaded_data()
        if df is None:
            return None

        rows = []
        for col in state.selected_columns():
            numeric = pd.to_numeric(df[col], errors="coerce").dropna()
            n_valid = len(numeric)
            rows.append({
                "Variable":   col,
                "N (Total)":  len(df[col]),
                "N (Valid)":  n_valid,
                "Min":        f"{numeric.min():.4g}"    if n_valid else "N/A",
                "Median":     f"{numeric.median():.4g}" if n_valid else "N/A",
                "Max":        f"{numeric.max():.4g}"    if n_valid else "N/A",
                "Mean":       f"{numeric.mean():.4g}"   if n_valid else "N/A",
                "Std Dev":    f"{numeric.std():.4g}"    if n_valid else "N/A",

            })

        return render.DataGrid(pd.DataFrame(rows), width="100%", filters=False)


    @render.ui
    def viz_diagnostic_badge():
        """
        Shows pass/fail status for the currently selected variable,
        taken from the diagnostics table.
        """
        diag = diagnostic_table()
        if diag is None or diag.empty:
            return ui.p(
                "Run the diagnostics to see this variable's status here.",
                class_="text-muted small fst-italic"
            )

        try:
            var = input.viz_variable()
        except Exception:
            return ui.div()

        var_rows = diag[diag["Variable"] == var]
        if var_rows.empty:
            return ui.div()

        if var_rows["Pass"].astype(bool).all():
            return ui.div(
                ui.span(
                    icon_svg("circle-check"), " All Diagnostics Passed",
                    class_="text-success fw-bold small"
                ),
                class_="mt-1"
            )

        failed_tests = set(var_rows.loc[~var_rows["Pass"].astype(bool), "Test"])
        return ui.div(
            ui.span(
                icon_svg("triangle-exclamation"),
                f" Failed: {', '.join(failed_tests)}",
                class_="text-danger fw-bold small"
            ),
            class_="mt-1"
        )


    @render.plot
    def viz_variable_plot():
        import matplotlib.pyplot as plt

        df = uploaded_data()
        if df is None:
            return None

        try:
            var = input.viz_variable()
            plot_type = input.viz_plot_type()
        except Exception:
            return None

        if var not in df.columns:
            return None

        outcome = input.outcome_col()

        # ── Find gap for this variable (from diagnostics if available) ─────────
        gap = None
        diag = diagnostic_table()
        if diag is not None and not diag.empty:
            fail_row = diag[
                (diag["Test"] == "Input Coverage") &
                (diag["Variable"] == var) &
                (~diag["Pass"].astype(bool))
            ]
            if not fail_row.empty:
                gap = largest_gap(df[var])

        fig, ax = plt.subplots(figsize=(8, 4))
        vals = df[var].dropna().values

        # ── Distribution ───────────────────────────────────────────────────────
        if plot_type == "Distribution" or var == outcome:
            ax.hist(
                vals, bins=25, color="#1f77b4", alpha=0.65,
                edgecolor="white", density=True, label="Distribution"
            )

            # KDE overlay
            if len(vals) > 3:
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(vals, bw_method="scott")
                    x_kde = np.linspace(vals.min(), vals.max(), 200)
                    ax.plot(x_kde, kde(x_kde), color="#006abc",
                            linewidth=2, label="KDE")
                except Exception:
                    logger.debug("KDE overlay skipped", exc_info=True)

            # Gap overlay
            if gap is not None:
                gap_start, gap_end = gap
                ax.axvspan(
                    gap_start, gap_end,
                    color="#d13438", alpha=0.15,
                    label=f"Coverage Gap ({gap_start:.3g} – {gap_end:.3g})"
                )

            ax.set_xlabel(var)
            ax.set_ylabel("Density")
            ax.set_title(f"Distribution of '{var}'")

        # ── vs Outcome scatter ─────────────────────────────────────────────────
        else:
            if outcome not in df.columns:
                ax.text(
                    0.5, 0.5, "Outcome column not configured",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#605e5c", fontsize=12
                )
            else:
                x_data = df[var].dropna()
                y_data = df[outcome].loc[x_data.index]

                ax.scatter(x_data, y_data, alpha=0.55, color="#1f77b4",
                        s=25, label="Data")

                # Linear trend
                if len(x_data) > 2:
                    try:
                        p = np.poly1d(np.polyfit(x_data, y_data, 1))
                        x_line = np.linspace(x_data.min(), x_data.max(), 100)
                        ax.plot(x_line, p(x_line), color="#d13438",
                                linewidth=1.5, linestyle="--", label="Linear Trend")
                    except Exception:
                        logger.debug("Trend overlay skipped", exc_info=True)

                # Gap overlay on x-axis
                if gap is not None:
                    gap_start, gap_end = gap
                    ax.axvspan(
                        gap_start, gap_end,
                        color="#d13438", alpha=0.12,
                        label=f"Coverage Gap ({gap_start:.3g} – {gap_end:.3g})"
                    )

                ax.set_xlabel(var)
                ax.set_ylabel(outcome)
                ax.set_title(f"'{var}'  vs  '{outcome}'")

        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


    @render.plot
    def viz_coverage_overview():
        """
        One histogram panel per input variable. Title is green (pass) or red (fail).
        Orange shading marks the largest gap when coverage fails.
        Computed from check_input_coverage directly so it's always up to date,
        even if the user hasn't explicitly run diagnostics.
        """
        import matplotlib.pyplot as plt

        from digiqual.diagnostics import check_input_coverage

        df = uploaded_data()
        if df is None:
            return None

        # Use selected inputs, falling back to all-but-outcome if none chosen yet
        input_cols_list = list(input.input_cols())
        outcome = input.outcome_col()
        if not input_cols_list:
            input_cols_list = [c for c in df.columns if c != outcome]

        # Ensure the UI hasn't fallen behind the dataset
        input_cols_list = [c for c in input_cols_list if c in df.columns]

        if not input_cols_list:
            return None

        # --- Fetch the dynamic gap threshold from the UI ---
        try:
            thresh_gap = input.ui_max_gap()
        except Exception:
            thresh_gap = MAX_GAP_RATIO  # Safe fallback during initialization

        try:
            # --- Pass the custom threshold to the diagnostic helper ---
            coverage_res = check_input_coverage(df, input_cols_list, thresh_gap)
        except Exception:
            # Without the coverage result no gaps can be highlighted; say so rather
            # than silently showing every input as passing.
            logger.exception("Coverage check failed")
            ui.notification_show("Coverage check failed; gaps are not highlighted in this plot.",
                                 type="warning", duration=8)
            coverage_res = {}

        n = len(input_cols_list)
        ncols = min(n, 3)
        nrows = -(-n // ncols)  # ceiling division

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 3 * nrows),
            squeeze=False,
            constrained_layout=True
        )

        for idx, col in enumerate(input_cols_list):
            row, c = divmod(idx, ncols)
            ax = axes[row][c]

            vals = df[col].dropna().values
            res = coverage_res.get(col, {})
            passed = res.get("sufficient_coverage", True)

            bar_color = "#107c10" if passed else "#d13438"
            ax.hist(vals, bins=20, color=bar_color, alpha=0.55, edgecolor="white")

            # Shade the largest gap when coverage fails
            gap = largest_gap(vals) if not passed else None
            if gap is not None:
                ax.axvspan(
                    *gap,
                    color="#ffb900", alpha=0.4,
                    label=f"Gap ratio: {res.get('max_gap_ratio', 0):.2f}"
                )
                ax.legend(fontsize=7, loc="upper right")

            status = "✓" if passed else "✗"
            ax.set_title(
                f"{col}  {status}",
                color="#107c10" if passed else "#d13438",
                fontweight="bold", fontsize=10
            )
            ax.set_ylabel("Count", fontsize=9)
            ax.grid(True, alpha=0.3)

        # Hide unused subplot slots
        for idx in range(n, nrows * ncols):
            row, c = divmod(idx, ncols)
            axes[row][c].set_visible(False)

        return fig

    @render.plot
    def viz_collinearity_matrix():
        study = current_study()
        if study is None or study.clean_data.empty:
            return None

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        study.plot_collinearity(ax=ax)
        return fig
