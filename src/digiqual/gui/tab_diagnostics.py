"""Simulation Diagnostics tab: upload results, run the sufficiency checks and inspect the data."""

import numpy as np
import pandas as pd
from faicons import icon_svg
from shiny import reactive, render, ui

from digiqual import SimulationStudy
from digiqual.defaults import MAX_ALLOWED_VIF, MAX_AVG_CV, MAX_GAP_RATIO, MAX_MAX_CV, MIN_R2_SCORE

from .state import AppState, logger

panel = ui.nav_panel(
    "Simulation Diagnostics",
    ui.div(
        ui.h3("Simulation Diagnostics", class_="mb-4 text-center"),
        ui.layout_columns(
            # --- LEFT: CONFIGURATION ---
            ui.card(
                ui.card_header("Diagnostic Configuration"),
                ui.div(
                    ui.input_file("upload_csv", "Upload CSV file", accept=[".csv"], multiple=False),
                    ui.input_selectize("input_cols", "Select Input Variables", choices=[], multiple=True),
                    ui.input_selectize("outcome_col", "Select Outcome Variable", choices=[], multiple=False),

                    ui.output_ui("selection_error_display"),

                    ui.accordion(
                        ui.accordion_panel(
                            "Advanced Diagnostic Thresholds",
                            ui.input_numeric("ui_max_gap", "Max Gap Ratio", value=MAX_GAP_RATIO, step=0.01),
                            ui.input_numeric("ui_min_r2", "Min R² Score", value=MIN_R2_SCORE, step=0.01),
                            ui.input_numeric("ui_avg_cv", "Max Allowed Avg CV", value=MAX_AVG_CV, step=0.01),
                            ui.input_numeric("ui_max_cv", "Max Allowed Peak CV", value=MAX_MAX_CV, step=0.01),
                            ui.input_numeric("ui_max_vif", "Max Allowed VIF", value=MAX_ALLOWED_VIF, step=0.1),
                        ),
                        open=False
                    ),

                    ui.input_task_button(
                        "btn_run_diagnostics", "Run Diagnostics",
                        class_="btn-primary w-100", icon=icon_svg("stethoscope")
                    ),
                    ui.output_ui("validation_status"),



                    class_="config-container"
                ),
                class_="h-100 mb-0"
            ),

            # --- RIGHT: PREVIEWS, REPORTS & REMEDIATION ---
            ui.div(
                ui.output_ui("dynamic_preview_card"),
                ui.card(
                    ui.card_header("Validation Report"),
                    ui.output_data_frame("validation_results_table"),
                    full_screen=True,
                    class_="mb-0"
                ),
                ui.output_ui("remediation_ui"),
                class_="d-flex flex-column gap-3 h-100"
            ),
            col_widths=[-1, 3, 7, -1]
        ),

        # --- BOTTOM: VISUALISATION ---
        # Wrapped in layout_columns to ensure perfectly matched left/right padding
        ui.layout_columns(
            ui.div(
                ui.output_ui("viz_content"),
                class_="mt-3"
            ),
            col_widths=[-1, 10, -1]
        ),
        class_="container-fluid py-3 overflow-auto h-100"
    ),
    icon=icon_svg("check-double")
)


def register(input, output, session, state: AppState):
    uploaded_data = state.uploaded_data
    validation_passed = state.validation_passed
    new_samples = state.new_samples
    diagnostic_table = state.diagnostic_table
    study_instance = state.study_instance
    current_study = state.current_study

    @reactive.effect
    def read_uploaded_csv():
        file_info = input.upload_csv()
        if file_info is None:
            uploaded_data.set(None)
            return
        try:
            df = pd.read_csv(file_info[0]["datapath"])

            # Header string check & cleaning
            # 1. Strip leading/trailing spaces
            df.columns = df.columns.str.strip()
            # 2. Replace any internal spaces with underscores
            df.columns = df.columns.str.replace(r'\s+', '_', regex=True)
            # 3. Remove any remaining special characters
            df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)

            # 4. FIX: Detect and disambiguate duplicate column names
            new_cols = []
            seen = set()
            for col in df.columns:
                new_col = col
                counter = 1
                # If we've already seen this name, add a number until it's unique
                while new_col in seen:
                    new_col = f"{col}_{counter}"
                    counter += 1

                seen.add(new_col)
                new_cols.append(new_col)

            df.columns = new_cols

            # 5. Keep only numeric columns. Text columns can't be used as inputs or
            # outcomes; stray text cells in a numeric column become blanks, which
            # validation later drops as invalid rows.
            numeric = df.apply(pd.to_numeric, errors="coerce")
            text_cols = [c for c in df.columns if numeric[c].isna().all() and df[c].notna().any()]
            if text_cols:
                ui.notification_show(
                    f"Ignored non-numeric column(s): {', '.join(text_cols)}.", type="warning", duration=8
                )
            n_bad_cells = int((numeric.isna() & df.notna()).drop(columns=text_cols).sum().sum())
            if n_bad_cells:
                ui.notification_show(
                    f"{n_bad_cells} non-numeric value(s) were found and will be treated as missing.",
                    type="warning", duration=8
                )
            numeric = numeric.drop(columns=text_cols)
            if numeric.shape[1] < 2:
                raise ValueError("the file needs at least two numeric columns (inputs and an outcome)")

            uploaded_data.set(numeric)
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
        """Shows result status or prompt to configure."""
        selected_inputs = list(input.input_cols())
        selected_outcome = input.outcome_col()
        conflict = selected_outcome in selected_inputs

        # 1. Missing selections or conflict
        if uploaded_data() is None or not selected_inputs or conflict:
            return ui.div(ui.p("Configure selections to run diagnostics.", class_="text-muted fst-italic"))

        # 2. Ready, but the button hasn't been clicked yet!
        if diagnostic_table() is None:
            return ui.div(ui.p("Ready to run diagnostics. Click the button above.", class_="text-primary fw-bold mt-3"))

        # 3. Button was clicked, show actual results
        if validation_passed():
            return ui.div(
                ui.h5(icon_svg("circle-check"), " Validation Passed"),
                class_="alert alert-success mt-3"
            )
        else:
            diag = diagnostic_table()
            warnings = []
            if diag is not None and not diag.empty:
                # 1. Input Coverage
                coverage_rows = diag[diag["Test"] == "Input Coverage"]
                if not coverage_rows.empty and not coverage_rows["Pass"].all():
                    failed_vars = coverage_rows.loc[~coverage_rows["Pass"].astype(bool), "Variable"].tolist()
                    warnings.append(
                        ui.p(
                            ui.span(icon_svg("triangle-exclamation"), class_="text-danger me-1"),
                            ui.tags.strong("Input Coverage Gaps: "),
                            f"Large gaps in parameter coverage detected for: {', '.join(failed_vars)}.",
                            class_="small mb-1 text-danger"
                        )
                    )

                # 2. Model Fit (CV)
                fit_rows = diag[diag["Test"] == "Model Fit (CV)"]
                if not fit_rows.empty and not fit_rows["Pass"].all():
                    warnings.append(
                        ui.p(
                            ui.span(icon_svg("triangle-exclamation"), class_="text-danger me-1"),
                            ui.tags.strong("Weak Model Fit: "),
                            "The surrogate model R² score is below the required threshold. The relationship might be too noisy or non-linear for the current sample size.",
                            class_="small mb-1 text-danger"
                        )
                    )

                # 3. Bootstrap Convergence
                boot_rows = diag[diag["Test"] == "Bootstrap Convergence"]
                if not boot_rows.empty and not boot_rows["Pass"].all():
                    warnings.append(
                        ui.p(
                            ui.span(icon_svg("triangle-exclamation"), class_="text-danger me-1"),
                            ui.tags.strong("Bootstrap Instability: "),
                            "Model predictions are highly sensitive to sample variance (insufficient convergence). Confidence intervals may be too wide.",
                            class_="small mb-1 text-danger"
                        )
                    )

                # 4. Collinearity Check
                col_rows = diag[diag["Test"] == "Collinearity Check"]
                if not col_rows.empty and not col_rows["Pass"].all():
                    failed_vars = col_rows.loc[~col_rows["Pass"].astype(bool), "Variable"].tolist()
                    warnings.append(
                        ui.p(
                            ui.span(icon_svg("triangle-exclamation"), class_="text-danger me-1"),
                            ui.tags.strong("High Multicollinearity: "),
                            f"Strong correlation detected among inputs for: {', '.join(failed_vars)}. This can make the surrogate model fit unstable and predictions unreliable.",
                            class_="small mb-1 text-danger"
                        )
                    )

            warning_content = ui.div(
                *warnings,
                ui.p("See the visualisations below and the Remediation options for next steps.", class_="small mb-0 text-muted mt-2")
            ) if warnings else ui.p("See the visualisations below and the Remediation options for next steps.", class_="small mb-0")

            return ui.div(
                ui.h5(icon_svg("triangle-exclamation"), " Issues Detected"),
                warning_content,
                class_="alert alert-danger mt-3"
            )

    @reactive.effect
    @reactive.event(input.btn_run_diagnostics)
    def run_validation_diagnostics():
        df = uploaded_data()
        new_samples.set(None)

        selected_inputs = list(input.input_cols())
        selected_outcome = input.outcome_col()

        # Guard: Stop if data is missing, selections are empty, OR there is a conflict
        if df is None or not selected_inputs or not selected_outcome or (selected_outcome in selected_inputs):
            validation_passed.set(False)
            diagnostic_table.set(None)
            return

        try:
            study = SimulationStudy()
            study.add_data(uploaded_data(), outcome_col=selected_outcome, input_cols=selected_inputs, overwrite=True)
            study_instance.set(study)

            # Pass the UI threshold values dynamically at runtime!
            diag_df = study.diagnose(
                max_gap_ratio=input.ui_max_gap(),
                min_r2_score=input.ui_min_r2(),
                max_avg_cv=input.ui_avg_cv(),
                max_max_cv=input.ui_max_cv(),
                max_allowed_vif=input.ui_max_vif()
            )

            if diag_df is None:
                diagnostic_table.set(None)
                return

            diagnostic_table.set(diag_df)
            all_passed = diag_df["Pass"].astype(bool).all()
            validation_passed.set(all_passed)

        except Exception as e:
            validation_passed.set(False)
            diagnostic_table.set(None)
            logger.exception("Diagnostics failed")
            ui.notification_show(f"Diagnostics failed: {e}", type="error", duration=10)


    # --- OUTPUTS ---

    @render.ui
    def dynamic_preview_card():
        df = uploaded_data()
        if df is None:
            return None

        return ui.card(
            ui.card_header("Uploaded Data Preview"),
            ui.output_data_frame("preview_uploaded_table")
        )

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
        """
        Only appears if diagnostics have been run AND they detected issues.
        """
        # 1. Hide if no data or if diagnostics haven't run yet
        if uploaded_data() is None or diagnostic_table() is None:
            return None

        # 2. Hide if there is currently a selection conflict
        conflict = input.outcome_col() in list(input.input_cols())
        if conflict:
            return None

        # 3. Hide if validation actually passed
        if validation_passed():
            return None

        diag = diagnostic_table()
        has_coverage_issues = not diag[diag["Test"] == "Input Coverage"]["Pass"].all()
        has_fit_issues = not diag[diag["Test"].isin(["Model Fit (CV)", "Bootstrap Convergence"])]["Pass"].all()
        has_collinearity_issues = not diag[diag["Test"] == "Collinearity Check"]["Pass"].all()

        remediation_texts = []
        if has_coverage_issues:
            remediation_texts.append("Your data has coverage issues (gaps in parameter space).")
        if has_fit_issues:
            remediation_texts.append("Your data has model fit or convergence issues (noisy or unstable predictions).")

        # If there are coverage or fit issues, show the Refine tool
        if has_coverage_issues or has_fit_issues:
            refine_section = ui.div(
                ui.p("Use the Refine tool below to generate targeted new samples to fix these issues:"),
                ui.layout_columns(
                    ui.input_numeric("n_new_samples", "Count", value=10, min=1),
                    ui.input_task_button("btn_refine", "Generate New Samples", icon=icon_svg("wand-magic-sparkles"), class_="btn-warning"),
                ),
                ui.output_ui("download_new_samples_ui")
            )
        else:
            refine_section = ui.div()

        collinearity_section = ui.div()
        if has_collinearity_issues:
            collinearity_section = ui.div(
                ui.p(
                    ui.span(icon_svg("triangle-exclamation"), class_="text-danger me-1"),
                    ui.tags.strong("Collinearity Detected: "),
                    "High multicollinearity (VIF > threshold) means your inputs are highly correlated. "
                    "Surrogate model fitting can be unstable or misleading. "
                    "Consider removing or decoupling highly correlated variables in your experimental design.",
                    class_="small text-danger bg-light rounded p-2 border border-danger-subtle mt-2"
                )
            )

        return ui.card(
            ui.card_header("Remediation & Suggestions"),
            ui.HTML("<br>".join(remediation_texts)) if remediation_texts else ui.div(),
            refine_section,
            collinearity_section,
            class_="border-warning shadow-sm"
        )


    @render.ui
    def download_new_samples_ui():
        # Only show the button if new_samples has been populated
        if new_samples() is None:
            return None

        return ui.div(
            ui.hr(),
            ui.p("Success! Download your targeted samples below:", class_="small"),
            ui.download_button(
                "download_new_samples",
                "Download Refined CSV",
                class_="btn-success w-100",
                icon=icon_svg("download")
            )
        )

    @render.download(filename="remediation_samples.csv")
    def download_new_samples():
        df = new_samples()
        if df is not None:
            yield df.to_csv(index=False).encode('utf-8')


    # --- VISUALISATION ---

    @render.ui
    def viz_content():
        """
        Master render for the entire viz tab.
        Hides completely until diagnostics have been run.
        """
        diag = diagnostic_table()
        if diag is None or diag.empty:
            # Return an empty div so the UI stays clean until 'Run Diagnostics' is pressed
            return ui.div()

        df = uploaded_data()

        # Gather only the selected input and outcome columns
        input_cols = list(input.input_cols())
        outcome = input.outcome_col()

        selected_cols = input_cols.copy()
        if outcome and outcome not in selected_cols:
            selected_cols.append(outcome)

        if not selected_cols:
             selected_cols = list(df.columns)

        return ui.div(
            # ── Row 1: Summary Statistics ──────────────────────────────────────
            ui.card(
                ui.card_header("Summary Statistics"),
                ui.output_data_frame("viz_summary_table"),
                full_screen=True,
                class_="mb-3"
            ),

            # ── Row 2: Variable Inspector ──────────────────────────────────────
            ui.layout_columns(
                # Left: Controls
                ui.card(
                    ui.card_header("Inspector Controls"),
                    ui.div(
                        ui.input_select(
                            "viz_variable", "Select Variable",
                            choices=selected_cols, # Use the filtered list here!
                            selected=selected_cols[0] if selected_cols else None,
                        ),
                        ui.input_select(
                            "viz_plot_type", "Plot Type",
                            choices=["Distribution", "vs Outcome"],
                            selected="Distribution",
                        ),
                        ui.hr(),
                        ui.output_ui("viz_diagnostic_badge"),
                        class_="p-2"
                    )
                ),
                # Right: Plot
                ui.card(
                    ui.card_header("Variable Plot"),
                    ui.output_plot("viz_variable_plot", height="360px"),
                    full_screen=True,
                ),
                col_widths=[3, 9],
                class_="mb-3"
            ),

            # ── Row 3: Coverage Overview (all inputs, one panel each) ──────────
            ui.card(
                ui.card_header("Input Space Coverage Overview"),
                ui.p(
                    "Distribution of each input variable. "
                    "Green title = coverage passed. "
                    "Red title + orange shading = gap detected.",
                    class_="small text-muted px-3 pt-2 mb-0"
                ),
                ui.output_plot("viz_coverage_overview", height="300px"),
                full_screen=True,
            ),

            # ── Row 4: Collinearity Matrix ─────────────────────────────────────
            ui.card(
                ui.card_header("Collinearity Analysis"),
                ui.p(
                    "Pearson correlation coefficients between input variables. "
                    "High collinearity (VIF > Max Allowed VIF) can make model fits unstable.",
                    class_="small text-muted px-3 pt-2 mb-0 text-center"
                ),
                ui.div(
                    ui.output_plot("viz_collinearity_matrix", width="500px", height="400px"),
                    style="display: flex; justify-content: center; width: 100%;"
                ),
                full_screen=True,
            ),
        )


    @render.data_frame
    def viz_summary_table():
        df = uploaded_data()
        if df is None:
            return None

        # Gather only the selected input and outcome columns
        input_cols = list(input.input_cols())
        outcome = input.outcome_col()

        selected_cols = input_cols.copy()
        if outcome and outcome not in selected_cols:
            selected_cols.append(outcome)

        # Fallback just in case
        if not selected_cols:
            selected_cols = list(df.columns)

        rows = []
        # Iterate over selected_cols instead of df.columns
        for col in selected_cols:
            if col not in df.columns:
                continue

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
        sourced from the diagnostic_table reactive already computed in the Simulation Diagnostics tab.
        """
        diag = diagnostic_table()
        if diag is None or diag.empty:
            return ui.p(
                "Run diagnostics in the Simulation Diagnostics tab to see variable status here.",
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
        gap_start = gap_end = None
        coverage_failed = False
        diag = diagnostic_table()
        if diag is not None and not diag.empty:
            fail_row = diag[
                (diag["Test"] == "Input Coverage") &
                (diag["Variable"] == var) &
                (~diag["Pass"].astype(bool))
            ]
            if not fail_row.empty:
                coverage_failed = True
                sorted_vals = np.sort(df[var].dropna().values)
                if len(sorted_vals) > 1:
                    diffs = np.diff(sorted_vals)
                    idx = np.argmax(diffs)
                    gap_start = sorted_vals[idx]
                    gap_end = sorted_vals[idx + 1]

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
            if coverage_failed and gap_start is not None:
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
                if coverage_failed and gap_start is not None:
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
            if not passed and len(vals) > 1:
                sorted_vals = np.sort(vals)
                diffs = np.diff(sorted_vals)
                gap_idx = np.argmax(diffs)
                ax.axvspan(
                    sorted_vals[gap_idx], sorted_vals[gap_idx + 1],
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
