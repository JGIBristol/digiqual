import matplotlib
matplotlib.use('Agg')

from shiny import App, ui, render, reactive
from faicons import icon_svg
import pandas as pd
import numpy as np
from digiqual.sampling import generate_lhs
from digiqual import SimulationStudy
import shinyswatch

#### App CSS ####
app_css = """
/* --- 1. CORE PALETTE (Modern Engineering) --- */
:root {
    --bs-primary: #0f3460;
    --bs-primary-rgb: 15, 52, 96;
    --bs-secondary: #536473;
    --bs-secondary-rgb: 83, 100, 115;
    --bs-success: #10b981;
    --bs-success-rgb: 16, 185, 129;
    --bs-warning: #f59e0b;
    --bs-warning-rgb: 245, 158, 11;
    --bs-danger: #e11d48;
    --bs-danger-rgb: 225, 29, 72;
    --bs-body-bg: #f3f4f6;
    --bs-body-color: #1f2937;
}

/* --- 2. GLOBAL TYPOGRAPHY --- */
h1, h2, h3, h4, h5, h6 {
    color: var(--bs-primary);
    font-weight: 700;
    letter-spacing: -0.01em;
}

.navbar-brand {
    font-weight: 800 !important;
    letter-spacing: 0.05em;
}

/* --- 3. COMPONENT POLISH --- */
.card {
    border: 1px solid rgba(0,0,0,0.08);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    border-radius: 8px;
    margin-bottom: 1rem;
    background-color: #ffffff;
}

.card-header {
    background-color: transparent;
    border-bottom: 1px solid #e5e7eb;
    font-weight: 600;
    color: var(--bs-primary);
    padding-top: 1rem;
    padding-bottom: 1rem;
}

.btn-primary {
    background-color: var(--bs-primary);
    border-color: var(--bs-primary);
    color: #ffffff;
}
.btn-primary:hover {
    background-color: #162a45;
}

.btn-success, .btn-warning, .btn-danger {
    color: #ffffff;
}

/* --- 4. ALERT FIXES --- */
.alert h1, .alert h2, .alert h3, .alert h4, .alert h5, .alert h6 {
    color: inherit;
}

/* --- 5. LAYOUT --- */
.sidebar {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}

@media (min-width: 1400px) {
    .sidebar.sidebar-navigation {
        margin-left: -20px;
    }
}
"""

#### UI Definition ####
app_ui = ui.page_navbar(

#### UI - Homepage ####
    ui.nav_panel(
        "Home",
        ui.div(
            ui.h2("DigiQual", class_="fw-bold text-primary mb-1 text-center"),
            ui.p("Statistical Toolkit for Reliability Assessment in NDT",
                class_="lead text-muted mb-0 text-center"),
            ui.hr(class_="my-4"),
            class_="mb-4 mt-3"
        ),

        ui.layout_columns(
            ui.div(
                ui.h4("Workflow Modules", class_="mb-3 text-primary border-bottom pb-2"),

                # Module 1: Design
                ui.card(
                    ui.card_header(
                        ui.span(icon_svg("table"), " 1. Experimental Design", class_="fw-bold")
                    ),
                    ui.p("Design efficient experimental frameworks using Latin Hypercube Sampling (LHS).", class_="small"),
                    ui.tags.ul(
                        ui.tags.li("Space-filling parameter generation.", class_="x-small"),
                        ui.tags.li("Automatic scaling to variable bounds.", class_="x-small"),
                        class_="text-muted mb-0"
                    ),
                    class_="mb-3 border-start border-4 border-primary shadow-sm"
                ),

                # Module 2: Diagnostics
                ui.card(
                    ui.card_header(
                        ui.span(icon_svg("check-double"), " 2. Simulation Diagnostics", class_="fw-bold")
                    ),
                    ui.p("Validate dataset integrity and simulation outputs.", class_="small"),
                    ui.tags.ul(
                        ui.tags.li("Detect input coverage gaps and sanity checks.", class_="x-small"),
                        ui.tags.li("Identify model instability or insufficient samples.", class_="x-small"),
                        class_="text-muted mb-0"
                    ),
                    class_="mb-3 border-start border-4 border-warning shadow-sm"
                ),

                # Module 3: Analysis
                ui.card(
                    ui.card_header(
                        ui.span(icon_svg("chart-line"), " 3. PoD Analysis", class_="fw-bold")
                    ),
                    ui.p("Construct Generalized Probability of Detection (PoD) curves.", class_="small"),
                    ui.tags.ul(
                        ui.tags.li("Robust statistics using the Generalized â-versus-a Method.", class_="x-small"),
                        ui.tags.li("Uncertainty quantification with bootstrap resampling.", class_="x-small"),
                        class_="text-muted mb-0"
                    ),
                    class_="mb-3 border-start border-4 border-success shadow-sm"
                ),
            ),

            ui.div(
                ui.h4("Project Information", class_="mb-3 text-primary border-bottom pb-2"),
                ui.card(
                    ui.div(
                        # About Section
                        ui.div(
                            ui.tags.strong("Version: "), "0.10.4", ui.br(),
                            ui.tags.strong("License: "), "MIT Open Source", ui.br(),
                            ui.tags.strong("Author: "), "Josh Tyler", ui.br(),
                            ui.tags.strong("Institution: "), "University of Bristol",
                            class_="mb-3"
                        ),
                        ui.hr(),
                        # Methodology
                        ui.h6("Methodology Reference:", class_="fw-bold small"),
                        ui.p(
                            "Malkiel, N., Croxford, A. J., & Wilcox, P. D. (2025). ",
                            ui.span("A generalized method for the reliability assessment of safety–critical inspection. ", class_="fst-italic"),
                            "Proceedings of the Royal Society A.",
                            class_="x-small text-muted mb-2"
                        ),
                        ui.a(
                            "View Paper",
                            href="https://doi.org/10.1098/rspa.2024.0654",
                            target="_blank",
                            class_="btn btn-sm btn-outline-secondary w-100 mb-3"
                        ),
                        ui.hr(),
                        # Disclaimer
                        ui.h6("Disclaimer:", class_="fw-bold small"),
                        ui.p(
                            "This software is provided 'as is', without warranty of any kind. ",
                            "In no event shall the authors be liable for any claim or damages.",
                            class_="x-small text-muted fst-italic mb-3"
                        ),
                        ui.hr(),
                        # Support
                        ui.p("Development supported by:", class_="x-small fw-bold text-center text-muted mb-2"),
                        ui.div(
                            ui.span("[ Funder Logo Placeholder ]", class_="text-muted small"),
                            class_="bg-light border rounded p-2 text-center"
                        ),
                        class_="p-3"
                    ),
                    class_="shadow-sm"
                )
            ),
            col_widths=[7, 5]
        ),
        icon=icon_svg("house")
    ),

#### UI - Experimental Design (Tab 2) ####
    ui.nav_panel(
        "Experimental Design",
        ui.layout_columns(
            # --- LEFT: VARIABLE INPUTS
            ui.card(
                ui.card_header("Experimental Design Variables"),
                ui.div(
                    ui.layout_columns(
                        ui.tags.label("Variable Name", class_="fw-bold"),
                        ui.tags.label("Min Value", class_="fw-bold"),
                        ui.tags.label("Max Value", class_="fw-bold"),
                        ui.div(),
                        col_widths=(4, 3, 3, 2)
                    ),
                    class_="mb-2 px-1"
                ),
                ui.div(
                    ui.div(id="variable_rows_container"),
                    ui.div(
                        ui.input_action_button(
                            "add_variable_btn", "Add Variable",
                            icon=icon_svg("plus"), class_="btn-outline-secondary btn-sm"
                        ),
                        class_="mt-3 d-flex justify-content-start"
                    ),
                ),
                height="100%"
            ),

            # --- RIGHT: PREVIEW & SETTINGS
            ui.div(
                ui.card(
                    ui.card_header("Framework Preview"),
                    ui.output_data_frame("preview_experimental_design"),
                    full_screen=True,
                ),
                ui.card(
                    ui.card_header("Generation Settings"),
                    ui.div(
                        ui.input_numeric("num_rows", "Number of samples", value=100, min=1, width="180px"),
                        class_="d-flex justify-content-center mb-3"
                    ),
                    ui.input_task_button(
                        "generate_btn", "Generate Framework",
                        class_="btn-primary w-100", icon=icon_svg("gears")
                    ),
                    ui.output_ui("download_btn_container", class_="mt-3")
                ),
                class_="d-flex flex-column gap-3"
            ),
            col_widths=(6, 6)
        ),
        icon=icon_svg("table")
    ),


#### UI - Simulation Diagnostics (Tab 3) ####
    ui.nav_panel(
        "Simulation Diagnostics",
        ui.layout_columns(
            # --- LEFT: CONFIGURATION ---
            ui.card(
                ui.card_header("Diagnostic Configuration"),
                ui.input_file("upload_csv", "Upload CSV file", accept=[".csv"], multiple=False),
                ui.hr(),
                ui.input_selectize("input_cols", "Select Input Variables", choices=[], multiple=True),
                ui.input_selectize("outcome_col", "Select Outcome Variable", choices=[], multiple=False),

                # Permanent Error Display for selection conflicts
                ui.output_ui("selection_error_display"),

                ui.hr(),
                # This is the primary status alert (Success/Failure)
                ui.output_ui("validation_status"),
                height="100%"
            ),

            # --- RIGHT: PREVIEWS & REPORTS ---
            ui.div(
                # Dynamic Preview (Hides when no data is present)
                ui.output_ui("dynamic_preview_card"),

                ui.card(
                    ui.card_header("Validation Report"),
                    ui.output_data_frame("validation_results_table"),
                    full_screen=True
                ),
                # Remediation logic is now anchored to the results
                ui.output_ui("remediation_ui"),
                class_="d-flex flex-column gap-3"
            ),
            col_widths=(6, 6) # Symmetrical layout
        ),
        icon=icon_svg("check-double")
    ),

#### UI - PoD Analysis (Tab 4) ####
    ui.nav_panel(
        "PoD Analysis",
        ui.card(
            ui.card_header("Analysis Results"),
            ui.output_ui("analysis_output")
        ),
        icon=icon_svg("chart-line")
    ),
    title="DigiQual",
    id="navbar",
    fillable=True,
    theme=shinyswatch.theme.flatly(),
    header=ui.tags.style(app_css)  # <-- CHANGED: pass the string here
)

#### Server Definition ####
def server(input, output, session):

#### Server - LHS generation (Tab 2) ####

    active_row_ids = reactive.Value([0])
    next_id = reactive.Value(1)
    final_generated_df = reactive.Value(None)

    def _add_row(idx):
        """Helper function to insert UI and its specific removal effect"""
        ui.insert_ui(
            selector="#variable_rows_container",
            where="beforeEnd",
            ui=ui.div(
                ui.layout_columns(
                    ui.input_text(f"var_name_{idx}", label=None, placeholder="Name"),
                    ui.input_numeric(f"var_min_{idx}", label=None, value=0),
                    ui.input_numeric(f"var_max_{idx}", label=None, value=10),
                    ui.input_action_button(
                        f"remove_{idx}", "Remove", icon=icon_svg("trash"),
                        class_="btn-outline-danger btn-sm"
                    ),
                    col_widths=(4, 3, 3, 2)
                ),
                id=f"row_container_{idx}",
                class_="mb-2"
            )
        )

        @reactive.effect
        @reactive.event(input[f"remove_{idx}"])
        def _():
            # 1. Clear UI
            ui.remove_ui(selector=f"#row_container_{idx}")
            # 2. Update tracking list
            current_ids = active_row_ids.get().copy()
            if idx in current_ids:
                current_ids.remove(idx)
                active_row_ids.set(current_ids)
            # 3. Clear existing generation
            final_generated_df.set(None)

    @reactive.effect
    @reactive.event(input.add_variable_btn)
    def add_variable_handler():
        new_id = next_id.get()
        current_ids = active_row_ids.get().copy()
        current_ids.append(new_id)
        active_row_ids.set(current_ids)
        next_id.set(new_id + 1)
        _add_row(new_id)

    @reactive.effect
    def init_rows():
        if next_id.get() == 1:
            _add_row(0)

    @reactive.effect
    @reactive.event(input.generate_btn)
    def generate_handler():
        final_generated_df.set(None)
        ranges = {}
        errors = []

        # LOOP OVER ACTIVE IDs ONLY
        for i in active_row_ids.get():
            name_val = input[f"var_name_{i}"]()
            min_val = input[f"var_min_{i}"]()
            max_val = input[f"var_max_{i}"]()

            if not name_val or str(name_val).strip() == "":
                errors.append("An active row is missing a variable name.")
                continue
            if min_val is None or max_val is None:
                errors.append(f"Variable '{name_val}' is missing min/max values.")
                continue
            if min_val >= max_val:
                errors.append(f"Variable '{name_val}': Min must be less than Max.")
                continue
            if name_val in ranges:
                errors.append(f"Duplicate variable name: '{name_val}'.")
                continue
            ranges[name_val] = [min_val, max_val]

        if not ranges and not errors:
            errors.append("Please define at least one variable.")

        if errors:
            ui.modal_show(ui.modal(
                ui.HTML("<ul><li>" + "</li><li>".join(errors) + "</li></ul>"),
                title="Validation Errors", easy_close=True
            ))
            return

        try:
            df = generate_lhs(n=input.num_rows(), ranges=ranges)
            final_generated_df.set(df)
            ui.notification_show("Success! Framework generated.", type="message")
        except Exception as e:
            ui.notification_show(f"Generation Error: {str(e)}", type="error")

    @render.data_frame
    def preview_experimental_design():
        df = final_generated_df()
        if df is not None:
            return render.DataGrid(df, selection_mode="none", filters=False, height="250px")
        return None

    @render.ui
    def download_btn_container():
        if final_generated_df() is None:
            return ui.div()
        return ui.download_button("download_lhs", "Download CSV", class_="btn-success w-100", icon=icon_svg("download")) # <--- CHANGED

    @render.download(filename="generated_sample.csv")
    def download_lhs():
        df = final_generated_df()
        if df is not None:
            yield df.to_csv(index=False)

    @reactive.effect
    @reactive.event(input.num_rows)
    def on_param_change():
        final_generated_df.set(None)

#### Server - Diagnostics & Validation (Tab 3) ####

    uploaded_data = reactive.Value(None)
    validation_passed = reactive.Value(False)
    new_samples = reactive.Value(None)
    diagnostic_table = reactive.Value(None)

    @reactive.effect
    def _():
        file_info = input.upload_csv()
        if file_info is None:
            uploaded_data.set(None)
            return
        try:
            df = pd.read_csv(file_info[0]["datapath"])
            uploaded_data.set(df)
        except Exception:
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

        if uploaded_data() is None or not selected_inputs or conflict:
            return ui.div(ui.p("Configure selections to run diagnostics.", class_="text-muted fst-italic"))

        if validation_passed():
            return ui.div(
                ui.h5(icon_svg("circle-check"), " Validation Passed"),
                class_="alert alert-success mt-3"
            )
        else:
            return ui.div(
                ui.h5(icon_svg("triangle-exclamation"), " Issues Detected"),
                ui.p("See Remediation options for next steps.", class_="small mb-0"),
                class_="alert alert-danger mt-3"
            )

    @reactive.effect
    @reactive.event(uploaded_data, input.input_cols, input.outcome_col)
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
            study = SimulationStudy(input_cols=selected_inputs, outcome_col=selected_outcome)
            study.add_data(df)
            diag_df = study.diagnose()

            if diag_df is None:
                diagnostic_table.set(None)
                return

            diagnostic_table.set(diag_df)
            all_passed = diag_df["Pass"].astype(bool).all()
            validation_passed.set(all_passed)

        except Exception as e:
            validation_passed.set(False)
            diagnostic_table.set(None)
            print(f"Diagnostic Error: {e}")

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

        # 4. Show the Remediation card
        return ui.card(
            ui.card_header("Remediation: Generate New Samples"),
            ui.p("Your data has coverage issues. Use the Refine tool to generate new samples specifically in the empty spaces."),

            ui.layout_columns(
                ui.input_numeric("n_new_samples", "Count", value=10, min=1),
                ui.input_task_button("btn_refine", "Generate New Samples", icon=icon_svg("wand-magic-sparkles"), class_="btn-warning"),
            ),
            ui.output_ui("download_new_samples_ui"),
            class_="border-warning shadow-sm"
        )



#### Server - PoD Generation (Tab 4) ####

    # --- SHARED CALCULATOR ---
    @reactive.calc
    def current_study():
        """
        Creates the study object for analysis.
        Allows access even if validation failed, provided data exists.
        """
        if uploaded_data() is None:
            return None

        # Initialize study with current UI selections
        study = SimulationStudy(
            input_cols=list(input.input_cols()),
            outcome_col=input.outcome_col()
        )
        study.add_data(uploaded_data())
        return study

    # --- REACTIVE VALUES ---
    # Store scalar results for the table (e.g., a90/95, bandwidth)
    pod_metrics = reactive.value(None)
    # Store the full curve data for downloading
    pod_export_data = reactive.value(None)
    # Trigger to refresh plots when analysis runs
    plot_trigger = reactive.value(0)


    # --- MAIN UI ---
    @render.ui
    def analysis_output():
        """
        Renders the Analysis UI with conditional warnings.
        """
        # 1. Hard Lock: No Data
        if uploaded_data() is None:
            return ui.div(
                ui.div(
                    ui.h4("No Data Available", class_="text-muted"),
                    ui.p("Please upload data in the 'Simulation Diagnostics' tab to begin."),
                    class_="text-center p-5 bg-light border rounded"
                )
            )

        # 2. Build UI Elements
        ui_elements = []

        # -- Conditional Warning Banner --
        if not validation_passed():
            ui_elements.append(
                ui.div(
                    ui.h5(icon_svg("triangle-exclamation"), " Caution: Validation Issues Detected"),
                    ui.p("The diagnostic tests found potential issues (e.g., coverage gaps). Results may be unreliable."),
                    class_="alert alert-warning"
                )
            )

        # -- Input Controls --
        inputs = list(input.input_cols())
        ui_elements.append(
            ui.card(
                ui.card_header("Probability of Detection (PoD) Settings"),
                ui.layout_columns(
                    ui.input_select("pod_param", "Parameter of Interest", choices=inputs),
                    ui.input_numeric("pod_threshold", "Threshold Value", value=5.0),
                    ui.div(
                        ui.input_task_button("btn_run_pod", "Run Analysis", class_="btn-primary w-100", icon=icon_svg("chart-line")),
                        class_="pt-4"
                    )
                ),
            )
        )

        # -- Results Container --
        ui_elements.append(ui.br())
        ui_elements.append(ui.output_ui("pod_results_container"))

        return ui.div(*ui_elements)


    # --- ANALYSIS LOGIC ---
    @reactive.effect
    @reactive.event(input.btn_run_pod)
    def compute_pod_analysis():
        """
        Runs .pod(), calculates a90/95, and triggers plotting.
        """
        study = current_study()
        if study is None:
            return

        try:
            # 1. Run the Analysis (Generates models and curves)
            results = study.pod(
                poi_col=input.pod_param(),
                threshold=input.pod_threshold()
            )

            # 2. Calculate a90/95 (Interpolate)
            val = results["a90_95"]
            a9095_str = f"{val:.3f}" if not np.isnan(val) else "Not Reached"



            # 3. Create Metrics Dictionary for the UI
            metrics = {
                "Parameter of Interest": results["poi_col"],
                "Threshold": results["threshold"],
                "a90/95": a9095_str,
                "Model Degree": results["mean_model"].best_degree_,
                "Smoothing Bandwidth": f"{results['bandwidth']:.4f}",
            }
            pod_metrics.set(metrics)

            # 4. Prepare Data for Download
            export_df = pd.DataFrame({
                "x_defect_size": results["X_eval"],
                "pod_mean": results["curves"]["pod"],
                "ci_lower": results["curves"]["ci_lower"],
                "ci_upper": results["curves"]["ci_upper"]
            })
            pod_export_data.set(export_df)

            # 5. Generate Plots (Visualise draws them internally)
            study.visualise(show=False)
            plot_trigger.set(plot_trigger() + 1)

        except Exception as e:
            ui.notification_show(f"Analysis Failed: {str(e)}", type="error")


    # --- RESULTS DISPLAY ---
    @render.ui
    def pod_results_container():
        """
        Renders the side-by-side plots and the metrics table.
        """
        if pod_metrics() is None:
            return ui.div()

        return ui.div(
            # Row 1: Plots
            ui.layout_columns(
                ui.card(
                    ui.card_header("Signal Model Fit"),
                    ui.output_plot("plot_signal"),
                    full_screen=True
                ),
                ui.card(
                    ui.card_header("PoD Curve (95% CI)"),
                    ui.output_plot("plot_curve"),
                    full_screen=True
                ),
                col_widths=[6, 6]
            ),
            # Row 2: Table and Download Actions
            ui.layout_columns(
                ui.card(
                    ui.card_header("Key Reliability Metrics"),
                    ui.output_data_frame("pod_stats_table")
                ),
                ui.card(
                    ui.card_header("Export Results"),
                    ui.div(
                        ui.p("Download the full curve data (PoD, CI) for external reporting."),
                        ui.download_button("download_pod_results", "Download Results CSV", class_="btn-success w-100")
                    ),
                    class_="d-flex align-items-center justify-content-center text-center h-100"
                ),
                col_widths=[8, 4]
            )
        )

    @render.plot
    def plot_signal():
        _ = plot_trigger() # Dependency on button click
        study = current_study()
        if study and "signal_model" in study.plots:
            return study.plots["signal_model"]
        return None

    @render.plot
    def plot_curve():
        _ = plot_trigger() # Dependency on button click
        study = current_study()
        if study and "pod_curve" in study.plots:
            return study.plots["pod_curve"]
        return None

    @render.data_frame
    def pod_stats_table():
        data = pod_metrics()
        if data is None:
            return None
        # Convert dictionary to DataFrame for display
        df = pd.DataFrame(list(data.items()), columns=["Metric", "Value"])
        return render.DataGrid(df, width="100%", filters=False)

    @render.download(filename="pod_results.csv")
    def download_pod_results():
        """
        Downloads the curve data.
        """
        df = pod_export_data()
        if df is not None:
            yield df.to_csv(index=False)

#### App ####
app = App(app_ui, server)
