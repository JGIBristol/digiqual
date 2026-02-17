from shiny import App, ui, render, reactive
from pathlib import Path
from faicons import icon_svg
import pandas as pd
import numpy as np
from digiqual.sampling import generate_lhs
from digiqual import SimulationStudy
import shinyswatch


css_path = Path(__file__).parent / "styles.css"

# UI Definition
app_ui = ui.page_navbar(
    ui.nav_panel(
        "Home",
        # --- 1. HEADER (Text on Grey) ---
        ui.div(
            ui.h2("DigiQual", class_="fw-bold text-primary mb-1 text-center"),
            ui.p("Statistical Toolkit for Reliability Assessment in NDT",
                class_="lead text-muted mb-0 text-center"),
            ui.hr(class_="my-4"),
            class_="mb-4 mt-3"
        ),

        # --- 2. MAIN CONTENT GRID ---
        ui.layout_columns(
            # --- LEFT COLUMN: NESTED WORKFLOW ---
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

            # --- RIGHT COLUMN: UNIFIED INFO ---
            ui.div(
                ui.h4("Project Information", class_="mb-3 text-primary border-bottom pb-2"),
                ui.card(
                    ui.div(
                        # About Section
                        ui.div(
                            ui.tags.strong("Version: "), "0.10.0", ui.br(),
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
    ui.nav_panel(
        "Experimental Design",
        ui.layout_columns(
            ui.div(
                ui.card(
                    ui.card_header("Experimental Design Variables"),

                    # -- HEADER ROW --
                    ui.div(
                        ui.layout_columns(
                            ui.tags.label("Variable Name", class_="fw-bold"),
                            ui.tags.label("Min Value", class_="fw-bold"),
                            ui.tags.label("Max Value", class_="fw-bold"),
                        ),
                        class_="mb-2 px-1"
                    ),

                    # -- Dynamic Rows Container --
                    ui.div(id="variable_rows_container"),

                    ui.div(
                        ui.input_action_button(
                            "add_variable_btn",
                            "Add Row",
                            icon=icon_svg("plus"),
                            class_="btn-outline-secondary btn-sm"
                        ),
                        class_="d-flex justify-content-center mt-3"
                    )
                ),
            ),
            ui.div(
                ui.card(
                    ui.card_header("Generation Settings"),
                    ui.input_numeric("num_rows", "Number of samples", value=100, min=1),
                    ui.hr(),
                    ui.input_task_button(
                        "generate_btn",
                        "Generate Framework",
                        class_="btn-primary w-100",
                        icon=icon_svg("gears") # <--- CHANGED
                    ),
                    ui.output_ui("download_btn_container", class_="mt-3")
                ),
            ),
            col_widths=[8, 4]
        ),
        icon=icon_svg("table")
    ),
    ui.nav_panel(
        "Simulation Diagnostics",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_file("upload_csv", "Upload CSV file", accept=[".csv"], multiple=False),
                ui.hr(),
                # -- COLUMN SELECTORS --
                ui.input_selectize("input_cols", "Select Input Variables", choices=[], multiple=True),
                ui.input_selectize("outcome_col", "Select Outcome Variable", choices=[], multiple=False),
                ui.hr(),
                # -- STATUS INDICATOR --
                ui.output_ui("validation_status"),
            ),
            ui.card(
                ui.card_header("Uploaded Data Preview"),
                ui.output_data_frame("preview_uploaded")
            ),
            ui.card(
                ui.card_header("Validation Report"),
                ui.output_data_frame("validation_results_table"),
            ),

            ui.output_ui("remediation_ui")
        ),
        icon=icon_svg("check-double")
    ),
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
    header=ui.include_css(css_path)
)


def server(input, output, session):

    # ==================== TAB 1: DATA GENERATOR ====================

    # Track the number of variable rows added
    row_count = reactive.value(0)
    # Store the successfully generated data here
    final_generated_df = reactive.value(None)

    def _add_row():
        """Helper function to insert UI"""
        current_count = row_count()
        new_count = current_count + 1
        row_count.set(new_count)
        idx = current_count

        ui.insert_ui(
            selector="#variable_rows_container",
            where="beforeEnd",
            ui=ui.div(
                ui.layout_columns(
                    ui.input_text(f"var_name_{idx}", label=None, placeholder="e.g. Length"),
                    ui.input_numeric(f"var_min_{idx}", label=None, value=0),
                    ui.input_numeric(f"var_max_{idx}", label=None, value=10),
                ),
                class_="mb-1"
            )
        )

    @reactive.effect
    @reactive.event(input.add_variable_btn)
    def add_variable_handler():
        _add_row()
        final_generated_df.set(None)

    @reactive.effect
    def init_rows():
        if row_count() == 0:
            _add_row()

    # 1. HANDLE GENERATION AND VALIDATION
    @reactive.effect
    @reactive.event(input.generate_btn)
    def generate_handler():
        """Validates inputs and generates data to reactive value"""
        # Clear previous results/buttons
        final_generated_df.set(None)

        count = row_count()
        ranges = {}
        errors = []

        # -- Validation Logic --
        for i in range(count):
            name_val = input[f"var_name_{i}"]()
            min_val = input[f"var_min_{i}"]()
            max_val = input[f"var_max_{i}"]()

            # Logic Check: Was this row touched?
            if not name_val or str(name_val).strip() == "":
                errors.append(f"Row {i+1}: Name is missing.")
                continue

            if min_val is None or max_val is None:
                errors.append(f"Row {i+1} ({name_val}): Missing min/max values.")
                continue

            if min_val >= max_val:
                errors.append(f"Row {i+1} ({name_val}): Min ({min_val}) must be < Max ({max_val}).")
                continue

            if name_val in ranges:
                errors.append(f"Row {i+1}: Duplicate name '{name_val}'.")
                continue

            ranges[name_val] = [min_val, max_val]

        if not ranges and not errors:
            errors.append("Please define at least one variable.")

        # -- Failure Case --
        if errors:
            ui.notification_show("Validation Failed", type="error")
            m = ui.modal(
                ui.HTML("<ul><li>" + "</li><li>".join(errors) + "</li></ul>"),
                title="Validation Errors",
                easy_close=True
            )
            ui.modal_show(m)
            return

        # -- Success Case --
        try:
            n = input.num_rows()
            df = generate_lhs(n=n, ranges=ranges)
            final_generated_df.set(df)
            ui.notification_show("Success! Data ready for download.", type="message")
        except Exception as e:
            ui.notification_show(f"Generation Error: {str(e)}", type="error")


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

# ==================== TAB 2: DATA VALIDATOR ====================

    uploaded_data = reactive.value(None)
    validation_passed = reactive.value(False)
    new_samples = reactive.value(None)

    diagnostic_table = reactive.value(None)

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
    def update_column_selectors():
        df = uploaded_data()
        if df is None:
            ui.update_selectize("input_cols", choices=[])
            ui.update_selectize("outcome_col", choices=[])
            return

        cols = list(df.columns)
        ui.update_selectize("input_cols", choices=cols, selected=cols[:-1] if len(cols) > 1 else cols)
        ui.update_selectize("outcome_col", choices=cols, selected=cols[-1] if len(cols) > 0 else None)

    @reactive.effect
    @reactive.event(uploaded_data, input.input_cols, input.outcome_col)
    def run_validation_diagnostics():
        df = uploaded_data()
        # Reset previous new samples if we change data/cols
        new_samples.set(None)

        if df is None or not input.input_cols() or not input.outcome_col():
            validation_passed.set(False)
            diagnostic_table.set(None)
            return

        # 1. Initialize Study
        study = SimulationStudy(
            input_cols=list(input.input_cols()),
            outcome_col=input.outcome_col()
        )
        study.add_data(df)

        # 2. Run Diagnostics
        diag_df = study.diagnose()
        if diag_df is None:
            return

        diagnostic_table.set(diag_df)

        # 3. Check "Pass" column
        try:
            all_passed = diag_df["Pass"].astype(bool).all()
            validation_passed.set(all_passed)
        except KeyError:
            validation_passed.set(False)

    # --- OUTPUTS ---

    @render.data_frame
    def preview_uploaded():
        df = uploaded_data()
        if df is not None:
            return render.DataGrid(df.round(3).head(5))
        return render.DataGrid(pd.DataFrame())

    @render.ui
    def validation_status():
        if uploaded_data() is None:
            return ui.div(ui.p("Upload a CSV to begin.", class_="text-muted fst-italic"))

        if validation_passed():
            return ui.div(
                ui.h5(icon_svg("circle-check"), " Validation Passed"),
                class_="alert alert-success mt-3"
            )
        else:
            return ui.div(
                ui.h5(icon_svg("triangle-exclamation"), " Issues Detected"),
                ui.p("Coverage gaps found. See options below.", class_="small mb-0"),
                class_="alert alert-danger mt-3"
            )

    @render.data_frame
    def validation_results_table():
        df = diagnostic_table()
        if df is not None:
            return render.DataGrid(df)
        return None

    # --- REMEDIATION SECTION ---

    @render.ui
    def remediation_ui():
        """
        Only appears if validation failed.
        """
        if validation_passed() or uploaded_data() is None:
            return ui.div()

        return ui.card(
            ui.card_header("Remediation: Generate New Samples"),
            ui.p("Your data has issues. Use the Refine tool to generate new samples specifically in the empty spaces."),

            ui.layout_columns(
                ui.input_numeric("n_new_samples", "Count", value=10, min=1),
                ui.input_task_button("btn_refine", "Generate New Samples", icon=icon_svg("wand-magic-sparkles"), class_="btn-warning"),
            ),
            ui.output_ui("download_new_samples_ui"),
            class_="border-warning mt-3"
        )

    @reactive.effect
    @reactive.event(input.btn_refine)
    def compute_new_samples():
        """
        Calls study.refine() to create NEW SAMPLES.
        """
        df = uploaded_data()
        if df is None:
            return

        study = SimulationStudy(
            input_cols=list(input.input_cols()),
            outcome_col=input.outcome_col()
        )
        study.add_data(df)

        try:
            # Generate the new samples
            generated_df = study.refine(n_points=input.n_new_samples())

            # Save to our renamed reactive value
            new_samples.set(generated_df)

            ui.notification_show(f"Success! Generated {len(generated_df)} new samples.", type="message")

        except Exception as e:
            ui.notification_show(f"Refinement Failed: {str(e)}", type="error")

    @render.ui
    def download_new_samples_ui():
        if new_samples() is not None:
            return ui.div(
                ui.hr(),
                ui.br(),
                ui.download_button("download_new_samples_csv", "Download New Samples CSV", class_="btn-success w-100")
            )
        return ui.div()

    @render.data_frame
    def preview_new_samples():
        df = new_samples()
        if df is not None:
            return render.DataGrid(df.round(3))
        return None

    @render.download(filename="new_samples.csv")
    def download_new_samples_csv():
        df = new_samples()
        if df is not None:
            yield df.to_csv(index=False)


# ==================== TAB 3: DATA ANALYZER ====================

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

# Create the app object
app = App(app_ui, server)
