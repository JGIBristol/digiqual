import matplotlib
matplotlib.use('Agg')

from shiny import App, ui, render, reactive
from faicons import icon_svg
import pandas as pd
import numpy as np
from digiqual.sampling import generate_lhs
from digiqual import SimulationStudy
from pathlib import Path

#### App CSS ####
app_css = """
/* --- 1. CORE PALETTE (Modern Fluent) --- */
:root {
    --bs-primary: #006abc; /* Fluent blue */
    --bs-primary-rgb: 0, 106, 188;
    --bs-success: #107c10;
    --bs-warning: #ffb900;
    --bs-danger: #d13438;
    --bs-body-bg: #faf9f8; /* Soft fluent gray */
    --bs-body-color: #242424;
    --bs-border-color: #edebe9;
}

/* --- 2. GLOBAL TYPOGRAPHY --- */
body {
    font-family: 'Segoe UI', 'Segoe UI Variable', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bs-body-bg);
    color: var(--bs-body-color);
}

h1, h2, h3, h4, h5, h6 {
    color: var(--bs-primary);
    font-weight: 700;
    letter-spacing: -0.01em;
}

.navbar-brand {
    font-weight: 600 !important;
    letter-spacing: normal;
}

/* --- 3. COMPONENT POLISH (Fluent Elevation & Acrylic) --- */
.card {
    border: 1px solid #edebe9;
    box-shadow: 0 1.6px 3.6px 0 rgba(0,0,0,0.132), 0 0.3px 0.9px 0 rgba(0,0,0,0.108); /* Fluent Elevation 2 */
    border-radius: 8px;
    margin-bottom: 24px;
    background-color: #ffffff;
}

.card-header {
    background-color: transparent;
    border-bottom: 1px solid var(--bs-border-color);
    font-weight: 600;
    color: #242424;
    padding: 16px 20px;
}

.card-body {
    padding: 24px 20px;
}

/* Buttons Fluent Style */
.btn {
    border-radius: 4px;
    font-weight: 600;
    padding: 6px 20px;
    transition: all 0.1s ease-in-out;
    border: 1px solid transparent;
}

.btn-primary {
    background-color: var(--bs-primary);
    color: #ffffff;
}
.btn-primary:hover {
    background-color: #005a9e;
    color: #ffffff;
}
.btn-primary:active {
    background-color: #004578;
    transform: scale(0.98);
}

.btn-success { background-color: var(--bs-success); color: #fff; }
.btn-success:hover { background-color: #0b5a0b; color: #fff; }
.btn-warning { background-color: var(--bs-warning); color: #000; }
.btn-warning:hover { background-color: #da9d00; color: #000; }
.btn-danger { background-color: var(--bs-danger); color: #fff; }
.btn-danger:hover { background-color: #a82a2d; color: #fff; }

.btn-outline-secondary {
    border: 1px solid #8a8886;
    color: #323130;
    background-color: transparent;
}
.btn-outline-secondary:hover {
    background-color: #f3f2f1;
    color: #201f1e;
}

/* --- 4. CONTROLS (Form inputs) --- */
.form-control, .form-select {
    border-radius: 4px;
    border: 1px solid #8a8886;
    border-bottom: 2px solid #8a8886;
    background-color: #ffffff;
    padding: 6px 12px;
    transition: all 0.2s ease;
}

.form-control:focus, .form-select:focus {
    border-color: var(--bs-primary);
    border-bottom: 2px solid var(--bs-primary);
    box-shadow: none;
    outline: none;
}

/* Better File Upload UI (Fluent-inspired Drop Zone) */
.shiny-input-container:has(input[type="file"]) .input-group {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    border: 2px dashed #c8c6c4 !important;
    border-radius: 6px !important;
    background-color: #faf9f8 !important;
    padding: 30px 20px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    text-align: center !important;
}

.shiny-input-container:has(input[type="file"]) .input-group:hover {
    border-color: var(--bs-primary) !important;
    background-color: #f3f2f1 !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
}

.shiny-input-container:has(input[type="file"]) .input-group-btn {
    width: 100% !important;
}

.shiny-input-container:has(input[type="file"]) .btn-file {
    background-color: transparent !important;
    border: none !important;
    color: var(--bs-primary) !important;
    font-weight: 600 !important;
    padding: 0 !important;
    width: 100% !important;
    display: block !important;
    font-size: 1.1em !important;
}

.shiny-input-container:has(input[type="file"]) .form-control {
    display: block !important;
    width: 100% !important;
    border: none !important;
    background: transparent !important;
    text-align: center !important;
    box-shadow: none !important;
    color: #605e5c !important;
    margin-top: 10px !important;
    padding: 0 !important;
    height: auto !important;
}

.shiny-input-container:has(input[type="file"]) label {
    margin-bottom: 8px;
    font-weight: 600;
}

/* Container for Centered Configuration Inputs */
.config-container {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    text-align: center !important;
    gap: 1.5rem !important; /* Better spacing than HRs */
}

/* Ensure individual input containers are centered */
.config-container .shiny-input-container {
    width: 100% !important;
    max-width: 400px !important; /* Constrain width for a tidy look */
    margin-left: auto !important;
    margin-right: auto !important;
    margin-bottom: 0 !important;
}

/* Fix for Selectize labels and control alignment */
.config-container .control-label {
    margin-bottom: 8px !important;
    width: 100% !important;
}

/* Align selectize itself (content stays left, container centered) */
.config-container .selectize-control {
    text-align: left !important;
}

/* --- 5. NAVIGATION (Top Bar) --- */
.navbar {
    background: #006abc !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
    color: #fff;
}

.navbar-brand {
    color: #fff !important;
}

.navbar .nav-link {
    color: rgba(255, 255, 255, 0.8) !important;
    font-weight: 500;
    padding: 10px 16px;
    margin: 0 4px;
    border-radius: 4px;
    transition: background-color 0.1s;
}

.navbar .nav-link:hover {
    color: #ffffff !important;
    background-color: rgba(255, 255, 255, 0.1);
}

.navbar .nav-link.active {
    color: #ffffff !important;
    font-weight: 600;
    background-color: rgba(255, 255, 255, 0.2);
}

/* --- 6. CUSTOM SCROLLBARS --- */
::-webkit-scrollbar { width: 12px; height: 12px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background-color: #c8c6c4;
    border-radius: 10px;
    border: 3px solid #faf9f8;
}
::-webkit-scrollbar-thumb:hover { background-color: #a19f9d; }

/* MISC TWEAKS */
.text-primary { color: var(--bs-primary) !important; }
.text-muted { color: #605e5c !important; }
.bg-light { background-color: #f3f2f1 !important; }

hr {
    border-top: 1px solid #edebe9;
    opacity: 1;
}
"""


#### UI Definition ####
app_ui = ui.page_navbar(

#### UI - Homepage ####
ui.nav_panel(
        "Home",
        # --- NEW SCROLLING WRAPPER ---
        ui.div(
            ui.div(
                ui.h2("DigiQual", class_="fw-bold text-primary mb-1 text-center"),
                ui.p("Statistical Toolkit for Reliability Assessment in NDT",
                    class_="lead text-muted mb-0 text-center"),
                ui.hr(class_="my-4"),
                class_="mb-4 mt-3"
            ),

            ui.layout_columns(
                # --- LEFT COLUMN: WORKFLOW MODULES ---
                ui.div(
                    ui.h4("Workflow Modules", class_="mb-3 text-primary border-bottom pb-2"),
                    ui.card(
                        ui.div(
                            # Module 1: Design
                            ui.div(
                                ui.h5(
                                    ui.span(icon_svg("table"), class_="text-primary me-2"),
                                    "1. Experimental Design", class_="fw-bold mb-2"
                                ),
                                ui.p("Design efficient experimental frameworks using Latin Hypercube Sampling (LHS).", class_="fw-semibold mb-2"),
                                ui.tags.ul(
                                    ui.tags.li("Space-filling parameter generation."),
                                    ui.tags.li("Automatic scaling to variable bounds."),
                                    class_="mb-0 ps-3 text-muted"
                                ),
                                class_="border-start border-3 border-primary ps-3 mb-3"
                            ),
                            ui.hr(class_="my-4"),

                            # Module 2: Diagnostics
                            ui.div(
                                ui.h5(
                                    ui.span(icon_svg("check-double"), class_="text-warning me-2"),
                                    "2. Simulation Diagnostics", class_="fw-bold mb-2"
                                ),
                                ui.p("Validate dataset integrity and simulation outputs.", class_="fw-semibold mb-2"),
                                ui.tags.ul(
                                    ui.tags.li("Detect input coverage gaps and sanity checks."),
                                    ui.tags.li("Identify model instability or insufficient samples."),
                                    class_="mb-0 ps-3 text-muted"
                                ),
                                class_="border-start border-3 border-warning ps-3 mb-3"
                            ),
                            ui.hr(class_="my-4"),


                            # Module 3: Visualisation
                            ui.div(
                                ui.h5(
                                    ui.span(icon_svg("display"), class_="text-info me-2"),
                                    "3. Data Visualisation", class_="fw-bold mb-2"
                                ),
                                ui.p("Inspect simulation variables and results.", class_="fw-semibold mb-2"),
                                ui.tags.ul(
                                    ui.tags.li("Per-variable distribution and gap inspection."),
                                    ui.tags.li("Model fit and bootstrap convergence visualised."),
                                    class_="mb-0 ps-3 text-muted"
                                ),
                                class_="border-start border-3 border-info ps-3 mb-3"
                            ),
                            ui.hr(class_="my-4"),


                            # Module 4: Analysis
                            ui.div(
                                ui.h5(
                                    ui.span(icon_svg("chart-line"), class_="text-success me-2"),
                                    "4. PoD Analysis", class_="fw-bold mb-2"
                                ),
                                ui.p("Construct Generalized Probability of Detection (PoD) curves.", class_="fw-semibold mb-2"),
                                ui.tags.ul(
                                    ui.tags.li("Robust statistics using the Generalized a-versus-a Method."),
                                    ui.tags.li("Uncertainty quantification with bootstrap resampling."),
                                    class_="mb-0 ps-3 text-muted"
                                ),
                                class_="border-start border-3 border-success ps-3 mb-1"
                            ),
                            class_="p-3"
                        )
                    )
                ),

                # --- RIGHT COLUMN: PROJECT INFORMATION ---
                ui.div(
                    ui.h4("Project Information", class_="mb-3 text-primary border-bottom pb-2"),
                    ui.card(
                        ui.div(
                            # About & Resources Section (Side-by-side)
                            ui.layout_columns(
                                ui.div(
                                    ui.h5("About", class_="fw-bold mb-2"),
                                    ui.tags.strong("Version: "), "0.14.0", ui.br(),
                                    ui.tags.strong("License: "), "MIT", ui.br(),
                                    ui.tags.strong("Author: "), "Josh Tyler", ui.br(),
                                    ui.tags.strong("Institution: "), "Univ. of Bristol",
                                ),
                                ui.div(
                                    ui.h5("Resources", class_="fw-bold mb-2"),
                                    ui.a(ui.span(icon_svg("github"), class_="me-1 text-primary"), " GitHub Repo", href="https://github.com/JGIBristol/digiqual", target="_blank", class_="d-block text-decoration-none text-body mb-1"),
                                    ui.a(ui.span(icon_svg("book"), class_="me-1 text-primary"), " Documentation", href="https://jgibristol.github.io/digiqual/", target="_blank", class_="d-block text-decoration-none text-body mb-1"),
                                    ui.a(ui.span(icon_svg("python"), class_="me-1 text-primary"), " PyPI Package", href="https://pypi.org/project/digiqual/", target="_blank", class_="d-block text-decoration-none text-body"),
                                ),
                                col_widths=[6, 6],
                                class_="mb-1"
                            ),
                            ui.hr(class_="my-4"),

                            # Methodology
                            ui.div(
                                ui.h5("Methodology References", class_="fw-bold mb-3"),

                                # Reference Block 1
                                ui.div(
                                    ui.p(
                                        "Malkiel, N., Croxford, A. J., & Wilcox, P. D. (2025). ",
                                        ui.span("A generalized method for the reliability assessment of safety–critical inspection. ", class_="fst-italic"),
                                        "Proceedings of the Royal Society A.",
                                        class_="text-muted mb-2"
                                    ),
                                    ui.a(
                                        "View Paper",
                                        href="https://doi.org/10.1098/rspa.2024.0654",
                                        target="_blank",
                                        class_="btn btn-outline-secondary w-100"
                                    ),
                                    class_="mb-4"
                                ),

                                # Reference Block 2
                                ui.div(
                                    ui.p(
                                        "Malkiel, N., Croxford, A. J., & Wilcox, P. D. (2026). ",
                                        ui.span("A comprehensive investigation of flexible and multi-dimensional simulation-based PoD analysis. ", class_="fst-italic"),
                                        "NDT & E International",
                                        class_="text-muted mb-2"
                                    ),
                                    ui.a(
                                        "View Paper",
                                        href="https://doi.org/10.1016/j.ndteint.2025.103596",
                                        target="_blank",
                                        class_="btn btn-outline-secondary w-100"
                                    ),
                                    class_="mb-2"
                                )
                            ),
                            ui.hr(class_="my-4"),

                            # Support
                            ui.div(
                                ui.p("Development supported by:", class_="fw-bold text-center text-muted mb-3"),
                                ui.div(
                                    ui.img(
                                        src="ukri-epsrc-square-logo.png",
                                        height="60px",
                                        alt="UKRI EPSRC Logo"
                                    ),
                                    ui.span("UKRI EPSRC", class_="text-muted d-block mt-2 fw-semibold"),
                                    class_="bg-light border rounded p-3 text-center"
                                )
                            ),
                            class_="p-3"
                        )
                    )
                ),
                col_widths=[-1,5,5,-1]
            ),

            # --- FOOTER: DISCLAIMER CARD ---
            ui.layout_columns(
                ui.card(
                    ui.div(
                        ui.p(
                            ui.tags.strong("Disclaimer & Data Privacy: "),
                            "This software is provided 'as is', without warranty of any kind. "
                            "In no event shall the authors be liable for any claim or damages. All processing is performed locally. "
                            "This application does not implement data persistence, nor does it facilitate the outbound transmission "
                            "of user-supplied datasets to external servers.",
                            class_="text-center text-muted small mb-0"
                        ),
                        class_="p-3 bg-light rounded"
                    ),
                    class_="mt-2 mb-4 shadow-sm border-0"
                ),
                col_widths=[-1, 10, -1]
            ),

            # This class allows just this tab to scroll while respecting your global fillable=True
            class_="overflow-auto h-100 px-3"
        ),
        icon=icon_svg("house")
    ),
#### UI - Experimental Design (Tab 2) ####
    ui.nav_panel(
        "Experimental Design",
        ui.div(
            ui.h3("Experimental Design", class_="mb-4 text-center"),
            ui.layout_columns(
                # --- LEFT: VARIABLE INPUTS
                ui.card(
                    ui.card_header("Experimental Design Variables"),
                    # Header Row
                    ui.div(
                        ui.layout_columns(
                            ui.tags.label("Variable Name", class_="fw-bold mb-0"),
                            ui.tags.label("Min Value", class_="fw-bold mb-0"),
                            ui.tags.label("Max Value", class_="fw-bold mb-0"),
                            ui.div(), # Spacer for the delete button column
                            col_widths=(4, 3, 3, 2),
                            gap="5px",
                            class_="mb-0" # Drops layout_columns default bottom margin
                        ),
                        class_="mb-0 px-1 text-center" # Removed border-bottom and pb-2
                    ),
                    # Container for Rows
                    ui.div(
                        # mt-0 and pt-0 eliminate the space above the rows
                        ui.div(id="variable_rows_container", class_="mt-0 pt-0"),
                        ui.div(
                            ui.input_action_button(
                                "add_variable_btn", "Add Variable",
                                icon=icon_svg("plus"), class_="btn-outline-secondary btn-sm"
                            ),
                            class_="mt-3 d-flex justify-content-start"
                        ),
                    ),
                    class_="mb-0"
                ),

                # --- RIGHT: PREVIEW & SETTINGS ---
                ui.div(
                    ui.card(
                        ui.card_header("Framework Preview"),
                        ui.output_data_frame("preview_experimental_design"),
                        full_screen=True,
                        class_="mb-3"
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
                        ui.output_ui("download_btn_container", class_="mt-3"),
                    ),
                    class_="d-flex flex-column"
                ),
                col_widths=[-1,5,5,-1]
            ),
            class_="container-fluid py-3"
        ),
        icon=icon_svg("table")
    ),


#### UI - Simulation Diagnostics (Tab 3) ####
    ui.nav_panel(
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

                        # Permanent Error Display for selection conflicts
                        ui.output_ui("selection_error_display"),

                        # This is the primary status alert (Success/Failure)
                        ui.output_ui("validation_status"),
                        class_="config-container"
                    ),
                    class_="mb-0"
                ),

                # --- RIGHT: PREVIEWS & REPORTS ---
                ui.div(
                    # Dynamic Preview (Hides when no data is present)
                    ui.output_ui("dynamic_preview_card"),

                    ui.card(
                        ui.card_header("Validation Report"),
                        ui.output_data_frame("validation_results_table"),
                        full_screen=True,
                        class_="mb-3"
                    ),
                    # Remediation logic is now anchored to the results
                    ui.output_ui("remediation_ui"),
                    class_="d-flex flex-column"
                ),
                col_widths=[-1,3,7,-1]
            ),
            class_="container-fluid py-3"
        ),
        icon=icon_svg("check-double")
    ),



#### UI - Data Visualisation (Tab 3.5) ####
    ui.nav_panel(
        "Data Visualisation",
        ui.div(
            ui.h3("Data Visualisation", class_="mb-4 text-center"),
            ui.output_ui("viz_content"),
            class_="container-fluid py-3 overflow-auto h-100"
        ),
        icon=icon_svg("display")
    ),

#### UI - PoD Analysis (Tab 4) ####
    ui.nav_panel(
        "PoD Analysis",
        ui.div(
            ui.h3("PoD Analysis", class_="mb-4 text-center"),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Analysis Results"),
                    ui.output_ui("analysis_output"),
                ),
                col_widths=[-1,10,-1]
            ),
            class_="container-fluid py-3"
        ),
        icon=icon_svg("chart-line")
    ),
    title="DigiQual",
    id="navbar",
    fillable=True,
    header=ui.tags.style(app_css)
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
                        f"remove_{idx}", "", icon=icon_svg("trash"),
                        class_="btn-outline-danger btn-sm"
                    ),
                    col_widths=(4, 3, 3, 2),
                    gap="5px",         # Matches the header gap
                    class_="mt-0 mb-0" # Kills default layout_columns margins
                ),
                id=f"row_container_{idx}",
                class_="mb-2 mt-0"     # Keeps a small gap between rows, but 0 on top
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
            yield df.to_csv(index=False).encode('utf-8')

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

    @reactive.effect
    @reactive.event(input.btn_refine)
    def handle_refinement():
        study = current_study()
        if study is None:
            return

        try:
            # We assume your SimulationStudy has a refine method
            # that targets the 'Length' gaps we discussed earlier
            n_to_gen = input.n_new_samples()
            refined_df = study.refine(n_points=n_to_gen) # Or your specific generation logic

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
                ui.p("See Data Visualisation Tab for more information and the Remediation options to the right for next steps.", class_="small mb-0"),
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


#### Server - Visualisation (Tab 3.5) ####

    @render.ui
    def viz_content():
        """
        Master render for the entire viz tab.
        Reads uploaded_data() and diagnostic_table() from the Diagnostics tab —
        no separate upload is needed here.
        """
        if uploaded_data() is None:
            return ui.div(
                ui.div(
                    ui.h4("No Data Available", class_="text-muted"),
                    ui.p("Upload a CSV in the 'Simulation Diagnostics' tab to begin."),
                    class_="text-center p-5 bg-light border rounded"
                )
            )

        df = uploaded_data()
        all_cols = list(df.columns)

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
                            choices=all_cols,
                            selected=all_cols[0] if all_cols else None,
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
            # ── Row 4: Full Outcome Diagnostic Overview ────────────────────────
            ui.card(
                ui.card_header("Outcome Diagnostic Overview"),
                ui.output_ui("viz_diagnostics_state"),
                ui.p(
                    "Left: Actual vs Predicted scatter for the degree-3 polynomial model. "
                    "Points hugging the diagonal indicate a good fit. "
                    "Right: Bootstrap convergence trace — running relative std dev across "
                    "iterations. Lines flattening below the thresholds indicate convergence.",
                    class_="small text-muted px-3 pt-2 mb-0"
                ),

                full_screen=True,
            ),
        )


    @render.data_frame
    def viz_summary_table():
        df = uploaded_data()
        if df is None:
            return None

        rows = []
        for col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce").dropna()
            n_valid = len(numeric)
            rows.append({
                "Variable":   col,
                "N (Total)":  len(df[col]),
                "N (Valid)":  n_valid,
                "Min":        f"{numeric.min():.4g}"    if n_valid else "N/A",
                "Median":     f"{numeric.median():.4g}" if n_valid else "N/A",
                "Mean":       f"{numeric.mean():.4g}"   if n_valid else "N/A",
                "Std Dev":    f"{numeric.std():.4g}"    if n_valid else "N/A",
                "Max":        f"{numeric.max():.4g}"    if n_valid else "N/A",
            })

        return render.DataGrid(pd.DataFrame(rows), width="100%", filters=False)


    @render.ui
    def viz_diagnostic_badge():
        """
        Shows pass/fail status for the currently selected variable,
        sourced from the diagnostic_table reactive already computed in Tab 3.
        """
        diag = diagnostic_table()
        if diag is None or diag.empty:
            return ui.p(
                "Run diagnostics in Tab 3 to see variable status here.",
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
        from matplotlib.transforms import blended_transform_factory

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
                    pass

            # Rug plot along the x-axis (blended transform: data x, axes y)
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            ax.plot(
                vals, np.full(len(vals), -0.04), "|",
                color="#242424", alpha=0.35, markersize=10,
                transform=trans, clip_on=False
            )

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
                        pass

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
        Computed from _check_input_coverage directly so it's always up to date,
        even if the user hasn't explicitly run diagnostics.
        """
        import matplotlib.pyplot as plt
        from digiqual.diagnostics import _check_input_coverage

        df = uploaded_data()
        if df is None:
            return None

        # Use selected inputs, falling back to all-but-outcome if none chosen yet
        input_cols_list = list(input.input_cols())
        outcome = input.outcome_col()
        if not input_cols_list:
            input_cols_list = [c for c in df.columns if c != outcome]
        if not input_cols_list:
            return None

        try:
            coverage_res = _check_input_coverage(df, input_cols_list)
        except Exception:
            coverage_res = {}

        n = len(input_cols_list)
        ncols = min(n, 3)
        nrows = -(-n // ncols)  # ceiling division

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 3 * nrows),
            squeeze=False
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
            ax.set_xlabel(col, fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.grid(True, alpha=0.3)

        # Hide unused subplot slots
        for idx in range(n, nrows * ncols):
            row, c = divmod(idx, ncols)
            axes[row][c].set_visible(False)

        fig.suptitle(
            "Input Space Coverage  (✓ Pass  ✗ Fail)",
            fontsize=12, fontweight="bold"
        )
        fig.tight_layout()
        return fig

    @render.ui
    def viz_diagnostics_state():
        """
        Guards the diagnostic overview plot. Shows a prompt if diagnostics
        haven't been run yet (diagnostic_table is None or empty).
        """
        diag = diagnostic_table()
        if diag is None or diag.empty:
            return ui.div(
                ui.p(
                    ui.span(icon_svg("circle-info"), class_="me-2 text-primary"),
                    "Diagnostics have not been run yet. "
                    "Configure your columns in the 'Simulation Diagnostics' tab — "
                    "results will appear here automatically.",
                    class_="text-muted fst-italic p-4 text-center"
                )
            )
        return ui.output_plot("viz_diagnostics_overview", height="500px")

    @render.plot
    def viz_diagnostics_overview():
        """
        Two diagnostic visualisations side by side:

        Left — Actual vs Predicted (Model Fit):
            Fits a degree-3 polynomial on the full dataset and plots observed y
            against predicted y. Points on the diagonal = perfect fit. Spread
            away from it explains why the CV R² may be low.

        Right — Bootstrap Convergence Trace:
            Runs 100 bootstrap iterations and plots the *running* average and
            maximum relative std dev as each iteration accumulates. A line that
            flattens below its threshold dashed line has converged; one that is
            still declining suggests more samples would improve stability.
        """
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import make_pipeline

        df = uploaded_data()
        if df is None:
            return None

        diag = diagnostic_table()
        if diag is None or diag.empty:
            return None

        input_cols_list = list(input.input_cols())
        outcome = input.outcome_col()

        if not input_cols_list or outcome not in df.columns:
            return None

        # Clean numeric data (same subset diagnostics used)
        required = input_cols_list + [outcome]
        df_num = df[required].apply(pd.to_numeric, errors="coerce").dropna()
        if len(df_num) < 10:
            return None

        X = df_num[input_cols_list].values
        y = df_num[outcome].values

        # Pull pass/fail status from existing diagnostic_table
        fit_row  = diag[diag["Test"] == "Model Fit (CV)"]
        boot_row = diag[diag["Test"] == "Bootstrap Convergence"]

        r2_val      = float(fit_row["Value"].values[0])      if not fit_row.empty  else None
        fit_passed  = bool(fit_row["Pass"].values[0])         if not fit_row.empty  else True
        boot_passed = bool(boot_row["Pass"].values[0])        if not boot_row.empty else True

        avg_cv_val = None
        max_cv_val = None
        if not boot_row.empty:
            avg_rows = boot_row[boot_row["Metric"] == "Avg CV (Rel Std Dev)"]
            max_rows = boot_row[boot_row["Metric"] == "Max CV (Rel Std Dev)"]
            if not avg_rows.empty:
                avg_cv_val = float(avg_rows["Value"].values[0])
            if not max_rows.empty:
                max_cv_val = float(max_rows["Value"].values[0])

        fig, (ax_fit, ax_boot) = plt.subplots(1, 2, figsize=(12, 5))

        # ── LEFT: Actual vs Predicted ─────────────────────────────────────────
        poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
        poly_model.fit(X, y)
        y_pred = poly_model.predict(X)

        ax_fit.scatter(y, y_pred, alpha=0.55, s=25, color="#1f77b4",
                        label="Simulations", zorder=3)

        # Perfect-fit diagonal
        y_lo = min(y.min(), y_pred.min())
        y_hi = max(y.max(), y_pred.max())
        pad  = (y_hi - y_lo) * 0.05
        diag_range = [y_lo - pad, y_hi + pad]
        ax_fit.plot(diag_range, diag_range, color="#d13438", linewidth=1.5,
                    linestyle="--", label="Perfect Fit", zorder=2)

        fit_colour = "#107c10" if fit_passed else "#d13438"
        fit_status = "✓ Pass" if fit_passed else "✗ Fail"
        ax_fit.set_title(f"Model Fit (CV R²)  —  {fit_status}",
                            color=fit_colour, fontweight="bold")
        ax_fit.set_xlabel(f"Actual  '{outcome}'", fontsize=9)
        ax_fit.set_ylabel(f"Predicted  '{outcome}'", fontsize=9)

        if r2_val is not None:
            ax_fit.annotate(
                f"CV R² = {r2_val:.3f}\nThreshold > 0.50",
                xy=(0.05, 0.95), xycoords="axes fraction",
                va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                            edgecolor=fit_colour, alpha=0.9)
            )

        ax_fit.legend(fontsize=8)
        ax_fit.grid(True, alpha=0.3)

        # ── RIGHT: Bootstrap Convergence Trace ────────────────────────────────
        n_boot = 100
        probe_points = np.percentile(X, [10, 50, 90], axis=0)

        running_avg = []
        running_max = []
        accumulated_preds = []

        np.random.seed(42)
        for _ in range(n_boot):
            idx = np.random.choice(len(y), len(y), replace=True)
            X_b, y_b = X[idx], y[idx]
            m = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
            m.fit(X_b, y_b)
            accumulated_preds.append(m.predict(probe_points))

            preds_arr  = np.array(accumulated_preds)
            stds       = np.std(preds_arr, axis=0)
            means      = np.abs(np.mean(preds_arr, axis=0))
            rel_widths = stds / (means + 1e-6)

            running_avg.append(float(np.mean(rel_widths)))
            running_max.append(float(np.max(rel_widths)))

        iters = np.arange(1, n_boot + 1)

        ax_boot.plot(iters, running_avg, color="#006abc", linewidth=2,
                        label="Running Avg CV")
        ax_boot.plot(iters, running_max, color="#1f77b4", linewidth=1.5,
                        linestyle=":", alpha=0.75, label="Running Max CV")

        # Threshold lines
        ax_boot.axhline(0.15, color="#107c10", linewidth=1.2, linestyle="--",
                        alpha=0.85, label="Avg Threshold (0.15)")
        ax_boot.axhline(0.30, color="#ffb900", linewidth=1.2, linestyle="--",
                        alpha=0.85, label="Max Threshold (0.30)")

        # Shade the converged zone
        ax_boot.fill_between(iters, 0, 0.15, color="#107c10", alpha=0.04)

        boot_colour = "#107c10" if boot_passed else "#d13438"
        boot_status = "✓ Pass" if boot_passed else "✗ Fail"
        ax_boot.set_title(f"Bootstrap Convergence  —  {boot_status}",
                            color=boot_colour, fontweight="bold")
        ax_boot.set_xlabel("Bootstrap Iteration", fontsize=9)
        ax_boot.set_ylabel("Relative Std Dev (CV)", fontsize=9)
        ax_boot.set_xlim(1, n_boot)
        ax_boot.set_ylim(bottom=0)

        if avg_cv_val is not None and max_cv_val is not None:
            ax_boot.annotate(
                f"Final Avg CV = {avg_cv_val:.3f}  (< 0.15)\n"
                f"Final Max CV = {max_cv_val:.3f}  (< 0.30)",
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


    @reactive.calc
    def outcome_stats():
        """Centrally managed outcome statistics to prevent NameErrors."""
        df = uploaded_data()
        if df is None or not input.outcome_col():
            return None

        vals = pd.to_numeric(df[input.outcome_col()], errors="coerce").dropna()
        if vals.empty:
            return None

        return {
            "min": float(vals.min()),
            "median": float(vals.median()),
            "max": float(vals.max()),
        }

    # --- REACTIVE VALUES ---
    # Store scalar results for the table (e.g., a90/95, bandwidth)
    pod_metrics = reactive.value(None)
    # Store the full curve data for downloading
    pod_export_data = reactive.value(None)
    # Trigger to refresh plots when analysis runs
    plot_trigger = reactive.value(0)


    # --- MAIN UI ---
    @reactive.effect
    @reactive.event(uploaded_data)
    def _reset_pod_on_new_data():
        pod_metrics.set(None)
        pod_export_data.set(None)

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

        # Get outcome summary stats from our shared reactive calculator
        stats = outcome_stats()
        if stats is None:
            return ui.div(ui.p("Processing data...", class_="text-muted text-center p-3"))

        ui_elements.append(
            ui.card(
                ui.card_header("Probability of Detection (PoD) Settings"),
                ui.layout_columns(
                    ui.div(
                        ui.h6("Data Selection", class_="text-muted"),
                        ui.input_select("pod_param", "Parameter of Interest", choices=inputs),
                    ),
                    ui.div(
                        ui.h6("Model Configuration", class_="text-muted"),
                        ui.input_select(
                            "pod_model_override", "Model Override",
                            choices=["Auto (Best Fit)", "Polynomial", "Kriging"],
                            selected="Auto (Best Fit)"
                        ),
                        ui.panel_conditional(
                            "input.pod_model_override === 'Polynomial'",
                            ui.input_slider("pod_poly_degree", "Polynomial Degree", min=1, max=10, value=3, step=1),
                        ),
                        ui.input_numeric(
                                "pod_threshold",
                                f"Detection Threshold ({input.outcome_col()})",
                                value=round(stats['median'], 1),
                                min=round(stats['min'], 3),
                                max=round(stats['max'], 3),
                            ),
                        ui.tags.small(
                            ui.tags.span("Min ", style="color:#6c757d;"),
                            ui.tags.span(f"{stats['min']:.3g}", style="font-weight:600;"),
                            ui.tags.span("  ·  Median ", style="color:#6c757d; margin-left:4px;"),
                            ui.tags.span(f"{stats['median']:.3g}", style="font-weight:600;"),
                            ui.tags.span("  ·  Max ", style="color:#6c757d; margin-left:4px;"),
                            ui.tags.span(f"{stats['max']:.3g}", style="font-weight:600;"),
                            class_="text-muted d-block",
                            style="margin-top:-8px; font-size:0.8em;",
                        ),
                    ),
                    col_widths=[6, 6],
                ),
                ui.input_task_button("btn_run_pod", "Run Analysis", class_="btn-primary w-100", icon=icon_svg("chart-line")),
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

        stats = outcome_stats()
        if stats is None:
            return

        threshold = input.pod_threshold()
        if not (stats['min'] <= threshold <= stats['max']):
            ui.notification_show(
                "Threshold is outside the range of the outcome variable — results may be uninformative.",
                type="warning"
            )
        # ------

        try:
            # 1. Map UI selection to backend parameters
            override_map = {
                "Auto (Best Fit)": "auto",
                "Polynomial": "polynomial",
                "Kriging": "kriging",
            }
            model_override = override_map.get(input.pod_model_override(), "auto")
            force_degree = None
            if model_override == "polynomial":
                try:
                    force_degree = int(input.pod_poly_degree())
                except Exception:
                    force_degree = None

            # 2. Run the Analysis (Generates models and curves)
            results = study.pod(
                poi_col=input.pod_param(),
                threshold=input.pod_threshold(),
                model_override=model_override,
                force_degree=force_degree,
            )

            # 3. Calculate a90/95 (Interpolate)
            val = results["a90_95"]
            a9095_str = f"{val:.3f}" if not np.isnan(val) else "Not Reached"

            # 4. Format the Mean Model string based on the new architecture
            mean_model = results["mean_model"]
            if mean_model.model_type_ == 'Polynomial':
                model_str = f"Polynomial (Degree {mean_model.model_params_})"
            else:
                model_str = "Kriging (Gaussian Process)"

            # 5. Create Metrics Dictionary for the UI

            # Unpack distribution info
            dist_name = results["dist_info"][0].capitalize()
            dist_params = results["dist_info"][1]
            formatted_params = ", ".join([f"{p:.4f}" for p in dist_params])

            # Extract the MSE for the model that was actually used.
            # Because CV always runs in full now, this is always a real value —
            # even when the user has forced a model override.
            mean_model = results["mean_model"]
            cv_scores = mean_model.cv_scores_
            if mean_model.model_type_ == 'Polynomial':
                used_key = ('Polynomial', mean_model.model_params_)
            else:
                used_key = ('Kriging', None)
            used_mse = cv_scores.get(used_key, np.nan)
            best_mse_str = f"{used_mse:.2e}" if not np.isnan(used_mse) else "N/A"

            # Calculate Sample Size
            n_samples = len(results["X"])

            metrics = {
                "Parameter of Interest": results["poi_col"],
                "Threshold": results["threshold"],
                "a90/95": a9095_str,
                "Sample Size (N)": n_samples,
                "Mean Model": model_str,
                "Model Fit (MSE)": best_mse_str,
                "Smoothing Bandwidth": f"{results['bandwidth']:.4f}",
                "Error Distribution": dist_name,
                "Distribution Parameters": formatted_params,
                "Bootstrap Iterations": results["n_boot"]
            }
            pod_metrics.set(metrics)

            # 6. Prepare Data for Download
            export_df = pd.DataFrame({
                "x_defect_size": results["X_eval"],
                "pod_mean": results["curves"]["pod"],
                "ci_lower": results["curves"]["ci_lower"],
                "ci_upper": results["curves"]["ci_upper"]
            })
            pod_export_data.set(export_df)

            # 7. Generate Plots (Visualise draws them internally)
            study.visualise(show=False)
            plot_trigger.set(plot_trigger() + 1)

        except Exception as e:
            ui.notification_show(f"Analysis Failed: {str(e)}", type="error")


# --- RESULTS DISPLAY ---
    @render.ui
    def pod_results_container():
        """
        Renders the model selection plot, side-by-side analysis plots, and the metrics table.
        """
        if pod_metrics() is None:
            return ui.div()

        return ui.div(
            # Row 1: Model Selection Plot (Full Width)
            ui.card(
                ui.card_header("Model Selection (Bias-Variance Tradeoff)"),
                ui.output_plot("plot_model_selection", height="400px"),
                full_screen=True,
                class_="mb-3"
            ),
            # Row 2: Signal Model and PoD Plots
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
                col_widths=[6, 6],
                class_="mb-3"
            ),
            # Row 3: Table and Download Actions
            ui.layout_columns(
                ui.card(
                    ui.card_header("Key Reliability Metrics"),
                    ui.output_data_frame("pod_stats_table")
                ),
                ui.card(
                    ui.card_header("Export Results"),
                    ui.div(
                        # Updated text explaining the two download options
                        ui.p("Download the key reliability metrics or the full curve data (PoD, CI) for external reporting.", class_="small text-muted mb-3"),

                        # New button for metrics
                        ui.download_button("download_pod_metrics", "Download Metrics", class_="btn-primary w-100 mb-2", icon=icon_svg("file-csv")),

                        # Updated existing button for the curve
                        ui.download_button("download_pod_results", "Download Full Curve", class_="btn-success w-100", icon=icon_svg("download"))
                    ),
                    # Changed to flex-column to stack the buttons and text nicely
                    class_="d-flex flex-column align-items-center justify-content-center text-center h-100 p-3"
                ),
                col_widths=[8, 4]
            )
        )

    @render.plot
    def plot_model_selection():
        _ = plot_trigger() # Dependency on button click
        study = current_study()
        if study and "model_selection" in study.plots:
            return study.plots["model_selection"]
        return None

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
            yield df.to_csv(index=False).encode('utf-8')

    @render.download(filename="pod_metrics_summary.csv")
    def download_pod_metrics():
        """
        Downloads the scalar key reliability metrics as a CSV.
        """
        metrics_dict = pod_metrics()
        if metrics_dict is not None:
            # Convert the stored dictionary into a DataFrame just like we do for the UI table
            df = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])
            yield df.to_csv(index=False).encode('utf-8')

#### App ####
# 1. Define the absolute path to your www directory
www_dir = Path(__file__).parent / "www"

# 2. Pass the directory to the App constructor
app = App(
    ui=app_ui,
    server=server,
    static_assets=www_dir
)
