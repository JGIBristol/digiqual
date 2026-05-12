import matplotlib
matplotlib.use('Agg')

from shiny import App, ui, render, reactive
from faicons import icon_svg
import pandas as pd
import numpy as np
import asyncio
import os
from digiqual.sampling import generate_lhs
from digiqual import SimulationStudy
from pathlib import Path
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*resource_tracker.*",
    category=UserWarning
)

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

                            # Module 2: Diagnostics & Visualisation
                            ui.div(
                                ui.h5(
                                    ui.span(icon_svg("check-double"), class_="text-warning me-2"),
                                    "2. Simulation Diagnostics", class_="fw-bold mb-2"
                                ),
                                ui.p("Validate dataset integrity, inspect coverage gaps, and visualise model fit.", class_="fw-semibold mb-2"),
                                ui.tags.ul(
                                    ui.tags.li("Identify model instability or insufficient samples."),
                                    ui.tags.li("Per-variable distribution and gap inspection."),
                                    class_="mb-0 ps-3 text-muted"
                                ),
                                class_="border-start border-3 border-warning ps-3 mb-3"
                            ),
                            ui.hr(class_="my-4"),

                            # Module 3: Analysis (Physics)
                            ui.div(
                                ui.h5(
                                    ui.span(icon_svg("chart-area"), class_="text-success me-2"),
                                    "3. Model Fit & Response", class_="fw-bold mb-2"
                                ),
                                ui.p("Determine the statistical structure and physics of your parameters rapidly.", class_="fw-semibold mb-2"),
                                ui.tags.ul(
                                    ui.tags.li("Automated model selection via Cross-Validation."),
                                    ui.tags.li("Extract mathematical equations and visualize the mean response surface."),
                                    class_="mb-0 ps-3 text-muted"
                                ),
                                class_="border-start border-3 border-success ps-3 mb-3"
                            ),
                            ui.hr(class_="my-4"),

                            # Module 4: Exploration (Reliability) - NEW
                            ui.div(
                                ui.h5(
                                    ui.span(icon_svg("magnifying-glass-chart"), class_="text-info me-2"),
                                    "4. PoD Explorer", class_="fw-bold mb-2"
                                ),
                                ui.p("Real-time reliability evaluation using pre-calculated Threshold Spectrums.", class_="fw-semibold mb-2"),
                                ui.tags.ul(
                                    ui.tags.li("Instantly observe PoD changes across different detection thresholds."),
                                    ui.tags.li("Interactively slice constant parameters without model refitting."),
                                    class_="mb-0 ps-3 text-muted"
                                ),
                                class_="border-start border-3 border-info ps-3 mb-3"
                            ),
                            ui.hr(class_="my-4"),

                            # Module 5: Uncertainty Quantification (Confidence)
                            ui.div(
                                ui.h5(
                                    ui.span(icon_svg("chart-line"), class_="text-danger me-2"),
                                    "5. Uncertainty Quantification", class_="fw-bold mb-2"
                                ),
                                ui.p("Lock the structural shape and construct rigorous Probability of Detection bounds.", class_="fw-semibold mb-2"),
                                ui.tags.ul(
                                    ui.tags.li("Parallelized bootstrap resampling for robust 95% Confidence Intervals."),
                                    ui.tags.li("Monte Carlo integration to marginalize over nuisance parameters."),
                                    class_="mb-0 ps-3 text-muted"
                                ),
                                class_="border-start border-3 border-danger ps-3 mb-1"
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
                                    ui.tags.strong("Version: "), "0.19.1", ui.br(),
                                    ui.tags.strong("License: "), "MIT", ui.br(),
                                    ui.tags.strong("Author: "), "Dr. Josh Tyler", ui.br(),
                                    ui.tags.strong("Institution: "), "University of Bristol",
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
                                    # UKRI EPSRC Logo
                                    ui.div(
                                        ui.img(
                                            src="ukri-epsrc-square-logo.png",
                                            height="60px",
                                            alt="UKRI EPSRC Logo"
                                        ),
                                        ui.span("UKRI EPSRC", class_="text-muted d-block mt-2 fw-semibold"),
                                    ),
                                    # RCNDE Logo
                                    ui.div(
                                        ui.img(
                                            # Ensure the filename matches what you save in the www folder
                                            src="RCNDE-Logo-100.png",
                                            height="60px",
                                            alt="RCNDE Logo",
                                            # Adding a slight top margin if the aspect ratio makes it look misaligned next to the square EPSRC logo
                                            class_="mt-1"
                                        ),
                                        ui.span("RCNDE", class_="text-muted d-block mt-2 fw-semibold"),
                                    ),
                                    class_="bg-light border rounded p-3 d-flex justify-content-center align-items-center gap-5 text-center"
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
                # Removed the wrapping div and added h-100 to the card so it stretches to match the right column
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
                                ui.input_numeric("ui_max_gap", "Max Gap Ratio", value=0.20, step=0.01),
                                ui.input_numeric("ui_min_r2", "Min R² Score", value=0.50, step=0.01),
                                ui.input_numeric("ui_avg_cv", "Max Allowed Avg CV", value=0.15, step=0.01),
                                ui.input_numeric("ui_max_cv", "Max Allowed Peak CV", value=0.30, step=0.01),
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
    ),


#### UI - Model Fit & Response (Tab 4) ####
    ui.nav_panel(
        "Model Fit",
        ui.div(
            ui.h3("Model Fit & Response", class_="mb-4 text-center"),
            ui.output_ui("fit_warnings_ui"),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Model Configuration"),
                    ui.output_ui("model_context_note"),
                    ui.layout_columns(
                        ui.div(
                            ui.input_selectize("pod_pois", "Parameters to plot (Select 1 or 2)", choices=[], multiple=True),
                            ui.input_selectize("pod_nuisance", "Parameters to integrate over (Max 2)", choices=[], multiple=True),
                        ),
                        ui.div(
                            ui.input_select("pod_model_override", "Model Override", choices=["Auto (Best Fit)", "Polynomial", "Kriging"], selected="Auto (Best Fit)"),
                            ui.panel_conditional("input.pod_model_override === 'Polynomial'",
                                ui.input_slider("pod_poly_degree", "Polynomial Degree", min=1, max=10, value=3, step=1),
                            ),
                        ),
                        col_widths=[6, 6]
                    ),
                    ui.input_task_button("btn_run_fit", "Step 1: Fit Physics Model", class_="btn-primary w-100", icon=icon_svg("bolt")),
                ),
                col_widths=[-1,10,-1]
            ),
            ui.output_ui("fit_results_ui"),
            class_="container-fluid py-3"
        ),
        icon=icon_svg("chart-area")
    ),

#### UI - PoD Explorer (Tab 5 - NEW) ####
    ui.nav_panel(
        "PoD Explorer",
        ui.div(
            ui.h3("Reliability Explorer", class_="mb-4 text-center"),
            ui.output_ui("explorer_warnings_ui"), # Shared warning logic
            ui.layout_columns(
                ui.div(
                    ui.card(
                        ui.card_header("Real-Time Reliability Configuration"),
                        ui.p("Adjust the detection threshold to see the impact on reliability across the parameter space.", class_="small text-muted mb-3"),
                        ui.input_slider("pod_threshold_slider", "Detection Threshold", min=0, max=100, value=50, step=0.1),
                    ),
                    ui.output_ui("dynamic_slice_sliders"), # The slice sliders move here
                ),
                col_widths=[-1,10,-1]
            ),
            ui.output_ui("explorer_results_ui"),
            class_="container-fluid py-3"
        ),
        icon=icon_svg("magnifying-glass-chart")
    ),

#### UI - Uncertainty Quantification (Tab 6 - Renumbered) ####
    ui.nav_panel(
        "UQ Analysis",
        ui.div(
            ui.h3("Uncertainty Quantification (PoD)", class_="mb-4 text-center"),
            ui.output_ui("uq_warnings_ui"),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Bootstrap Configuration"),
                    ui.layout_columns(
                        ui.input_numeric("pod_n_boot", "Bootstrap Iterations", value=1000, min=10, step=100),
                        ui.div(
                            ui.input_checkbox("pod_parallel", "Enable Parallel Compute (Faster)", value=True),
                            class_="pt-4"
                        ),
                        col_widths=[6, 6]
                    ),
                    ui.input_task_button("btn_run_uq", "Run Uncertainty Quantification", class_="btn-danger w-100", icon=icon_svg("layer-group")),
                ),
                col_widths=[-1,10,-1]
            ),
            ui.output_ui("uq_results_ui"),
            class_="container-fluid py-3"
        ),
        icon=icon_svg("chart-line")
    ),
    title="DigiQual",
    id="navbar",
    fillable=True,
    # --- ADDED MATHJAX SCRIPT TO HEADER ---
    header=ui.tags.head(
        ui.tags.style(app_css),
        ui.tags.script(src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js")
    )
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
        if input.num_rows() is None or input.num_rows() < 1:
            errors.append("Please enter a valid number of samples.")

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

            # --- NEW: Header String Check & Cleaning ---
            # 1. Strip leading/trailing spaces
            df.columns = df.columns.str.strip()
            # 2. Replace any internal spaces with underscores
            df.columns = df.columns.str.replace(r'\s+', '_', regex=True)
            # 3. Remove any remaining special characters (keep only letters, numbers, and underscores)
            df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)

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

        # 1. Missing selections or conflict
        if uploaded_data() is None or not selected_inputs or conflict:
            return ui.div(ui.p("Configure selections to run diagnostics.", class_="text-muted fst-italic"))

        # 2. NEW: Ready, but the button hasn't been clicked yet!
        if diagnostic_table() is None:
            return ui.div(ui.p("Ready to run diagnostics. Click the button above.", class_="text-primary fw-bold mt-3"))

        # 3. Button was clicked, show actual results
        if validation_passed():
            return ui.div(
                ui.h5(icon_svg("circle-check"), " Validation Passed"),
                class_="alert alert-success mt-3"
            )
        else:
            return ui.div(
                ui.h5(icon_svg("triangle-exclamation"), " Issues Detected"),
                ui.p("See the visualisations below and the Remediation options for next steps.", class_="small mb-0"),
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
            # NEW: Initialize empty, then add_data with overwrite=True
            study = SimulationStudy(
            max_gap_ratio=input.ui_max_gap(), min_r2_score=input.ui_min_r2(),
            max_avg_cv=input.ui_avg_cv(), max_max_cv=input.ui_max_cv()
            )
            study.add_data(uploaded_data(), outcome_col=selected_outcome, input_cols=selected_inputs, overwrite=True)
            study_instance.set(study)
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


#### Server - Visualisation (Tab 3) ####

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
                "Max":        f"{numeric.max():.4g}"    if n_valid else "N/A",
                "Mean":       f"{numeric.mean():.4g}"   if n_valid else "N/A",
                "Std Dev":    f"{numeric.std():.4g}"    if n_valid else "N/A",

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

        # --- ADD THIS NEW SAFETY GUARD ---
        # Ensure the UI hasn't fallen behind the dataset
        input_cols_list = [c for c in input_cols_list if c in df.columns]

        if not input_cols_list:
            return None

        # --- Fetch the dynamic gap threshold from the UI ---
        try:
            thresh_gap = input.ui_max_gap()
        except Exception:
            thresh_gap = 0.20  # Safe fallback during initialization

        try:
            # --- Pass the custom threshold to the diagnostic helper ---
            coverage_res = _check_input_coverage(df, input_cols_list, thresh_gap)
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
            ax.set_ylabel("Count", fontsize=9)
            ax.grid(True, alpha=0.3)

        # Hide unused subplot slots
        for idx in range(n, nrows * ncols):
            row, c = divmod(idx, ncols)
            axes[row][c].set_visible(False)

        fig.tight_layout(pad=2.0, h_pad=3.0)
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

        # --- NEW: Fetch dynamic thresholds from the UI ---
        thresh_r2 = input.ui_min_r2()
        thresh_avg = input.ui_avg_cv()
        thresh_max = input.ui_max_cv()

        # Pull pass/fail status from existing diagnostic_table
        fit_row  = diag[diag["Test"] == "Model Fit (CV)"]
        boot_row = diag[diag["Test"] == "Bootstrap Convergence"]

        r2_val      = float(fit_row["Value"].values[0])      if not fit_row.empty  else None
        fit_passed  = bool(fit_row["Pass"].values[0])         if not fit_row.empty  else True
        boot_passed = bool(boot_row["Pass"].all())            if not boot_row.empty else True

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
                # --- CHANGED: Dynamic annotation ---
                f"CV R² = {r2_val:.3f}\nThreshold > {thresh_r2:.2f}",
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

        rng = np.random.default_rng(42)
        for _ in range(n_boot):
            idx = rng.choice(len(y), len(y), replace=True)
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

        # --- CHANGED: Dynamic Threshold lines ---
        ax_boot.axhline(thresh_avg, color="#107c10", linewidth=1.2, linestyle="--",
                        alpha=0.85, label=f"Avg Threshold ({thresh_avg:.2f})")
        ax_boot.axhline(thresh_max, color="#ffb900", linewidth=1.2, linestyle="--",
                        alpha=0.85, label=f"Max Threshold ({thresh_max:.2f})")

        # --- CHANGED: Dynamic Shading ---
        ax_boot.fill_between(iters, 0, thresh_avg, color="#107c10", alpha=0.04)

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
                # --- CHANGED: Dynamic Annotations ---
                f"Final Avg CV = {avg_cv_val:.3f}  (< {thresh_avg:.2f})\n"
                f"Final Max CV = {max_cv_val:.3f}  (< {thresh_max:.2f})",
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

#### Server - Model Fit & UQ Logic (Tabs 4 & 5) ####

    # --- SHARED CALCULATOR ---
    study_instance = reactive.value(None)
    locked_model_type = reactive.value(None)
    locked_model_degree = reactive.value(None)

    fit_metrics = reactive.value(None)
    uq_metrics = reactive.value(None)
    pod_export_data = reactive.value(None)
    plot_trigger_fit = reactive.value(0)
    plot_trigger_uq = reactive.value(0)

    @reactive.effect
    @reactive.event(uploaded_data, input.input_cols, input.outcome_col)
    def update_study():
        df = uploaded_data()
        selected_inputs = list(input.input_cols())
        selected_outcome = input.outcome_col()

        # Guard 1: Missing basic selections
        if df is None or not selected_inputs or not selected_outcome or (selected_outcome in selected_inputs):
            study_instance.set(None)
            return

        # Guard 2: Reactive Race Condition Guard
        # Ensure the UI selections actually exist in the CURRENT dataset before processing.
        required_cols = selected_inputs + [selected_outcome]
        if not all(col in df.columns for col in required_cols):
            # The UI hasn't caught up to the newly uploaded data yet. Safely abort.
            study_instance.set(None)
            return

        # Initialize empty, then add_data with overwrite=True
        study = SimulationStudy(
            max_gap_ratio=input.ui_max_gap(), min_r2_score=input.ui_min_r2(),
            max_avg_cv=input.ui_avg_cv(), max_max_cv=input.ui_max_cv()
        )
        study.add_data(df, outcome_col=selected_outcome, input_cols=selected_inputs, overwrite=True)
        study_instance.set(study)

    # -----------------------------------------------------------------
    # TWO-TIER RESET LOGIC
    # -----------------------------------------------------------------
    @reactive.effect
    @reactive.event(
        uploaded_data, input.input_cols, input.outcome_col,
        input.ui_max_gap, input.ui_min_r2, input.ui_avg_cv, input.ui_max_cv
    )
    def reset_core_data_state():
        """TIER 1: Wipes EVERYTHING when the physical data or core definitions change."""
        diagnostic_table.set(None)
        validation_passed.set(False)
        new_samples.set(None)

        fit_metrics.set(None)
        uq_metrics.set(None)
        locked_model_type.set(None)
        pod_export_data.set(None)

    @reactive.effect
    @reactive.event(input.pod_pois, input.pod_nuisance)
    def reset_analysis_results():
        """TIER 2: Clears just the downstream math when plot targets change, keeping the core study intact."""
        fit_metrics.set(None)
        uq_metrics.set(None)
        locked_model_type.set(None)
        pod_export_data.set(None)
    @reactive.calc
    def current_study(): return study_instance()

    @reactive.calc
    def outcome_stats():
        df = uploaded_data()
        if df is None or not input.outcome_col():
            return None
        vals = pd.to_numeric(df[input.outcome_col()], errors="coerce").dropna()
        if vals.empty:
            return None
        return {"min": float(vals.min()), "median": float(vals.median()), "max": float(vals.max())}

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



    @reactive.effect
    @reactive.event(uploaded_data, input.outcome_col)
    def update_pod_ui_choices():
        study = current_study()
        if study is not None:
            # We don't need to check df columns anymore, the study knows!
            ui.update_selectize("pod_pois", choices=study.inputs)
            ui.update_selectize("pod_nuisance", choices=study.inputs)

            # --- THE FIX: Restrict Kriging for large datasets ---
            n_samples = len(study.data)
            if n_samples > 1000:
                ui.update_select(
                    "pod_model_override",
                    choices=["Auto (Best Fit)", "Polynomial"],
                    selected="Auto (Best Fit)"
                )
            else:
                ui.update_select(
                    "pod_model_override",
                    choices=["Auto (Best Fit)", "Polynomial", "Kriging"],
                    selected="Auto (Best Fit)"
                )

            if study.outcome:
                # Ask the package for the min/max/median instantly
                summary = study.get_data_summary(study.outcome)
                if summary["median"] is not None:
                    ui.update_slider(
                        "pod_threshold_slider",
                        label=f"Detection Threshold ({study.outcome})",
                        value=round(summary["median"], 2),
                        min=round(summary["min"], 2),
                        max=round(summary["max"], 2)
                    )

    @render.ui
    def leftover_params_note():
        study = current_study()
        if study is None or not study.inputs:
            return ui.div() # Hide if nothing is configured yet

        selected_pois = list(input.pod_pois()) if input.pod_pois() else []
        selected_nuis = list(input.pod_nuisance()) if input.pod_nuisance() else []

        # Instantly get unassigned parameters from the package
        leftovers = study.get_unassigned_parameters(selected_pois, selected_nuis)

        if leftovers:
            leftovers_str = ", ".join([f"'{c}'" for c in leftovers])
            return ui.div(
                ui.span(icon_svg("sliders"), class_="text-primary me-2", style="font-size: 1.1em;"),
                ui.div(
                    ui.tags.strong("Interactive Sliders: ", class_="d-block"),
                    ui.span(f"Unassigned parameters ({leftovers_str}) will appear as interactive sliders after fitting to let you slice the surface.", class_="small text-muted")
                ),
                class_="mt-3 p-2 bg-light border rounded d-flex align-items-start shadow-sm"
            )
        else:
            return ui.div(
                ui.span(icon_svg("check"), class_="text-success me-2", style="font-size: 1.1em;"),
                ui.div(
                    ui.tags.strong("All Parameters Assigned", class_="d-block text-success"),
                    ui.span("No slice sliders will be generated.", class_="small text-muted")
                ),
                class_="mt-3 p-2 bg-light border rounded d-flex align-items-start shadow-sm"
            )


    @render.ui
    def dynamic_slice_sliders():
        study = current_study()

        # LAZY RENDER: Only show if a model has been successfully fitted!
        if study is None or fit_metrics() is None:
            return ui.div()

        selected_pois = list(input.pod_pois()) if input.pod_pois() else []
        selected_nuis = list(input.pod_nuisance()) if input.pod_nuisance() else []

        leftovers = study.get_unassigned_parameters(selected_pois, selected_nuis)

        if not leftovers:
            return ui.div()

        sliders = []
        for col in leftovers:
            # Get clean bounds from the package
            summary = study.get_data_summary(col)
            if summary["min"] is None:
                continue

            col_min = round(summary["min"], 3)
            col_max = round(summary["max"], 3)
            col_med = round(summary["median"], 3)

            step_size = round((col_max - col_min) / 100, 3)
            if step_size == 0:
                step_size = 0.001

            sliders.append(
                ui.input_slider(f"slice_{col}", col, min=col_min, max=col_max, value=col_med, step=step_size)
            )

        return ui.card(
            ui.card_header(
                ui.span(icon_svg("sliders"), class_="text-primary me-2"),
                "Real-Time Slice Explorer"
            ),
            ui.p(
                "Adjust these constant parameters to instantly update the surface plot below. "
                "There is no need to re-fit the model.",
                class_="small text-muted mb-3"
            ),
            ui.layout_columns(*sliders, col_widths=6),
            class_="mt-3 shadow-sm"
        )

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


    # ─────────────────────────────────────────────────────────────────
    # TAB 4: MODEL FIT LOGIC
    # ─────────────────────────────────────────────────────────────────
    @render.ui
    def fit_warnings_ui():
        if uploaded_data() is None:
            return ui.layout_columns(ui.div(ui.p("Please upload data in the 'Simulation Diagnostics' tab.", class_="text-center p-4 text-muted bg-light rounded border")), col_widths=[-1,10,-1])

        if not validation_passed():
            return ui.layout_columns(ui.div(ui.h5(icon_svg("triangle-exclamation"), " Caution: Validation Issues"),
                       ui.p("The diagnostic tests found potential issues. Results may be unreliable.", class_="mb-0"), class_="alert alert-warning shadow-sm"), col_widths=[-1,10,-1])
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
                # --- CHANGED: Removed the Export card and let Fit Statistics take full width ---
                ui.card(
                    ui.card_header("Fit Statistics"),
                    ui.output_ui("mathjax_equation_ui"),
                    ui.output_data_frame("fit_stats_table")
                )
            ),
            col_widths=[-1,10,-1]
        )

    @render.ui
    def explorer_warnings_ui():
        if fit_metrics() is None:
            return ui.layout_columns(ui.div(ui.p("Please fit a model in the 'Model Fit' tab first.", class_="text-center p-4 text-muted bg-light rounded border")), col_widths=[-1,10,-1])
        return ui.div()

    @render.ui
    def explorer_results_ui():
        if fit_metrics() is None:
            return ui.div()

        study = current_study()
        is_multi_dim = len(study.pod_results.get("poi_cols", [])) > 1

        return ui.layout_columns(
            # --- NEW: Added the Signal Model Plot to the left ---
            ui.card(
                ui.card_header(f"{input.outcome_col()} Surface" if is_multi_dim else "Model Fit"),
                ui.output_plot("plot_signal_explorer", height="450px"),
                full_screen=True, class_="mt-3"
            ),
            # --- The existing PoD Plot on the right ---
            ui.card(
                ui.card_header("PoD Surface Heatmap" if is_multi_dim else "PoD Reliability Curve"),
                ui.output_plot("plot_explorer", height="450px"),
                full_screen=True, class_="mt-3"
            ),
            col_widths=[6, 6]
        )

    @render.plot
    def plot_explorer():
        """Dedicated PoD plot renderer for Tab 5."""
        _ = plot_trigger_fit() # Shares the same fast-trigger as Tab 4
        study = current_study()
        return study.plots["pod_curve"] if study and "pod_curve" in study.plots else None

    @render.plot
    def plot_signal_explorer():
        """Dedicated Signal Model plot renderer for Tab 5."""
        _ = plot_trigger_fit() # Triggers instantly when sliders move
        study = current_study()
        return study.plots["signal_model"] if study and "signal_model" in study.plots else None

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

        # --- TAB 4 TIME ESTIMATION HEURISTIC ---
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
            # 1. Run the standard fit (n_boot=0) to establish Layer 1, 2, 3
            results = study.pod(
                poi_col=poi_cols, threshold=float(study.get_data_summary(study.outcome)["median"]),
                nuisance_col=nuisance_cols, slice_values=slice_values,
                model_override=model_override, force_degree=force_degree, n_boot=0
            )

            # 2. NEW: Trigger the Threshold Spectrum calculation (Layer 4)
            ui.notification_show("Generating Instant Threshold Spectrum...", id="spec_toast", duration=None)
            study.compute_pod_spectrum(
                poi_col=poi_cols, nuisance_col=nuisance_cols,
                slice_values=slice_values, n_threshold_points=100,
                model_override=model_override, force_degree=force_degree
            )
            ui.notification_remove("spec_toast")

            # 3. Update the Tab 5 slider range based on actual data
            summary = study.get_data_summary(study.outcome)
            ui.update_slider("pod_threshold_slider",
                             min=round(summary["min"], 2),
                             max=round(summary["max"], 2),
                             value=round(summary["median"], 2))

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
                "Smoothing Bandwidth": f"{results['bandwidth']:.4f}",
                "Error Distribution": f"{dist_name} (Params: {tuple(dist_params)})",
            }
            fit_metrics.set(metrics)

            # --- EXPORT PRELIMINARY DATA ---
            export_data = {}
            if len(poi_cols) == 1:
                export_data["x_defect_size"] = results["X_eval"].flatten()
            else:
                for i, col in enumerate(poi_cols):
                    export_data[col] = results["X_eval"][:, i]
            export_data["pod_mean"] = results["curves"]["pod"]
            export_data["ci_lower"] = np.nan
            export_data["ci_upper"] = np.nan
            pod_export_data.set(pd.DataFrame(export_data))

            study.visualise(show=False)
            plot_trigger_fit.set(plot_trigger_fit() + 1)
            ui.notification_show("Model Fit Complete. Proceed to Uncertainty Quantification.", type="success")

        except Exception as e:
            ui.notification_show(f"Fit Failed: {str(e)}", type="error")
        finally:
            ui.notification_remove("fit_toast")


    @reactive.effect
    def realtime_slice_update():
        """Listens to the dynamic sliders and instantly updates the plots without refitting."""
        study = current_study()
        if study is None or fit_metrics() is None:
            return

        poi_cols = study.pod_results.get("poi_cols", [])
        nuis_cols = study.pod_results.get("nuisance_cols", [])

        leftovers = study.get_unassigned_parameters(poi_cols, nuis_cols)
        if not leftovers:
            return

        slice_values = {}
        for col in leftovers:
            try:
                val = input[f"slice_{col}"]()
                slice_values[col] = val if val is not None else study.get_data_summary(col)["median"]
            except Exception:
                slice_values[col] = study.get_data_summary(col)["median"]

        if not slice_values:
            return

        # 1. Update the math instantly using the Layer 3 Cache
        study.update_slice(slice_values)

        # 2. Re-generate the visualisations in memory
        study.visualise(show=False)

        # 3. Trigger the UI to redraw (Isolated to prevent infinite loops!)
        with reactive.isolate():
            current_trigger = plot_trigger_fit()
        plot_trigger_fit.set(current_trigger + 1)

        # 4. Clear UQ bounds
        uq_metrics.set(None)


    @reactive.effect
    @reactive.event(input.pod_threshold_slider)
    def realtime_threshold_update():
        """Instantly updates plots using Layer 4 Spectrum Interpolation."""
        study = current_study()
        if study is None or fit_metrics() is None or not study.pod_results:
            return

        # 1. Grab the currently active slice values so they don't reset!
        current_slices = study.pod_results.get("slice_values", {})

        # 2. Lock the model so it doesn't attempt to run auto-CV again
        current_model = study.pod_results["mean_model"]
        override = "polynomial" if current_model.model_type_ == "Polynomial" else "kriging"
        degree = current_model.model_params_ if current_model.model_type_ == "Polynomial" else None

        # 3. Re-run .pod() with the new threshold
        # Since Layer 4 is primed, this takes microseconds.
        study.pod(
            poi_col=list(input.pod_pois()),
            threshold=input.pod_threshold_slider(),
            nuisance_col=list(input.pod_nuisance()),
            slice_values=current_slices,
            model_override=override,
            force_degree=degree,
            n_boot=0
        )

        study.visualise(show=False)

        with reactive.isolate():
            current_trigger = plot_trigger_fit()
        plot_trigger_fit.set(current_trigger + 1)


    # ─────────────────────────────────────────────────────────────────
    # TAB 5: UQ LOGIC
    # ─────────────────────────────────────────────────────────────────
    @render.ui
    def uq_warnings_ui():
        if locked_model_type() is None:
            return ui.layout_columns(ui.div(ui.p("Please fit a model in the 'Model Fit' tab first.", class_="text-center p-4 text-muted bg-light rounded border")), col_widths=[-1,10,-1])

        model_str = f"Polynomial (Degree {locked_model_degree()})" if locked_model_type() == 'Polynomial' else "Kriging"
        return ui.layout_columns(
            ui.div(
                ui.h5(icon_svg("lock"), f" Structural Shape Locked: {model_str}"),
                ui.p("Note: The confidence bounds generated here reflect parameter uncertainty assuming the mathematical shape chosen in Tab 4 is correct. If you add more data later, the 'Auto' selector may choose a different shape, causing new curves to fall outside these bounds.", class_="small mb-0"),
                class_="alert alert-info shadow-sm"
            ),
            col_widths=[-1,10,-1]
        )

    @render.ui
    def uq_results_ui():
        if uq_metrics() is None:
            return ui.div()

        study = current_study()
        is_multi_dim = len(study.pod_results.get("poi_cols", [])) > 1

        return ui.layout_columns(
            ui.div(
                # --- NEW SIDE-BY-SIDE PLOT LAYOUT ---
                ui.layout_columns(
                    ui.card(
                        ui.card_header(f"{input.outcome_col()} Surface" if is_multi_dim else "Model Fit"),
                        ui.output_plot("plot_signal_uq", height="400px"),
                        full_screen=True, class_="mt-3"
                    ),
                    ui.card(
                        ui.card_header("PoD Surface Heatmap" if is_multi_dim else "PoD Curve (95% CI)"),
                        ui.output_plot("plot_curve", height="400px"),
                        full_screen=True, class_="mt-3"
                    ),
                    col_widths=[-1, 5, 5, -1]
                ),
                # --- METRICS & EXPORT ---
                ui.layout_columns(
                    ui.card(ui.card_header("Reliability Metrics"), ui.output_data_frame("uq_stats_table")),
                    ui.card(
                        ui.card_header("Export Results"),
                        ui.p("Download a comprehensive Excel workbook containing all configuration metrics and full curve data across separate tabs.", class_="small text-muted mb-3"),
                        ui.download_button("download_excel_report", "Download Excel Report", class_="btn-success w-100", icon=icon_svg("file-excel")),
                        class_="text-center"
                    ),
                    col_widths=[8, 4]
                )
            ),
            col_widths=[-1,10,-1]
        )

    @reactive.effect
    @reactive.event(input.btn_run_uq)
    async def compute_uq():
        uq_metrics.set(None)
        study = current_study()
        if study is None:
            return

        poi_cols, nuisance_cols = list(input.pod_pois()), list(input.pod_nuisance())
        actual_cores = max((os.cpu_count() or 1) - 1, 1) if input.pod_parallel() else 1

        slice_values = {}
        leftovers = [c for c in study.inputs if c not in poi_cols and c not in nuisance_cols]
        for col in leftovers:
            try:
                slice_values[col] = input[f"slice_{col}"]()
            except Exception:
                pass

        # --- TAB 5 TIME ESTIMATION HEURISTIC ---
        # Get the currently locked model type
        override = "polynomial" if locked_model_type() == "Polynomial" else "kriging"

        # Call the package!
        est_seconds = study.estimate_compute_time(
            model_type=override,
            n_boot=input.pod_n_boot(),
            n_nuisances=len(nuisance_cols),
            n_jobs=-1 if input.pod_parallel() else 1
        )

        time_str = f"~{max(1, int(est_seconds))} seconds" if est_seconds < 90 else f"~{int(est_seconds / 60)} minutes"

        ui.notification_show(f"Running Bootstrap on {actual_cores} core(s). Estimated time: {time_str}...", id="uq_toast", duration=None, type="message")
        await asyncio.sleep(0.1)

        try:
            override = "polynomial" if locked_model_type() == "Polynomial" else "kriging"
            results = study.pod(
                poi_col=poi_cols,
                threshold=input.pod_threshold_slider(),
                nuisance_col=nuisance_cols,
                slice_values=slice_values,
                model_override=override, force_degree=locked_model_degree(),
                n_boot=input.pod_n_boot(), n_jobs=-1 if input.pod_parallel() else 1
            )

            val = results["a90_95"]
            a9095_str = "N/A (Surface)" if len(poi_cols) > 1 else (f"{val:.3f}" if not np.isnan(val) else "Not Reached")

            # --- EXPANDED UQ METRICS ---
            metrics = {
                "Detection Threshold": results["threshold"],
                "Bootstrap Iterations": results["n_boot"],
                "a90/95 Reliability Index": a9095_str
            }
            uq_metrics.set(metrics)

            export_data = {}
            if len(poi_cols) == 1:
                export_data["x_defect_size"] = results["X_eval"].flatten()
            else:
                for i, col in enumerate(poi_cols):
                    export_data[col] = results["X_eval"][:, i]
            export_data["pod_mean"] = results["curves"]["pod"]
            export_data["ci_lower"] = results["curves"]["ci_lower"]
            export_data["ci_upper"] = results["curves"]["ci_upper"]
            pod_export_data.set(pd.DataFrame(export_data))

            study.visualise(show=False)
            plot_trigger_uq.set(plot_trigger_uq() + 1)
            ui.notification_show("Uncertainty Quantification Complete!", type="success")

        except Exception as e:
            ui.notification_show(f"UQ Failed: {str(e)}", type="error")
        finally:
            ui.notification_remove("uq_toast")

# --- RESULTS DISPLAY ---

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
        _ = plot_trigger_fit()
        study = current_study()
        return study.plots["model_selection"] if study and "model_selection" in study.plots else None

    @render.plot
    def plot_signal():
        _ = plot_trigger_fit()
        study = current_study()
        return study.plots["signal_model"] if study and "signal_model" in study.plots else None

    @render.plot
    def plot_signal_uq():
        _ = plot_trigger_uq()
        study = current_study()
        return study.plots["signal_model"] if study and "signal_model" in study.plots else None

    @render.plot
    def plot_curve():
        _ = plot_trigger_uq()
        study = current_study()
        return study.plots["pod_curve"] if study and "pod_curve" in study.plots else None

    @render.data_frame
    def fit_stats_table():
        data = fit_metrics()
        if data is None:
            return None
        # Hide the equations from the generic UI table
        display_data = {k: v for k, v in data.items() if k not in ["LaTeX Equation", "Model Equation"]}
        return render.DataGrid(pd.DataFrame(list(display_data.items()), columns=["Metric", "Value"]), width="100%", filters=False)

    @render.data_frame
    def uq_stats_table():
        data = uq_metrics()
        return render.DataGrid(pd.DataFrame(list(data.items()), columns=["Metric", "Value"]), width="100%", filters=False) if data else None

    @render.download(filename="digiqual_pod_report.xlsx")
    def download_excel_report():
        """
        Generates an in-memory Excel workbook containing multiple tabs
        for both the configuration metrics and the raw curve data.
        """
        import io

        output = io.BytesIO()

        # Use pandas ExcelWriter to create multiple sheets
        with pd.ExcelWriter(output, engine='openpyxl') as writer:

            # --- TAB 1: Summary Metrics ---
            combined_metrics = {}
            fit_data = fit_metrics()
            if fit_data is not None:
                # Strip out the LaTeX, but keep "Model Equation" (the plain text)
                excel_data = {k: v for k, v in fit_data.items() if k != "LaTeX Equation"}
                combined_metrics.update(excel_data) # Note: For download_excel_fit, this line is df_metrics = pd.DataFrame(list(excel_data.items())...)
            uq_data = uq_metrics()
            if uq_data is not None:
                combined_metrics.update(uq_data)

            if combined_metrics:
                df_metrics = pd.DataFrame(list(combined_metrics.items()), columns=["Metric", "Value"])
                df_metrics.to_excel(writer, sheet_name="Summary Metrics", index=False)

            # --- TAB 2: Full Curve Data ---
            df_curve = pod_export_data()
            if df_curve is not None:
                df_curve.to_excel(writer, sheet_name="PoD Curve Data", index=False)

        # Return the bytes to trigger the download
        yield output.getvalue()

#### App ####
# 1. Define the absolute path to your www directory
www_dir = Path(__file__).parent / "www"

# 2. Pass the directory to the App constructor
app = App(
    ui=app_ui,
    server=server,
    static_assets=www_dir
)
