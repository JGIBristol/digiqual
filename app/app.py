from shiny import App, ui, render, reactive
from faicons import icon_svg  # <--- NEW IMPORT
import pandas as pd
from digiqual.sampling import generate_lhs
import shinyswatch

# UI Definition
app_ui = ui.page_navbar(
    ui.nav_panel(
        "Home",
        ui.layout_columns(
            ui.card(
                ui.h2("Welcome to DigiQual"),
                ui.p("This application provides a workflow framework for MAPoD."),
                ui.hr(),
                ui.h4("Available Tools"),
                ui.div(
                    ui.tags.ul(
                        ui.tags.li(
                            ui.tags.strong("Sample Generator: "),
                            "Design experimental frameworks using Latin Hypercube Sampling."
                        ),
                        ui.tags.li(
                            ui.tags.strong("Simulation Diagnostics: "),
                            "Upload datasets to check for integrity and sampling sufficiency."
                        ),
                        ui.tags.li(
                            ui.tags.strong("PoD Analysis: "),
                            "Construct Probability of Detection Curves."
                        ),
                        class_="lead fs-6"
                    )
                ),
                class_="p-4"
            )
        ),
        icon=icon_svg("house")
    ),
    ui.nav_panel(
        "Sample Generator",
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
                # -- NEW COLUMN SELECTORS --
                ui.input_selectize("input_cols", "Select Input Variables", choices=[], multiple=True),
                ui.input_selectize("outcome_col", "Select Outcome Variable", choices=[], multiple=False),
                ui.hr(),
                # --------------------------
                ui.output_ui("validation_status"),
                ui.output_ui("download_corrected_ui", class_="mt-3")
            ),
            ui.card(
                ui.card_header("Uploaded Data Preview"),
                ui.output_data_frame("preview_uploaded"),
                height="300px"
            ),
            ui.card(
                ui.card_header("Validation Report"),
                ui.output_text_verbatim("validation_results"),
                class_="font-monospace"
            ),
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
    theme=shinyswatch.theme.flatly()
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

    @output
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
    validation_passed = reactive.value(True)
    corrected_data = reactive.value(None)

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
    def _():
        """
        Update column selectors when data is uploaded.
        """
        df = uploaded_data()
        if df is None:
            # Reset if no data
            ui.update_selectize("input_cols", choices=[])
            ui.update_selectize("outcome_col", choices=[])
            return

        cols = list(df.columns)

        # Smart defaults: Try to guess outcome if last column, otherwise empty
        # Update Input Columns (Default: All except last)
        ui.update_selectize(
            "input_cols",
            choices=cols,
            selected=cols[:-1] if len(cols) > 1 else cols
        )

        # Update Outcome Column (Default: Last column)
        ui.update_selectize(
            "outcome_col",
            choices=cols,
            selected=cols[-1] if len(cols) > 0 else None
        )

        has_missing = df.isnull().any().any()
        has_duplicates = df.duplicated().any()

        if has_missing or has_duplicates:
            validation_passed.set(False)
            corrected_df = df.copy().fillna(0).drop_duplicates()
            corrected_data.set(corrected_df)
        else:
            validation_passed.set(True)
            corrected_data.set(None)

    @output
    @render.data_frame
    def preview_uploaded():
        df = uploaded_data()
        if df is not None:
            return render.DataGrid(df.head(20))
        return render.DataGrid(pd.DataFrame())

    @output
    @render.ui
    def validation_status():
        df = uploaded_data()
        if df is None:
            return ui.div(ui.p("Upload a CSV to begin.", class_="text-muted fst-italic"))

        if validation_passed():
            return ui.div(
                ui.h5(icon_svg("circle-check"), " Validation Passed", class_="text-success"), # <--- CHANGED
                class_="alert alert-success mt-3"
            )
        else:
            return ui.div(
                ui.h5(icon_svg("triangle-exclamation"), " Issues Detected", class_="text-danger"), # <--- CHANGED
                ui.p("Missing values or duplicates found.", class_="small mb-0"),
                class_="alert alert-danger mt-3"
            )

    @output
    @render.text
    def validation_results():
        df = uploaded_data()
        if df is None:
            return "Waiting for file..."

        results = []
        results.append(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} cols")
        results.append("-" * 20)
        results.append(f"Missing Values:  {df.isnull().sum().sum()}")
        results.append(f"Duplicate Rows:  {df.duplicated().sum()}")
        return "\n".join(results)

    @output
    @render.ui
    def download_corrected_ui():
        if not validation_passed() and corrected_data() is not None:
            return ui.download_button("download_corrected", "Download Corrected CSV", class_="btn-danger w-100", icon=icon_svg("file-csv")) # <--- CHANGED
        return ui.div()

    @render.download(filename="corrected_data.csv")
    def download_corrected():
        df = corrected_data()
        if df is not None:
            yield df.to_csv(index=False)


    # ==================== TAB 3: DATA ANALYZER ====================

    @output
    @render.ui
    def analysis_output():
        df = uploaded_data()

        if df is None:
            return ui.div(
                ui.div(
                    ui.h4("No Data Available", class_="text-muted"),
                    ui.p("Please upload valid data in the Validator tab first."),
                    class_="text-center p-5 bg-light border rounded"
                )
            )

        analysis_df = corrected_data() if not validation_passed() else df
        numeric_cols = analysis_df.select_dtypes(include=['number']).columns

        if len(numeric_cols) == 0:
            return ui.p("No numeric columns found in the uploaded data.")

        # Calculate summary statistics
        summary_df = analysis_df[numeric_cols].describe().reset_index()
        summary_df = summary_df.rename(columns={"index": "Statistic"})

        return ui.div(
            ui.h4("Summary Statistics", class_="mb-3"),
            ui.p(f"Analyzing {len(analysis_df)} rows and {len(numeric_cols)} numeric columns."),
            # Render the summary table as a nice DataGrid
            ui.output_data_frame("summary_grid"),
        )

    @output
    @render.data_frame
    def summary_grid():
        df = uploaded_data()
        if df is None:
            return None

        analysis_df = corrected_data() if not validation_passed() else df
        numeric_cols = analysis_df.select_dtypes(include=['number']).columns

        if len(numeric_cols) == 0:
            return None

        val = analysis_df[numeric_cols].describe().reset_index()
        val = val.rename(columns={"index": "Statistic"})

        return render.DataGrid(val.round(4), width="100%", filters=False)

# Create the app object
app = App(app_ui, server)
