"""Experimental Design tab: build a Latin Hypercube sample from variable ranges."""

from faicons import icon_svg
from shiny import reactive, render, ui

from digiqual.sampling import generate_lhs

from .state import AppState

panel = ui.nav_panel(
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
)


def register(input, output, session, state: AppState):
    final_generated_df = state.final_generated_df
    active_row_ids = reactive.Value([0])
    next_id = reactive.Value(1)

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
        @reactive.event(input[f"var_name_{idx}"], input[f"var_min_{idx}"], input[f"var_max_{idx}"],
                        ignore_init=True)
        def _clear_on_edit():
            # Editing a variable makes any generated design out of date
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

        # Loop over active IDs only
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
        return ui.download_button("download_lhs", "Download CSV", class_="btn-success w-100", icon=icon_svg("download"))

    @render.download(filename="generated_sample.csv")
    def download_lhs():
        df = final_generated_df()
        if df is not None:
            yield df.to_csv(index=False).encode('utf-8')

    @reactive.effect
    @reactive.event(input.num_rows)
    def on_param_change():
        final_generated_df.set(None)
