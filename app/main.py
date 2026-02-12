import flet as ft
from digiqual.sampling import generate_lhs
# from digiqual import SimulationStudy

def main(page: ft.Page):
    page.title = "Digiqual"
    page.theme_mode = ft.ThemeMode.LIGHT

    # --- 1. THE STATE ---
    variable_inputs = [] # List of dictionaries: {"name": tf, "min": tf, "max": tf}
    variable_rows = ft.Column()

    def add_variable_row(e=None):
        # Create the objects first
        name_tf = ft.TextField(label="Variable Name", expand=2)
        min_tf = ft.TextField(label="Min", value="0", expand=1)
        max_tf = ft.TextField(label="Max", value="10", expand=1)

        # Store them in our tracker list
        row_data = {"name": name_tf, "min": min_tf, "max": max_tf}
        variable_inputs.append(row_data)

        # Create the visual row
        new_row = ft.Row([
            name_tf, min_tf, max_tf,
            ft.IconButton(
                icon=ft.Icons.DELETE_OUTLINE,
                on_click=lambda _: remove_variable_row(new_row, row_data)
            )
        ])

        variable_rows.controls.append(new_row)
        page.update()

    def remove_variable_row(row_view, row_data):
        variable_rows.controls.remove(row_view)
        variable_inputs.remove(row_data) # Remove from tracker too
        page.update()

    def run_lhs(e):
        # --- START LOADING STATE ---
        e.control.disabled = True
        # Using a white spinner for better visibility on the blue button
        e.control.content = ft.ProgressRing(width=16, height=16, stroke_width=2, color="white")
        page.update()

        ranges = {}
        has_error = False

        for entry in variable_inputs:
            # 1. Reset to "Normal" state
            entry["name"].border_color = None
            entry["name"].helper = None
            entry["min"].border_color = None
            entry["min"].helper = None
            entry["max"].border_color = None
            entry["max"].helper = None

            # 2. Validation Checks
            if not entry["name"].value.strip():
                entry["name"].border_color = "red"
                entry["name"].helper = "Name required"
                entry["name"].helper_style = ft.TextStyle(color="red")
                has_error = True

            try:
                v_min = float(entry["min"].value)
            except ValueError:
                entry["min"].border_color = "red"
                entry["min"].helper = "Invalid"
                entry["min"].helper_style = ft.TextStyle(color="red")
                has_error = True
                v_min = None

            try:
                v_max = float(entry["max"].value)
            except ValueError:
                entry["max"].border_color = "red"
                entry["max"].helper = "Invalid"
                entry["max"].helper_style = ft.TextStyle(color="red")
                has_error = True
                v_max = None

            if v_min is not None and v_max is not None:
                if v_min >= v_max:
                    entry["max"].border_color = "red"
                    entry["max"].helper = "Max must be > Min"
                    entry["max"].helper_style = ft.TextStyle(color="red")
                    has_error = True
                else:
                    ranges[entry["name"].value] = (v_min, v_max)

        if len(ranges)==0:
            has_error = True

        if has_error:
            print("Validation Failed - See Errors in Application")
        else:
            try:
                print(f"Validation Success: {ranges}")
                n_val = int(samples_input.value)

                # RUN THE ENGINE
                samples = generate_lhs(n=n_val, ranges=ranges)
                print(f"Successfully generated {len(samples)} samples!")

                # Visual feedback
                page.snack_bar = ft.SnackBar(
                    content=ft.Text(f"Generated {n_val} samples successfully!"),
                    bgcolor="green"
                )
                page.snack_bar.open = True
            except ValueError:
                samples_input.error_text = "N must be an integer"
                has_error = True

        # --- RESET BUTTON STATE ---
        e.control.disabled = False
        # Put the original Row back into the content property
        e.control.content = ft.Row(
            [ft.Icon(ft.Icons.PLAY_ARROW), ft.Text("Generate LHS Samples")],
            alignment=ft.MainAxisAlignment.CENTER,
        )
        page.update()

    # --- 2. UI STRUCTURE ---
    samples_input = ft.TextField(label="Number of Samples (n)", value="100", width=200)

    # Initialize the generator button with a Row in 'content' to allow for state swapping
    generate_btn = ft.Button(
        content=ft.Row(
            [ft.Icon(ft.Icons.PLAY_ARROW), ft.Text("Generate LHS Samples")],
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        on_click=run_lhs,
        bgcolor="blue",
        color="white"
    )

    tabs = ft.Tabs(
        selected_index=0,
        length=4,
        expand=True,
        content=ft.Column(
            expand=True,
            controls=[
                ft.TabBar(
                    tabs=[
                        ft.Tab(label="Home", icon=ft.Icons.HOME),
                        ft.Tab(label="Sample Generator", icon=ft.Icons.DATASET),
                        ft.Tab(label="Data Validation", icon=ft.Icons.FACT_CHECK),
                        ft.Tab(label="PoD Analysis", icon=ft.Icons.INSIGHTS),
                    ]
                ),
                ft.TabBarView(
                    expand=True,
                    controls=[
                        # Page 1: Home
                        ft.Container(
                            padding=40,
                            content=ft.Column([
                                ft.Text("Digiqual: NDT Statistical Toolkit", size=32, weight="bold"),
                                ft.Text("Welcome! Use the tabs above to navigate the workflow.", size=16),
                            ])
                        ),
                        # Page 2: Sample Generator
                        ft.Container(
                            padding=40,
                            content=ft.Column([
                                ft.Text("LHS Configuration", size=24, weight="bold"),
                                samples_input,
                                ft.Divider(),
                                ft.Row([
                                    ft.Text("Variables & Ranges", size=18, weight="bold"),
                                    ft.IconButton(ft.Icons.ADD_CIRCLE, on_click=add_variable_row, icon_color="blue")
                                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                                variable_rows,
                                ft.Divider(),
                                generate_btn # Using the new button variable
                            ], scroll=ft.ScrollMode.AUTO, spacing=20)
                        ),
                        # Page 3: Validation
                        ft.Container(
                            padding=40,
                            content=ft.Text("Check data and refine samples, if needed", size=20)
                        ),
                        # Page 4: PoD
                        ft.Container(
                            padding=40,
                            content=ft.Text("Calculate PoD Curves and Statistics", size=20)
                        ),
                    ],
                ),
            ],
        ),
    )

    page.add(tabs)

if __name__ == "__main__":
    ft.run(main)
