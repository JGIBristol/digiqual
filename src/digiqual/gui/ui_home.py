"""Home page: overview of the workflow modules and project information."""

from faicons import icon_svg
from shiny import ui

from digiqual import __version__

from .components import TABS


def _module_cards():
    """One entry per workflow tab, numbered in tab order."""
    cards = []
    for i, tab in enumerate(TABS, start=1):
        if i > 1:
            cards.append(ui.hr(class_="my-4"))
        cards.append(ui.div(
            ui.h5(ui.span(icon_svg(tab.icon), class_=f"text-{tab.colour} me-2"), f"{i}. {tab.title}",
                  class_="fw-bold mb-2"),
            ui.p(tab.lead, class_="fw-semibold mb-2"),
            ui.tags.ul(*[ui.tags.li(b) for b in tab.bullets], class_="mb-0 ps-3 text-muted"),
            class_=f"border-start border-3 border-{tab.colour} ps-3 mb-1"
        ))
    return cards


home_panel = ui.nav_panel(
    "Home",
    # Scrolling wrapper
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
                    ui.div(*_module_cards(), class_="p-3")
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
                                ui.tags.strong("Version: "), __version__, ui.br(),
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
                        ui.hr(class_="my-4"),

                        # Disclaimer & Privacy
                        ui.div(
                            ui.h5("Disclaimer & Data Privacy", class_="fw-bold mb-2 text-warning"),
                            ui.p(
                                "This software is provided 'as is', without warranty of any kind. "
                                "In no event shall the authors be liable for any claim or damages. All processing is performed locally. "
                                "This application does not implement data persistence, nor does it facilitate the outbound transmission "
                                "of user-supplied datasets to external servers.",
                                class_="text-muted small mb-0"
                            ),
                            class_="p-3 rounded border-start border-3 border-warning"
                        ),
                        class_="p-3"
                    )
                )
            ),
            col_widths=[-1,5,5,-1]
        ),

        # This class allows just this tab to scroll while respecting your global fillable=True
        class_="overflow-auto h-100 px-3"
    ),
    icon=icon_svg("house")
)
