"""About page: version, links, methodology papers, funders and the disclaimer.

Styled like the start page: each card has a coloured top border and icon tile, using
the stage colours.
"""

from faicons import icon_svg
from shiny import ui

from digiqual import __version__

from .components import STAGE, accent_style, hero, icon_tile

PAPERS = (
    ("Malkiel, N., Croxford, A. J., & Wilcox, P. D. (2025). ",
     "A generalized method for the reliability assessment of safety–critical inspection. ",
     "Proceedings of the Royal Society A, 481: 20240654.", "https://doi.org/10.1098/rspa.2024.0654"),
    ("Malkiel, N., Croxford, A. J., & Wilcox, P. D. (2026). ",
     "A comprehensive investigation of flexible and multi-dimensional simulation-based PoD analysis. ",
     "NDT & E International, 159: 103596.", "https://doi.org/10.1016/j.ndteint.2025.103596"),
)

LINKS = (
    ("github", "GitHub repository", "https://github.com/JGIBristol/digiqual"),
    ("book", "Documentation", "https://jgibristol.github.io/digiqual/"),
    ("python", "PyPI package", "https://pypi.org/project/digiqual/"),
)


def _card(title: str, icon: str, colour: str, *body):
    return ui.div(
        icon_tile(icon),
        ui.h5(title, class_="mb-3"),
        *body,
        class_="dq-info-card card p-4 h-100", style=accent_style(colour),
    )


def _fact(label, value):
    return ui.div(ui.span(label, class_="text-muted"), ui.span(value, class_="fw-semibold"),
                  class_="d-flex justify-content-between border-bottom py-1")


panel = ui.nav_panel(
    "About",
    ui.div(
        hero("About DigiQual", "Simulation-based reliability assessment for non-destructive testing"),
        ui.layout_columns(
            ui.div(
                ui.layout_columns(
                    _card("DigiQual", "circle-info", STAGE["design"].colour,
                          _fact("Version", __version__),
                          _fact("Licence", "MIT"),
                          _fact("Author", "Dr. Josh Tyler"),
                          _fact("Institution", "University of Bristol")),
                    _card("Resources", "link", STAGE["explorer"].colour,
                          *[ui.a(icon_svg(icon, margin_left="0", margin_right="0.5em"), label, href=url,
                                 target="_blank", class_="dq-link d-block mb-2") for icon, label, url in LINKS]),
                    _card("Supported by", "handshake", STAGE["data"].colour,
                          ui.div(
                              ui.div(ui.img(src="ukri-epsrc-square-logo.png", height="60px", alt="UKRI EPSRC logo"),
                                     ui.span("UKRI EPSRC", class_="text-muted d-block mt-2 small fw-semibold")),
                              ui.div(ui.img(src="RCNDE-Logo-100.png", height="60px", alt="RCNDE logo", class_="mt-1"),
                                     ui.span("RCNDE", class_="text-muted d-block mt-2 small fw-semibold")),
                              class_="d-flex justify-content-around align-items-center text-center"
                          )),
                    col_widths=[4, 4, 4], gap="1.5rem"
                ),
                ui.layout_columns(
                    _card("Methodology", "book-open", STAGE["model"].colour,
                          *[ui.div(
                              ui.p(authors, ui.span(title, class_="fst-italic"), journal, class_="text-muted mb-2"),
                              ui.a("View paper ", icon_svg("arrow-up-right-from-square"), href=url,
                                   target="_blank", class_="btn btn-sm dq-accent-btn"),
                              class_="mb-3"
                          ) for authors, title, journal, url in PAPERS]),
                    _card("Disclaimer & data privacy", "shield-halved", STAGE["pod"].colour,
                          ui.p("This software is provided 'as is', without warranty of any kind. In no event shall "
                               "the authors be liable for any claim or damages.", class_="text-muted small"),
                          ui.p("All processing is performed locally. This application does not implement data "
                               "persistence, nor does it facilitate the outbound transmission of user-supplied "
                               "datasets to external servers.", class_="text-muted small mb-0")),
                    col_widths=[7, 5], gap="1.5rem", class_="mt-4 mb-5"
                ),
            ),
            col_widths=[-1, 10, -1]
        ),
        class_="dq-start"
    ),
    value="about",
)
