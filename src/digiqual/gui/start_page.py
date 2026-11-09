"""Start page: the two ways into the workflow and an overview of the stages."""

from faicons import icon_svg
from shiny import ui

from .components import STAGE, STAGES, _goto_js, accent_style, hero, icon_tile


def _entry(value, title, text, icon):
    """A way into the workflow, in the colour of the stage it starts at."""
    stage = STAGE[value]
    return ui.div(
        icon_tile(icon),
        ui.h4(title, class_="mb-2"),
        ui.p(text, class_="text-muted"),
        ui.tags.button(title, " ", icon_svg("arrow-right"), type="button", onclick=_goto_js(value),
                       class_="btn dq-accent-btn mt-auto align-self-start"),
        class_="dq-entry card p-4 h-100 d-flex flex-column", style=accent_style(stage.colour),
    )


def _stage_card(stage):
    """One stage of the workflow, in its colour. An overview only: the stages are reached
    through the entry cards and the stepper, which know whether a stage is ready."""
    return ui.div(
        ui.div(ui.span(str(stage.number), class_="dq-stage-card-number"), icon_svg(stage.icon),
               class_="d-flex align-items-center justify-content-between mb-2"),
        ui.div(stage.title, class_="fw-semibold mb-1"),
        ui.div(stage.purpose, class_="small text-muted"),
        class_="dq-stage-card card p-3 h-100",
        style=accent_style(stage.colour),
    )


panel = ui.nav_panel(
    "Start",
    ui.div(
        hero("DigiQual", "Statistical toolkit for reliability assessment in NDT"),
        ui.layout_columns(
            _entry("design", "Plan simulations",
                   "Start from your variable ranges and build a space-filling design to run in your solver.",
                   "table"),
            _entry("data", "Analyse results",
                   "Upload simulation results and go from data checks to a PoD curve with confidence bounds.",
                   "chart-line"),
            col_widths=[-2, 4, 4, -2], gap="1.5rem", class_="dq-entries"
        ),
        ui.layout_columns(
            ui.div(
                ui.h5("The workflow", class_="text-primary mt-2 mb-2"),
                ui.layout_column_wrap(*[_stage_card(s) for s in STAGES], width=1 / 5, gap="12px",
                                      class_="dq-stage-cards mb-5"),
            ),
            col_widths=[-1, 10, -1]
        ),
        class_="dq-start"
    ),
    value="start",
)
