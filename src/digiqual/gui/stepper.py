"""The workflow stepper: one item per stage with its status and a summary of what was decided.

`stage_statuses` is a pure function of a small snapshot of the app state, so the rules
can be tested without a Shiny session. `register` builds the snapshot, renders the
stepper and handles all stage navigation (stepper items and every `nav_button`).
"""

from faicons import icon_svg
from shiny import reactive, render, ui

from .components import GOTO_INPUT, STAGES, _goto_js, fmt_value
from .state import AppState, read_input

STATUS_ICONS = {"done": "circle-check", "warning": "triangle-exclamation", "stale": "rotate",
                "todo": "circle", "locked": "lock", "running": "spinner"}


def stage_statuses(s: dict) -> dict[str, tuple[str, str]]:
    """{stage: (status, summary)}. Status is 'done', 'warning', 'stale', 'todo', 'locked' or 'running'."""
    out = {}

    out["design"] = (("done", f"{s['generated_rows']} runs") if s.get("generated_rows")
                     else ("todo", "optional"))

    if not s.get("n_rows"):
        out["data"] = ("todo", "upload results")
    else:
        size = f"{s['n_rows']} rows · {s['n_inputs']} inputs"
        diag = s.get("diagnostics")
        if diag is None:
            out["data"] = ("todo", f"{size} · run checks")
        elif s.get("diagnostics_stale"):
            out["data"] = ("stale", f"{size} · re-run checks")
        elif diag == "failed":
            out["data"] = ("warning", f"{size} · issues found")
        else:
            out["data"] = ("done", size)

    if s.get("fitting"):
        out["model"] = ("running", "fitting models…")
    elif s.get("model_name") is None:
        out["model"] = (("todo", "run the models") if s.get("diagnostics") is not None
                        else ("locked", "check the data first"))
    else:
        label = f"{s['model_name']} · R² {s['cv_r2']:.2f}"
        out["model"] = ("warning" if s["cv_r2"] < s["min_r2"] else "done", label)

    if s.get("model_name") is None:
        out["explorer"] = ("locked", "choose a model first")
    elif s.get("pod_running"):
        out["explorer"] = ("running", "computing PoD…")
    elif not s.get("applied"):
        out["explorer"] = ("todo", "set up the PoD")
    else:
        parts = [" × ".join(s["plot"])] + [f"{c} = {fmt_value(v)}" for c, v in s.get("slice", {}).items()]
        out["explorer"] = ("stale", "update needed") if s.get("dirty") else ("done", " · ".join(parts))

    if s.get("uq_running"):
        out["pod"] = ("running", "bootstrap running…")
    elif not s.get("applied") or s.get("dirty"):
        out["pod"] = ("locked", "update the PoD first")
    elif s.get("uq_value") is None:
        out["pod"] = ("todo", "run the bootstrap")
    elif s.get("uq_stale"):
        out["pod"] = ("stale", f"{s['uq_value']} · out of date")
    else:
        out["pod"] = ("done", s["uq_value"])
    return out


def register(input, output, session, state: AppState):

    @reactive.effect
    @reactive.event(input[GOTO_INPUT])
    def go_to_stage():
        ui.update_navset("stage", selected=input[GOTO_INPUT]())

    @reactive.calc
    def snapshot():
        study = state.current_study()
        diag = state.diagnostic_table()
        fit = state.model_fit()
        config = state.explorer_config()
        applied = state.applied_structure()
        uq = state.uq_result()
        generated = state.final_generated_df()

        uq_value = None
        if uq is not None:
            res = uq["results"]
            # The reliability index for the target and confidence currently selected
            target = int(read_input(input, "pod_target_val", uq["settings"]["target"]))
            conf = int(read_input(input, "pod_conf_val", uq["settings"]["conf"]))
            if len(res["poi_cols"]) > 1:
                uq_value = f"PoD surface · {res['n_boot']} bootstraps"
            else:
                val = res["reliability_table"].get((target, conf))
                uq_value = f"a{target}/{conf} = " + (fmt_value(val) if val is not None and val == val else "not reached")

        return {
            "generated_rows": None if generated is None else len(generated),
            "n_rows": len(study.data) if study is not None else None,
            "n_inputs": len(study.inputs) if study is not None else None,
            "diagnostics": None if diag is None else ("passed" if state.validation_passed() else "failed"),
            "diagnostics_stale": state.diagnostics_stale(),
            "fitting": state.fit_running(),
            "pod_running": state.pod_running(),
            "uq_running": state.uq_running(),
            "model_name": fit["short_name"] if fit else None,
            "cv_r2": fit["cv_r2"] if fit else None,
            "min_r2": read_input(input, "ui_min_r2", 0.0),
            "applied": applied is not None,
            "dirty": applied is not None and state.structure(config) != applied,
            "plot": config["plot"] if config else [],
            "slice": config["slice"] if config else {},
            "uq_value": uq_value,
            "uq_stale": uq is not None and config != uq["config"],
        }

    @render.ui
    def stepper_ui():
        current = input.stage()
        if current in ("start", "about"):
            return ui.div()
        statuses = stage_statuses(snapshot())
        items = []
        for stage in STAGES:
            status, summary = statuses[stage.value]
            classes = f"dq-step is-{status}" + (" is-current" if stage.value == current else "")
            items.append(ui.tags.button(
                ui.span(str(stage.number), class_="dq-step-number"),
                ui.span(
                    ui.span(stage.title, class_="dq-step-title"),
                    ui.span(icon_svg(STATUS_ICONS[status]), " ", summary, class_="dq-step-summary"),
                    class_="dq-step-text"
                ),
                type="button", class_=classes, onclick=_goto_js(stage.value),
                id=f"step_{stage.value}", title=f"{stage.title}: {summary}",
            ))
        return ui.tags.nav(*items, class_="dq-stepper", **{"aria-label": "Workflow steps"})
