"""Model Explorer stage: choose each variable's role and explore the PoD live.

Every input is either plotted (1 or 2 of them), integrated over (with a distribution
and range) or sliced at a fixed value. Changing the model (in Model Diagnostics), which
variables are plotted or integrated, or the integration settings needs the Update PoD
button (Monte Carlo integration and the threshold spectrum). Slice values and the
detection threshold update the plots live.
"""


from faicons import icon_svg
from shiny import reactive, render, ui

from digiqual.defaults import N_THRESHOLD_POINTS
from digiqual.integration import NUISANCE_DISTRIBUTIONS, estimate_nuisance_distribution

from .background import TaskProgress, progress_panel, push_progress, step_states
from .components import (
    _goto_js,
    config_errors,
    fmt_value,
    nav_button,
    notice,
    slider_precision,
    stage_page,
)
from .state import AppState, logger, read_input

ROLES = {"plot": "Plot", "integrate": "Integrate", "slice": "Slice"}

panel = stage_page(
    "explorer",
    [ui.output_ui("explorer_sidebar_ui")],
    ui.output_ui("explorer_warnings_ui"),
    ui.output_ui("explorer_setup_ui"),
    ui.output_ui("explorer_results_ui"),
)


def _section(title, *children):
    return ui.div(
        ui.tags.h6(title, class_="text-secondary small text-uppercase fw-bold mb-2"),
        *children,
        class_="pb-3 mb-3 border-bottom"
    )


def register(input, output, session, state: AppState):
    current_study = state.current_study
    model_fit = state.model_fit
    slice_values = state.slice_values
    threshold = state.threshold
    applied_structure = state.applied_structure
    explorer_config = state.explorer_config


    @reactive.calc
    def ranges():
        """{col: (min, max, median, step, decimals)} for every input and the outcome."""
        study = current_study()
        if study is None or not state.candidates_ready():
            return {}
        out = {}
        for col in study.inputs + [study.outcome]:
            vals = state.data_values(col)
            lo, hi = float(vals.min()), float(vals.max())
            step, decimals = slider_precision(lo, hi)
            out[col] = (lo, hi, round(float(vals.median()), decimals), step, decimals)
        return out

    # ── Defaults once the candidate models are fitted ───────────────────────
    @reactive.effect
    @reactive.event(state.candidates_ready)
    def init_values():
        r = ranges()
        study = current_study()
        if not r or study is None or slice_values():
            return
        slice_values.set({col: r[col][2] for col in study.inputs})
        threshold.set(r[study.outcome][2])

    def _default_roles(study):
        """Keep the roles shown before; otherwise plot the most influential input."""
        sobol = (model_fit() or {}).get("sobol_indices") or {}
        top = max(study.inputs, key=lambda c: sobol.get(c, 0.0)) if sobol else study.inputs[0]
        return {col: read_input(input, f"role_{col}") or ("plot" if col == top else "slice")
                for col in study.inputs}

    # ── Layout ──────────────────────────────────────────────────────────────
    @render.ui
    def explorer_warnings_ui():
        if current_study() is None:
            return notice("Upload data and choose the input and outcome columns first.", action=nav_button("data"))
        if model_fit() is None:
            return notice("Choose a model in Model Diagnostics first.", action=nav_button("model"))
        return ui.div()

    def _variable_row(col, role, r):
        """One row of the PoD setup table: the variable, its role, and that role's settings."""
        lo, hi, med, step, decimals = r
        rnd = lambda v: round(v, decimals)  # noqa: E731
        exact_step = 10 ** -decimals
        with reactive.isolate():
            current = slice_values().get(col, med)
            dist = read_input(input, f"dist_{col}", "Uniform")
            int_lo = read_input(input, f"int_lo_{col}", rnd(lo))
            int_hi = read_input(input, f"int_hi_{col}", rnd(hi))

        def labelled(label, control, class_=""):
            return ui.div(ui.span(label, class_="small text-muted"), control, class_=f"dq-inline-field {class_}")

        return ui.div(
            ui.div(ui.tags.strong(col), ui.div(f"{fmt_value(lo)} to {fmt_value(hi)}", class_="small text-muted"),
                   class_="dq-role-name"),
            ui.div(ui.input_radio_buttons(f"role_{col}", None, choices=ROLES, selected=role, inline=True),
                   class_="dq-role"),
            ui.div(
                ui.panel_conditional(
                    f"input.role_{col} === 'plot'",
                    ui.span("On an axis of the plots, across its full data range.", class_="text-muted small"),
                ),
                ui.panel_conditional(
                    f"input.role_{col} === 'slice'",
                    ui.div(
                        ui.input_slider(f"slice_{col}", None, min=rnd(lo), max=rnd(hi), value=rnd(current),
                                        step=step, ticks=False),
                        labelled("Value", ui.input_numeric(f"slice_num_{col}", None, value=rnd(current),
                                                           min=rnd(lo), max=rnd(hi), step=exact_step)),
                        class_="dq-slice-controls"
                    ),
                ),
                ui.panel_conditional(
                    f"input.role_{col} === 'integrate'",
                    ui.div(
                        ui.input_select(f"dist_{col}", None, choices=list(NUISANCE_DISTRIBUTIONS), selected=dist),
                        labelled("from", ui.input_numeric(f"int_lo_{col}", None, value=int_lo, min=rnd(lo),
                                                          max=rnd(hi), step=exact_step)),
                        labelled("to", ui.input_numeric(f"int_hi_{col}", None, value=int_hi, min=rnd(lo),
                                                        max=rnd(hi), step=exact_step)),
                        class_="dq-integrate-controls"
                    ),
                ),
                class_="dq-role-settings"
            ),
            class_="dq-role-row"
        )

    @render.ui
    def explorer_setup_ui():
        """The PoD setup table: one row per input variable."""
        study = current_study()
        if study is None or not state.candidates_ready() or model_fit() is None:
            return ui.div()
        r = ranges()
        with reactive.isolate():
            roles = _default_roles(study)
        return ui.card(
            ui.card_header(
                ui.span(icon_svg("sliders"), class_="text-primary me-2"), "PoD setup",
                ui.span("Plot 1 or 2 variables, integrate over up to 2, and slice the rest at a fixed value.",
                        class_="small text-muted fw-normal ms-2"),
            ),
            ui.div(
                ui.div(ui.span("Variable"), ui.span("Role"), ui.span("Settings"), class_="dq-role-row dq-role-head"),
                *[_variable_row(col, roles[col], r[col]) for col in study.inputs],
                class_="dq-role-table"
            ),
            ui.output_ui("integration_notes_ui"),
        )

    @render.ui
    def explorer_sidebar_ui():
        study = current_study()
        if study is None or not state.candidates_ready():
            return ui.p("Available once a model is chosen.", class_="small text-muted")
        r = ranges()
        with reactive.isolate():
            thresh = threshold() if threshold() is not None else r[study.outcome][2]
        o_lo, o_hi, _o_med, o_step, o_dec = r[study.outcome]

        return ui.div(
            ui.output_ui("explorer_model_ui"),
            _section(
                f"Detection threshold ({study.outcome})",
                ui.input_slider("pod_threshold", None, min=round(o_lo, o_dec), max=round(o_hi, o_dec),
                                value=round(thresh, o_dec), step=o_step, ticks=False, width="100%"),
                ui.div(ui.span("Value", class_="small text-muted"),
                       ui.input_numeric("pod_threshold_num", None, value=round(thresh, o_dec),
                                        min=round(o_lo, o_dec), max=round(o_hi, o_dec), step=10 ** -o_dec),
                       class_="dq-inline-field"),
            ),
            ui.output_ui("explorer_status_ui"),
            ui.input_task_button("btn_update_pod", "Update PoD", class_="btn-primary w-100",
                                 icon=icon_svg("arrows-rotate")),
        )

    @render.ui
    def explorer_model_ui():
        fit = model_fit()
        if fit is None:
            return ui.div()
        return ui.div(
            ui.span("Model: ", class_="text-muted small"), ui.tags.strong(fit["model_name"], class_="small"),
            ui.tags.button("change", type="button", class_="btn btn-link btn-sm p-0 ms-2 align-baseline",
                           onclick=_goto_js("model")),
            class_="pb-3 mb-3 border-bottom"
        )

    @render.ui
    def integration_notes_ui():
        config = explorer_config()
        if not config or not config["integrate"]:
            return ui.div()
        notes = []
        for col, spec in config["integrate"].items():
            lo, hi = spec["bounds"]
            if lo is None or hi is None:
                continue
            d_lo, d_hi = ranges()[col][:2]
            text = f"sampled within [{fmt_value(max(lo, d_lo))}, {fmt_value(min(hi, d_hi))}]"
            if lo < d_lo or hi > d_hi:
                text += " (limited to the data range)"
            if spec["dist"] == "Uniform":
                text = f"Uniform, {text}."
            else:
                try:
                    import scipy.stats as stats
                    name, params = estimate_nuisance_distribution(state.data_values(col), spec["dist"])
                    p1, p2 = (params[0], params[1]) if name == "norm" else (params[0], params[2])
                    labels = {"norm": ("μ", "σ"), "lognorm": ("shape", "scale"), "weibull_min": ("shape", "scale")}[name]
                    dist = getattr(stats, name)(*params)
                    text = (f"{spec['dist']} fitted to the data ({labels[0]} = {p1:.4g}, {labels[1]} = {p2:.4g}; "
                            f"95% of values in [{fmt_value(dist.ppf(0.025))}, {fmt_value(dist.ppf(0.975))}]), {text}.")
                except ValueError as e:
                    text = f"{e} Uniform will be used instead, {text}."
            notes.append(ui.div(ui.tags.strong(f"{col}: "), text, class_="small text-muted mb-1"))
        return ui.div(*notes, class_="mt-3 p-2 bg-light border-start border-primary border-3 rounded-end")

    def _dirty(config):
        return applied_structure() is None or state.structure(config) != applied_structure()

    @render.ui
    def explorer_status_ui():
        if state.pod_running():
            return ui.p(ui.span(icon_svg("spinner"), class_="me-1"), "Computing the PoD...",
                        class_="small text-primary fw-semibold mb-2")
        config = explorer_config()
        errors = config_errors(config)
        if errors:
            return ui.div(*[ui.p(ui.span(icon_svg("circle-xmark"), class_="me-1"), e, class_="small text-danger mb-1")
                            for e in errors], class_="mb-2")
        if _dirty(config):
            return ui.p(ui.span(icon_svg("circle-info"), class_="me-1"),
                        "Press Update PoD to compute the PoD for these settings." if applied_structure() is None
                        else "The model, plot or integration settings changed: press Update PoD.",
                        class_="small text-primary fw-semibold mb-2")
        return ui.p(ui.span(icon_svg("circle-check"), class_="me-1"),
                    "Up to date. Slices and threshold update live.", class_="small text-success mb-2")

    @render.ui
    def explorer_results_ui():
        study = current_study()
        if study is None or model_fit() is None:
            return ui.div()
        if state.pod_running():
            return _pod_panel()
        if applied_structure() is None:
            return notice("Choose what to plot, integrate over and slice in the sidebar, then press Update PoD.")
        is_multi_dim = len(applied_structure()["plot"]) > 1
        return ui.layout_columns(
            ui.card(
                ui.card_header(f"{study.outcome} Surface" if is_multi_dim else "Signal Model"),
                ui.output_plot("plot_signal_explorer", height="480px"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header("PoD Surface Heatmap" if is_multi_dim else "PoD Curve"),
                ui.output_plot("plot_pod_explorer", height="480px"),
                full_screen=True,
            ),
            col_widths=[6, 6]
        )

    @render.plot
    def plot_signal_explorer():
        return state.study_plot("signal_model")

    @render.plot
    def plot_pod_explorer():
        return state.study_plot("pod_curve")

    # ── Slider <-> number box sync (both write the exact value to state) ────
    def _sync(slider_id, number_id, get, put, r):
        """Slider moves beyond half a step and typed numbers (clamped, rounded) both set the value."""
        lo, hi, _med, step, decimals = r
        current = get()
        moved = read_input(input, slider_id)
        raw = read_input(input, number_id)
        typed = None if raw is None else round(min(max(float(raw), lo), hi), decimals)
        if typed is not None and (current is None or abs(typed - current) > 10 ** -(decimals + 3)):
            put(typed)
            ui.update_slider(slider_id, value=typed)  # may snap to the step; ignored below
            if typed != raw:
                ui.update_numeric(number_id, value=typed)
        elif moved is not None and current is not None and abs(float(moved) - current) > step / 2:
            value = round(float(moved), decimals)
            put(value)
            ui.update_numeric(number_id, value=value)

    @reactive.effect
    def sync_slices():
        study = current_study()
        r = ranges()
        if study is None or not r:
            return
        for col in study.inputs:
            read_input(input, f"slice_{col}")
            read_input(input, f"slice_num_{col}")  # take dependencies on both
        with reactive.isolate():
            values = dict(slice_values())
            for col in study.inputs:
                _sync(f"slice_{col}", f"slice_num_{col}", lambda c=col: values.get(c),
                      lambda v, c=col: values.__setitem__(c, v), r[col])
            if values != slice_values():
                slice_values.set(values)

    @reactive.effect
    def sync_threshold():
        study = current_study()
        r = ranges()
        if study is None or not r:
            return
        read_input(input, "pod_threshold")
        read_input(input, "pod_threshold_num")
        with reactive.isolate():
            _sync("pod_threshold", "pod_threshold_num", threshold.get, threshold.set, r[study.outcome])

    # ── PoD computation ─────────────────────────────────────────────────────
    # Update PoD runs in a worker thread (an extended task), so the page stays live and
    # shows its progress. The plots are drawn on the main thread once it finishes.
    progress = TaskProgress()

    @ui.bind_task_button(button_id="btn_update_pod")
    @reactive.extended_task
    async def pod_task(study, config, kwargs):
        def work():
            # Spectrum first (it runs pod() at the median threshold), then the chosen threshold
            study.compute_pod_spectrum(n_threshold_points=N_THRESHOLD_POINTS, **kwargs)
            study.pod(threshold=config["threshold"], n_boot=0, **kwargs)
        await progress.run(work)
        return study, config

    @reactive.effect
    @reactive.event(input.btn_update_pod)
    def update_pod():
        study = current_study()
        config = explorer_config()
        errors = config_errors(config)
        if study is None or errors:
            ui.notification_show(" ".join(errors) or "Fit a model first.", type="error")
            return
        if state.pod_running() or state.uq_running():
            return
        progress.reset(integrated=list(config["integrate"]))
        state.pod_running.set(True)
        pod_task.invoke(study, config, state.pod_kwargs(config))

    @reactive.effect
    def finish_pod():
        status = pod_task.status()
        if status not in ("success", "error"):
            return
        with reactive.isolate():
            state.pod_running.set(False)
            if status == "error":
                try:
                    pod_task.result()
                except Exception as e:
                    logger.exception("PoD update failed")
                    ui.notification_show(f"PoD failed: {str(e) or type(e).__name__}", type="error")
                return
            study, config = pod_task.result()
            if study is not current_study():
                return  # the data changed meanwhile
            study.visualise(show=False)
            applied_structure.set(state.structure(config))
            state.bump_plots()

    def _pod_steps():
        spectrum_done = progress.seen("--- Spectrum Generation Complete", "4. Threshold Spectrum already cached")
        reached = 2 if spectrum_done else 1 if progress.seen("4.") else 0
        integrated = ", ".join(progress.info["integrated"])
        steps = [
            ("Computing the PoD curve",
             f"Monte Carlo integration over {integrated}" if integrated else "Nothing to integrate: a direct evaluation"),
            (f"Computing the threshold spectrum ({N_THRESHOLD_POINTS} thresholds)",
             "So the threshold slider can move the PoD instantly"),
            ("PoD at the chosen threshold", ""),
        ]
        return [(label, detail, st) for (label, detail), st in zip(steps, step_states(reached, 3), strict=True)]

    def _pod_panel():
        return progress_panel("pod-progress", "Computing the PoD", progress, _pod_steps(),
                              note="Slices and the threshold update live again once this finishes.")

    @reactive.effect
    async def tick_pod_progress():
        """Update the panel in place twice a second (redrawing it would restart its spinners)."""
        if not state.pod_running():
            return
        reactive.invalidate_later(0.5)
        await push_progress(session, "pod-progress", progress, _pod_steps())

    @reactive.effect
    def live_update():
        """Slice and threshold changes: recompute from the caches without the button."""
        config = explorer_config()
        if (config is None or applied_structure() is None or _dirty(config) or config_errors(config)
                or state.pod_running() or state.uq_running()):
            return  # (a running task changes the study; this runs again when it finishes)
        with reactive.isolate():
            study = current_study()
            res = study.pod_results if study else None
            if not res or (res["slice_values"] == config["slice"] and res["threshold"] == config["threshold"]):
                return
            kwargs = state.pod_kwargs(config)
            try:
                if not config["integrate"]:
                    # Without integration the spectrum is one vectorised pass, so build it for
                    # the new slice and keep threshold changes instant (cached per slice)
                    study.compute_pod_spectrum(n_threshold_points=N_THRESHOLD_POINTS, **kwargs)
                study.pod(threshold=config["threshold"], n_boot=0, **kwargs)
                study.visualise(show=False)
                state.bump_plots()
            except Exception as e:
                logger.exception("Live PoD update failed")
                ui.notification_show(f"Could not update the PoD: {e}", type="error")
