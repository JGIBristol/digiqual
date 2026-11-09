"""Long computations in a worker thread, with a progress panel.

The slow steps (fitting the candidate models, Update PoD, the bootstrap) run as Shiny
extended tasks on a worker thread, so the page stays live while they work. `TaskProgress`
follows such a task: it collects the library's log messages from that thread (which
mark its steps) and any ``(done, total, label)`` progress callbacks. `progress_panel`
draws what it has found.
"""

import asyncio
import logging
import threading
import time

from faicons import icon_svg
from shiny import ui

_library_logger = logging.getLogger("digiqual")


class TaskProgress(logging.Handler):
    """The progress of one task running in a worker thread."""

    def __init__(self):
        super().__init__()
        self.thread = None
        self.reset()

    def reset(self, **info):
        """Start following a new run; `info` holds its settings, for the panel."""
        self.info = info
        self.started = time.monotonic()
        self.messages: list[str] = []
        self.count: tuple[int, int, str] | None = None  # (done, total, label)

    def on_count(self, done: int, total: int, label: str = ""):
        """A progress callback, e.g. for `fit_model` or `pod`."""
        self.count = (done, total, label)

    def emit(self, record):
        if record.thread == self.thread:
            self.messages.append(record.getMessage().strip())

    def seen(self, *prefixes: str) -> bool:
        """Whether the task has logged a message starting with one of `prefixes`."""
        return any(m.startswith(prefixes) for m in list(self.messages))

    def elapsed(self) -> int:
        return int(time.monotonic() - self.started)

    async def run(self, fn, *args, **kwargs):
        """Run ``fn(*args, **kwargs)`` in a worker thread, collecting its log messages."""
        def work():
            self.thread = threading.get_ident()
            return fn(*args, **kwargs)
        _library_logger.addHandler(self)
        try:
            return await asyncio.to_thread(work)
        finally:
            _library_logger.removeHandler(self)


def step_states(reached: int, n: int) -> list[str]:
    """'done' before step `reached`, 'active' at it and 'todo' after it."""
    return ["done" if i < reached else "active" if i == reached else "todo" for i in range(n)]


def progress_panel(panel_id: str, title: str, progress: TaskProgress, steps: list[tuple[str, str, str]],
                   fraction: float | None = None, note: str = ""):
    """A panel for a running task: spinner, elapsed time, bar and steps.

    The panel is drawn once; `push_progress` then updates it in place, so its spinners
    keep turning smoothly instead of restarting on every redraw. `steps` are
    (label, detail, state), state 'done', 'active' or 'todo'; the detail shows under
    the active step. Without a `fraction` the bar is full and animated.
    """
    rows = []
    for i, (label, detail, state) in enumerate(steps):
        rows.append(ui.div(
            # All three icons are present; CSS shows the one for the row's state
            *[ui.span(icon_svg(icon, margin_left="0", margin_right="0.6em"), class_=f"dq-icon-{kind}")
              for kind, icon in (("done", "circle-check"), ("active", "spinner"), ("todo", "circle"))],
            ui.div(ui.div(label), ui.div(detail, id=f"{panel_id}-detail-{i}", class_="small text-muted dq-step-detail")),
            id=f"{panel_id}-step-{i}", class_=f"dq-fit-step is-{state} d-flex align-items-start"
        ))
    return ui.div(
        ui.div(
            ui.div(class_="spinner-border text-primary me-3", role="status"),
            ui.div(ui.h5(title, class_="mb-0"),
                   ui.span(f"{progress.elapsed()} s elapsed", id=f"{panel_id}-elapsed", class_="small text-muted")),
            class_="d-flex align-items-center mb-3"
        ),
        ui.div(ui.div(id=f"{panel_id}-bar", class_="progress-bar progress-bar-striped progress-bar-animated",
                      style=f"width: {_bar_width(fraction)}%"),
               class_="progress mb-3", style="height: 8px;"),
        *rows,
        ui.p(note or "You can look at other stages meanwhile; the stepper shows when it's done.",
             class_="small text-muted mt-3 mb-0"),
        id=panel_id, class_="dq-fit-progress card p-4 mb-3", **{"aria-live": "polite"},
    )


async def push_progress(session, panel_id: str, progress: TaskProgress, steps: list[tuple[str, str, str]],
                        fraction: float | None = None):
    """Update a `progress_panel` in the browser without redrawing it."""
    await session.send_custom_message("dq_progress", {
        "id": panel_id,
        "elapsed": f"{progress.elapsed()} s elapsed",
        "width": _bar_width(fraction),
        "steps": [{"state": state, "detail": detail} for _label, detail, state in steps],
    })


def _bar_width(fraction: float | None) -> int:
    return 100 if fraction is None else max(2, min(100, int(100 * fraction)))


# Browser side of `push_progress`: update the panel's text, bar and step states in place
PROGRESS_JS = """
Shiny.addCustomMessageHandler('dq_progress', function(m) {
    var panel = document.getElementById(m.id);
    if (!panel) return;
    document.getElementById(m.id + '-elapsed').textContent = m.elapsed;
    document.getElementById(m.id + '-bar').style.width = m.width + '%';
    m.steps.forEach(function(step, i) {
        var row = document.getElementById(m.id + '-step-' + i);
        if (!row) return;
        var cls = 'dq-fit-step is-' + step.state + ' d-flex align-items-start';
        if (row.className !== cls) row.className = cls;
        var detail = document.getElementById(m.id + '-detail-' + i);
        if (detail.textContent !== step.detail) detail.textContent = step.detail;
    });
});
"""
