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

    def reset(self):
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


def progress_panel(title: str, progress: TaskProgress, steps: list[tuple[str, str, str]],
                   fraction: float | None = None, note: str = ""):
    """A live panel for a running task: spinner, elapsed time, bar and steps.

    `steps` are (label, detail, state) with state 'done', 'active' or 'todo'; the detail
    shows under the active step. Without a `fraction` the bar is full and animated.
    """
    rows = []
    for label, detail, state in steps:
        icon = {"done": "circle-check", "active": "spinner", "todo": "circle"}[state]
        rows.append(ui.div(
            icon_svg(icon, margin_left="0", margin_right="0.6em"),
            ui.div(ui.div(label), ui.div(detail, class_="small text-muted") if state == "active" and detail else ""),
            class_=f"dq-fit-step is-{state} d-flex align-items-start"
        ))
    width = 100 if fraction is None else max(2, min(100, int(100 * fraction)))
    return ui.div(
        ui.div(
            ui.div(class_="spinner-border text-primary me-3", role="status"),
            ui.div(ui.h5(title, class_="mb-0"), ui.span(f"{progress.elapsed()} s elapsed", class_="small text-muted")),
            class_="d-flex align-items-center mb-3"
        ),
        ui.div(ui.div(class_="progress-bar progress-bar-striped progress-bar-animated", style=f"width: {width}%"),
               class_="progress mb-3", style="height: 8px;"),
        *rows,
        ui.p(note or "You can look at other stages meanwhile; the stepper shows when it's done.",
             class_="small text-muted mt-3 mb-0"),
        class_="dq-fit-progress card p-4 mb-3"
    )
