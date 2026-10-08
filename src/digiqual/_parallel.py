"""Shared helpers for parallel (bootstrap) execution."""

import os


def resolve_n_jobs(n_jobs: int | None) -> int:
    """
    Converts a user ``n_jobs`` setting into the number of worker processes.

    - ``None`` or ``1``: run sequentially in the current process (1 worker).
    - ``-1``: use all CPU cores except two, so the machine (and the GUI) stays
      responsive while a long bootstrap runs. Always at least 1.
    - ``n > 1``: use ``n`` workers, capped at the number of CPU cores.

    This single rule is used by the bootstrap itself, by the progress messages and
    by `SimulationStudy.estimate_compute_time`, so they always agree.
    """
    total_cores = os.cpu_count() or 1
    if n_jobs is None or n_jobs == 1:
        return 1
    if n_jobs == -1:
        return max(1, total_cores - 2)
    return min(max(1, int(n_jobs)), total_cores)


def _call_single_threaded(step, *args, **kwargs):
    """Runs ``step`` with native thread pools (BLAS/OpenMP) limited to one thread.

    Used inside worker processes so N workers don't each start N BLAS threads.
    """
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return step(*args, **kwargs)


def run_bootstrap(
    step,
    args: tuple,
    n_boot: int,
    n_jobs: int | None,
    n_points: int,
    label: str = "Bootstrap",
    progress_callback=None,
    chunk_size: int = 50,
    step_kwargs: dict | None = None,
):
    """
    Runs ``n_boot`` bootstrap iterations of ``step`` and stacks their results.

    ``step(*args, seed=i, **step_kwargs)`` must return a 1-D array of length
    ``n_points`` (one PoD curve). Iteration ``i`` always gets seed ``i``, so results
    are reproducible and independent of ``n_jobs``.

    Work runs in chunks so progress can be reported (and memory released) between
    them. With several workers, one process pool is created and reused for every
    chunk rather than respawned each time.

    Returns:
        np.ndarray: Array of shape ``(n_boot, n_points)``.
    """
    import gc
    import logging

    import numpy as np
    from joblib import Parallel, delayed

    logger = logging.getLogger("digiqual")
    step_kwargs = step_kwargs or {}
    n_workers = resolve_n_jobs(n_jobs)
    logger.info(f"   -> [{label}] Running {n_boot} iterations on {n_workers} worker core(s)...")

    results = np.empty((n_boot, n_points))

    def _report(completed):
        pct = int((completed / n_boot) * 100)
        logger.info(f"   -> [{label} Progress] Completed {completed}/{n_boot} iterations ({pct}%)...")
        if progress_callback is not None:
            try:
                progress_callback(completed, n_boot)
            except Exception as e:  # noqa: BLE001 - user-supplied callback must not abort the run
                logger.warning("Progress callback raised an exception: %s", e)

    def _chunks():
        for start in range(0, n_boot, chunk_size):
            yield start, min(start + chunk_size, n_boot)

    if n_workers > 1:
        with Parallel(n_jobs=n_workers, backend="multiprocessing", verbose=0) as parallel:
            for start, end in _chunks():
                chunk = parallel(
                    delayed(_call_single_threaded)(step, *args, seed=i, **step_kwargs)
                    for i in range(start, end)
                )
                results[start:end] = np.asarray(chunk)
                del chunk
                gc.collect()
                _report(end)
    else:
        for start, end in _chunks():
            for i in range(start, end):
                results[i] = step(*args, seed=i, **step_kwargs)
            _report(end)

    return results


def percentile_bounds(pod_matrix, confidence_levels=None):
    """
    Two-sided percentile bounds of bootstrap PoD curves.

    Returns ``(lower_95, upper_95)`` if ``confidence_levels`` is None, otherwise a
    dict ``{level: (lower, upper)}`` with the central ``level`` % interval for each.
    """
    import numpy as np

    if confidence_levels is None:
        return np.percentile(pod_matrix, 2.5, axis=0), np.percentile(pod_matrix, 97.5, axis=0)
    bounds = {}
    for cl in confidence_levels:
        low_p = (100.0 - cl) / 2.0
        bounds[cl] = (np.percentile(pod_matrix, low_p, axis=0), np.percentile(pod_matrix, 100.0 - low_p, axis=0))
    return bounds
