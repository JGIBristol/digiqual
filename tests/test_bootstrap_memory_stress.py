import os
import gc
import time
import threading
import psutil
import pytest
import numpy as np
from digiqual.pod import bootstrap_pod_ci, fit_all_robust_mean_models, fit_variance_model


def measure_memory_during_task(task_func, sample_interval_sec=0.05):
    """
    Executes a function while sampling process RSS memory usage via psutil.
    Returns (result, initial_rss_mb, peak_rss_mb, final_rss_mb).
    """
    process = psutil.Process(os.getpid())
    gc.collect()
    time.sleep(0.1)

    initial_rss = process.memory_info().rss / (1024 * 1024)
    peak_rss = initial_rss
    stop_event = threading.Event()

    def monitor():
        nonlocal peak_rss
        while not stop_event.is_set():
            try:
                current_rss = process.memory_info().rss / (1024 * 1024)
                if current_rss > peak_rss:
                    peak_rss = current_rss
            except Exception:
                pass
            time.sleep(sample_interval_sec)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    try:
        res = task_func()
    finally:
        stop_event.set()
        monitor_thread.join(timeout=1.0)

    gc.collect()
    time.sleep(0.1)
    final_rss = process.memory_info().rss / (1024 * 1024)

    return res, initial_rss, peak_rss, final_rss


@pytest.mark.stress
def test_bootstrap_memory_stress_bounded_ram():
    """
    Stress Test: Run 1,000 bootstrap iterations on a 3,000-row dataset
    across all CPU cores and verify that peak memory growth is flat and bounded (< 500 MB).
    """
    np.random.seed(42)
    n_samples = 3000
    X1 = np.random.uniform(0.1, 5.0, n_samples)
    X2 = np.random.uniform(10.0, 50.0, n_samples)
    X = np.column_stack([X1, X2])

    noise = np.random.normal(0, 0.5 * (1 + 0.2 * X1), n_samples)
    y = 2.0 * X1 + 0.05 * X2 + noise

    model_type = 'Polynomial'
    model_params = 2
    bw = 1.5

    X_eval = np.column_stack([
        np.linspace(0.1, 5.0, 50),
        np.full(50, 30.0)
    ])
    dist_info = ('norm', (0.0, 1.0))
    n_boot = 100

    def run_bootstrap():
        return bootstrap_pod_ci(
            X=X,
            y=y,
            X_eval=X_eval,
            threshold=5.0,
            model_type=model_type,
            model_params=model_params,
            bandwidth=bw,
            dist_info=dist_info,
            n_boot=n_boot,
            n_jobs=-1,
            feature_names=['Length', 'Angle'],
            poi_names=['Length']
        )

    print(f"\n--- Starting Bootstrap Memory Stress Test (Dataset: {n_samples} rows, Bootstraps: {n_boot}, Cores: {os.cpu_count()}) ---")
    bounds, initial_mb, peak_mb, final_mb = measure_memory_during_task(run_bootstrap)
    growth_mb = peak_mb - initial_mb
    net_leak_mb = final_mb - initial_mb

    print(f"Initial RSS RAM: {initial_mb:.2f} MB")
    print(f"Peak RSS RAM:    {peak_mb:.2f} MB")
    print(f"Final RSS RAM:   {final_mb:.2f} MB")
    print(f"Peak RAM Growth: {growth_mb:.2f} MB")
    print(f"Net RAM Leak:    {net_leak_mb:.2f} MB")

    assert bounds is not None
    assert len(bounds) == 2
    # Ensure memory growth remains strictly bounded (< 500 MB) during 1000 iterations on 3000 rows
    assert growth_mb < 500.0, f"Memory growth exceeded limit: {growth_mb:.2f} MB > 500.0 MB"
    assert net_leak_mb < 100.0, f"Memory leak detected: {net_leak_mb:.2f} MB > 100.0 MB"
