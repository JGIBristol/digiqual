import time
import cProfile
import pstats
import io
import numpy as np
import pandas as pd
from digiqual.pod import (
    fit_all_robust_mean_models,
    optimise_bandwidth,
    fit_variance_model,
    infer_best_distribution,
    compute_pod_curve,
    bootstrap_pod_ci,
    predict_local_std,
)
from digiqual.integration import compute_multi_dim_pod
from digiqual.adaptive import run_adaptive_search, generate_targeted_samples
from digiqual.executors import PythonExecutor

def generate_benchmark_dataset(n_samples=200, seed=42):
    np.random.seed(seed)
    length = np.random.uniform(0.5, 10.0, n_samples)
    angle = np.random.uniform(-45.0, 45.0, n_samples)
    roughness = np.random.uniform(0.0, 1.0, n_samples)

    base_signal = 5.0 + 3.0 * length - 0.8 * (length ** 2) + 0.1 * (length ** 3)
    angle_effect = 0.1 * angle - 0.05 * length * np.abs(angle)
    roughness_effect = -5.0 * roughness
    noise_scale = 0.5 + 0.4 * length + 1.0 * roughness
    noise = np.random.gumbel(loc=0, scale=noise_scale) - noise_scale * 0.57721

    signal = base_signal + angle_effect + roughness_effect + noise

    df = pd.DataFrame({
        "Length": length,
        "Angle": angle,
        "Roughness": roughness,
        "Signal": signal
    })
    return df

def profile_component(name, func, *args, **kwargs):
    print(f"\n==================================================")
    print(f"PROFILING: {name}")
    print(f"==================================================")

    # Measure exact runtime
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    t1 = time.perf_counter()
    duration = t1 - t0
    print(f"Elapsed Time: {duration:.4f} seconds")

    # Measure cProfile stats
    pr = cProfile.Profile()
    pr.enable()
    func(*args, **kwargs)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(25) # top 25 functions
    print(s.getvalue())

    return result, duration

def mock_solver(row):
    l, a, r = row['Length'], row['Angle'], row['Roughness']
    return 5.0 + 3.0*l - 0.8*(l**2) + 0.1*(l**3) + 0.1*a - 5.0*r + np.random.normal(0, 0.5)

def run_benchmarks():
    print("Generating benchmark dataset (n=200, 3D inputs)...")
    df = generate_benchmark_dataset(n_samples=200)
    X = df[["Length", "Angle", "Roughness"]].values
    y = df["Signal"].values
    feature_names = ["Length", "Angle", "Roughness"]
    poi_names = ["Length", "Angle"]

    # 1. Model Fitting & Bandwidth Optimisation
    def bench_model_fit():
        models, scores, best_key = fit_all_robust_mean_models(X, y, max_degree=4)
        mean_model = models[best_key]
        residuals, bw = fit_variance_model(X, y, mean_model)
        dist_info = infer_best_distribution(residuals, X, bw)
        return mean_model, residuals, bw, dist_info

    (mean_model, residuals, bw, dist_info), t_fit = profile_component(
        "1. Model Fitting, Bandwidth & Dist Inference", bench_model_fit
    )

    # 2. Monte Carlo Marginalisation (Fast Path vs Slow Path)
    poi_grid = np.column_stack([
        np.linspace(1.0, 9.0, 50),
        np.linspace(-30.0, 30.0, 50)
    ])
    nuisance_ranges_slice = {"Roughness": (0.5, 0.5)}

    _, t_mc_fast = profile_component(
        "2a. Monte Carlo Marginalisation - Fast Path (50 grid points, constant slice)",
        compute_multi_dim_pod,
        poi_grid, nuisance_ranges_slice, mean_model, X, residuals, bw, dist_info,
        thresholds=20.0, n_mc_samples=1000, feature_names=feature_names, poi_names=poi_names
    )

    nuisance_ranges_active = {"Roughness": (0.0, 1.0)}

    _, t_mc_slow = profile_component(
        "2b. Monte Carlo Marginalisation - Slow Path (50 grid points, 1000 MC samples per point)",
        compute_multi_dim_pod,
        poi_grid, nuisance_ranges_active, mean_model, X, residuals, bw, dist_info,
        thresholds=20.0, n_mc_samples=1000, feature_names=feature_names, poi_names=poi_names
    )

    # 3. Bootstrap Uncertainty Estimation (Single-threaded to measure CPU time accurately)
    model_type = getattr(mean_model, 'model_type_', 'Polynomial')
    model_params = getattr(mean_model, 'model_params_', 3)

    _, t_boot = profile_component(
        "3. Bootstrap PoD CI (n_boot=100, 50 grid points, 1000 MC samples per bootstrap step)",
        bootstrap_pod_ci,
        X=X, y=y, X_eval=poi_grid, threshold=20.0,
        model_type=model_type, model_params=model_params,
        bandwidth=bw, dist_info=dist_info, n_boot=100,
        nuisance_ranges=nuisance_ranges_active, n_jobs=1,
        feature_names=feature_names, poi_names=poi_names
    )

    # 4. Nadaraya-Watson predict_local_std standalone microbenchmark
    X_target_micro = np.tile(X[:50], (20, 1)) # 1000 target points
    _, t_nadaraya = profile_component(
        "4. predict_local_std (Nadaraya-Watson Kernel Smoothing on 1000 target points, 200 source points)",
        predict_local_std,
        X, residuals, X_target_micro, bw
    )

    # 5. Active Learning / Adaptive Search
    executor = PythonExecutor(solver_func=mock_solver, outcome_col="Signal")
    ranges = {"Length": (0.5, 10.0), "Angle": (-45.0, 45.0), "Roughness": (0.0, 1.0)}
    _, t_active = profile_component(
        "5. Active Learning Adaptive Search (n_start=40, n_step=10, max_iter=3)",
        run_adaptive_search,
        executor=executor, input_cols=["Length", "Angle", "Roughness"], ranges=ranges, outcome_col="Signal",
        n_start=40, n_step=10, max_iter=3
    )

    # Summary Table
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY & BASELINE METRICS")
    print("="*70)
    print(f"1. Model Fitting & Dist Inference:            {t_fit:.4f} s")
    print(f"2a. MC Marginalisation (Fast Path):            {t_mc_fast:.4f} s")
    print(f"2b. MC Marginalisation (Slow Path, 50 grid pts):{t_mc_slow:.4f} s")
    print(f"3. Bootstrap PoD CI (n_boot=100):             {t_boot:.4f} s")
    print(f"4. Nadaraya-Watson predict_local_std (1000 pts): {t_nadaraya:.4f} s")
    print(f"5. Active Learning Search (3 iterations):     {t_active:.4f} s")
    print("="*70)

if __name__ == "__main__":
    run_benchmarks()
