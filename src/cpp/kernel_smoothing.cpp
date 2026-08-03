#include "kernel_smoothing.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

void predict_local_std_cpp(
    const double* X_train,
    const double* residuals,
    const double* X_eval,
    size_t N_train,
    size_t N_eval,
    size_t D,
    double bandwidth,
    double* out
) {
    if (N_train == 0 || N_eval == 0 || D == 0 || bandwidth <= 0.0) {
        return;
    }

    const double inv_2bw2 = 1.0 / (2.0 * bandwidth * bandwidth);
    const double inv_sqrt_2pi_bw = 1.0 / (std::sqrt(2.0 * M_PI) * bandwidth);

    // Precompute squared residuals
    std::vector<double> sq_residuals(N_train);
    for (size_t j = 0; j < N_train; ++j) {
        sq_residuals[j] = residuals[j] * residuals[j];
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N_eval; ++i) {
        const double* eval_pt = X_eval + i * D;
        double weight_sum = 0.0;
        double weighted_sq_res_sum = 0.0;

        for (size_t j = 0; j < N_train; ++j) {
            const double* train_pt = X_train + j * D;
            double sq_dist = 0.0;
            for (size_t k = 0; k < D; ++k) {
                double diff = eval_pt[k] - train_pt[k];
                sq_dist += diff * diff;
            }
            double w = inv_sqrt_2pi_bw * std::exp(-sq_dist * inv_2bw2);
            weight_sum += w;
            weighted_sq_res_sum += w * sq_residuals[j];
        }

        if (weight_sum <= 1e-12) {
            weight_sum = 1e-10;
        }

        out[i] = std::sqrt(std::max(0.0, weighted_sq_res_sum / weight_sum));
    }
#else
    // Fallback: Multithreading via std::thread across hardware cores
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    num_threads = std::min(num_threads, static_cast<unsigned int>(N_eval));

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    size_t chunk_size = (N_eval + num_threads - 1) / num_threads;

    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, N_eval);
        if (start >= end) break;

        threads.emplace_back([=, &sq_residuals]() {
            for (size_t i = start; i < end; ++i) {
                const double* eval_pt = X_eval + i * D;
                double weight_sum = 0.0;
                double weighted_sq_res_sum = 0.0;

                for (size_t j = 0; j < N_train; ++j) {
                    const double* train_pt = X_train + j * D;
                    double sq_dist = 0.0;
                    for (size_t k = 0; k < D; ++k) {
                        double diff = eval_pt[k] - train_pt[k];
                        sq_dist += diff * diff;
                    }
                    double w = inv_sqrt_2pi_bw * std::exp(-sq_dist * inv_2bw2);
                    weight_sum += w;
                    weighted_sq_res_sum += w * sq_residuals[j];
                }

                if (weight_sum <= 1e-12) {
                    weight_sum = 1e-10;
                }

                out[i] = std::sqrt(std::max(0.0, weighted_sq_res_sum / weight_sum));
            }
        });
    }

    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }
#endif
}
