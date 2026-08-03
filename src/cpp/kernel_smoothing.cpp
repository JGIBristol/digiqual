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

    double stack_sq_res[4096];
    std::vector<double> heap_sq_res;
    const double* sq_res_ptr = nullptr;
    if (N_train <= 4096) {
        for (size_t j = 0; j < N_train; ++j) {
            stack_sq_res[j] = residuals[j] * residuals[j];
        }
        sq_res_ptr = stack_sq_res;
    } else {
        heap_sq_res.resize(N_train);
        for (size_t j = 0; j < N_train; ++j) {
            heap_sq_res[j] = residuals[j] * residuals[j];
        }
        sq_res_ptr = heap_sq_res.data();
    }

    auto compute_i = [&](size_t i) {
        const double* eval_pt = X_eval + i * D;
        double weight_sum = 0.0;
        double weighted_sq_res_sum = 0.0;

        if (D == 1) {
            double eval0 = eval_pt[0];
            for (size_t j = 0; j < N_train; ++j) {
                double diff = eval0 - X_train[j];
                double sq_dist = diff * diff;
                double w = inv_sqrt_2pi_bw * std::exp(-sq_dist * inv_2bw2);
                weight_sum += w;
                weighted_sq_res_sum += w * sq_res_ptr[j];
            }
        } else if (D == 2) {
            double eval0 = eval_pt[0];
            double eval1 = eval_pt[1];
            for (size_t j = 0; j < N_train; ++j) {
                const double* train_pt = X_train + j * 2;
                double diff0 = eval0 - train_pt[0];
                double diff1 = eval1 - train_pt[1];
                double sq_dist = diff0 * diff0 + diff1 * diff1;
                double w = inv_sqrt_2pi_bw * std::exp(-sq_dist * inv_2bw2);
                weight_sum += w;
                weighted_sq_res_sum += w * sq_res_ptr[j];
            }
        } else {
            for (size_t j = 0; j < N_train; ++j) {
                const double* train_pt = X_train + j * D;
                double sq_dist = 0.0;
                for (size_t k = 0; k < D; ++k) {
                    double diff = eval_pt[k] - train_pt[k];
                    sq_dist += diff * diff;
                }
                double w = inv_sqrt_2pi_bw * std::exp(-sq_dist * inv_2bw2);
                weight_sum += w;
                weighted_sq_res_sum += w * sq_res_ptr[j];
            }
        }

        if (weight_sum <= 1e-12) {
            weight_sum = 1e-10;
        }

        out[i] = std::sqrt(std::max(0.0, weighted_sq_res_sum / weight_sum));
    };

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N_eval; ++i) {
        compute_i(i);
    }
#else
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    num_threads = std::min(num_threads, static_cast<unsigned int>(N_eval));

    if (num_threads <= 1 || N_eval < 64) {
        for (size_t i = 0; i < N_eval; ++i) {
            compute_i(i);
        }
    } else {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        size_t chunk_size = (N_eval + num_threads - 1) / num_threads;

        for (unsigned int t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, N_eval);
            if (start >= end) break;

            threads.emplace_back([=]() {
                for (size_t i = start; i < end; ++i) {
                    compute_i(i);
                }
            });
        }

        for (auto& th : threads) {
            if (th.joinable()) {
                th.join();
            }
        }
    }
#endif
}
