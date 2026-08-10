#include "mc_integration.hpp"
#include <cmath>
#include <algorithm>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

static inline double calc_prob(double z_std, const std::string& dist_name) {
    if (dist_name == "norm") {
        return 0.5 * std::erfc(z_std * M_SQRT1_2);
    } else if (dist_name == "gumbel_r") {
        return 1.0 - std::exp(-std::exp(-z_std));
    } else if (dist_name == "gumbel_l") {
        return std::exp(-std::exp(z_std));
    } else if (dist_name == "logistic") {
        return 1.0 / (1.0 + std::exp(z_std));
    } else if (dist_name == "laplace") {
        if (z_std < 0.0) {
            return 1.0 - 0.5 * std::exp(z_std);
        } else {
            return 0.5 * std::exp(-z_std);
        }
    } else {
        return 0.5 * std::erfc(z_std * M_SQRT1_2);
    }
}

void compute_pod_probs_cpp(
    const double* mean_resp,
    const double* sigma_resp,
    double threshold,
    const std::string& dist_name,
    double loc,
    double scale,
    size_t N_points,
    double* out
) {
    if (N_points == 0 || scale <= 0.0) {
        return;
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N_points; ++i) {
        double sig = std::max(sigma_resp[i], 1e-10);
        double z = (threshold - mean_resp[i]) / sig;
        double z_std = (z - loc) / scale;
        out[i] = std::clamp(calc_prob(z_std, dist_name), 0.0, 1.0);
    }
#else
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    num_threads = std::min(num_threads, static_cast<unsigned int>(N_points));

    if (num_threads <= 1 || N_points < 64) {
        for (size_t i = 0; i < N_points; ++i) {
            double sig = std::max(sigma_resp[i], 1e-10);
            double z = (threshold - mean_resp[i]) / sig;
            double z_std = (z - loc) / scale;
            out[i] = std::clamp(calc_prob(z_std, dist_name), 0.0, 1.0);
        }
    } else {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        size_t chunk_size = (N_points + num_threads - 1) / num_threads;

        for (unsigned int t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, N_points);
            if (start >= end) break;

            threads.emplace_back([=]() {
                for (size_t i = start; i < end; ++i) {
                    double sig = std::max(sigma_resp[i], 1e-10);
                    double z = (threshold - mean_resp[i]) / sig;
                    double z_std = (z - loc) / scale;
                    out[i] = std::clamp(calc_prob(z_std, dist_name), 0.0, 1.0);
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
