#ifndef MC_INTEGRATION_HPP
#define MC_INTEGRATION_HPP

#include <vector>
#include <string>
#include <cmath>

/**
 * Fast multi-threaded calculation of PoD CDF probabilities.
 *
 * @param mean_resp Pointer to array of mean predictions (N_points)
 * @param sigma_resp Pointer to array of noise std predictions (N_points)
 * @param threshold Threshold value
 * @param dist_name Distribution name ("norm", "gumbel_r", "logistic", "laplace")
 * @param loc Distribution location parameter
 * @param scale Distribution scale parameter
 * @param N_points Total number of points
 * @param out Pointer to output array of probabilities (N_points)
 */
void compute_pod_probs_cpp(
    const double* mean_resp,
    const double* sigma_resp,
    double threshold,
    const std::string& dist_name,
    double loc,
    double scale,
    size_t N_points,
    double* out
);

#endif // MC_INTEGRATION_HPP
