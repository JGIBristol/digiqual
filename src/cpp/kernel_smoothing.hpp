#ifndef KERNEL_SMOOTHING_HPP
#define KERNEL_SMOOTHING_HPP

#include <vector>
#include <cmath>

/**
 * Predict local standard deviation using Nadaraya-Watson Kernel Smoothing.
 *
 * @param X_train Pointer to flat row-major X_train array (N_train x D)
 * @param residuals Pointer to flat residuals array (N_train)
 * @param X_eval Pointer to flat row-major X_eval array (N_eval x D)
 * @param N_train Number of training points
 * @param N_eval Number of evaluation points
 * @param D Dimensionality (number of columns)
 * @param bandwidth Bandwidth (sigma) of Gaussian kernel
 * @param out Pointer to output array (N_eval)
 */
void predict_local_std_cpp(
    const double* X_train,
    const double* residuals,
    const double* X_eval,
    size_t N_train,
    size_t N_eval,
    size_t D,
    double bandwidth,
    double* out
);

#endif // KERNEL_SMOOTHING_HPP
