#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "kernel_smoothing.hpp"
#include "mc_integration.hpp"

namespace py = pybind11;

py::array_t<double> py_predict_local_std(
    py::array_t<double, py::array::c_style | py::array::forcecast> X_train,
    py::array_t<double, py::array::c_style | py::array::forcecast> residuals,
    py::array_t<double, py::array::c_style | py::array::forcecast> X_eval,
    double bandwidth,
    py::object out_arr = py::none()
) {
    py::buffer_info buf_X_train = X_train.request();
    py::buffer_info buf_res = residuals.request();
    py::buffer_info buf_X_eval = X_eval.request();

    size_t N_train = buf_X_train.shape[0];
    size_t D = (buf_X_train.ndim > 1) ? buf_X_train.shape[1] : 1;
    size_t N_eval = buf_X_eval.shape[0];

    py::array_t<double, py::array::c_style> out;
    if (!out_arr.is_none()) {
        out = py::cast<py::array_t<double, py::array::c_style>>(out_arr);
    } else {
        out = py::array_t<double>(N_eval);
    }
    py::buffer_info buf_out = out.request();

    const double* ptr_X_train = static_cast<const double*>(buf_X_train.ptr);
    const double* ptr_res = static_cast<const double*>(buf_res.ptr);
    const double* ptr_X_eval = static_cast<const double*>(buf_X_eval.ptr);
    double* ptr_out = static_cast<double*>(buf_out.ptr);

    {
        py::gil_scoped_release release;
        predict_local_std_cpp(ptr_X_train, ptr_res, ptr_X_eval, N_train, N_eval, D, bandwidth, ptr_out);
    }

    return out;
}

py::array_t<double> py_compute_pod_probs(
    py::array_t<double, py::array::c_style | py::array::forcecast> mean_resp,
    py::array_t<double, py::array::c_style | py::array::forcecast> sigma_resp,
    double threshold,
    const std::string& dist_name,
    py::tuple dist_params,
    py::object out_arr = py::none()
) {
    py::buffer_info buf_mean = mean_resp.request();
    py::buffer_info buf_sigma = sigma_resp.request();

    size_t N_points = buf_mean.size;

    double loc = 0.0;
    double scale = 1.0;
    if (dist_params.size() >= 2) {
        loc = dist_params[dist_params.size() - 2].cast<double>();
        scale = dist_params[dist_params.size() - 1].cast<double>();
    } else if (dist_params.size() == 1) {
        scale = dist_params[0].cast<double>();
    }

    py::array_t<double, py::array::c_style> out;
    if (!out_arr.is_none()) {
        out = py::cast<py::array_t<double, py::array::c_style>>(out_arr);
    } else {
        out = py::array_t<double>(N_points);
    }
    py::buffer_info buf_out = out.request();

    const double* ptr_mean = static_cast<const double*>(buf_mean.ptr);
    const double* ptr_sigma = static_cast<const double*>(buf_sigma.ptr);
    double* ptr_out = static_cast<double*>(buf_out.ptr);

    {
        py::gil_scoped_release release;
        compute_pod_probs_cpp(ptr_mean, ptr_sigma, threshold, dist_name, loc, scale, N_points, ptr_out);
    }

    return out;
}

PYBIND11_MODULE(_digiqual_cpp, m) {
    m.doc() = "High-performance C++ backend for DigiQual";

    m.def(
        "predict_local_std",
        &py_predict_local_std,
        py::arg("X_train"),
        py::arg("residuals"),
        py::arg("X_eval"),
        py::arg("bandwidth"),
        py::arg("out") = py::none(),
        "Predict local standard deviation using OpenMP Nadaraya-Watson kernel smoothing."
    );

    m.def(
        "compute_pod_probs",
        &py_compute_pod_probs,
        py::arg("mean_resp"),
        py::arg("sigma_resp"),
        py::arg("threshold"),
        py::arg("dist_name"),
        py::arg("dist_params"),
        py::arg("out") = py::none(),
        "Compute PoD survival CDF probabilities."
    );
}
