// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PYGMO_EXPOSE_PROBLEMS_HPP
#define PYGMO_EXPOSE_PROBLEMS_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <pagmo/problem.hpp>

#include "common_utils.hpp"

namespace pygmo
{

namespace py = pybind11;

// Split problem exposition functions.
void expose_problems_0(py::module &, py::class_<pagmo::problem> &, py::module &);
void expose_problems_1(py::module &, py::class_<pagmo::problem> &, py::module &);

// C++ UDP exposition function.
template <typename Prob>
inline py::class_<Prob> expose_problem(py::module &m, py::class_<pagmo::problem> &prob, py::module &p_module,
                                       const char *name, const char *descr)
{
    py::class_<Prob> c(m, name, descr);

    // We require all problems to be def-ctible at the bare minimum.
    c.def(py::init<>());

    // Mark it as a C++ problem.
    c.attr("_pygmo_cpp_problem") = true;

    // Expose the problem constructor from Prob.
    prob.def(py::init<const Prob &>(), py::arg("udp"));

    // Expose extract.
    prob.def("_cpp_extract", &generic_cpp_extract<pagmo::problem, Prob>, py::return_value_policy::reference_internal);

    // Add the problem to the problems submodule.
    p_module.attr(name) = c;

    return c;
}

// Wrapper for the best known method.
template <typename Prob>
inline py::array_t<double> best_known_wrapper(const Prob &p)
{
    return vector_to_ndarr<py::array_t<double>>(p.best_known());
}

} // namespace pygmo

#endif
