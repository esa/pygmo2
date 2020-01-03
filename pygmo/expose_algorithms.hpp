// Copyright 2019 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PYGMO_EXPOSE_ALGORITHMS_HPP
#define PYGMO_EXPOSE_ALGORITHMS_HPP

#include <pybind11/pybind11.h>

#include <pagmo/algorithm.hpp>

#include "common_utils.hpp"

namespace pygmo
{

namespace py = pybind11;

// Split algorithm exposition functions.
void expose_algorithms_0(py::module &, py::class_<pagmo::algorithm> &, py::module &);
void expose_algorithms_1(py::module &, py::class_<pagmo::algorithm> &, py::module &);

// C++ UDA exposition function.
template <typename Algo>
inline py::class_<Algo> expose_algorithm(py::module &m, py::class_<pagmo::algorithm> &algo, py::module &a_module,
                                         const char *name, const char *descr)
{
    py::class_<Algo> c(m, name, descr);

    // We require all algos to be def-ctible at the bare minimum.
    c.def(py::init<>());

    // Mark it as a C++ algorithm.
    c.attr("_pygmo_cpp_algorithm") = true;

    // Expose the algorithm constructor from Algo.
    algo.def(py::init<const Algo &>(), py::arg("uda"));

    // Expose extract.
    algo.def("_cpp_extract", &generic_cpp_extract<pagmo::algorithm, Algo>, py::return_value_policy::reference_internal);

    // Add the algorithm to the algorithms submodule.
    a_module.attr(name) = c;

    return c;
}

} // namespace pygmo

#endif
