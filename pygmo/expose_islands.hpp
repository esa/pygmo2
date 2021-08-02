// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PYGMO_EXPOSE_ISLANDS_HPP
#define PYGMO_EXPOSE_ISLANDS_HPP

#include <pybind11/pybind11.h>

#include <pagmo/algorithm.hpp>
#include <pagmo/island.hpp>
#include <pagmo/population.hpp>
#include <pagmo/r_policy.hpp>
#include <pagmo/s_policy.hpp>

#include "common_utils.hpp"

namespace pygmo
{

namespace py = pybind11;

// Islands exposition function.
void expose_islands(py::module &, py::class_<pagmo::island> &, py::module &);

// Main island exposition function.
template <typename Isl>
inline py::class_<Isl> expose_island(py::module &m, py::class_<pagmo::island> &isl, py::module &isl_module,
                                     const char *name, const char *descr)
{
    py::class_<Isl> c(m, name, descr);

    // We require all islands to be def-ctible at the bare minimum.
    c.def(py::init<>());

    // Mark it as a C++ island.
    c.attr("_pygmo_cpp_island") = true;

    // Expose the island constructor from Isl.
    isl.def(py::init<const Isl &, const pagmo::algorithm &, const pagmo::population &, const pagmo::r_policy &,
                     const pagmo::s_policy &>());

    // Expose extract.
    isl.def("_cpp_extract", &generic_cpp_extract<pagmo::island, Isl>, py::return_value_policy::reference_internal);

    // Add the island to the islands submodule.
    isl_module.attr(name) = c;

    return c;
}

} // namespace pygmo

#endif
