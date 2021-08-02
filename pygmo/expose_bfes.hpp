// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PYGMO_EXPOSE_BFES_HPP
#define PYGMO_EXPOSE_BFES_HPP

#include <pybind11/pybind11.h>

#include <pagmo/bfe.hpp>

#include "common_utils.hpp"

namespace pygmo
{

namespace py = pybind11;

// Bfes exposition function.
void expose_bfes(py::module &, py::class_<pagmo::bfe> &, py::module &);

// Main bfe exposition function.
template <typename Bfe>
inline py::class_<Bfe> expose_bfe(py::module &m, py::class_<pagmo::bfe> &b, py::module &b_module, const char *name,
                                  const char *descr)
{
    py::class_<Bfe> c(m, name, descr);

    // We require all bfes to be def-ctible at the bare minimum.
    c.def(py::init<>());

    // Mark it as a C++ bfe.
    c.attr("_pygmo_cpp_bfe") = true;

    // Expose the bfe constructor from Bfe.
    b.def(py::init<const Bfe &>(), py::arg("udbfe"));

    // Expose extract.
    b.def("_cpp_extract", &generic_cpp_extract<pagmo::bfe, Bfe>, py::return_value_policy::reference_internal);

    // Add the bfe to the batch_evaluators submodule.
    b_module.attr(name) = c;

    return c;
}

} // namespace pygmo

#endif
