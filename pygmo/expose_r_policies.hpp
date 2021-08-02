// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PYGMO_EXPOSE_R_POLICIES_HPP
#define PYGMO_EXPOSE_R_POLICIES_HPP

#include <pybind11/pybind11.h>

#include <pagmo/r_policy.hpp>

#include "common_utils.hpp"

namespace pygmo
{

namespace py = pybind11;

// Replacement policies exposition function.
void expose_r_policies(py::module &, py::class_<pagmo::r_policy> &, py::module &);

// C++ UDRP exposition function.
template <typename RPol>
inline py::class_<RPol> expose_r_policy(py::module &m, py::class_<pagmo::r_policy> &r_pol, py::module &r_module,
                                        const char *name, const char *descr)
{
    py::class_<RPol> c(m, name, descr);

    // We require all policies to be def-ctible at the bare minimum.
    c.def(py::init<>());

    // Mark it as a C++ policy.
    c.attr("_pygmo_cpp_r_policy") = true;

    // Expose the r_policy constructor from RPol.
    r_pol.def(py::init<const RPol &>(), py::arg("udrp"));

    // Expose extract.
    r_pol.def("_cpp_extract", &generic_cpp_extract<pagmo::r_policy, RPol>, py::return_value_policy::reference_internal);

    // Add the policy to the policies submodule.
    r_module.attr(name) = c;

    return c;
}

} // namespace pygmo

#endif
