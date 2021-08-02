// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PYGMO_EXPOSE_S_POLICIES_HPP
#define PYGMO_EXPOSE_S_POLICIES_HPP

#include <pybind11/pybind11.h>

#include <pagmo/s_policy.hpp>

#include "common_utils.hpp"

namespace pygmo
{

namespace py = pybind11;

// Selection policies exposition function.
void expose_s_policies(py::module &, py::class_<pagmo::s_policy> &, py::module &);

// C++ UDSP exposition function.
template <typename SPol>
inline py::class_<SPol> expose_s_policy(py::module &m, py::class_<pagmo::s_policy> &s_pol, py::module &s_module,
                                        const char *name, const char *descr)
{
    py::class_<SPol> c(m, name, descr);

    // We require all policies to be def-ctible at the bare minimum.
    c.def(py::init<>());

    // Mark it as a C++ policy.
    c.attr("_pygmo_cpp_s_policy") = true;

    // Expose the s_policy constructor from SPol.
    s_pol.def(py::init<const SPol &>(), py::arg("udsp"));

    // Expose extract.
    s_pol.def("_cpp_extract", &generic_cpp_extract<pagmo::s_policy, SPol>, py::return_value_policy::reference_internal);

    // Add the policy to the policies submodule.
    s_module.attr(name) = c;

    return c;
}

} // namespace pygmo

#endif
