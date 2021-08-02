// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PYGMO_EXPOSE_TOPOLOGIES_HPP
#define PYGMO_EXPOSE_TOPOLOGIES_HPP

#include <pybind11/pybind11.h>

#include <pagmo/topology.hpp>

#include "common_utils.hpp"

namespace pygmo
{

namespace py = pybind11;

// Topologies exposition function.
void expose_topologies(py::module &, py::class_<pagmo::topology> &, py::module &);

// C++ UDT exposition function.
template <typename Topo>
inline py::class_<Topo> expose_topology(py::module &m, py::class_<pagmo::topology> &topo, py::module &t_module,
                                        const char *name, const char *descr)
{
    py::class_<Topo> c(m, name, descr);

    // We require all topologies to be def-ctible at the bare minimum.
    c.def(py::init<>());

    // Mark it as a C++ topology.
    c.attr("_pygmo_cpp_topology") = true;

    // Expose the topology constructor from Topo.
    topo.def(py::init<const Topo &>(), py::arg("udt"));

    // Expose extract.
    topo.def("_cpp_extract", &generic_cpp_extract<pagmo::topology, Topo>, py::return_value_policy::reference_internal);

    // Add the topology to the topologies submodule.
    t_module.attr(name) = c;

    return c;
}

} // namespace pygmo

#endif
