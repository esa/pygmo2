// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

#include <pagmo/topologies/free_form.hpp>
#include <pagmo/topologies/fully_connected.hpp>
#include <pagmo/topologies/ring.hpp>
#include <pagmo/topologies/unconnected.hpp>
#include <pagmo/topology.hpp>
#include <pagmo/types.hpp>

#include "common_utils.hpp"
#include "docstrings.hpp"
#include "expose_topologies.hpp"

namespace pygmo
{

namespace py = pybind11;

namespace detail
{

namespace
{

// A test topology.
struct test_topology {
    std::pair<std::vector<std::size_t>, pagmo::vector_double> get_connections(std::size_t) const
    {
        return std::pair<std::vector<std::size_t>, pagmo::vector_double>{};
    }
    void push_back() {}
    // Set/get an internal value to test extraction semantics.
    void set_n(int n)
    {
        m_n = n;
    }
    int get_n() const
    {
        return m_n;
    }
    int m_n = 1;
};

// Expose the methods from the base BGL topology.
template <typename Topo>
void expose_base_bgl_topo(py::class_<Topo> &c)
{
    c.def("num_vertices", &Topo::num_vertices, pygmo::base_bgl_num_vertices_docstring().c_str());
    c.def("are_adjacent", &Topo::are_adjacent, pygmo::base_bgl_are_adjacent_docstring().c_str(), py::arg("i"),
          py::arg("j"));
    c.def("add_vertex", &Topo::add_vertex, pygmo::base_bgl_add_vertex_docstring().c_str());
    c.def("add_edge", &Topo::add_edge, pygmo::base_bgl_add_edge_docstring().c_str(), py::arg("i"), py::arg("j"),
          py::arg("w") = 1.);
    c.def("remove_edge", &Topo::remove_edge, pygmo::base_bgl_remove_edge_docstring().c_str(), py::arg("i"),
          py::arg("j"));
    c.def("set_weight", &Topo::set_weight, pygmo::base_bgl_set_weight_docstring().c_str(), py::arg("i"), py::arg("j"),
          py::arg("w"));
    c.def("set_all_weights", &Topo::set_all_weights, pygmo::base_bgl_set_all_weights_docstring().c_str(), py::arg("w"));
    c.def("get_edge_weight", &Topo::get_edge_weight, pygmo::base_bgl_get_edge_weight_docstring().c_str(), py::arg("i"),
          py::arg("j"));
}

} // namespace

} // namespace detail

void expose_topologies(py::module &m, py::class_<pagmo::topology> &topo, py::module &t_module)
{
    // Test topology.
    auto t_topology = expose_topology<detail::test_topology>(m, topo, t_module, "_test_topology", "A test topology.");
    t_topology.def("get_n", &detail::test_topology::get_n);
    t_topology.def("set_n", &detail::test_topology::set_n);

    // Unconnected topology.
    expose_topology<pagmo::unconnected>(m, topo, t_module, "unconnected", unconnected_docstring().c_str());

    // Ring.
    auto ring_ = expose_topology<pagmo::ring>(m, topo, t_module, "ring", ring_docstring().c_str());
    ring_.def(py::init<std::size_t, double>(), py::arg("n") = std::size_t(0), py::arg("w") = 1.)
        .def("get_weight", &pagmo::ring::get_weight, pygmo::ring_get_weight_docstring().c_str());
    detail::expose_base_bgl_topo(ring_);

    // Fully connected.
    auto fully_connected_ = expose_topology<pagmo::fully_connected>(m, topo, t_module, "fully_connected",
                                                                    fully_connected_docstring().c_str());
    fully_connected_.def(py::init<std::size_t, double>(), py::arg("n") = std::size_t(0), py::arg("w") = 1.)
        .def("get_weight", &pagmo::fully_connected::get_weight, pygmo::fully_connected_get_weight_docstring().c_str())
        .def("num_vertices", &pagmo::fully_connected::num_vertices,
             pygmo::fully_connected_num_vertices_docstring().c_str());

    // Free-form.
    auto free_form_ = expose_topology<pagmo::free_form>(m, topo, t_module, "free_form", free_form_docstring().c_str());
    free_form_.def(py::init<const pagmo::topology &>()).def(py::init([](const py::object &o) {
        return std::make_unique<pagmo::free_form>(networkx_to_bgl_graph_t(o));
    }));
    detail::expose_base_bgl_topo(free_form_);
}

} // namespace pygmo
