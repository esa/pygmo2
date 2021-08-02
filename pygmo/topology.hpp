// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PYGMO_TOPOLOGY_HPP
#define PYGMO_TOPOLOGY_HPP

#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

#include <pagmo/s11n.hpp>
#include <pagmo/topology.hpp>
#include <pagmo/types.hpp>

#include "common_base.hpp"

namespace pagmo
{

namespace detail
{

namespace py = pybind11;

// Disable the static UDT checks for py::object.
template <>
struct disable_udt_checks<py::object> : std::true_type {
};

template <>
struct topo_inner<py::object> final : topo_inner_base, pygmo::common_base {
    // Just need the def ctor, delete everything else.
    topo_inner() = default;
    topo_inner(const topo_inner &) = delete;
    topo_inner(topo_inner &&) = delete;
    topo_inner &operator=(const topo_inner &) = delete;
    topo_inner &operator=(topo_inner &&) = delete;
    explicit topo_inner(const py::object &);
    std::unique_ptr<topo_inner_base> clone() const final;
    // Mandatory methods.
    std::pair<std::vector<std::size_t>, vector_double> get_connections(std::size_t) const final;
    void push_back() final;
    // Optional methods.
    std::string get_name() const final;
    std::string get_extra_info() const final;

    bgl_graph_t to_bgl() const final;
    std::type_index get_type_index() const final;
    const void *get_ptr() const final;
    void *get_ptr() final;

    template <typename Archive>
    void save(Archive &, unsigned) const;
    template <typename Archive>
    void load(Archive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    py::object m_value;
};

} // namespace detail

} // namespace pagmo

// Register the topo_inner specialisation for py::object.
PAGMO_S11N_TOPOLOGY_EXPORT_KEY(pybind11::object)

#endif
