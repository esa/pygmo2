// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PYGMO_BFE_HPP
#define PYGMO_BFE_HPP

#include <memory>
#include <string>
#include <type_traits>
#include <typeindex>

#include <pybind11/pybind11.h>

#include <pagmo/bfe.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

#include "common_base.hpp"

namespace pagmo
{

namespace detail
{

namespace py = pybind11;

// Disable the static UDBFE checks for py::object.
template <>
struct disable_udbfe_checks<py::object> : std::true_type {
};

template <>
struct bfe_inner<py::object> final : bfe_inner_base, pygmo::common_base {
    // Just need the def ctor, delete everything else.
    bfe_inner() = default;
    bfe_inner(const bfe_inner &) = delete;
    bfe_inner(bfe_inner &&) = delete;
    bfe_inner &operator=(const bfe_inner &) = delete;
    bfe_inner &operator=(bfe_inner &&) = delete;
    explicit bfe_inner(const py::object &);
    std::unique_ptr<bfe_inner_base> clone() const final;
    // Mandatory methods.
    vector_double operator()(const problem &, const vector_double &) const final;
    // Optional methods.
    thread_safety get_thread_safety() const final;
    std::string get_name() const final;
    std::string get_extra_info() const final;

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

// Register the bfe_inner specialisation for py::object.
PAGMO_S11N_BFE_EXPORT_KEY(pybind11::object)

#endif
