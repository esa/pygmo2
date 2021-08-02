// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PYGMO_S_POLICY_HPP
#define PYGMO_S_POLICY_HPP

#include <memory>
#include <string>
#include <type_traits>
#include <typeindex>

#include <pybind11/pybind11.h>

#include <pagmo/s11n.hpp>
#include <pagmo/s_policy.hpp>
#include <pagmo/types.hpp>

#include "common_base.hpp"

namespace pagmo
{

namespace detail
{

namespace py = pybind11;

// Disable the static UDSP checks for py::object.
template <>
struct disable_udsp_checks<py::object> : std::true_type {
};

template <>
struct s_pol_inner<py::object> final : s_pol_inner_base, pygmo::common_base {
    // Just need the def ctor, delete everything else.
    s_pol_inner() = default;
    s_pol_inner(const s_pol_inner &) = delete;
    s_pol_inner(s_pol_inner &&) = delete;
    s_pol_inner &operator=(const s_pol_inner &) = delete;
    s_pol_inner &operator=(s_pol_inner &&) = delete;
    explicit s_pol_inner(const py::object &);
    std::unique_ptr<s_pol_inner_base> clone() const final;
    // Mandatory methods.
    individuals_group_t select(const individuals_group_t &, const vector_double::size_type &,
                               const vector_double::size_type &, const vector_double::size_type &,
                               const vector_double::size_type &, const vector_double::size_type &,
                               const vector_double &) const final;
    // Optional methods.
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

// Register the s_pol_inner specialisation for py::object.
PAGMO_S11N_S_POLICY_EXPORT_KEY(pybind11::object)

#endif
