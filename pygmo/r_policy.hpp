// Copyright 2020 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PYGMO_R_POLICY_HPP
#define PYGMO_R_POLICY_HPP

#include <memory>
#include <string>
#include <type_traits>
#include <typeindex>

#include <pybind11/pybind11.h>

#include <pagmo/config.hpp>
#include <pagmo/r_policy.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/types.hpp>

#include "common_base.hpp"

namespace pagmo
{

namespace detail
{

namespace py = pybind11;

// Disable the static UDRP checks for py::object.
template <>
struct disable_udrp_checks<py::object> : std::true_type {
};

template <>
struct r_pol_inner<py::object> final : r_pol_inner_base, pygmo::common_base {
    // Just need the def ctor, delete everything else.
    r_pol_inner() = default;
    r_pol_inner(const r_pol_inner &) = delete;
    r_pol_inner(r_pol_inner &&) = delete;
    r_pol_inner &operator=(const r_pol_inner &) = delete;
    r_pol_inner &operator=(r_pol_inner &&) = delete;
    explicit r_pol_inner(const py::object &);
    virtual std::unique_ptr<r_pol_inner_base> clone() const override final;
    // Mandatory methods.
    virtual individuals_group_t replace(const individuals_group_t &, const vector_double::size_type &,
                                        const vector_double::size_type &, const vector_double::size_type &,
                                        const vector_double::size_type &, const vector_double::size_type &,
                                        const vector_double &, const individuals_group_t &) const override final;
    // Optional methods.
    virtual std::string get_name() const override final;
    virtual std::string get_extra_info() const override final;

#if PAGMO_VERSION_MAJOR > 2 || (PAGMO_VERSION_MAJOR == 2 && PAGMO_VERSION_MINOR >= 15)
    virtual std::type_index get_type_index() const override final;
    virtual const void *get_ptr() const override final;
    virtual void *get_ptr() override final;
#endif

    template <typename Archive>
    void save(Archive &, unsigned) const;
    template <typename Archive>
    void load(Archive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()
    py::object m_value;
};

} // namespace detail

} // namespace pagmo

// Register the r_pol_inner specialisation for py::object.
PAGMO_S11N_R_POLICY_EXPORT_KEY(pybind11::object)

namespace pygmo
{

namespace py = pybind11;

py::tuple r_policy_pickle_getstate(const pagmo::r_policy &);
pagmo::r_policy r_policy_pickle_setstate(py::tuple);

} // namespace pygmo

#endif
