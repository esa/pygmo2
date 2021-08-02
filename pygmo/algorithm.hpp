// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PYGMO_ALGORITHM_HPP
#define PYGMO_ALGORITHM_HPP

#include <memory>
#include <string>
#include <type_traits>
#include <typeindex>

#include <pybind11/pybind11.h>

#include <pagmo/algorithm.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>

#include "common_base.hpp"

namespace pagmo
{

namespace detail
{

namespace py = pybind11;

// Disable the static UDA checks for py::object.
template <>
struct disable_uda_checks<py::object> : std::true_type {
};

template <>
struct algo_inner<py::object> final : algo_inner_base, pygmo::common_base {
    // Just need the def ctor, delete everything else.
    algo_inner() = default;
    algo_inner(const algo_inner &) = delete;
    algo_inner(algo_inner &&) = delete;
    algo_inner &operator=(const algo_inner &) = delete;
    algo_inner &operator=(algo_inner &&) = delete;
    explicit algo_inner(const py::object &);
    std::unique_ptr<algo_inner_base> clone() const final;
    // Mandatory methods.
    population evolve(const population &) const final;
    // Optional methods.
    void set_seed(unsigned) final;
    bool has_set_seed() const final;
    thread_safety get_thread_safety() const final;
    std::string get_name() const final;
    std::string get_extra_info() const final;
    void set_verbosity(unsigned) final;
    bool has_set_verbosity() const final;

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

// Register the algo_inner specialisation for py::object.
PAGMO_S11N_ALGORITHM_EXPORT_KEY(pybind11::object)

#endif
