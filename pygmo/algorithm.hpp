// Copyright 2020 PaGMO development team
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
#include <pagmo/config.hpp>
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
    virtual std::unique_ptr<algo_inner_base> clone() const override final;
    // Mandatory methods.
    virtual population evolve(const population &) const override final;
    // Optional methods.
    virtual void set_seed(unsigned) override final;
    virtual bool has_set_seed() const override final;
    virtual thread_safety get_thread_safety() const override final;
    virtual std::string get_name() const override final;
    virtual std::string get_extra_info() const override final;
    virtual void set_verbosity(unsigned) override final;
    virtual bool has_set_verbosity() const override final;

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

// Register the algo_inner specialisation for py::object.
PAGMO_S11N_ALGORITHM_EXPORT_KEY(pybind11::object)

namespace pygmo
{

namespace py = pybind11;

py::tuple algorithm_pickle_getstate(const pagmo::algorithm &);
pagmo::algorithm algorithm_pickle_setstate(py::tuple);

} // namespace pygmo

#endif
