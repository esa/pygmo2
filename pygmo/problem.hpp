// Copyright 2020 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PYGMO_PROBLEM_HPP
#define PYGMO_PROBLEM_HPP

#include <memory>
#include <string>
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

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

// Disable the static UDP checks for py::object.
template <>
struct disable_udp_checks<py::object> : std::true_type {
};

// NOTE: here we are specialising the prob_inner implementation template for py::object.
// We need to do this because the default implementation works on C++ types by detecting
// their methods via type-traits at compile-time, but here we need to check the presence
// of methods at runtime. That is, we need to replace the type-traits with runtime
// inspection of Python objects.
//
// We cannot be as precise as in C++ detecting the methods' signatures (it might be
// possible with the inspect module in principle, but it looks messy and it might break if the methods
// are implemented as C/C++ extensions). The main policy adopted here is: if the py::object
// has a callable attribute with the required name, then the "runtime type-trait" is considered
// satisfied, otherwise not.
template <>
struct prob_inner<py::object> final : prob_inner_base, pygmo::common_base {
    // Just need the def ctor, delete everything else.
    prob_inner() = default;
    prob_inner(const prob_inner &) = delete;
    prob_inner(prob_inner &&) = delete;
    prob_inner &operator=(const prob_inner &) = delete;
    prob_inner &operator=(prob_inner &&) = delete;
    explicit prob_inner(const py::object &);
    virtual std::unique_ptr<prob_inner_base> clone() const override final;
    // Mandatory methods.
    virtual vector_double fitness(const vector_double &) const override final;
    virtual std::pair<vector_double, vector_double> get_bounds() const override final;
    // Optional methods.
    virtual vector_double batch_fitness(const vector_double &) const override final;
    virtual bool has_batch_fitness() const override final;
    virtual vector_double::size_type get_nobj() const override final;
    virtual vector_double::size_type get_nec() const override final;
    virtual vector_double::size_type get_nic() const override final;
    virtual vector_double::size_type get_nix() const override final;
    virtual std::string get_name() const override final;
    virtual std::string get_extra_info() const override final;
    virtual bool has_gradient() const override final;
    virtual vector_double gradient(const vector_double &) const override final;
    virtual bool has_gradient_sparsity() const override final;
    virtual sparsity_pattern gradient_sparsity() const override final;
    virtual bool has_hessians() const override final;
    virtual std::vector<vector_double> hessians(const vector_double &) const override final;
    virtual bool has_hessians_sparsity() const override final;
    virtual std::vector<sparsity_pattern> hessians_sparsity() const override final;
    virtual void set_seed(unsigned) override final;
    virtual bool has_set_seed() const override final;
    virtual thread_safety get_thread_safety() const override final;
    virtual std::type_index get_type_index() const override final;
    template <typename Archive>
    void save(Archive &, unsigned) const;
    template <typename Archive>
    void load(Archive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()
    py::object m_value;
};

} // namespace detail

} // namespace pagmo

// Register the prob_inner specialisation for py::object.
PAGMO_S11N_PROBLEM_EXPORT_KEY(pybind11::object)

namespace pygmo
{

namespace py = pybind11;

py::tuple problem_pickle_getstate(const pagmo::problem &);
pagmo::problem problem_pickle_setstate(py::tuple);

} // namespace pygmo

#endif
