// Copyright 2020, 2021 PaGMO development team
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
    std::unique_ptr<prob_inner_base> clone() const final;
    // Mandatory methods.
    vector_double fitness(const vector_double &) const final;
    std::pair<vector_double, vector_double> get_bounds() const final;
    // Optional methods.
    vector_double batch_fitness(const vector_double &) const final;
    bool has_batch_fitness() const final;
    vector_double::size_type get_nobj() const final;
    vector_double::size_type get_nec() const final;
    vector_double::size_type get_nic() const final;
    vector_double::size_type get_nix() const final;
    std::string get_name() const final;
    std::string get_extra_info() const final;
    bool has_gradient() const final;
    vector_double gradient(const vector_double &) const final;
    bool has_gradient_sparsity() const final;
    sparsity_pattern gradient_sparsity() const final;
    bool has_hessians() const final;
    std::vector<vector_double> hessians(const vector_double &) const final;
    bool has_hessians_sparsity() const final;
    std::vector<sparsity_pattern> hessians_sparsity() const final;
    void set_seed(unsigned) final;
    bool has_set_seed() const final;
    thread_safety get_thread_safety() const final;

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

// Register the prob_inner specialisation for py::object.
PAGMO_S11N_PROBLEM_EXPORT_KEY(pybind11::object)

#endif
