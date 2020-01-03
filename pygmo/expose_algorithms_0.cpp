// Copyright 2019 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <pybind11/pybind11.h>

#include <pagmo/algorithm.hpp>
#include <pagmo/population.hpp>
#include <pagmo/threading.hpp>

#include "expose_algorithms.hpp"

namespace pygmo
{

namespace py = pybind11;

namespace detail
{

namespace
{

// A test algo.
struct test_algorithm {
    pagmo::population evolve(const pagmo::population &pop) const
    {
        return pop;
    }
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

// A thread unsafe test algo.
struct tu_test_algorithm {
    pagmo::population evolve(const pagmo::population &pop) const
    {
        return pop;
    }
    pagmo::thread_safety get_thread_safety() const
    {
        return pagmo::thread_safety::none;
    }
};

} // namespace

} // namespace detail

void expose_algorithms_0(py::module &m, py::class_<pagmo::algorithm> &algo, py::module &a_module)
{
    // Test algo.
    auto test_a = expose_algorithm<detail::test_algorithm>(m, algo, a_module, "_test_algorithm", "A test algorithm.");
    test_a.def("get_n", &detail::test_algorithm::get_n);
    test_a.def("set_n", &detail::test_algorithm::set_n);

    // Thread unsafe test algo.
    expose_algorithm<detail::tu_test_algorithm>(m, algo, a_module, "_tu_test_algorithm",
                                                "A thread unsafe test algorithm.");
}

} // namespace pygmo
