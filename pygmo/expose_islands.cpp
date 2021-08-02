// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <pybind11/pybind11.h>

#include <pagmo/island.hpp>
#include <pagmo/islands/thread_island.hpp>

#include "docstrings.hpp"
#include "expose_islands.hpp"

namespace pygmo
{

namespace py = pybind11;

namespace detail
{

namespace
{

// A test island.
struct test_island {
    void run_evolve(pagmo::island &) const {}
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

} // namespace

} // namespace detail

// Islands exposition function.
void expose_islands(py::module &m, py::class_<pagmo::island> &isl, py::module &isl_module)
{
    // Test island.
    auto test_isl = expose_island<detail::test_island>(m, isl, isl_module, "_test_island", "A test island.");
    test_isl.def("get_n", &detail::test_island::get_n);
    test_isl.def("set_n", &detail::test_island::set_n);

    // Thread island.
    auto thread_island_
        = expose_island<pagmo::thread_island>(m, isl, isl_module, "thread_island", thread_island_docstring().c_str());
    thread_island_.def(py::init<bool>(), py::arg("use_pool") = true);
}

} // namespace pygmo
