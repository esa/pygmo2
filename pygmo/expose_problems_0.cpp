// Copyright 2019 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <utility>

#include <pybind11/pybind11.h>

#include <pagmo/problem.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/null_problem.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

#include "docstrings.hpp"
#include "expose_problems.hpp"

namespace pygmo
{

namespace py = pybind11;

namespace detail
{

namespace
{

// A test problem.
struct test_problem {
    test_problem(unsigned nobj = 1) : m_nobj(nobj) {}
    pagmo::vector_double fitness(const pagmo::vector_double &) const
    {
        return {1.};
    }
    std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const
    {
        return {{0.}, {1.}};
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
    pagmo::vector_double::size_type get_nobj() const
    {
        return m_nobj;
    }
    int m_n = 1;
    unsigned m_nobj;
};

// A thread-unsafe test problem.
struct tu_test_problem {
    pagmo::vector_double fitness(const pagmo::vector_double &) const
    {
        return {1.};
    }
    std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const
    {
        return {{0.}, {1.}};
    }
    pagmo::thread_safety get_thread_safety() const
    {
        return pagmo::thread_safety::none;
    }
};

} // namespace

} // namespace detail

void expose_problems_0(py::module &m, py::class_<pagmo::problem> &prob, py::module &p_module)
{
    // Exposition of C++ problems.

    // Test problem.
    auto test_p = expose_problem<detail::test_problem>(m, prob, p_module, "_test_problem", "A test problem.");
    test_p.def(py::init<unsigned>(), py::arg("nobj"));
    test_p.def("get_n", &detail::test_problem::get_n);
    test_p.def("set_n", &detail::test_problem::set_n);

    // Thread unsafe test problem.
    expose_problem<detail::tu_test_problem>(m, prob, p_module, "_tu_test_problem", "A thread unsafe test problem.");

    // Null problem.
    auto np = expose_problem<pagmo::null_problem>(m, prob, p_module, "null_problem", null_problem_docstring().c_str());
    np.def(
        py::init<pagmo::vector_double::size_type, pagmo::vector_double::size_type, pagmo::vector_double::size_type>(),
        py::arg("nobj") = 1, py::arg("nec") = 0, py::arg("nic") = 0);

    // Hock-Schittkowsky 71
    auto hs71 = expose_problem<pagmo::hock_schittkowsky_71>(m, prob, p_module, "hock_schittkowsky_71",
                                                            "__init__()\n\nThe Hock-Schittkowsky 71 problem.\n\n"
                                                            "See :cpp:class:`pagmo::hock_schittkowsky_71`.\n\n");
    hs71.def("best_known", &best_known_wrapper<pagmo::hock_schittkowsky_71>,
             problem_get_best_docstring("Hock-Schittkowsky 71").c_str());
}

} // namespace pygmo
