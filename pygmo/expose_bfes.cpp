// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <pybind11/pybind11.h>

#include <pagmo/batch_evaluators/default_bfe.hpp>
#include <pagmo/batch_evaluators/member_bfe.hpp>
#include <pagmo/batch_evaluators/thread_bfe.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

#include "docstrings.hpp"
#include "expose_bfes.hpp"

namespace pygmo
{

namespace py = pybind11;

namespace detail
{

namespace
{

// A test bfe.
struct test_bfe {
    pagmo::vector_double operator()(const pagmo::problem &p, const pagmo::vector_double &dvs) const
    {
        pagmo::vector_double retval;
        const auto nx = p.get_nx();
        const auto n_dvs = dvs.size() / nx;
        for (decltype(dvs.size()) i = 0; i < n_dvs; ++i) {
            const auto f = p.fitness(pagmo::vector_double(dvs.data() + i * nx, dvs.data() + (i + 1u) * nx));
            retval.insert(retval.end(), f.begin(), f.end());
        }
        return retval;
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

// A thread-unsafe test bfe.
struct tu_test_bfe : test_bfe {
    pagmo::thread_safety get_thread_safety() const
    {
        return pagmo::thread_safety::none;
    }
};

} // namespace

} // namespace detail

// Bfes exposition function.
void expose_bfes(py::module &m, py::class_<pagmo::bfe> &b, py::module &b_module)
{
    // Test bfe.
    auto t_bfe = expose_bfe<detail::test_bfe>(m, b, b_module, "_test_bfe", "A test bfe.");
    t_bfe.def("get_n", &detail::test_bfe::get_n);
    t_bfe.def("set_n", &detail::test_bfe::set_n);

    // Thread unsafe test bfe.
    auto tu_bfe = expose_bfe<detail::tu_test_bfe>(m, b, b_module, "_tu_test_bfe", "A thread-unsafe test bfe.");
    tu_bfe.def("get_n", &detail::tu_test_bfe::get_n);
    tu_bfe.def("set_n", &detail::tu_test_bfe::set_n);

    // Default bfe.
    expose_bfe<pagmo::default_bfe>(m, b, b_module, "default_bfe", default_bfe_docstring().c_str());

    // Thread bfe.
    expose_bfe<pagmo::thread_bfe>(m, b, b_module, "thread_bfe", thread_bfe_docstring().c_str());

    // Member bfe.
    expose_bfe<pagmo::member_bfe>(m, b, b_module, "member_bfe", member_bfe_docstring().c_str());
}

} // namespace pygmo
