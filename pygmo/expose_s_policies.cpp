// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <pybind11/pybind11.h>

#include <pagmo/s_policies/select_best.hpp>
#include <pagmo/s_policy.hpp>
#include <pagmo/types.hpp>

#include "docstrings.hpp"
#include "expose_s_policies.hpp"
#include "sr_policy_add_rate_constructor.hpp"

namespace pygmo
{

namespace py = pybind11;

namespace detail
{

namespace
{

// A test s_policy.
struct test_s_policy {
    pagmo::individuals_group_t select(const pagmo::individuals_group_t &inds, const pagmo::vector_double::size_type &,
                                      const pagmo::vector_double::size_type &, const pagmo::vector_double::size_type &,
                                      const pagmo::vector_double::size_type &, const pagmo::vector_double::size_type &,
                                      const pagmo::vector_double &) const
    {
        return inds;
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

} // namespace

} // namespace detail

void expose_s_policies(py::module &m, py::class_<pagmo::s_policy> &s_pol, py::module &s_module)
{
    // Test s_policy.
    auto t_s_policy
        = expose_s_policy<detail::test_s_policy>(m, s_pol, s_module, "_test_s_policy", "A test selection policy.");
    t_s_policy.def("get_n", &detail::test_s_policy::get_n);
    t_s_policy.def("set_n", &detail::test_s_policy::set_n);

    // Select best policy.
    auto select_best_
        = expose_s_policy<pagmo::select_best>(m, s_pol, s_module, "select_best", select_best_docstring().c_str());
    detail::sr_policy_add_rate_constructor(select_best_);
}

} // namespace pygmo
