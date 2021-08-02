// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <pybind11/pybind11.h>

#include <pagmo/r_policies/fair_replace.hpp>
#include <pagmo/r_policy.hpp>
#include <pagmo/types.hpp>

#include "docstrings.hpp"
#include "expose_r_policies.hpp"
#include "sr_policy_add_rate_constructor.hpp"

namespace pygmo
{

namespace py = pybind11;

namespace detail
{

namespace
{

// A test r_policy.
struct test_r_policy {
    pagmo::individuals_group_t replace(const pagmo::individuals_group_t &inds, const pagmo::vector_double::size_type &,
                                       const pagmo::vector_double::size_type &, const pagmo::vector_double::size_type &,
                                       const pagmo::vector_double::size_type &, const pagmo::vector_double::size_type &,
                                       const pagmo::vector_double &, const pagmo::individuals_group_t &) const
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

void expose_r_policies(py::module &m, py::class_<pagmo::r_policy> &r_pol, py::module &r_module)
{
    // Test r_policy.
    auto t_r_policy
        = expose_r_policy<detail::test_r_policy>(m, r_pol, r_module, "_test_r_policy", "A test replacement policy.");
    t_r_policy.def("get_n", &detail::test_r_policy::get_n);
    t_r_policy.def("set_n", &detail::test_r_policy::set_n);

    // Fair replacement policy.
    auto fair_replace_
        = expose_r_policy<pagmo::fair_replace>(m, r_pol, r_module, "fair_replace", fair_replace_docstring().c_str());
    detail::sr_policy_add_rate_constructor(fair_replace_);
}

} // namespace pygmo
