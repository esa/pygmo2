// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>

#include <pybind11/pybind11.h>

#include <pagmo/population.hpp>
#include <pagmo/threading.hpp>

#include "algorithm.hpp"
#include "common_utils.hpp"
#include "s11n_wrappers.hpp"

namespace pagmo
{

namespace detail
{

namespace py = pybind11;

algo_inner<py::object>::algo_inner(const py::object &o)
{
    // Forbid the use of a pygmo.algorithm as a UDA.
    // The motivation here is consistency with C++. In C++, the use of
    // a pagmo::algorithm as a UDA is forbidden and prevented by the fact
    // that the generic constructor from UDA is disabled if the input
    // object is a pagmo::algorithm (the copy/move constructor is
    // invoked instead). In order to achieve an equivalent behaviour
    // in pygmo, we throw an error if o is a algorithm, and instruct
    // the user to employ the standard copy/deepcopy facilities
    // for creating a copy of the input algorithm.
    if (pygmo::type(o).equal(py::module::import("pygmo").attr("algorithm"))) {
        pygmo::py_throw(
            PyExc_TypeError,
            ("a pygmo.algorithm cannot be used as a UDA for another pygmo.algorithm (if you need to copy an "
             "algorithm please use the standard Python copy()/deepcopy() functions)"));
    }
    // Check that o is an instance of a class, and not a type.
    check_not_type(o, "algorithm");
    check_mandatory_method(o, "evolve", "algorithm");
    m_value = pygmo::deepcopy(o);
}

std::unique_ptr<algo_inner_base> algo_inner<py::object>::clone() const
{
    // This will make a deep copy using the ctor above.
    return std::make_unique<algo_inner>(m_value);
}

population algo_inner<py::object>::evolve(const population &pop) const
{
    return py::cast<population>(m_value.attr("evolve")(pop));
}

void algo_inner<py::object>::set_seed(unsigned n)
{
    auto ss = pygmo::callable_attribute(m_value, "set_seed");
    if (ss.is_none()) {
        pygmo::py_throw(PyExc_NotImplementedError,
                        ("set_seed() has been invoked but it is not implemented "
                         "in the user-defined Python algorithm '"
                         + pygmo::str(m_value) + "' of type '" + pygmo::str(pygmo::type(m_value))
                         + "': the method is either not present or not callable")
                            .c_str());
    }
    ss(n);
}

bool algo_inner<py::object>::has_set_seed() const
{
    auto ss = pygmo::callable_attribute(m_value, "set_seed");
    if (ss.is_none()) {
        return false;
    }
    auto hss = pygmo::callable_attribute(m_value, "has_set_seed");
    if (hss.is_none()) {
        return true;
    }
    return py::cast<bool>(hss());
}

thread_safety algo_inner<py::object>::get_thread_safety() const
{
    return thread_safety::none;
}

std::string algo_inner<py::object>::get_name() const
{
    return getter_wrapper<std::string>(m_value, "get_name", pygmo::str(pygmo::type(m_value)));
}

std::string algo_inner<py::object>::get_extra_info() const
{
    return getter_wrapper<std::string>(m_value, "get_extra_info", std::string{});
}

void algo_inner<py::object>::set_verbosity(unsigned n)
{
    auto sv = pygmo::callable_attribute(m_value, "set_verbosity");
    if (sv.is_none()) {
        pygmo::py_throw(PyExc_NotImplementedError,
                        ("set_verbosity() has been invoked but it is not implemented "
                         "in the user-defined Python algorithm '"
                         + pygmo::str(m_value) + "' of type '" + pygmo::str(pygmo::type(m_value))
                         + "': the method is either not present or not callable")
                            .c_str());
    }
    sv(n);
}

bool algo_inner<py::object>::has_set_verbosity() const
{
    auto sv = pygmo::callable_attribute(m_value, "set_verbosity");
    if (sv.is_none()) {
        return false;
    }
    auto hsv = pygmo::callable_attribute(m_value, "has_set_verbosity");
    if (hsv.is_none()) {
        return true;
    }
    return py::cast<bool>(hsv());
}

std::type_index algo_inner<py::object>::get_type_index() const
{
    return std::type_index(typeid(py::object));
}

const void *algo_inner<py::object>::get_ptr() const
{
    return &m_value;
}

void *algo_inner<py::object>::get_ptr()
{
    return &m_value;
}

template <typename Archive>
void algo_inner<py::object>::save(Archive &ar, unsigned) const
{
    pygmo::inner_class_save<algo_inner_base>(ar, *this);
}

template <typename Archive>
void algo_inner<py::object>::load(Archive &ar, unsigned)
{
    pygmo::inner_class_load<algo_inner_base>(ar, *this);
}

} // namespace detail

} // namespace pagmo

PAGMO_S11N_ALGORITHM_IMPLEMENT(pybind11::object)
