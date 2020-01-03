// Copyright 2019 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <pybind11/pybind11.h>

#include <pagmo/detail/make_unique.hpp>
#include <pagmo/population.hpp>
#include <pagmo/s11n.hpp>
#include <pagmo/threading.hpp>

#include "algorithm.hpp"
#include "common_utils.hpp"
#include "object_serialization.hpp"

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
    if (pygmo::type(o).is(py::module::import("pygmo").attr("algorithm"))) {
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
    return detail::make_unique<algo_inner>(m_value);
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

pagmo::thread_safety algo_inner<py::object>::get_thread_safety() const
{
    return pagmo::thread_safety::none;
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

template <typename Archive>
void algo_inner<py::object>::save(Archive &ar, unsigned) const
{
    ar << boost::serialization::base_object<algo_inner_base>(*this);
    ar << pygmo::object_to_vchar(m_value);
}

template <typename Archive>
void algo_inner<py::object>::load(Archive &ar, unsigned)
{
    ar >> boost::serialization::base_object<algo_inner_base>(*this);
    std::vector<char> v;
    ar >> v;
    m_value = pygmo::vchar_to_object(v);
}

} // namespace detail

} // namespace pagmo

PAGMO_S11N_ALGORITHM_IMPLEMENT(pybind11::object)

namespace pygmo
{

namespace py = pybind11;

// Serialization support for the algorithm class.
py::tuple algorithm_pickle_getstate(const pagmo::algorithm &a)
{
    // The idea here is that first we extract a char array
    // into which a has been serialized, then we turn
    // this object into a Python bytes object and return that.
    std::ostringstream oss;
    {
        boost::archive::binary_oarchive oarchive(oss);
        oarchive << a;
    }
    auto s = oss.str();
    return py::make_tuple(py::bytes(s.data(), boost::numeric_cast<py::size_t>(s.size())));
}

pagmo::algorithm algorithm_pickle_setstate(py::tuple state)
{
    // Similarly, first we extract a bytes object from the Python state,
    // and then we build a C++ string from it. The string is then used
    // to deserialized the object.
    if (py::len_hint(state) != 1) {
        pygmo::py_throw(PyExc_ValueError, ("the state tuple passed for algorithm deserialization "
                                           "must have 1 element, but instead it has "
                                           + std::to_string(py::len_hint(state)) + " element(s)")
                                              .c_str());
    }

    auto ptr = PyBytes_AsString(py::object(state[0]).ptr());
    if (!ptr) {
        pygmo::py_throw(PyExc_TypeError, "a bytes object is needed to deserialize an algorithm");
    }

    std::istringstream iss;
    iss.str(std::string(ptr, ptr + py::len_hint(state[0])));
    pagmo::algorithm a;
    {
        boost::archive::binary_iarchive iarchive(iss);
        iarchive >> a;
    }

    return a;
}

} // namespace pygmo
