// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PYGMO_S11N_WRAPPERS_HPP
#define PYGMO_S11N_WRAPPERS_HPP

#include <cstddef>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/binary_object.hpp>

#include <pybind11/pybind11.h>

#include "common_utils.hpp"

namespace pygmo
{

namespace py = pybind11;

// Two helpers to implement s11n for the *_inner classes
// specialisations for py::object. d is an instance of the *_inner
// class, Base its base type.
template <typename Base, typename Archive, typename Derived>
inline void inner_class_save(Archive &ar, const Derived &d)
{
    static_assert(std::is_base_of_v<Base, Derived>);

    // Serialize the base class.
    ar << boost::serialization::base_object<Base>(d);

    // This will dump m_value into a bytes object..
    auto tmp = py::module::import("pygmo").attr("get_serialization_backend")().attr("dumps")(d.m_value);

    // This gives a null-terminated char * to the internal
    // content of the bytes object.
    auto ptr = PyBytes_AsString(tmp.ptr());
    if (!ptr) {
        py_throw(PyExc_TypeError, "The serialization backend's dumps() function did not return a bytes object");
    }

    // NOTE: this will be the length of the bytes object *without* the terminator.
    const auto size = boost::numeric_cast<std::size_t>(py::len(tmp));

    // Save the binary size.
    ar << size;

    // Save the binary object.
    ar << boost::serialization::make_binary_object(ptr, size);
}

template <typename Base, typename Archive, typename Derived>
inline void inner_class_load(Archive &ar, Derived &d)
{
    static_assert(std::is_base_of_v<Base, Derived>);

    // Deserialize the base class.
    ar >> boost::serialization::base_object<Base>(d);

    // Recover the size.
    std::size_t size{};
    ar >> size;

    // Recover the binary object.
    std::vector<char> tmp;
    tmp.resize(boost::numeric_cast<decltype(tmp.size())>(size));
    ar >> boost::serialization::make_binary_object(tmp.data(), size);

    // Deserialise and assign.
    auto b = py::bytes(tmp.data(), boost::numeric_cast<py::size_t>(size));
    d.m_value = py::module::import("pygmo").attr("get_serialization_backend")().attr("loads")(b);
}

// Helpers to implement pickling on top of Boost.Serialization.
template <typename T>
inline py::tuple pickle_getstate_wrapper(const T &x)
{
    std::ostringstream oss;
    {
        boost::archive::binary_oarchive oa(oss);
        oa << x;
    }

    return py::make_tuple(py::bytes(oss.str()));
}

template <typename T>
inline T pickle_setstate_wrapper(py::tuple state)
{
    if (py::len(state) != 1) {
        py_throw(PyExc_ValueError, ("The state tuple passed to the deserialization wrapper "
                                    "must have 1 element, but instead it has "
                                    + std::to_string(py::len(state)) + " element(s)")
                                       .c_str());
    }

    auto ptr = PyBytes_AsString(state[0].ptr());
    if (!ptr) {
        py_throw(PyExc_TypeError, "A bytes object is needed in the deserialization wrapper");
    }

    std::istringstream iss;
    iss.str(std::string(ptr, ptr + py::len(state[0])));
    T x;
    {
        boost::archive::binary_iarchive iarchive(iss);
        iarchive >> x;
    }

    return x;
}

} // namespace pygmo

#endif
