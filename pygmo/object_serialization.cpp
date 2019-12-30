// Copyright 2019 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <pybind11/pybind11.h>

#include "common_utils.hpp"
#include "object_serialization.hpp"

namespace pygmo
{

namespace py = pybind11;

std::vector<char> object_to_vchar(const py::object &o)
{
    // This will dump to a bytes object.
    auto tmp = py::module::import("pygmo").attr("get_serialization_backend")().attr("dumps")(o);

    // This gives a null-terminated char * to the internal
    // content of the bytes object.
    auto ptr = PyBytes_AsString(tmp.ptr());
    if (!ptr) {
        py_throw(PyExc_TypeError, "the serialization backend's dumps() function did not return a bytes object");
    }

    // NOTE: this will be the length of the bytes object *without* the terminator.
    const auto size = py::len_hint(tmp);

    // NOTE: we store as char here because that's what is returned by the CPython function.
    // From Python it seems like these are unsigned chars, but this should not concern us.
    return std::vector<char>(ptr, ptr + size);
}

py::object vchar_to_object(const std::vector<char> &v)
{
    auto b = py::bytes(v.data(), boost::numeric_cast<std::size_t>(v.size()));
    return py::module::import("pygmo").attr("get_serialization_backend")().attr("loads")(b);
}

} // namespace pygmo
