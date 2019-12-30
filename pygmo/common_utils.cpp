// Copyright 2019 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <string>

#include <pybind11/pybind11.h>

#include "common_utils.hpp"

namespace pygmo
{

namespace py = pybind11;

// Import and return the builtins module.
py::object builtins()
{
    return py::module::import("builtins");
}

// Throw a Python exception of type "type" with associated
// error message "msg".
void py_throw(PyObject *type, const char *msg)
{
    PyErr_SetString(type, msg);
    throw py::error_already_set();
}

gil_thread_ensurer::gil_thread_ensurer()
{
    // NOTE: follow the Python C API to the letter,
    // using an assignment rather than constructing
    // m_state in the init list above.
    m_state = PyGILState_Ensure();
}

gil_thread_ensurer::~gil_thread_ensurer()
{
    PyGILState_Release(m_state);
}

// Check if 'o' has a callable attribute (i.e., a method) named 's'. If so, it will
// return the attribute, otherwise it will return None.
py::object callable_attribute(const py::object &o, const char *s)
{
    if (py::hasattr(o, s)) {
        auto retval = o.attr(s);
        if (callable(retval)) {
            return retval;
        }
    }
    return py::none();
}

// Check if type is callable.
bool callable(const py::object &o)
{
    return py::cast<bool>(builtins().attr("callable")(o));
}

// Get the string representation of an object.
std::string str(const py::object &o)
{
    return py::cast<std::string>(py::str(o));
}

// Get the type of an object.
py::object type(const py::object &o)
{
    return builtins().attr("type")(o);
}

} // namespace pygmo
