// Copyright 2019 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <string>

#include <boost/numeric/conversion/cast.hpp>

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

gil_releaser::gil_releaser()
{
    m_thread_state = PyEval_SaveThread();
}

gil_releaser::~gil_releaser()
{
    PyEval_RestoreThread(m_thread_state);
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

// Perform a deep copy of input object o.
py::object deepcopy(const py::object &o)
{
    return py::module::import("copy").attr("deepcopy")(o);
}

// Convert a numpy array into a sparsity pattern.
pagmo::sparsity_pattern ndarr_to_sp(const py::array_t<pagmo::vector_double::size_type> &a)
{
    // Get a two-dimensional view on the array.
    // If the array is not 2D, this will throw.
    auto r = a.template unchecked<2>();

    if (r.shape(1) != 2) {
        py_throw(PyExc_ValueError, ("when converting a numpy array into a sparsity pattern, the number of columns in "
                                    "the array must be 2, but it is "
                                    + std::to_string(r.shape(1)) + " instead")
                                       .c_str());
    }

    // Prepare the retval.
    pagmo::sparsity_pattern retval(boost::numeric_cast<pagmo::sparsity_pattern::size_type>(r.shape(0)));

    for (py::ssize_t i = 0; i < r.shape(0); ++i) {
        retval[static_cast<pagmo::sparsity_pattern::size_type>(i)].first = r(i, 0);
        retval[static_cast<pagmo::sparsity_pattern::size_type>(i)].second = r(i, 1);
    }

    return retval;
}

// Convert a sparsity pattern into a numpy array.
py::array_t<pagmo::vector_double::size_type> sp_to_ndarr(const pagmo::sparsity_pattern &sp)
{
    // Create the output array, of shape n x 2.
    py::array_t<pagmo::vector_double::size_type> retval({boost::numeric_cast<py::ssize_t>(sp.size()), py::ssize_t(2)});

    // Get a mutable view into it and copy the data from sp.
    auto r = retval.mutable_unchecked<2>();
    for (decltype(sp.size()) i = 0; i < sp.size(); ++i) {
        r(static_cast<py::ssize_t>(i), 0) = sp[i].first;
        r(static_cast<py::ssize_t>(i), 1) = sp[i].second;
    }

    return retval;
}

} // namespace pygmo
