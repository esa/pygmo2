// Copyright 2019 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PYGMO_COMMON_UTILS_HPP
#define PYGMO_COMMON_UTILS_HPP

#include <sstream>
#include <string>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <pagmo/types.hpp>

namespace pygmo
{

namespace py = pybind11;

// Import and return the builtins module.
py::object builtins();

// Throw a Python exception.
[[noreturn]] void py_throw(PyObject *, const char *);

// This RAII struct ensures that the Python interpreter
// can be used from a thread created from outside Python
// (e.g., a pthread/std::thread/etc. created from C/C++).
// On creation, it will register the C/C++ thread with
// the Python interpreter and lock the GIL. On destruction,
// it will release any resource acquired on construction
// and unlock the GIL.
//
// See: https://docs.python.org/3/c-api/init.html
struct gil_thread_ensurer {
    gil_thread_ensurer();
    ~gil_thread_ensurer();

    // Make sure we don't accidentally try to copy/move it.
    gil_thread_ensurer(const gil_thread_ensurer &) = delete;
    gil_thread_ensurer(gil_thread_ensurer &&) = delete;
    gil_thread_ensurer &operator=(const gil_thread_ensurer &) = delete;
    gil_thread_ensurer &operator=(gil_thread_ensurer &&) = delete;

    PyGILState_STATE m_state;
};

// This RAII struct will unlock the GIL on construction,
// and lock it again on destruction.
//
// See: https://docs.python.org/3/c-api/init.html
struct gil_releaser {
    gil_releaser();
    ~gil_releaser();

    // Make sure we don't accidentally try to copy/move it.
    gil_releaser(const gil_releaser &) = delete;
    gil_releaser(gil_releaser &&) = delete;
    gil_releaser &operator=(const gil_releaser &) = delete;
    gil_releaser &operator=(gil_releaser &&) = delete;

    PyThreadState *m_thread_state;
};

// Check if 'o' has a callable attribute (i.e., a method) named 's'. If so, it will
// return the attribute, otherwise it will return None.
py::object callable_attribute(const py::object &, const char *);

// Check if type is callable.
bool callable(const py::object &);

// Get the string representation of an object.
std::string str(const py::object &);

// Get the type of an object.
py::object type(const py::object &);

// Perform a deep copy of input object o.
py::object deepcopy(const py::object &);

// repr() via ostream.
template <typename T>
inline std::string ostream_repr(const T &x)
{
    std::ostringstream oss;
    oss << x;
    return oss.str();
}

// Generic copy wrappers.
template <typename T>
inline T generic_copy_wrapper(const T &x)
{
    return x;
}

template <typename T>
inline T generic_deepcopy_wrapper(const T &x, py::dict)
{
    return x;
}

// Convert an input 1D numpy array into a C++ vector.
template <typename Vector, typename T, int ExtraFlags>
inline Vector ndarr_to_vector(const py::array_t<T, ExtraFlags> &a)
{
    // Get a one-dimensional view on the array.
    // If the array is not 1D, this will throw.
    auto r = a.template unchecked<1>();

    // Prepare the output vector with the
    // correct size.
    Vector retval(boost::numeric_cast<typename Vector::size_type>(r.shape(0)));

    // Copy the values from a into retval.
    for (py::ssize_t i = 0; i < r.shape(0); ++i) {
        retval[static_cast<typename Vector::size_type>(i)] = r(i);
    }

    return retval;
}

// Convert a vector of something into a 1D numpy array.
template <typename Array, typename T, typename Allocator>
inline Array vector_to_ndarr(const std::vector<T, Allocator> &v)
{
    return Array(v.size(), v.data());
}

// Convert a numpy array into a sparsity pattern.
pagmo::sparsity_pattern ndarr_to_sp(const py::array_t<pagmo::vector_double::size_type> &);

// Convert a sparsity pattern into a numpy array.
py::array_t<pagmo::vector_double::size_type> sp_to_ndarr(const pagmo::sparsity_pattern &);

} // namespace pygmo

#endif
