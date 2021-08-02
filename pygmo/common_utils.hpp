// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PYGMO_COMMON_UTILS_HPP
#define PYGMO_COMMON_UTILS_HPP

#include <initializer_list>
#include <sstream>
#include <string>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <pagmo/topology.hpp>
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
    return Array(boost::numeric_cast<py::ssize_t>(v.size()), v.data());
}

// Convert a numpy array into a sparsity pattern.
pagmo::sparsity_pattern ndarr_to_sp(const py::array_t<pagmo::vector_double::size_type> &);

// Convert a sparsity pattern into a numpy array.
py::array_t<pagmo::vector_double::size_type> sp_to_ndarr(const pagmo::sparsity_pattern &);

// Generic extract() wrappers.
template <typename C, typename T>
inline T *generic_cpp_extract(C &c, const T &)
{
    return c.template extract<T>();
}

template <typename C>
inline py::object generic_py_extract(C &c, const py::object &t)
{
    auto ptr = c.template extract<py::object>();
    if (ptr && (t.equal(type(*ptr)) || t.equal(builtins().attr("object")))) {
        // c contains a user-defined pythonic entity and either:
        // - the type passed in by the user is the exact type of the user-defined
        //   entity, or
        // - the user supplied as t the builtin 'object' type (which we use as a
        //   wildcard for any Python type).
        // Let's return the extracted object.
        return *ptr;
    }

    // Either the user-defined entity is not pythonic, or the user specified the
    // wrong type. Return None.
    return py::none();
}

// Convert a vector of vectors into a 2D numpy array.
template <typename Array, typename T, typename A1, typename A2>
inline Array vvector_to_ndarr(const std::vector<std::vector<T, A2>, A1> &v)
{
    // The dimensions of the array to be created.
    const auto nrows = v.size();
    const auto ncols = nrows ? v[0].size() : 0u;

    // Create the output array.
    Array retval({boost::numeric_cast<py::ssize_t>(nrows), boost::numeric_cast<py::ssize_t>(ncols)});

    // Get a mutable view into it and copy the data from sp.
    auto r = retval.template mutable_unchecked<2>();
    for (decltype(v.size()) i = 0; i < nrows; ++i) {
        if (v[i].size() != ncols) {
            py_throw(PyExc_ValueError, "cannot convert a vector of vectors to a NumPy 2D array "
                                       "if the vector instances don't have all the same size");
        }
        for (decltype(v[i].size()) j = 0; j < ncols; ++j) {
            r(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j)) = v[i][j];
        }
    }

    return retval;
}

// Convert an input 2D numpy array into a vector of vectors.
template <typename Vector, typename T, int ExtraFlags>
inline Vector ndarr_to_vvector(const py::array_t<T, ExtraFlags> &a)
{
    // Get a 2D view on the array.
    // If the array is not 2D, this will throw.
    auto r = a.template unchecked<2>();

    // Prepare the output vector with the
    // correct size.
    Vector retval(boost::numeric_cast<typename Vector::size_type>(r.shape(0)));

    // Copy the values from a into retval.
    for (py::ssize_t i = 0; i < r.shape(0); ++i) {
        retval[static_cast<decltype(retval.size())>(i)].resize(
            boost::numeric_cast<decltype(retval[i].size())>(r.shape(1)));

        for (py::ssize_t j = 0; j < r.shape(1); ++j) {
            retval[static_cast<decltype(retval.size())>(i)][static_cast<decltype(retval[i].size())>(j)] = r(i, j);
        }
    }

    return retval;
}

// Convert an individuals_group_t into a Python tuple of:
// - 1D integral array of IDs,
// - 2D float array of dvs,
// - 2D float array of fvs.
py::tuple inds_to_tuple(const pagmo::individuals_group_t &);

// Convert a Python iterable into an individuals_group_t.
pagmo::individuals_group_t iterable_to_inds(const py::iterable &);

// Convert a Python iterable into a problem bounds.
std::pair<pagmo::vector_double, pagmo::vector_double> iterable_to_bounds(const py::iterable &o);

// Conversion between BGL and NetworkX.
py::object bgl_graph_t_to_networkx(const pagmo::bgl_graph_t &);
pagmo::bgl_graph_t networkx_to_bgl_graph_t(const py::object &);

} // namespace pygmo

#endif
