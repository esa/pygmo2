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

#include <pagmo/algorithm.hpp>
#include <pagmo/island.hpp>
#include <pagmo/population.hpp>

#include "common_utils.hpp"
#include "handle_thread_py_exception.hpp"
#include "island.hpp"
#include "s11n_wrappers.hpp"

namespace pagmo
{

namespace detail
{

namespace py = pybind11;

isl_inner<py::object>::isl_inner(const py::object &o)
{
    // NOTE: unlike pygmo.problem/algorithm, we don't have to enforce
    // that o is not a pygmo.island, because pygmo.island does not satisfy
    // the UDI interface and thus check_mandatory_method() below will fail
    // if o is a pygmo.island.
    //
    // Check that o is an instance of a class, and not a type.
    check_not_type(o, "island");
    check_mandatory_method(o, "run_evolve", "island");
    m_value = pygmo::deepcopy(o);
}

std::unique_ptr<isl_inner_base> isl_inner<py::object>::clone() const
{
    // This will make a deep copy using the ctor above.
    return std::make_unique<isl_inner>(m_value);
}

void isl_inner<py::object>::run_evolve(island &isl) const
{
    // NOTE: run_evolve() is called from a separate thread in pagmo::island, need to construct a GTE before
    // doing anything with the interpreter (including the throws in the checks below).
    pygmo::gil_thread_ensurer gte;

    // NOTE: every time we call into the Python interpreter from a separate thread, we need to
    // handle Python exceptions in a special way.
    std::string isl_name;
    try {
        isl_name = get_name();
    } catch (const py::error_already_set &eas) {
        pygmo::handle_thread_py_exception("Could not fetch the name of a pythonic island. The error is:\n", eas);
    }

    try {
        auto ret = m_value.attr("run_evolve")(isl.get_algorithm(), isl.get_population());

        py::tuple ret_tup;
        try {
            ret_tup = py::cast<py::tuple>(ret);
        } catch (const py::cast_error &) {
            pygmo::py_throw(PyExc_TypeError, ("the 'run_evolve()' method of a user-defined island "
                                              "must return a tuple, but it returned an object of type '"
                                              + pygmo::str(pygmo::type(ret)) + "' instead")
                                                 .c_str());
        }
        if (py::len(ret_tup) != 2) {
            pygmo::py_throw(PyExc_ValueError,
                            ("the tuple returned by the 'run_evolve()' method of a user-defined island "
                             "must have 2 elements, but instead it has "
                             + std::to_string(py::len(ret_tup)) + " element(s)")
                                .c_str());
        }

        algorithm ret_algo;
        try {
            ret_algo = py::cast<algorithm>(ret_tup[0]);
        } catch (const py::cast_error &) {
            pygmo::py_throw(PyExc_TypeError,
                            ("the first value returned by the 'run_evolve()' method of a user-defined island "
                             "must be an algorithm, but an object of type '"
                             + pygmo::str(pygmo::type(ret_tup[0])) + "' was returned instead")
                                .c_str());
        }

        population ret_pop;
        try {
            ret_pop = py::cast<population>(ret_tup[1]);
        } catch (const py::cast_error &) {
            pygmo::py_throw(PyExc_TypeError,
                            ("the second value returned by the 'run_evolve()' method of a user-defined island "
                             "must be a population, but an object of type '"
                             + pygmo::str(pygmo::type(ret_tup[1])) + "' was returned instead")
                                .c_str());
        }

        isl.set_algorithm(ret_algo);
        isl.set_population(ret_pop);
    } catch (const py::error_already_set &eas) {
        pygmo::handle_thread_py_exception(
            "The asynchronous evolution of a pythonic island of type '" + isl_name + "' raised an error:\n", eas);
    }
}

std::string isl_inner<py::object>::get_name() const
{
    return getter_wrapper<std::string>(m_value, "get_name", pygmo::str(pygmo::type(m_value)));
}

std::string isl_inner<py::object>::get_extra_info() const
{
    return getter_wrapper<std::string>(m_value, "get_extra_info", std::string{});
}

std::type_index isl_inner<py::object>::get_type_index() const
{
    return std::type_index(typeid(py::object));
}

const void *isl_inner<py::object>::get_ptr() const
{
    return &m_value;
}

void *isl_inner<py::object>::get_ptr()
{
    return &m_value;
}

template <typename Archive>
void isl_inner<py::object>::save(Archive &ar, unsigned) const
{
    pygmo::inner_class_save<isl_inner_base>(ar, *this);
}

template <typename Archive>
void isl_inner<py::object>::load(Archive &ar, unsigned)
{
    pygmo::inner_class_load<isl_inner_base>(ar, *this);
}

} // namespace detail

} // namespace pagmo

PAGMO_S11N_ISLAND_IMPLEMENT(pybind11::object)
