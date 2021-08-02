// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PYGMO_EXPOSE_ALGORITHMS_HPP
#define PYGMO_EXPOSE_ALGORITHMS_HPP

#include <string>

#include <boost/any.hpp>

#include <pybind11/pybind11.h>

#include <pagmo/algorithm.hpp>
#include <pagmo/population.hpp>

#include "common_utils.hpp"
#include "docstrings.hpp"

namespace pygmo
{

namespace py = pybind11;

// Split algorithm exposition functions.
void expose_algorithms_0(py::module &, py::class_<pagmo::algorithm> &, py::module &);
void expose_algorithms_1(py::module &, py::class_<pagmo::algorithm> &, py::module &);

// C++ UDA exposition function.
template <typename Algo>
inline py::class_<Algo> expose_algorithm(py::module &m, py::class_<pagmo::algorithm> &algo, py::module &a_module,
                                         const char *name, const char *descr)
{
    py::class_<Algo> c(m, name, descr);

    // We require all algos to be def-ctible at the bare minimum.
    c.def(py::init<>());

    // Mark it as a C++ algorithm.
    c.attr("_pygmo_cpp_algorithm") = true;

    // Expose the algorithm constructor from Algo.
    algo.def(py::init<const Algo &>(), py::arg("uda"));

    // Expose extract.
    algo.def("_cpp_extract", &generic_cpp_extract<pagmo::algorithm, Algo>, py::return_value_policy::reference_internal);

    // Add the algorithm to the algorithms submodule.
    a_module.attr(name) = c;

    return c;
}

// Utils to expose algo log.
template <typename Algo>
inline py::list generic_log_getter(const Algo &a)
{
    py::list retval;
    for (const auto &t : a.get_log()) {
        retval.append(t);
    }
    return retval;
}

template <typename Algo>
inline void expose_algo_log(py::class_<Algo> &algo_class, const char *doc)
{
    algo_class.def("get_log", &generic_log_getter<Algo>, doc);
}

// Helper for the exposition of algorithms
// inheriting from not_population_based.
template <typename T>
inline void expose_not_population_based(py::class_<T> &c, const std::string &algo_name)
{
    // Selection/replacement.
    c.def_property(
        "selection",
        [](const T &n) -> py::object {
            auto s = n.get_selection();
            if (boost::any_cast<std::string>(&s)) {
                return py::str(boost::any_cast<std::string>(s));
            }
            return py::cast(boost::any_cast<pagmo::population::size_type>(s));
        },
        [](T &n, const py::object &o) {
            try {
                n.set_selection(py::cast<std::string>(o));
                return;
            } catch (const py::cast_error &) {
            }
            try {
                n.set_selection(py::cast<pagmo::population::size_type>(o));
                return;
            } catch (const py::cast_error &) {
            }
            py_throw(PyExc_TypeError,
                     ("cannot convert the input object '" + str(o) + "' of type '" + str(type(o))
                      + "' to either a selection policy (one of ['best', 'worst', 'random']) or an individual index")
                         .c_str());
        },
        bls_selection_docstring(algo_name).c_str());
    c.def_property(
        "replacement",
        [](const T &n) -> py::object {
            auto s = n.get_replacement();
            if (boost::any_cast<std::string>(&s)) {
                return py::str(boost::any_cast<std::string>(s));
            }
            return py::cast(boost::any_cast<pagmo::population::size_type>(s));
        },
        [](T &n, const py::object &o) {
            try {
                n.set_replacement(py::cast<std::string>(o));
                return;
            } catch (const py::cast_error &) {
            }
            try {
                n.set_replacement(py::cast<pagmo::population::size_type>(o));
                return;
            } catch (const py::cast_error &) {
            }
            py_throw(PyExc_TypeError,
                     ("cannot convert the input object '" + str(o) + "' of type '" + str(type(o))
                      + "' to either a replacement policy (one of ['best', 'worst', 'random']) or an individual index")
                         .c_str());
        },
        bls_replacement_docstring(algo_name).c_str());
    c.def("set_random_sr_seed", &T::set_random_sr_seed, bls_set_random_sr_seed_docstring(algo_name).c_str());
}

} // namespace pygmo

#endif
