// Copyright 2019 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <memory>

#include <pybind11/pybind11.h>

#include <pagmo/algorithm.hpp>
#include <pagmo/detail/make_unique.hpp>
#include <pagmo/island.hpp>
#include <pagmo/islands/thread_island.hpp>
#include <pagmo/population.hpp>
#include <pagmo/threading.hpp>

#include "common_utils.hpp"
#include "island.hpp"

namespace py = pybind11;
namespace pg = pagmo;

PYBIND11_MODULE(core, m)
{
    // This function needs to be called before doing anything with threads.
    // https://docs.python.org/3/c-api/init.html
    PyEval_InitThreads();

    // Check if cloudpickle is available.
    try {
        py::module::import("cloudpickle");
    } catch (...) {
        py::module::import("warnings")
            .attr("warn")("The 'cloudpickle' module could not be imported. Please make sure that cloudpickle has been "
                          "correctly installed, since pygmo depends on it.",
                          pygmo::builtins().attr("RuntimeWarning"));
        PyErr_Clear();
    }

    // Check if numpy is available.
    try {
        py::module::import("numpy");
    } catch (...) {
        py::module::import("warnings")
            .attr("warn")("The 'numpy' module could not be imported. Please make sure that numpy has been "
                          "correctly installed, since pygmo depends on it.",
                          pygmo::builtins().attr("RuntimeWarning"));
        PyErr_Clear();
    }

    // Expose some internal functions for testing.
    m.def("_callable", &pygmo::callable);
    m.def("_callable_attribute", &pygmo::callable_attribute);
    m.def("_str", &pygmo::str);
    m.def("_type", &pygmo::type);

    // Override the default implementation of the island factory.
    pg::detail::island_factory
        = [](const pg::algorithm &algo, const pg::population &pop, std::unique_ptr<pg::detail::isl_inner_base> &ptr) {
              if (algo.get_thread_safety() >= pg::thread_safety::basic
                  && pop.get_problem().get_thread_safety() >= pg::thread_safety::basic) {
                  // Both algo and prob have at least the basic thread safety guarantee. Use the thread island.
                  ptr = pg::detail::make_unique<pg::detail::isl_inner<pg::thread_island>>();
              } else {
                  // NOTE: here we are re-implementing a piece of code that normally
                  // is pure C++. We are calling into the Python interpreter, so, in order to handle
                  // the case in which we are invoking this code from a separate C++ thread, we construct a GIL ensurer
                  // in order to guard against concurrent access to the interpreter. The idea here is that this piece
                  // of code normally would provide a basic thread safety guarantee, and in order to continue providing
                  // it we use the ensurer.
                  pygmo::gil_thread_ensurer gte;
                  auto py_island = py::module::import("pygmo").attr("mp_island");
                  // ptr = pg::detail::make_unique<pg::detail::isl_inner<py::object>>(py_island());
              }
          };
}
