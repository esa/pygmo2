// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <exception>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>

#include "handle_thread_py_exception.hpp"

namespace pygmo
{

namespace py = pybind11;

// Helper to handle Python exceptions thrown in a separate thread of execution not managed
// by Python.
void handle_thread_py_exception(const std::string &err, const py::error_already_set &eas)
{
    // NOTE: this function must be called only in a catch block.
    assert(std::current_exception());

    // Make sure to clean up the Python error indicator
    // for the current thread.
    ::PyErr_Clear();

    // NOTE: this helper is needed because sometimes one of the attributes
    // of eas is a default-constructed py::object(), which causes an error
    // when passed to format_exception() below. In such a case, this helper
    // will return None instead. See:
    // https://github.com/pybind/pybind11/issues/1543
    auto obj_or_none = [](const py::object &o) { return o ? o : py::none(); };

    // Try to extract a string description of the exception using the "traceback" module.
    std::string tmp(err);
    try {
        // NOTE: we are about to go back into the Python interpreter. Here Python could throw an exception
        // and set again the error indicator. In case of any issue,
        // we will give up any attempt of producing a meaningful error message, reset the error indicator,
        // and throw a pure C++ exception with a generic error message.
        tmp += py::cast<std::string>(
            py::str("").attr("join")(py::module::import("traceback")
                                         .attr("format_exception")(obj_or_none(eas.type()), obj_or_none(eas.value()),
                                                                   obj_or_none(eas.trace()))));
    } catch (const py::error_already_set &) {
        // The block above threw from Python. There's not much we can do.
        ::PyErr_Clear();
        throw std::runtime_error("While trying to analyze the error message of a Python exception raised in a "
                                 "separate thread, another Python exception was raised. Giving up now.");
    }

    // Throw the C++ exception.
    throw std::runtime_error(tmp);
}

} // namespace pygmo
