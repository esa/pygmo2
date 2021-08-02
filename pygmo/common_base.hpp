// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PYGMO_COMMON_BASE_HPP
#define PYGMO_COMMON_BASE_HPP

#include <pybind11/pybind11.h>

#include "common_utils.hpp"

namespace pygmo
{

namespace py = pybind11;

// A common base class with methods useful in the implementation of
// the pythonic problem, algorithm, etc.
struct common_base {
    static void check_mandatory_method(const py::object &, const char *, const char *);
    // A simple wrapper for getters. It will try to:
    // - get the attribute "name" from the object o,
    // - call it without arguments,
    // - extract an instance from the ret value and return it.
    // If the attribute is not there or it is not callable, the value "def_value" will be returned.
    template <typename RetType>
    static RetType getter_wrapper(const py::object &o, const char *name, const RetType &def_value)
    {
        auto a = callable_attribute(o, name);
        if (a.is_none()) {
            return def_value;
        }
        return py::cast<RetType>(a());
    }
    static void check_not_type(const py::object &, const char *);
};

} // namespace pygmo

#endif
