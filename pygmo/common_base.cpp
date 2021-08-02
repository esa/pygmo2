// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <string>

#include <pybind11/pybind11.h>

#include "common_base.hpp"
#include "common_utils.hpp"

namespace pygmo
{

namespace py = pybind11;

// Check if a mandatory method is present in a user-defined entity.
void common_base::check_mandatory_method(const py::object &o, const char *s, const char *target)
{
    if (callable_attribute(o, s).is_none()) {
        py_throw(PyExc_NotImplementedError,
                 ("the mandatory '" + std::string(s) + "()' method has not been detected in the user-defined Python "
                  + std::string(target) + " '" + str(o) + "' of type '" + str(type(o))
                  + "': the method is either not present or not callable")
                     .c_str());
    }
}

// Check if the user is trying to construct a pagmo object from a type, rather than from an object.
// This is an easy error to commit, and it is sneaky because the callable_attribute() machinery will detect
// the methods of the *class* (rather than instance methods), and it will thus not error out.
void common_base::check_not_type(const py::object &o, const char *target)
{
    if (py::isinstance(o, builtins().attr("type"))) {
        py_throw(PyExc_TypeError, ("it seems like you are trying to instantiate a pygmo " + std::string(target)
                                   + " using a type rather than an object instance: please construct an object "
                                     "and use that instead of the type in the "
                                   + std::string(target) + " constructor")
                                      .c_str());
    }
}

} // namespace pygmo
