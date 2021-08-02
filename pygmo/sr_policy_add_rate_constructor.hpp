// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PYGMO_SR_POLICY_ADD_RATE_CONSTRUCTOR_HPP
#define PYGMO_SR_POLICY_ADD_RATE_CONSTRUCTOR_HPP

#include <memory>

#include <pybind11/pybind11.h>

#include "common_utils.hpp"

namespace pygmo
{

namespace detail
{

namespace py = pybind11;

// An helper to add a constructor from a migration rate to a
// replacement/selection policy.
template <typename Pol>
inline void sr_policy_add_rate_constructor(py::class_<Pol> &c)
{
    c.def(py::init([](const py::object &o) -> std::unique_ptr<Pol> {
              if (py::isinstance(o, builtins().attr("int"))) {
                  return std::make_unique<Pol>(py::cast<int>(o));
              } else if (py::isinstance(o, builtins().attr("float"))) {
                  return std::make_unique<Pol>(py::cast<double>(o));
              } else {
                  py_throw(PyExc_TypeError,
                           ("cannot construct a replacement/selection policy from a migration rate of type '"
                            + str(type(o)) + "': the migration rate must be an integral or floating-point value")
                               .c_str());
              }
          }),
          py::arg("rate"));
}

} // namespace detail

} // namespace pygmo

#endif
