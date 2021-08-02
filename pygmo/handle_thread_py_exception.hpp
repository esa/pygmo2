// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PYGMO_HANDLE_THREAD_PY_EXCEPTION_HPP
#define PYGMO_HANDLE_THREAD_PY_EXCEPTION_HPP

#include <string>

#include <pybind11/pybind11.h>

namespace pygmo
{

namespace py = pybind11;

[[noreturn]] void handle_thread_py_exception(const std::string &, const py::error_already_set &);

} // namespace pygmo

#endif
