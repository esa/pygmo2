// Copyright 2019 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef PYGMO_OBJECT_SERIALIZATION_HPP
#define PYGMO_OBJECT_SERIALIZATION_HPP

#include <vector>

#include <pybind11/pybind11.h>

namespace pygmo
{

namespace py = pybind11;

std::vector<char> object_to_vchar(const py::object &);

py::object vchar_to_object(const std::vector<char> &);

} // namespace pygmo

#endif
