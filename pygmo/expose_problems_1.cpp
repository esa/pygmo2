// Copyright 2019 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <pagmo/detail/make_unique.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/schwefel.hpp>
#include <pagmo/problems/translate.hpp>
#include <pagmo/types.hpp>

#include "common_utils.hpp"
#include "docstrings.hpp"
#include "expose_problems.hpp"

namespace pygmo
{

namespace py = pybind11;

void expose_problems_1(py::module &m, py::class_<pagmo::problem> &prob, py::module &p_module)
{
    // Rosenbrock.
    auto rb = expose_problem<pagmo::rosenbrock>(m, prob, p_module, "rosenbrock", rosenbrock_docstring().c_str());
    rb.def(py::init<pagmo::vector_double::size_type>(), py::arg("dim"));
    rb.def("best_known", &best_known_wrapper<pagmo::rosenbrock>, problem_get_best_docstring("Rosenbrock").c_str());

    // Translate meta-problem
    auto translate_ = expose_problem<pagmo::translate>(m, prob, p_module, "translate", translate_docstring().c_str());
    translate_
        // NOTE: An __init__ wrapper on the Python side will take care of cting a pagmo::problem from the input UDP,
        // and then invoke this ctor. This way we avoid having to expose a different ctor for every exposed C++ prob.
        .def(py::init([](const pagmo::problem &p, const py::array_t<double> &tv) {
            return pagmo::detail::make_unique<pagmo::translate>(p, pygmo::ndarr_to_vector<pagmo::vector_double>(tv));
        }))
        .def_property_readonly(
            "translation",
            [](const pagmo::translate &t) { return pygmo::vector_to_ndarr<py::array_t<double>>(t.get_translation()); },
            translate_translation_docstring().c_str())
        .def_property_readonly(
            "inner_problem", [](pagmo::translate &udp) -> pagmo::problem & { return udp.get_inner_problem(); },
            py::return_value_policy::reference_internal, generic_udp_inner_problem_docstring().c_str());

    // Schwefel.
    auto sch = expose_problem<pagmo::schwefel>(m, prob, p_module, "schwefel",
                                               "__init__(dim = 1)\n\nThe Schwefel problem.\n\n"
                                               "See :cpp:class:`pagmo::schwefel`.\n\n");
    sch.def(py::init<unsigned>(), py::arg("dim"));
    sch.def("best_known", &pygmo::best_known_wrapper<pagmo::schwefel>, problem_get_best_docstring("Schwefel").c_str());
}

} // namespace pygmo
