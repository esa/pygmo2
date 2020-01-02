// Copyright 2019 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <utility>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <pagmo/detail/make_unique.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/translate.hpp>
#include <pagmo/types.hpp>

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
    // NOTE: An __init__ wrapper on the Python side will take care of cting a pagmo::problem from the input UDP,
    // and then invoke this ctor. This way we avoid having to expose a different ctor for every exposed C++ prob.
    translate_.def(py::init([](const pagmo::problem &p, const py::array_t<double> &tv) {
        return pagmo::detail::make_unique<pagmo::translate>(p, pygmo::ndarr_to_vector<pagmo::vector_double>(tv));
    }));

#if 0
    // NOTE: An __init__ wrapper on the Python side will take care of cting a pagmo::problem from the input UDP,
    // and then invoke this ctor. This way we avoid having to expose a different ctor for every exposed C++ prob.
    translate_.def("__init__", bp::make_constructor(lcast([](const problem &p, const bp::object &tv) {
                                                        return ::new translate(p, obj_to_vector<vector_double>(tv));
                                                    }),
                                                    bp::default_call_policies()));
    add_property(translate_, "translation",
                 lcast([](const translate &t) { return vector_to_ndarr(t.get_translation()); }),
                 translate_translation_docstring().c_str());
    add_property(translate_, "inner_problem",
                 bp::make_function(lcast([](translate &udp) -> problem & { return udp.get_inner_problem(); }),
                                   bp::return_internal_reference<>()),
                 generic_udp_inner_problem_docstring().c_str());
#endif
}

} // namespace pygmo
