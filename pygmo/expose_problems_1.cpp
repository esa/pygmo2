// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <memory>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/cec2013.hpp>
#include <pagmo/problems/golomb_ruler.hpp>
#include <pagmo/problems/luksan_vlcek1.hpp>
#include <pagmo/problems/minlp_rastrigin.hpp>
#include <pagmo/problems/rastrigin.hpp>
#include <pagmo/problems/rosenbrock.hpp>
#include <pagmo/problems/schwefel.hpp>
#include <pagmo/problems/translate.hpp>
#include <pagmo/problems/unconstrain.hpp>
#include <pagmo/problems/wfg.hpp>
#include <pagmo/problems/zdt.hpp>
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
            return std::make_unique<pagmo::translate>(p, pygmo::ndarr_to_vector<pagmo::vector_double>(tv));
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

    // ZDT.
    auto zdt_p = expose_problem<pagmo::zdt>(m, prob, p_module, "zdt",
                                            "__init__(prob_id = 1, param = 30)\n\nThe ZDT problem.\n\n"
                                            "See :cpp:class:`pagmo::zdt`.\n\n");
    zdt_p.def(py::init<unsigned, unsigned>(), py::arg("prob_id") = 1u, py::arg("param") = 30u);
    zdt_p.def("p_distance", [](const pagmo::zdt &z, const py::array_t<double> &x) {
        return z.p_distance(pygmo::ndarr_to_vector<pagmo::vector_double>(x));
    });
    zdt_p.def(
        "p_distance", [](const pagmo::zdt &z, const pagmo::population &pop) { return z.p_distance(pop); },
        zdt_p_distance_docstring().c_str());

    // Golomb Ruler
    auto gr = expose_problem<pagmo::golomb_ruler>(m, prob, p_module, "golomb_ruler",
                                                  "__init__(order, upper_bound)\n\nThe Golomb Ruler Problem.\n\n"
                                                  "See :cpp:class:`pagmo::golomb_ruler`.\n\n");
    gr.def(py::init<unsigned, unsigned>(), py::arg("order"), py::arg("upper_bound"));

    auto cec2013_ = expose_problem<pagmo::cec2013>(m, prob, p_module, "cec2013", cec2013_docstring().c_str());
    cec2013_.def(py::init<unsigned, unsigned>(), py::arg("prob_id") = 1, py::arg("dim") = 2);

    // Luksan Vlcek 1
    auto lv_
        = expose_problem<pagmo::luksan_vlcek1>(m, prob, p_module, "luksan_vlcek1", luksan_vlcek1_docstring().c_str());
    lv_.def(py::init<unsigned>(), py::arg("dim"));

    // MINLP-Rastrigin.
    auto minlp_rastr = expose_problem<pagmo::minlp_rastrigin>(m, prob, p_module, "minlp_rastrigin",
                                                              minlp_rastrigin_docstring().c_str());
    minlp_rastr.def(py::init<unsigned, unsigned>(), py::arg("dim_c") = 1u, py::arg("dim_i") = 1u);

    // Rastrigin.
    auto rastr = expose_problem<pagmo::rastrigin>(m, prob, p_module, "rastrigin",
                                                  "__init__(dim = 1)\n\nThe Rastrigin problem.\n\n"
                                                  "See :cpp:class:`pagmo::rastrigin`.\n\n");
    rastr.def(py::init<unsigned>(), py::arg("dim") = 1);
    rastr.def("best_known", &best_known_wrapper<pagmo::rastrigin>, problem_get_best_docstring("Rastrigin").c_str());

    // Unconstrain meta-problem.
    auto unconstrain_
        = expose_problem<pagmo::unconstrain>(m, prob, p_module, "unconstrain", unconstrain_docstring().c_str());
    // NOTE: An __init__ wrapper on the Python side will take care of cting a pagmo::problem from the input UDP,
    // and then invoke this ctor. This way we avoid having to expose a different ctor for every exposed C++ prob.
    unconstrain_
        .def(py::init([](const pagmo::problem &p, const std::string &method, const py::array_t<double> &weights) {
            return std::make_unique<pagmo::unconstrain>(p, method, ndarr_to_vector<pagmo::vector_double>(weights));
        }))
        .def_property_readonly(
            "inner_problem", [](pagmo::unconstrain &udp) -> pagmo::problem & { return udp.get_inner_problem(); },
            py::return_value_policy::reference_internal, generic_udp_inner_problem_docstring().c_str());

    // WFG.
    auto wfg_p = expose_problem<pagmo::wfg>(m, prob, p_module, "wfg", wfg_docstring().c_str());
    wfg_p.def(py::init<unsigned, pagmo::vector_double::size_type, pagmo::vector_double::size_type,
                       pagmo::vector_double::size_type>(),
              py::arg("prob_id") = 1u, py::arg("dim_dvs") = 5u, py::arg("dim_obj") = 3u, py::arg("dim_k") = 4u);
}

} // namespace pygmo
