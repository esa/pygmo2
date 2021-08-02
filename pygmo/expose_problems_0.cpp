// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <memory>
#include <string>
#include <utility>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/problems/ackley.hpp>
#include <pagmo/problems/cec2006.hpp>
#include <pagmo/problems/cec2009.hpp>
#include <pagmo/problems/cec2014.hpp>
#include <pagmo/problems/decompose.hpp>
#include <pagmo/problems/dtlz.hpp>
#include <pagmo/problems/griewank.hpp>
#include <pagmo/problems/hock_schittkowsky_71.hpp>
#include <pagmo/problems/inventory.hpp>
#include <pagmo/problems/lennard_jones.hpp>
#include <pagmo/problems/null_problem.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

#include "common_utils.hpp"
#include "docstrings.hpp"
#include "expose_problems.hpp"

namespace pygmo
{

namespace py = pybind11;

namespace detail
{

namespace
{

// A test problem.
struct test_problem {
    test_problem(unsigned nobj = 1) : m_nobj(nobj) {}
    pagmo::vector_double fitness(const pagmo::vector_double &) const
    {
        return {1.};
    }
    std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const
    {
        return {{0.}, {1.}};
    }
    // Set/get an internal value to test extraction semantics.
    void set_n(int n)
    {
        m_n = n;
    }
    int get_n() const
    {
        return m_n;
    }
    pagmo::vector_double::size_type get_nobj() const
    {
        return m_nobj;
    }
    int m_n = 1;
    unsigned m_nobj;
};

// A thread-unsafe test problem.
struct tu_test_problem {
    pagmo::vector_double fitness(const pagmo::vector_double &) const
    {
        return {1.};
    }
    std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const
    {
        return {{0.}, {1.}};
    }
    pagmo::thread_safety get_thread_safety() const
    {
        return pagmo::thread_safety::none;
    }
};

} // namespace

} // namespace detail

void expose_problems_0(py::module &m, py::class_<pagmo::problem> &prob, py::module &p_module)
{
    // Exposition of C++ problems.

    // Test problem.
    auto test_p = expose_problem<detail::test_problem>(m, prob, p_module, "_test_problem", "A test problem.");
    test_p.def(py::init<unsigned>(), py::arg("nobj"));
    test_p.def("get_n", &detail::test_problem::get_n);
    test_p.def("set_n", &detail::test_problem::set_n);

    // Thread unsafe test problem.
    expose_problem<detail::tu_test_problem>(m, prob, p_module, "_tu_test_problem", "A thread unsafe test problem.");

    // Null problem.
    auto np = expose_problem<pagmo::null_problem>(m, prob, p_module, "null_problem", null_problem_docstring().c_str());
    np.def(
        py::init<pagmo::vector_double::size_type, pagmo::vector_double::size_type, pagmo::vector_double::size_type>(),
        py::arg("nobj") = 1, py::arg("nec") = 0, py::arg("nic") = 0);

    // Hock-Schittkowsky 71
    auto hs71 = expose_problem<pagmo::hock_schittkowsky_71>(m, prob, p_module, "hock_schittkowsky_71",
                                                            "__init__()\n\nThe Hock-Schittkowsky 71 problem.\n\n"
                                                            "See :cpp:class:`pagmo::hock_schittkowsky_71`.\n\n");
    hs71.def("best_known", &best_known_wrapper<pagmo::hock_schittkowsky_71>,
             problem_get_best_docstring("Hock-Schittkowsky 71").c_str());

    // Decompose meta-problem.
    auto decompose_ = expose_problem<pagmo::decompose>(m, prob, p_module, "decompose", decompose_docstring().c_str());
    decompose_
        // NOTE: An __init__ wrapper on the Python side will take care of cting a pagmo::problem from the input UDP,
        // and then invoke this ctor. This way we avoid having to expose a different ctor for every exposed C++ prob.
        .def(py::init([](const pagmo::problem &p, const py::array_t<double> &weight, const py::array_t<double> &z,
                         const std::string &method, bool adapt_ideal) {
            return std::make_unique<pagmo::decompose>(p, pygmo::ndarr_to_vector<pagmo::vector_double>(weight),
                                                      pygmo::ndarr_to_vector<pagmo::vector_double>(z), method,
                                                      adapt_ideal);
        }))
        .def(
            "original_fitness",
            [](const pagmo::decompose &p, const py::array_t<double> &x) {
                return pygmo::vector_to_ndarr<py::array_t<double>>(
                    p.original_fitness(pygmo::ndarr_to_vector<pagmo::vector_double>(x)));
            },
            decompose_original_fitness_docstring().c_str(), py::arg("x"))
        .def_property_readonly(
            "z", [](const pagmo::decompose &p) { return pygmo::vector_to_ndarr<py::array_t<double>>(p.get_z()); },
            decompose_z_docstring().c_str())
        .def_property_readonly(
            "inner_problem", [](pagmo::decompose &udp) -> pagmo::problem & { return udp.get_inner_problem(); },
            py::return_value_policy::reference_internal, generic_udp_inner_problem_docstring().c_str());

    // Inventory.
    auto inv = expose_problem<pagmo::inventory>(
        m, prob, p_module, "inventory",
        "__init__(weeks = 4,sample_size = 10,seed = random)\n\nThe inventory problem.\n\n"
        "See :cpp:class:`pagmo::inventory`.\n\n");
    inv.def(py::init<unsigned, unsigned>(), py::arg("weeks") = 4u, py::arg("sample_size") = 10u);
    inv.def(py::init<unsigned, unsigned, unsigned>(), py::arg("weeks") = 4u, py::arg("sample_size") = 10u,
            py::arg("seed"));

    // Ackley.
    auto ack = expose_problem<pagmo::ackley>(m, prob, p_module, "ackley",
                                             "__init__(dim = 1)\n\nThe Ackley problem.\n\n"
                                             "See :cpp:class:`pagmo::ackley`.\n\n");
    ack.def(py::init<unsigned>(), py::arg("dim"));
    ack.def("best_known", &best_known_wrapper<pagmo::ackley>, problem_get_best_docstring("Ackley").c_str());

    // Lennard Jones
    auto lj = expose_problem<pagmo::lennard_jones>(m, prob, p_module, "lennard_jones",
                                                   "__init__(atoms = 3)\n\nThe Lennard Jones Cluster problem.\n\n"
                                                   "See :cpp:class:`pagmo::lennard_jones`.\n\n");
    lj.def(py::init<unsigned>(), py::arg("atoms") = 3u);

    // DTLZ.
    auto dtlz_p = expose_problem<pagmo::dtlz>(m, prob, p_module, "dtlz", dtlz_docstring().c_str());
    dtlz_p.def(py::init<unsigned, unsigned, unsigned, unsigned>(), py::arg("prob_id") = 1u, py::arg("dim") = 5u,
               py::arg("fdim") = 3u, py::arg("alpha") = 100u);
    dtlz_p.def("p_distance", [](const pagmo::dtlz &z, const py::array_t<double> &x) {
        return z.p_distance(ndarr_to_vector<pagmo::vector_double>(x));
    });
    dtlz_p.def(
        "p_distance", [](const pagmo::dtlz &z, const pagmo::population &pop) { return z.p_distance(pop); },
        dtlz_p_distance_docstring().c_str());

    // CEC 2006
    auto cec2006_ = expose_problem<pagmo::cec2006>(m, prob, p_module, "cec2006", cec2006_docstring().c_str());
    cec2006_.def(py::init<unsigned>(), py::arg("prob_id"));
    cec2006_.def("best_known", &best_known_wrapper<pagmo::cec2006>, problem_get_best_docstring("CEC 2006").c_str());

    // CEC 2009
    auto cec2009_ = expose_problem<pagmo::cec2009>(m, prob, p_module, "cec2009", cec2009_docstring().c_str());
    cec2009_.def(py::init<unsigned, bool, unsigned>(), py::arg("prob_id") = 1u, py::arg("is_constrained") = false,
                 py::arg("dim") = 30u);

    auto cec2014_ = expose_problem<pagmo::cec2014>(m, prob, p_module, "cec2014", cec2014_docstring().c_str());
    cec2014_.def(py::init<unsigned, unsigned>(), py::arg("prob_id") = 1, py::arg("dim") = 2);

    // Griewank.
    auto griew = expose_problem<pagmo::griewank>(m, prob, p_module, "griewank",
                                                 "__init__(dim = 1)\n\nThe Griewank problem.\n\n"
                                                 "See :cpp:class:`pagmo::griewank`.\n\n");
    griew.def(py::init<unsigned>(), py::arg("dim"));
    griew.def("best_known", &best_known_wrapper<pagmo::griewank>, problem_get_best_docstring("Griewank").c_str());
}

} // namespace pygmo
