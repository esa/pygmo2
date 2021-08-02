// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <string>
#include <tuple>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/gaco.hpp>
#include <pagmo/algorithms/gwo.hpp>
#include <pagmo/algorithms/ihs.hpp>
#include <pagmo/algorithms/maco.hpp>
#include <pagmo/algorithms/nsga2.hpp>
#include <pagmo/algorithms/nspso.hpp>
#include <pagmo/algorithms/null_algorithm.hpp>
#include <pagmo/algorithms/pso.hpp>
#include <pagmo/algorithms/pso_gen.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/algorithms/sea.hpp>
#include <pagmo/algorithms/sga.hpp>
#include <pagmo/algorithms/simulated_annealing.hpp>
#include <pagmo/config.hpp>

#if defined(PAGMO_WITH_NLOPT)
#include <pagmo/algorithms/nlopt.hpp>
#endif

#include "common_utils.hpp"
#include "docstrings.hpp"
#include "expose_algorithms.hpp"

namespace pygmo
{

namespace py = pybind11;

void expose_algorithms_1(py::module &m, py::class_<pagmo::algorithm> &algo, py::module &a_module)
{
    // Null algo.
    auto na = expose_algorithm<pagmo::null_algorithm>(m, algo, a_module, "null_algorithm",
                                                      null_algorithm_docstring().c_str());

    // NSGA2
    auto nsga2_ = expose_algorithm<pagmo::nsga2>(m, algo, a_module, "nsga2", nsga2_docstring().c_str());
    nsga2_.def(py::init<unsigned, double, double, double, double>(), py::arg("gen") = 1u, py::arg("cr") = 0.95,
               py::arg("eta_c") = 10., py::arg("m") = 0.01, py::arg("eta_m") = 50.);
    nsga2_.def(py::init<unsigned, double, double, double, double, unsigned>(), py::arg("gen") = 1u,
               py::arg("cr") = 0.95, py::arg("eta_c") = 10., py::arg("m") = 0.01, py::arg("eta_m") = 50.,
               py::arg("seed"));
    // nsga2 needs an ad hoc exposition for the log as one entry is a vector (ideal_point)
    nsga2_.def(
        "get_log",
        [](const pagmo::nsga2 &a) -> py::list {
            py::list retval;
            for (const auto &t : a.get_log()) {
                retval.append(py::make_tuple(std::get<0>(t), std::get<1>(t),
                                             vector_to_ndarr<py::array_t<double>>(std::get<2>(t))));
            }
            return retval;
        },
        nsga2_get_log_docstring().c_str());

    nsga2_.def("get_seed", &pagmo::nsga2::get_seed, generic_uda_get_seed_docstring().c_str());
    nsga2_.def("set_bfe", &pagmo::nsga2::set_bfe, nsga2_set_bfe_docstring().c_str(), py::arg("b"));

    // GACO
    auto gaco_ = expose_algorithm<pagmo::gaco>(m, algo, a_module, "gaco", gaco_docstring().c_str());
    gaco_.def(
        py::init<unsigned, unsigned, double, double, double, unsigned, unsigned, unsigned, unsigned, double, bool>(),
        py::arg("gen") = 100u, py::arg("ker") = 63u, py::arg("q") = 1.0, py::arg("oracle") = 0., py::arg("acc") = 0.01,
        py::arg("threshold") = 1u, py::arg("n_gen_mark") = 7u, py::arg("impstop") = 100000u,
        py::arg("evalstop") = 100000u, py::arg("focus") = 0., py::arg("memory") = false);
    gaco_.def(py::init<unsigned, unsigned, double, double, double, unsigned, unsigned, unsigned, unsigned, double, bool,
                       unsigned>(),
              py::arg("gen") = 100u, py::arg("ker") = 63u, py::arg("q") = 1.0, py::arg("oracle") = 0.,
              py::arg("acc") = 0.01, py::arg("threshold") = 1u, py::arg("n_gen_mark") = 7u,
              py::arg("impstop") = 100000u, py::arg("evalstop") = 100000u, py::arg("focus") = 0.,
              py::arg("memory") = false, py::arg("seed"));
    expose_algo_log(gaco_, gaco_get_log_docstring().c_str());
    gaco_.def("get_seed", &pagmo::gaco::get_seed, generic_uda_get_seed_docstring().c_str());
    gaco_.def("set_bfe", &pagmo::gaco::set_bfe, gaco_set_bfe_docstring().c_str(), py::arg("b"));

    // GWO
    auto gwo_ = expose_algorithm<pagmo::gwo>(m, algo, a_module, "gwo", gwo_docstring().c_str());
    gwo_.def(py::init<unsigned>(), py::arg("gen") = 1u);
    gwo_.def(py::init<unsigned, unsigned>(), py::arg("gen") = 1u, py::arg("seed"));
    expose_algo_log(gwo_, gwo_get_log_docstring().c_str());
    gwo_.def("get_seed", &pagmo::gwo::get_seed, generic_uda_get_seed_docstring().c_str());

    // SEA
    auto sea_ = expose_algorithm<pagmo::sea>(m, algo, a_module, "sea", sea_docstring().c_str());
    sea_.def(py::init<unsigned>(), py::arg("gen") = 1u);
    sea_.def(py::init<unsigned, unsigned>(), py::arg("gen") = 1u, py::arg("seed"));
    expose_algo_log(sea_, sea_get_log_docstring().c_str());
    sea_.def("get_seed", &pagmo::sea::get_seed, generic_uda_get_seed_docstring().c_str());

    // PSO
    auto pso_ = expose_algorithm<pagmo::pso>(m, algo, a_module, "pso", pso_docstring().c_str());
    pso_.def(py::init<unsigned, double, double, double, double, unsigned, unsigned, unsigned, bool>(),
             py::arg("gen") = 1u, py::arg("omega") = 0.7298, py::arg("eta1") = 2.05, py::arg("eta2") = 2.05,
             py::arg("max_vel") = 0.5, py::arg("variant") = 5u, py::arg("neighb_type") = 2u,
             py::arg("neighb_param") = 4u, py::arg("memory") = false);
    pso_.def(py::init<unsigned, double, double, double, double, unsigned, unsigned, unsigned, bool, unsigned>(),
             py::arg("gen") = 1u, py::arg("omega") = 0.7298, py::arg("eta1") = 2.05, py::arg("eta2") = 2.05,
             py::arg("max_vel") = 0.5, py::arg("variant") = 5u, py::arg("neighb_type") = 2u,
             py::arg("neighb_param") = 4u, py::arg("memory") = false, py::arg("seed"));
    expose_algo_log(pso_, pso_get_log_docstring().c_str());
    pso_.def("get_seed", &pagmo::pso::get_seed, generic_uda_get_seed_docstring().c_str());

    // PSO (generational)
    auto pso_gen_ = expose_algorithm<pagmo::pso_gen>(m, algo, a_module, "pso_gen", pso_gen_docstring().c_str());
    pso_gen_.def(py::init<unsigned, double, double, double, double, unsigned, unsigned, unsigned, bool>(),
                 py::arg("gen") = 1u, py::arg("omega") = 0.7298, py::arg("eta1") = 2.05, py::arg("eta2") = 2.05,
                 py::arg("max_vel") = 0.5, py::arg("variant") = 5u, py::arg("neighb_type") = 2u,
                 py::arg("neighb_param") = 4u, py::arg("memory") = false);
    pso_gen_.def(py::init<unsigned, double, double, double, double, unsigned, unsigned, unsigned, bool, unsigned>(),
                 py::arg("gen") = 1u, py::arg("omega") = 0.7298, py::arg("eta1") = 2.05, py::arg("eta2") = 2.05,
                 py::arg("max_vel") = 0.5, py::arg("variant") = 5u, py::arg("neighb_type") = 2u,
                 py::arg("neighb_param") = 4u, py::arg("memory") = false, py::arg("seed"));
    expose_algo_log(pso_gen_, pso_gen_get_log_docstring().c_str());
    pso_gen_.def("get_seed", &pagmo::pso_gen::get_seed, generic_uda_get_seed_docstring().c_str());
    pso_gen_.def("set_bfe", &pagmo::pso_gen::set_bfe, pso_gen_set_bfe_docstring().c_str(), py::arg("b"));

    // SIMULATED ANNEALING
    auto simulated_annealing_ = expose_algorithm<pagmo::simulated_annealing>(m, algo, a_module, "simulated_annealing",
                                                                             simulated_annealing_docstring().c_str());
    simulated_annealing_.def(py::init<double, double, unsigned, unsigned, unsigned, double>(), py::arg("Ts") = 10.,
                             py::arg("Tf") = 0.1, py::arg("n_T_adj") = 10u, py::arg("n_range_adj") = 1u,
                             py::arg("bin_size") = 20u, py::arg("start_range") = 1.);
    simulated_annealing_.def(py::init<double, double, unsigned, unsigned, unsigned, double, unsigned>(),
                             py::arg("Ts") = 10., py::arg("Tf") = 0.1, py::arg("n_T_adj") = 10u,
                             py::arg("n_range_adj") = 10u, py::arg("bin_size") = 10u, py::arg("start_range") = 1.,
                             py::arg("seed"));
    expose_algo_log(simulated_annealing_, simulated_annealing_get_log_docstring().c_str());
    simulated_annealing_.def("get_seed", &pagmo::simulated_annealing::get_seed,
                             generic_uda_get_seed_docstring().c_str());
    expose_not_population_based(simulated_annealing_, "simulated_annealing");

    // SGA
    auto sga_ = expose_algorithm<pagmo::sga>(m, algo, a_module, "sga", sga_docstring().c_str());
    sga_.def(py::init<unsigned, double, double, double, double, unsigned, std::string, std::string, std::string>(),
             py::arg("gen") = 1u, py::arg("cr") = 0.9, py::arg("eta_c") = 1., py::arg("m") = 0.02,
             py::arg("param_m") = 1., py::arg("param_s") = 2u, py::arg("crossover") = "exponential",
             py::arg("mutation") = "polynomial", py::arg("selection") = "tournament");
    sga_.def(
        py::init<unsigned, double, double, double, double, unsigned, std::string, std::string, std::string, unsigned>(),
        py::arg("gen") = 1u, py::arg("cr") = 0.9, py::arg("eta_c") = 1., py::arg("m") = 0.02, py::arg("param_m") = 1.,
        py::arg("param_s") = 2u, py::arg("crossover") = "exponential", py::arg("mutation") = "polynomial",
        py::arg("selection") = "tournament", py::arg("seed"));
    expose_algo_log(sga_, sga_get_log_docstring().c_str());
    sga_.def("get_seed", &pagmo::sga::get_seed, generic_uda_get_seed_docstring().c_str());

    // IHS
    auto ihs_ = expose_algorithm<pagmo::ihs>(m, algo, a_module, "ihs", ihs_docstring().c_str());
    ihs_.def(py::init<unsigned, double, double, double, double, double>(), py::arg("gen") = 1u, py::arg("phmcr") = 0.85,
             py::arg("ppar_min") = 0.35, py::arg("ppar_max") = 0.99, py::arg("bw_min") = 1E-5, py::arg("bw_max") = 1.);
    ihs_.def(py::init<unsigned, double, double, double, double, double, unsigned>(), py::arg("gen") = 1u,
             py::arg("phmcr") = 0.85, py::arg("ppar_min") = 0.35, py::arg("ppar_max") = 0.99, py::arg("bw_min") = 1E-5,
             py::arg("bw_max") = 1., py::arg("seed"));
    // ihs needs an ad hoc exposition for the log as one entry is a vector (ideal_point)
    ihs_.def(
        "get_log",
        [](const pagmo::ihs &a) -> py::list {
            py::list retval;
            for (const auto &t : a.get_log()) {
                retval.append(py::make_tuple(std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t),
                                             std::get<4>(t), std::get<5>(t), std::get<6>(t),
                                             vector_to_ndarr<py::array_t<double>>(std::get<7>(t))));
            }
            return retval;
        },
        ihs_get_log_docstring().c_str());
    ihs_.def("get_seed", &pagmo::ihs::get_seed, generic_uda_get_seed_docstring().c_str());

    // SADE
    auto sade_ = expose_algorithm<pagmo::sade>(m, algo, a_module, "sade", sade_docstring().c_str());
    sade_.def(py::init<unsigned, unsigned, unsigned, double, double, bool>(), py::arg("gen") = 1u,
              py::arg("variant") = 2u, py::arg("variant_adptv") = 1u, py::arg("ftol") = 1e-6, py::arg("xtol") = 1e-6,
              py::arg("memory") = false);
    sade_.def(py::init<unsigned, unsigned, unsigned, double, double, bool, unsigned>(), py::arg("gen") = 1u,
              py::arg("variant") = 2u, py::arg("variant_adptv") = 1u, py::arg("ftol") = 1e-6, py::arg("xtol") = 1e-6,
              py::arg("memory") = false, py::arg("seed"));
    expose_algo_log(sade_, sade_get_log_docstring().c_str());
    sade_.def("get_seed", &pagmo::sade::get_seed, generic_uda_get_seed_docstring().c_str());

    // MACO
    auto maco_ = expose_algorithm<pagmo::maco>(m, algo, a_module, "maco", maco_docstring().c_str());
    maco_.def(py::init<unsigned, unsigned, double, unsigned, unsigned, unsigned, double, bool>(), py::arg("gen") = 1u,
              py::arg("ker") = 63u, py::arg("q") = 1.0, py::arg("threshold") = 1u, py::arg("n_gen_mark") = 7u,
              py::arg("evalstop") = 100000u, py::arg("focus") = 0., py::arg("memory") = false);
    maco_.def(py::init<unsigned, unsigned, double, unsigned, unsigned, unsigned, double, bool, unsigned>(),
              py::arg("gen") = 1u, py::arg("ker") = 63u, py::arg("q") = 1.0, py::arg("threshold") = 1u,
              py::arg("n_gen_mark") = 7u, py::arg("evalstop") = 100000u, py::arg("focus") = 0.,
              py::arg("memory") = false, py::arg("seed"));
    // maco needs an ad hoc exposition for the log as one entry is a vector (ideal_point)
    maco_.def(
        "get_log",
        [](const pagmo::maco &a) -> py::list {
            py::list retval;
            for (const auto &t : a.get_log()) {
                retval.append(py::make_tuple(std::get<0>(t), std::get<1>(t),
                                             vector_to_ndarr<py::array_t<double>>(std::get<2>(t))));
            }
            return retval;
        },
        maco_get_log_docstring().c_str());
    maco_.def("get_seed", &pagmo::maco::get_seed, generic_uda_get_seed_docstring().c_str());
    maco_.def("set_bfe", &pagmo::maco::set_bfe, maco_set_bfe_docstring().c_str(), py::arg("b"));

    // NSPSO
    auto nspso_ = expose_algorithm<pagmo::nspso>(m, algo, a_module, "nspso", nspso_docstring().c_str());
    nspso_.def(py::init<unsigned, double, double, double, double, double, unsigned, std::string, bool>(),
               py::arg("gen") = 1u, py::arg("omega") = 0.6, py::arg("c1") = 0.01, py::arg("c2") = 0.5,
               py::arg("chi") = 0.5, py::arg("v_coeff") = 0.5, py::arg("leader_selection_range") = 2u,
               py::arg("diversity_mechanism") = "crowding distance", py::arg("memory") = false);
    nspso_.def(py::init<unsigned, double, double, double, double, double, unsigned, std::string, bool, unsigned>(),
               py::arg("gen") = 1u, py::arg("omega") = 0.6, py::arg("c1") = 0.01, py::arg("c2") = 0.5,
               py::arg("chi") = 0.5, py::arg("v_coeff") = 0.5, py::arg("leader_selection_range") = 2u,
               py::arg("diversity_mechanism") = "crowding distance", py::arg("memory") = false, py::arg("seed"));
    // nspso needs an ad hoc exposition for the log as one entry is a vector (ideal_point)
    nspso_.def(
        "get_log",
        [](const pagmo::nspso &a) -> py::list {
            py::list retval;
            for (const auto &t : a.get_log()) {
                retval.append(py::make_tuple(std::get<0>(t), std::get<1>(t),
                                             vector_to_ndarr<py::array_t<double>>(std::get<2>(t))));
            }
            return retval;
        },
        nspso_get_log_docstring().c_str());
    nspso_.def("get_seed", &pagmo::nspso::get_seed, generic_uda_get_seed_docstring().c_str());
    nspso_.def("set_bfe", &pagmo::nspso::set_bfe, nspso_set_bfe_docstring().c_str(), py::arg("b"));

#if defined(PAGMO_WITH_NLOPT)
    // NLopt.
    auto nlopt_ = expose_algorithm<pagmo::nlopt>(m, algo, a_module, "nlopt", nlopt_docstring().c_str());
    nlopt_
        .def(py::init<const std::string &>(), py::arg("solver"))
        // Properties for the stopping criteria.
        .def_property("stopval", &pagmo::nlopt::get_stopval, &pagmo::nlopt::set_stopval,
                      nlopt_stopval_docstring().c_str())
        .def_property("ftol_rel", &pagmo::nlopt::get_ftol_rel, &pagmo::nlopt::set_ftol_rel,
                      nlopt_ftol_rel_docstring().c_str())
        .def_property("ftol_abs", &pagmo::nlopt::get_ftol_abs, &pagmo::nlopt::set_ftol_abs,
                      nlopt_ftol_abs_docstring().c_str())
        .def_property("xtol_rel", &pagmo::nlopt::get_xtol_rel, &pagmo::nlopt::set_xtol_rel,
                      nlopt_xtol_rel_docstring().c_str())
        .def_property("xtol_abs", &pagmo::nlopt::get_xtol_abs, &pagmo::nlopt::set_xtol_abs,
                      nlopt_xtol_abs_docstring().c_str())
        .def_property("maxeval", &pagmo::nlopt::get_maxeval, &pagmo::nlopt::set_maxeval,
                      nlopt_maxeval_docstring().c_str())
        .def_property("maxtime", &pagmo::nlopt::get_maxtime, &pagmo::nlopt::set_maxtime,
                      nlopt_maxtime_docstring().c_str());
    expose_not_population_based(nlopt_, "nlopt");
    expose_algo_log(nlopt_, nlopt_get_log_docstring().c_str());
    nlopt_.def(
        "get_last_opt_result", [](const pagmo::nlopt &n) { return static_cast<int>(n.get_last_opt_result()); },
        nlopt_get_last_opt_result_docstring().c_str());
    nlopt_.def("get_solver_name", &pagmo::nlopt::get_solver_name, nlopt_get_solver_name_docstring().c_str());
    nlopt_.def_property(
        "local_optimizer", [](pagmo::nlopt &n) { return n.get_local_optimizer(); },
        [](pagmo::nlopt &n, const pagmo::nlopt *ptr) {
            if (ptr) {
                n.set_local_optimizer(*ptr);
            } else {
                n.unset_local_optimizer();
            }
        },
        py::return_value_policy::reference_internal, nlopt_local_optimizer_docstring().c_str());
#endif
}

} // namespace pygmo
