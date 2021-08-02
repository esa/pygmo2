// Copyright 2020, 2021 PaGMO development team
//
// This file is part of the pygmo library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <pagmo/algorithm.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/batch_evaluators/default_bfe.hpp>
#include <pagmo/batch_evaluators/member_bfe.hpp>
#include <pagmo/batch_evaluators/thread_bfe.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/config.hpp>
#include <pagmo/detail/gte_getter.hpp>
#include <pagmo/exceptions.hpp>
#include <pagmo/island.hpp>
#include <pagmo/islands/thread_island.hpp>
#include <pagmo/population.hpp>
#include <pagmo/r_policy.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/s_policy.hpp>
#include <pagmo/threading.hpp>
#include <pagmo/topology.hpp>
#include <pagmo/types.hpp>
#include <pagmo/utils/constrained.hpp>
#include <pagmo/utils/generic.hpp>
#include <pagmo/utils/genetic_operators.hpp>
#include <pagmo/utils/gradients_and_hessians.hpp>
#include <pagmo/utils/hv_algos/hv_bf_approx.hpp>
#include <pagmo/utils/hv_algos/hv_bf_fpras.hpp>
#include <pagmo/utils/hv_algos/hv_hv2d.hpp>
#include <pagmo/utils/hv_algos/hv_hv3d.hpp>
#include <pagmo/utils/hv_algos/hv_hvwfg.hpp>
#include <pagmo/utils/hypervolume.hpp>
#include <pagmo/utils/multi_objective.hpp>

#include "algorithm.hpp"
#include "bfe.hpp"
#include "common_utils.hpp"
#include "docstrings.hpp"
#include "expose_algorithms.hpp"
#include "expose_bfes.hpp"
#include "expose_islands.hpp"
#include "expose_problems.hpp"
#include "expose_r_policies.hpp"
#include "expose_s_policies.hpp"
#include "expose_topologies.hpp"
#include "island.hpp"
#include "problem.hpp"
#include "r_policy.hpp"
#include "s11n_wrappers.hpp"
#include "s_policy.hpp"
#include "topology.hpp"

namespace py = pybind11;
namespace pg = pagmo;

namespace pygmo
{

namespace detail
{

namespace
{

// NOTE: we need to provide a custom raii waiter in the island. The reason is the following.
// When we call wait() from Python, the calling thread will be holding the GIL and then we will be waiting
// for evolutions in the island to finish. During this time, no
// Python code will be executed because the GIL is locked. This means that if we have a Python thread doing background
// work (e.g., managing the task queue in pythonic islands), it will have to wait before doing any progress. By
// unlocking the GIL before calling thread_island::wait(), we give the chance to other Python threads to continue
// doing some work.
// NOTE: here we have 2 RAII classes interacting with the GIL. The GIL releaser is the *second* one,
// and it is the one that is responsible for unlocking the Python interpreter while wait() is running.
// The *first* one, the GIL thread ensurer, does something else: it makes sure that we can call the Python
// interpreter from the current C++ thread. In a normal situation, in which islands are just instantiated
// from the main thread, the gte object is superfluous. However, if we are interacting with islands from a
// separate C++ thread, then we need to make sure that every time we call into the Python interpreter (e.g., by
// using the GIL releaser below) we inform Python we are about to call from a separate thread. This is what
// the GTE object does. This use case is, for instance, what happens with the PADE algorithm when, algo, prob,
// etc. are all C++ objects (when at least one object is pythonic, we will not end up using the thread island).
// NOTE: by ordering the class members in this way we ensure that gte is constructed before gr, which is essential
// (otherwise we might be calling into the interpreter with a releaser before informing Python we are calling
// from a separate thread).
struct py_wait_locks {
    gil_thread_ensurer gte;
    gil_releaser gr;
};

} // namespace

} // namespace detail

} // namespace pygmo

PYBIND11_MODULE(core, m)
{
    using namespace pybind11::literals;

#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 9
    // This function needs to be called before doing anything with threads.
    // https://docs.python.org/3/c-api/init.html
    // NOTE: this is deprecated and does nothing since Python 3.9.
    PyEval_InitThreads();
#endif

    // Disable automatic function signatures in the docs.
    // NOTE: the 'options' object needs to stay alive
    // throughout the whole definition of the module.
    py::options options;
    options.disable_function_signatures();

    // Export the pagmo version.
    m.attr("_pagmo_version_major") = PAGMO_VERSION_MAJOR;
    m.attr("_pagmo_version_minor") = PAGMO_VERSION_MINOR;
    m.attr("_pagmo_version_patch") = PAGMO_VERSION_PATCH;

    // Expose some internal functions for testing.
    m.def("_callable", &pygmo::callable);
    m.def("_callable_attribute", &pygmo::callable_attribute);
    m.def("_str", &pygmo::str);
    m.def("_type", &pygmo::type);
    m.def("_builtins", &pygmo::builtins);
    m.def("_deepcopy", &pygmo::deepcopy);
    m.def("_max_unsigned", []() {
        // Small helper function to get the max value of unsigned.
        return std::numeric_limits<unsigned>::max();
    });

    // The random_device_next() helper.
    m.def("_random_device_next", []() { return pg::random_device::next(); });

    // Global random number generator
    m.def(
        "set_global_rng_seed", [](unsigned seed) { pg::random_device::set_seed(seed); },
        pygmo::set_global_rng_seed_docstring().c_str(), py::arg("seed"));

    // Override the default implementation of the island factory.
    pg::detail::island_factory
        = [](const pg::algorithm &algo, const pg::population &pop, std::unique_ptr<pg::detail::isl_inner_base> &ptr) {
              if (algo.get_thread_safety() >= pg::thread_safety::basic
                  && pop.get_problem().get_thread_safety() >= pg::thread_safety::basic) {
                  // Both algo and prob have at least the basic thread safety guarantee. Use the thread island.
                  ptr = std::make_unique<pg::detail::isl_inner<pg::thread_island>>();
              } else {
                  // NOTE: here we are re-implementing a piece of code that normally
                  // is pure C++. We are calling into the Python interpreter, so, in order to handle
                  // the case in which we are invoking this code from a separate C++ thread, we construct a GIL ensurer
                  // in order to guard against concurrent access to the interpreter. The idea here is that this piece
                  // of code normally would provide a basic thread safety guarantee, and in order to continue providing
                  // it we use the ensurer.
                  pygmo::gil_thread_ensurer gte;
                  auto py_island = py::module::import("pygmo").attr("mp_island");
                  ptr = std::make_unique<pg::detail::isl_inner<py::object>>(py_island());
              }
          };

    // Override the default implementation of default_bfe.
    pg::detail::default_bfe_impl = [](const pg::problem &p, const pg::vector_double &dvs) -> pg::vector_double {
        // The member function batch_fitness() of p, if present, has priority.
        if (p.has_batch_fitness()) {
            return pg::member_bfe{}(p, dvs);
        }

        // Otherwise, we run the generic thread-based bfe, if the problem
        // is thread-safe enough.
        if (p.get_thread_safety() >= pg::thread_safety::basic) {
            return pg::thread_bfe{}(p, dvs);
        }

        // NOTE: in this last bit of the implementation we need to call
        // into the Python interpreter. In order to ensure that default_bfe
        // still works also from a C++ thread of which Python knows nothing about,
        // we will be using a thread ensurer, so that the thread safety
        // guarantee provided by default_bfe is still respected.
        // NOTE: the original default_bfe code is thread safe in the sense that the
        // code directly implemented within that class is thread safe. Invoking the call
        // operator of default_bfe might still end up being thread unsafe if p
        // itself is thread unsafe (the same happens, e.g., in a thread-safe algorithm
        // which uses a thread-unsafe problem in its evolve()).
        pygmo::gil_thread_ensurer gte;
        // Otherwise, we go for the multiprocessing bfe.
        return pygmo::ndarr_to_vector<pg::vector_double>(
            py::cast<py::array_t<double>>(py::module::import("pygmo").attr("mp_bfe")().attr("__call__")(
                p, pygmo::vector_to_ndarr<py::array_t<double>>(dvs))));
    };

    // Override the default RAII waiter. We need to use shared_ptr because we don't want to move/copy/destroy
    // the locks when invoking this from island::wait(), we need to instaniate exactly 1 py_wait_lock and have it
    // destroyed at the end of island::wait().
    pg::detail::wait_raii_getter = []() { return std::make_shared<pygmo::detail::py_wait_locks>(); };

    // NOTE: set the gte getter.
    pg::detail::gte_getter = []() { return std::make_shared<pygmo::gil_thread_ensurer>(); };

    // Register pagmo's custom exceptions.
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const pg::not_implemented_error &nie) {
            PyErr_SetString(PyExc_NotImplementedError, nie.what());
        }
    });

    // The thread_safety enum.
    py::enum_<pg::thread_safety>(m, "thread_safety", pygmo::thread_safety_docstring().c_str())
        .value("none", pg::thread_safety::none, pygmo::thread_safety_none_docstring().c_str())
        .value("basic", pg::thread_safety::basic, pygmo::thread_safety_basic_docstring().c_str())
        .value("constant", pg::thread_safety::constant, pygmo::thread_safety_constant_docstring().c_str());

    // The evolve_status enum.
    py::enum_<pg::evolve_status>(m, "evolve_status", pygmo::evolve_status_docstring().c_str())
        .value("idle", pg::evolve_status::idle, pygmo::evolve_status_idle_docstring().c_str())
        .value("busy", pg::evolve_status::busy, pygmo::evolve_status_busy_docstring().c_str())
        .value("idle_error", pg::evolve_status::idle_error, pygmo::evolve_status_idle_error_docstring().c_str())
        .value("busy_error", pg::evolve_status::busy_error, pygmo::evolve_status_busy_error_docstring().c_str());

    // Migration type enum.
    py::enum_<pg::migration_type>(m, "migration_type", pygmo::migration_type_docstring().c_str())
        .value("p2p", pg::migration_type::p2p, pygmo::migration_type_p2p_docstring().c_str())
        .value("broadcast", pg::migration_type::broadcast, pygmo::migration_type_broadcast_docstring().c_str());

    // Migrant handling policy enum.
    py::enum_<pg::migrant_handling>(m, "migrant_handling", pygmo::migrant_handling_docstring().c_str())
        .value("preserve", pg::migrant_handling::preserve, pygmo::migrant_handling_preserve_docstring().c_str())
        .value("evict", pg::migrant_handling::evict, pygmo::migrant_handling_evict_docstring().c_str());

    // Generic utilities
    m.def(
        "random_decision_vector",
        [](const pg::problem &p) -> py::array_t<double> {
            using reng_t = pg::detail::random_engine_type;
            reng_t tmp_rng(static_cast<reng_t::result_type>(pg::random_device::next()));
            auto retval = pg::random_decision_vector(p, tmp_rng);
            return pygmo::vector_to_ndarr<py::array_t<double>>(retval);
        },
        pygmo::random_decision_vector_docstring().c_str(), py::arg("prob"));
    m.def(
        "batch_random_decision_vector",
        [](const pg::problem &p, pg::vector_double::size_type n) -> py::array_t<double> {
            using reng_t = pg::detail::random_engine_type;
            reng_t tmp_rng(static_cast<reng_t::result_type>(pg::random_device::next()));
            auto retval = pg::batch_random_decision_vector(p, n, tmp_rng);
            return pygmo::vector_to_ndarr<py::array_t<double>>(retval);
        },
        pygmo::batch_random_decision_vector_docstring().c_str(), py::arg("prob"), py::arg("n"));
    // Genetic operators
    m.def(
        "sbx_crossover",
        [](const py::array_t<double> &parent1, const py::array_t<double> &parent2, const py::iterable &bounds,
           pg::vector_double::size_type nix, const double p_cr, const double eta_c, unsigned seed) {
            auto pg_bounds = pygmo::iterable_to_bounds(bounds);
            using reng_t = pg::detail::random_engine_type;
            reng_t tmp_rng(static_cast<reng_t::result_type>(seed));
            auto retval = pagmo::sbx_crossover(pygmo::ndarr_to_vector<pg::vector_double>(parent1),
                                               pygmo::ndarr_to_vector<pg::vector_double>(parent2), pg_bounds, nix, p_cr,
                                               eta_c, tmp_rng);
            return py::make_tuple(pygmo::vector_to_ndarr<py::array_t<double>>(retval.first),
                                  pygmo::vector_to_ndarr<py::array_t<double>>(retval.second));
        },
        pygmo::sbx_crossover_docstring().c_str(), py::arg("parent1"), py::arg("parent2"), py::arg("bounds"),
        py::arg("nix"), py::arg("p_cr"), py::arg("eta_c"), py::arg("seed"));

    m.def(
        "polynomial_mutation",
        [](const py::array_t<double> &dv, const py::iterable &bounds, pg::vector_double::size_type nix,
           const double p_m, const double eta_m, unsigned seed) {
            auto pg_bounds = pygmo::iterable_to_bounds(bounds);
            using reng_t = pg::detail::random_engine_type;
            reng_t tmp_rng(static_cast<reng_t::result_type>(seed));
            auto dv_c = pygmo::ndarr_to_vector<pg::vector_double>(dv);
            pagmo::polynomial_mutation(dv_c, pg_bounds, nix, p_m, eta_m, tmp_rng);
            return pygmo::vector_to_ndarr<py::array_t<double>>(dv_c);
        },
        pygmo::polynomial_mutation_docstring().c_str(), py::arg("dv"), py::arg("bounds"), py::arg("nix"),
        py::arg("p_m"), py::arg("eta_m"), py::arg("seed"));
    // Hypervolume class
    py::class_<pg::hypervolume> hv_class(m, "hypervolume", "Hypervolume Class");
    hv_class
        .def(py::init([](const py::array_t<double> &points) {
                 return std::make_unique<pg::hypervolume>(
                     pygmo::ndarr_to_vvector<std::vector<pg::vector_double>>(points), true);
             }),
             py::arg("points"), pygmo::hv_init2_docstring().c_str())
        .def(py::init([](const pg::population &pop) { return std::make_unique<pg::hypervolume>(pop, true); }),
             py::arg("pop"), pygmo::hv_init1_docstring().c_str())
        .def(
            "compute",
            [](const pg::hypervolume &hv, const py::array_t<double> &r_point) {
                return hv.compute(pygmo::ndarr_to_vector<pg::vector_double>(r_point));
            },
            py::arg("ref_point"))
        .def(
            "compute",
            [](const pg::hypervolume &hv, const py::array_t<double> &r_point, pg::hv_algorithm &hv_algo) {
                return hv.compute(pygmo::ndarr_to_vector<pg::vector_double>(r_point), hv_algo);
            },
            pygmo::hv_compute_docstring().c_str(), py::arg("ref_point"), py::arg("hv_algo"))
        .def(
            "exclusive",
            [](const pg::hypervolume &hv, unsigned p_idx, const py::array_t<double> &r_point) {
                return hv.exclusive(p_idx, pygmo::ndarr_to_vector<pg::vector_double>(r_point));
            },
            py::arg("idx"), py::arg("ref_point"))
        .def(
            "exclusive",
            [](const pg::hypervolume &hv, unsigned p_idx, const py::array_t<double> &r_point,
               pg::hv_algorithm &hv_algo) {
                return hv.exclusive(p_idx, pygmo::ndarr_to_vector<pg::vector_double>(r_point), hv_algo);
            },
            pygmo::hv_exclusive_docstring().c_str(), py::arg("idx"), py::arg("ref_point"), py::arg("hv_algo"))
        .def(
            "least_contributor",
            [](const pg::hypervolume &hv, const py::array_t<double> &r_point) {
                return hv.least_contributor(pygmo::ndarr_to_vector<pg::vector_double>(r_point));
            },
            py::arg("ref_point"))
        .def(
            "least_contributor",
            [](const pg::hypervolume &hv, const py::array_t<double> &r_point, pg::hv_algorithm &hv_algo) {
                return hv.least_contributor(pygmo::ndarr_to_vector<pg::vector_double>(r_point), hv_algo);
            },
            pygmo::hv_least_contributor_docstring().c_str(), py::arg("ref_point"), py::arg("hv_algo"))
        .def(
            "greatest_contributor",
            [](const pg::hypervolume &hv, const py::array_t<double> &r_point) {
                return hv.greatest_contributor(pygmo::ndarr_to_vector<pg::vector_double>(r_point));
            },
            py::arg("ref_point"))
        .def(
            "greatest_contributor",
            [](const pg::hypervolume &hv, const py::array_t<double> &r_point, pg::hv_algorithm &hv_algo) {
                return hv.greatest_contributor(pygmo::ndarr_to_vector<pg::vector_double>(r_point), hv_algo);
            },
            pygmo::hv_greatest_contributor_docstring().c_str(), py::arg("ref_point"), py::arg("hv_algo"))
        .def(
            "contributions",
            [](const pg::hypervolume &hv, const py::array_t<double> &r_point) {
                return pygmo::vector_to_ndarr<py::array_t<double>>(
                    hv.contributions(pygmo::ndarr_to_vector<pg::vector_double>(r_point)));
            },
            py::arg("ref_point"))
        .def(
            "contributions",
            [](const pg::hypervolume &hv, const py::array_t<double> &r_point, pg::hv_algorithm &hv_algo) {
                return pygmo::vector_to_ndarr<py::array_t<double>>(
                    hv.contributions(pygmo::ndarr_to_vector<pg::vector_double>(r_point), hv_algo));
            },
            pygmo::hv_contributions_docstring().c_str(), py::arg("ref_point"), py::arg("hv_algo"))
        .def("get_points",
             [](const pg::hypervolume &hv) { return pygmo::vvector_to_ndarr<py::array_t<double>>(hv.get_points()); })
        .def(
            "refpoint",
            [](const pg::hypervolume &hv, double offset) {
                return pygmo::vector_to_ndarr<py::array_t<double>>(hv.refpoint(offset));
            },
            pygmo::hv_refpoint_docstring().c_str(), py::arg("offset") = 0)
        .def_property("copy_points", &pg::hypervolume::get_copy_points, &pg::hypervolume::set_copy_points);

    // Hypervolume algorithms
    py::class_<pg::hv_algorithm> hv_algorithm_class(m, "_hv_algorithm");
    hv_algorithm_class.def("get_name", &pg::hv_algorithm::get_name);

    py::class_<pg::hvwfg, pg::hv_algorithm> hvwfg_class(m, "hvwfg", pygmo::hvwfg_docstring().c_str());
    hvwfg_class.def(py::init<unsigned>(), py::arg("stop_dimension") = 2);

    py::class_<pg::bf_approx, pg::hv_algorithm> bf_approx_class(m, "bf_approx", pygmo::bf_approx_docstring().c_str());
    bf_approx_class
        .def(py::init<bool, unsigned, double, double, double, double, double, double>(), py::arg("use_exact") = true,
             py::arg("trivial_subcase_size") = 1u, py::arg("eps") = 1e-2, py::arg("delta") = 1e-6,
             py::arg("delta_multiplier") = 0.775, py::arg("alpha") = 0.2, py::arg("initial_delta_coeff") = 0.1,
             py::arg("gamma") = 0.25)
        .def(py::init<bool, unsigned, double, double, double, double, double, double, unsigned>(),
             py::arg("use_exact") = true, py::arg("trivial_subcase_size") = 1u, py::arg("eps") = 1e-2,
             py::arg("delta") = 1e-6, py::arg("delta_multiplier") = 0.775, py::arg("alpha") = 0.2,
             py::arg("initial_delta_coeff") = 0.1, py::arg("gamma") = 0.25, py::arg("seed"));

    py::class_<pg::bf_fpras, pg::hv_algorithm> bf_fpras_class(m, "bf_fpras", pygmo::bf_fpras_docstring().c_str());
    bf_fpras_class.def(py::init<double, double>(), py::arg("eps") = 1e-2, py::arg("delta") = 1e-2)
        .def(py::init<double, double, unsigned>(), py::arg("eps") = 1e-2, py::arg("delta") = 1e-2, py::arg("seed"));

    py::class_<pg::hv2d, pg::hv_algorithm> hv2d_class(m, "hv2d", pygmo::hv2d_docstring().c_str());
    hv2d_class.def(py::init<>());

    py::class_<pg::hv3d, pg::hv_algorithm> hv3d_class(m, "hv3d", pygmo::hv3d_docstring().c_str());
    hv3d_class.def(py::init<>());

    // Multi-objective utilities
    m.def(
        "fast_non_dominated_sorting",
        [](const py::array_t<double> &x) -> py::tuple {
            auto fnds = pg::fast_non_dominated_sorting(pygmo::ndarr_to_vvector<std::vector<pg::vector_double>>(x));
            // the non-dominated fronts
            py::list ndf_py;
            for (const auto &front : std::get<0>(fnds)) {
                ndf_py.append(pygmo::vector_to_ndarr<py::array_t<pg::pop_size_t>>(front));
            }
            // the domination list
            py::list dl_py;
            for (const auto &item : std::get<1>(fnds)) {
                dl_py.append(pygmo::vector_to_ndarr<py::array_t<pg::pop_size_t>>(item));
            }
            return py::make_tuple(ndf_py, dl_py, pygmo::vector_to_ndarr<py::array_t<pg::pop_size_t>>(std::get<2>(fnds)),
                                  pygmo::vector_to_ndarr<py::array_t<pg::pop_size_t>>(std::get<3>(fnds)));
        },
        pygmo::fast_non_dominated_sorting_docstring().c_str(), py::arg("points"));

    m.def(
        "pareto_dominance",
        [](const py::array_t<double> &obj1, const py::array_t<double> &obj2) {
            return pg::pareto_dominance(pygmo::ndarr_to_vector<pg::vector_double>(obj1),
                                        pygmo::ndarr_to_vector<pg::vector_double>(obj2));
        },
        pygmo::pareto_dominance_docstring().c_str(), py::arg("obj1"), py::arg("obj2"));

    m.def(
        "non_dominated_front_2d",
        [](const py::array_t<double> &points) {
            return pygmo::vector_to_ndarr<py::array_t<pg::pop_size_t>>(
                pg::non_dominated_front_2d(pygmo::ndarr_to_vvector<std::vector<pg::vector_double>>(points)));
        },
        pygmo::non_dominated_front_2d_docstring().c_str(), py::arg("points"));

    m.def(
        "crowding_distance",
        [](const py::array_t<double> &points) {
            return pygmo::vector_to_ndarr<py::array_t<double>>(
                pg::crowding_distance(pygmo::ndarr_to_vvector<std::vector<pg::vector_double>>(points)));
        },
        pygmo::crowding_distance_docstring().c_str(), py::arg("points"));

    m.def(
        "sort_population_mo",
        [](const py::array_t<double> &input_f) {
            return pygmo::vector_to_ndarr<py::array_t<pg::pop_size_t>>(
                pg::sort_population_mo(pygmo::ndarr_to_vvector<std::vector<pg::vector_double>>(input_f)));
        },
        pygmo::sort_population_mo_docstring().c_str(), py::arg("points"));

    m.def(
        "select_best_N_mo",
        [](const py::array_t<double> &input_f, unsigned N) {
            return pygmo::vector_to_ndarr<py::array_t<pg::pop_size_t>>(
                pg::select_best_N_mo(pygmo::ndarr_to_vvector<std::vector<pg::vector_double>>(input_f), N));
        },
        pygmo::select_best_N_mo_docstring().c_str(), py::arg("points"), py::arg("N"));

    m.def(
        "decomposition_weights",
        [](pg::vector_double::size_type n_f, pg::vector_double::size_type n_w, const std::string &method,
           unsigned seed) {
            using reng_t = pg::detail::random_engine_type;
            reng_t tmp_rng(static_cast<reng_t::result_type>(seed));
            return pygmo::vvector_to_ndarr<py::array_t<double>>(pg::decomposition_weights(n_f, n_w, method, tmp_rng));
        },
        pygmo::decomposition_weights_docstring().c_str(), py::arg("n_f"), py::arg("n_w"), py::arg("method"),
        py::arg("seed"));

    m.def(
        "decompose_objectives",
        [](const py::array_t<double> &objs, const py::array_t<double> &weights, const py::array_t<double> &ref_point,
           const std::string &method) {
            return pygmo::vector_to_ndarr<py::array_t<double>>(pg::decompose_objectives(
                pygmo::ndarr_to_vector<pg::vector_double>(objs), pygmo::ndarr_to_vector<pg::vector_double>(weights),
                pygmo::ndarr_to_vector<pg::vector_double>(ref_point), method));
        },
        pygmo::decompose_objectives_docstring().c_str(), py::arg("objs"), py::arg("weights"), py::arg("ref_point"),
        py::arg("method"));

    m.def(
        "nadir",
        [](const py::array_t<double> &p) {
            return pygmo::vector_to_ndarr<py::array_t<double>>(
                pg::nadir(pygmo::ndarr_to_vvector<std::vector<pg::vector_double>>(p)));
        },
        pygmo::nadir_docstring().c_str(), py::arg("points"));

    m.def(
        "ideal",
        [](const py::array_t<double> &p) {
            return pygmo::vector_to_ndarr<py::array_t<double>>(
                pg::ideal(pygmo::ndarr_to_vvector<std::vector<pg::vector_double>>(p)));
        },
        pygmo::ideal_docstring().c_str(), py::arg("points"));

    // Gradient and Hessians utilities
    m.def(
        "estimate_sparsity",
        [](const py::object &func, const py::array_t<double> &x,
           double dx) -> py::array_t<pg::vector_double::size_type> {
            auto f = [&func](const pg::vector_double &x_) {
                return pygmo::ndarr_to_vector<pg::vector_double>(
                    py::cast<py::array_t<double>>(func(pygmo::vector_to_ndarr<py::array_t<double>>(x_))));
            };
            return pygmo::sp_to_ndarr(pg::estimate_sparsity(f, pygmo::ndarr_to_vector<pg::vector_double>(x), dx));
        },
        pygmo::estimate_sparsity_docstring().c_str(), py::arg("callable"), py::arg("x"), py::arg("dx") = 1e-8);

    m.def(
        "estimate_gradient",
        [](const py::object &func, const py::array_t<double> &x, double dx) -> py::array_t<double> {
            auto f = [&func](const pg::vector_double &x_) {
                return pygmo::ndarr_to_vector<pg::vector_double>(
                    py::cast<py::array_t<double>>(func(pygmo::vector_to_ndarr<py::array_t<double>>(x_))));
            };
            return pygmo::vector_to_ndarr<py::array_t<double>>(
                pg::estimate_gradient(f, pygmo::ndarr_to_vector<pg::vector_double>(x), dx));
        },
        pygmo::estimate_gradient_docstring().c_str(), py::arg("callable"), py::arg("x"), py::arg("dx") = 1e-8);

    m.def(
        "estimate_gradient_h",
        [](const py::object &func, const py::array_t<double> &x, double dx) -> py::array_t<double> {
            auto f = [&func](const pg::vector_double &x_) {
                return pygmo::ndarr_to_vector<pg::vector_double>(
                    py::cast<py::array_t<double>>(func(pygmo::vector_to_ndarr<py::array_t<double>>(x_))));
            };
            return pygmo::vector_to_ndarr<py::array_t<double>>(
                pg::estimate_gradient_h(f, pygmo::ndarr_to_vector<pg::vector_double>(x), dx));
        },
        pygmo::estimate_gradient_h_docstring().c_str(), py::arg("callable"), py::arg("x"), py::arg("dx") = 1e-2);

    // Constrained optimization utilities
    m.def(
        "compare_fc",
        [](const py::array_t<double> &f1, const py::array_t<double> &f2, pg::vector_double::size_type nec,
           const py::array_t<double> &tol) {
            return pg::compare_fc(pygmo::ndarr_to_vector<pg::vector_double>(f1),
                                  pygmo::ndarr_to_vector<pg::vector_double>(f2), nec,
                                  pygmo::ndarr_to_vector<pg::vector_double>(tol));
        },
        pygmo::compare_fc_docstring().c_str(), py::arg("f1"), py::arg("f2"), py::arg("nec"), py::arg("tol"));

    m.def(
        "sort_population_con",
        [](const py::array_t<double> &input_f, pg::vector_double::size_type nec, const py::array_t<double> &tol) {
            return pygmo::vector_to_ndarr<py::array_t<pg::pop_size_t>>(
                pg::sort_population_con(pygmo::ndarr_to_vvector<std::vector<pg::vector_double>>(input_f), nec,
                                        pygmo::ndarr_to_vector<pg::vector_double>(tol)));
        },
        pygmo::sort_population_con_docstring().c_str(), py::arg("input_f"), py::arg("nec"), py::arg("tol"));

    // Add the submodules.
    auto problems_module = m.def_submodule("problems");
    auto algorithms_module = m.def_submodule("algorithms");
    auto islands_module = m.def_submodule("islands");
    auto batch_evaluators_module = m.def_submodule("batch_evaluators");
    auto topologies_module = m.def_submodule("topologies");
    auto r_policies_module = m.def_submodule("r_policies");
    auto s_policies_module = m.def_submodule("s_policies");

    // Population class.
    py::class_<pg::population> pop_class(m, "population", pygmo::population_docstring().c_str());
    pop_class
        // Def ctor.
        .def(py::init<>())
        // Ctors from problem.
        // NOTE: we expose only the ctors from pagmo::problem, not from C++ or Python UDPs. An __init__ wrapper
        // on the Python side will take care of cting a pagmo::problem from the input UDP, and then invoke this ctor.
        // This way we avoid having to expose a different ctor for every exposed C++ prob. Same idea with
        // the bfe argument.
        .def(py::init<const pg::problem &, pg::population::size_type, unsigned>())
        .def(py::init<const pg::problem &, const pg::bfe &, pg::population::size_type, unsigned>())
        // repr().
        .def("__repr__", &pygmo::ostream_repr<pg::population>)
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<pg::population>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<pg::population>, "memo"_a)
        // Pickle support.
        .def(py::pickle(&pygmo::pickle_getstate_wrapper<pg::population>,
                        &pygmo::pickle_setstate_wrapper<pg::population>))
        .def(
            "push_back",
            [](pg::population &pop, const py::array_t<double> &x, const py::object &f) {
                if (f.is_none()) {
                    pop.push_back(pygmo::ndarr_to_vector<pg::vector_double>(x));
                } else {
                    pop.push_back(pygmo::ndarr_to_vector<pg::vector_double>(x),
                                  pygmo::ndarr_to_vector<pg::vector_double>(py::cast<py::array_t<double>>(f)));
                }
            },
            pygmo::population_push_back_docstring().c_str(), py::arg("x"), py::arg("f") = py::none())
        .def(
            "random_decision_vector",
            [](const pg::population &pop) {
                return pygmo::vector_to_ndarr<py::array_t<double>>(pop.random_decision_vector());
            },
            pygmo::population_random_decision_vector_docstring().c_str())
        .def(
            "best_idx",
            [](const pg::population &pop, const py::array_t<double> &tol) {
                return pop.best_idx(pygmo::ndarr_to_vector<pg::vector_double>(tol));
            },
            py::arg("tol"))
        .def(
            "best_idx", [](const pg::population &pop, double tol) { return pop.best_idx(tol); }, py::arg("tol"))
        .def(
            "best_idx", [](const pg::population &pop) { return pop.best_idx(); },
            pygmo::population_best_idx_docstring().c_str())
        .def(
            "worst_idx",
            [](const pg::population &pop, const py::array_t<double> &tol) {
                return pop.worst_idx(pygmo::ndarr_to_vector<pg::vector_double>(tol));
            },
            py::arg("tol"))
        .def(
            "worst_idx", [](const pg::population &pop, double tol) { return pop.worst_idx(tol); }, py::arg("tol"))
        .def(
            "worst_idx", [](const pg::population &pop) { return pop.worst_idx(); },
            pygmo::population_worst_idx_docstring().c_str())
        .def("__len__", &pg::population::size)
        .def(
            "set_xf",
            [](pg::population &pop, pg::population::size_type i, const py::array_t<double> &x,
               const py::array_t<double> &f) {
                pop.set_xf(i, pygmo::ndarr_to_vector<pg::vector_double>(x),
                           pygmo::ndarr_to_vector<pg::vector_double>(f));
            },
            pygmo::population_set_xf_docstring().c_str())
        .def(
            "set_x",
            [](pg::population &pop, pg::population::size_type i, const py::array_t<double> &x) {
                pop.set_x(i, pygmo::ndarr_to_vector<pg::vector_double>(x));
            },
            pygmo::population_set_x_docstring().c_str())
        .def(
            "get_f",
            [](const pg::population &pop) { return pygmo::vvector_to_ndarr<py::array_t<double>>(pop.get_f()); },
            pygmo::population_get_f_docstring().c_str())
        .def(
            "get_x",
            [](const pg::population &pop) { return pygmo::vvector_to_ndarr<py::array_t<double>>(pop.get_x()); },
            pygmo::population_get_x_docstring().c_str())
        .def(
            "get_ID",
            [](const pg::population &pop) {
                return pygmo::vector_to_ndarr<py::array_t<unsigned long long>>(pop.get_ID());
            },
            pygmo::population_get_ID_docstring().c_str())
        .def("get_seed", &pg::population::get_seed, pygmo::population_get_seed_docstring().c_str())
        .def_property_readonly(
            "champion_x",
            [](const pg::population &pop) { return pygmo::vector_to_ndarr<py::array_t<double>>(pop.champion_x()); },
            pygmo::population_champion_x_docstring().c_str())
        .def_property_readonly(
            "champion_f",
            [](const pg::population &pop) { return pygmo::vector_to_ndarr<py::array_t<double>>(pop.champion_f()); },
            pygmo::population_champion_f_docstring().c_str())
        .def_property_readonly(
            "problem", [](pg::population &pop) -> pg::problem & { return pop.get_problem(); },
            py::return_value_policy::reference_internal, pygmo::population_problem_docstring().c_str());

    // Archi.
    py::class_<pg::archipelago> archi_class(m, "archipelago", pygmo::archipelago_docstring().c_str());
    archi_class
        // Def ctor.
        .def(py::init<>())
        .def(py::init<const pg::topology &>())
        // repr().
        .def("__repr__", &pygmo::ostream_repr<pg::archipelago>)
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<pg::archipelago>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<pg::archipelago>, "memo"_a)
        // Pickle support.
        .def(py::pickle(&pygmo::pickle_getstate_wrapper<pg::archipelago>,
                        &pygmo::pickle_setstate_wrapper<pg::archipelago>))
        // Size.
        .def("__len__", &pg::archipelago::size)
        .def(
            "evolve", [](pg::archipelago &archi, unsigned n) { archi.evolve(n); },
            pygmo::archipelago_evolve_docstring().c_str(), py::arg("n") = 1u)
        .def("wait", &pg::archipelago::wait, pygmo::archipelago_wait_docstring().c_str())
        .def("wait_check", &pg::archipelago::wait_check, pygmo::archipelago_wait_check_docstring().c_str())
        .def(
            "__getitem__",
            [](pg::archipelago &archi, pg::archipelago::size_type n) -> pg::island & { return archi[n]; },
            pygmo::archipelago_getitem_docstring().c_str(), py::return_value_policy::reference_internal)
        // NOTE: docs for push_back() are in the Python reimplementation.
        .def("_push_back", [](pg::archipelago &archi, const pg::island &isl) { archi.push_back(isl); })
        // Champions.
        .def(
            "get_champions_f",
            [](const pg::archipelago &archi) -> py::list {
                py::list retval;
                auto fs = archi.get_champions_f();
                for (const auto &f : fs) {
                    retval.append(pygmo::vector_to_ndarr<py::array_t<double>>(f));
                }
                return retval;
            },
            pygmo::archipelago_get_champions_f_docstring().c_str())
        .def(
            "get_champions_x",
            [](const pg::archipelago &archi) -> py::list {
                py::list retval;
                auto xs = archi.get_champions_x();
                for (const auto &x : xs) {
                    retval.append(pygmo::vector_to_ndarr<py::array_t<double>>(x));
                }
                return retval;
            },
            pygmo::archipelago_get_champions_x_docstring().c_str())
        .def(
            "get_migrants_db",
            [](const pg::archipelago &archi) -> py::list {
                py::list retval;
                const auto tmp = archi.get_migrants_db();
                for (const auto &ig : tmp) {
                    retval.append(pygmo::inds_to_tuple(ig));
                }
                return retval;
            },
            pygmo::archipelago_get_migrants_db_docstring().c_str())
        .def(
            "set_migrants_db",
            [](pg::archipelago &archi, const py::list &mig) {
                pg::archipelago::migrants_db_t mig_db;

                for (auto o : mig) {
                    mig_db.push_back(pygmo::iterable_to_inds(py::cast<py::iterable>(o)));
                }

                archi.set_migrants_db(mig_db);
            },
            pygmo::archipelago_set_migrants_db_docstring().c_str())
        .def(
            "get_migration_log",
            [](const pg::archipelago &archi) -> py::list {
                py::list retval;
                const auto tmp = archi.get_migration_log();
                for (const auto &le : tmp) {
                    retval.append(py::make_tuple(std::get<0>(le), std::get<1>(le),
                                                 pygmo::vector_to_ndarr<py::array_t<double>>(std::get<2>(le)),
                                                 pygmo::vector_to_ndarr<py::array_t<double>>(std::get<3>(le)),
                                                 std::get<4>(le), std::get<5>(le)));
                }
                return retval;
            },
            pygmo::archipelago_get_migration_log_docstring().c_str())
        .def("get_topology", &pg::archipelago::get_topology, pygmo::archipelago_get_topology_docstring().c_str())
        .def("_set_topology", &pg::archipelago::set_topology)
        .def("set_migration_type", &pg::archipelago::set_migration_type,
             pygmo::archipelago_set_migration_type_docstring().c_str(), py::arg("mt"))
        .def("set_migrant_handling", &pg::archipelago::set_migrant_handling,
             pygmo::archipelago_set_migrant_handling_docstring().c_str(), py::arg("mh"))
        .def("get_migration_type", &pg::archipelago::get_migration_type,
             pygmo::archipelago_get_migration_type_docstring().c_str())
        .def("get_migrant_handling", &pg::archipelago::get_migrant_handling,
             pygmo::archipelago_get_migrant_handling_docstring().c_str())
        .def_property_readonly("status", &pg::archipelago::status, pygmo::archipelago_status_docstring().c_str());

    // Problem class.
    py::class_<pg::problem> problem_class(m, "problem", pygmo::problem_docstring().c_str());
    problem_class
        // Def ctor.
        .def(py::init<>())
        // repr().
        .def("__repr__", &pygmo::ostream_repr<pg::problem>)
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<pg::problem>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<pg::problem>, "memo"_a)
        // Pickle support.
        .def(py::pickle(&pygmo::pickle_getstate_wrapper<pg::problem>, &pygmo::pickle_setstate_wrapper<pg::problem>))
        // UDP extraction.
        .def("_py_extract", &pygmo::generic_py_extract<pg::problem>)
        // Problem methods.
        .def(
            "fitness",
            [](const pg::problem &p, const py::array_t<double> &dv) {
                return pygmo::vector_to_ndarr<py::array_t<double>>(
                    p.fitness(pygmo::ndarr_to_vector<pg::vector_double>(dv)));
            },
            pygmo::problem_fitness_docstring().c_str(), py::arg("dv"))
        .def(
            "get_bounds",
            [](const pg::problem &p) {
                return py::make_tuple(pygmo::vector_to_ndarr<py::array_t<double>>(p.get_lb()),
                                      pygmo::vector_to_ndarr<py::array_t<double>>(p.get_ub()));
            },
            pygmo::problem_get_bounds_docstring().c_str())
        .def(
            "get_lb", [](const pg::problem &p) { return pygmo::vector_to_ndarr<py::array_t<double>>(p.get_lb()); },
            pygmo::problem_get_lb_docstring().c_str())
        .def(
            "get_ub", [](const pg::problem &p) { return pygmo::vector_to_ndarr<py::array_t<double>>(p.get_ub()); },
            pygmo::problem_get_ub_docstring().c_str())
        .def(
            "batch_fitness",
            [](const pg::problem &p, const py::array_t<double> &dvs) {
                return pygmo::vector_to_ndarr<py::array_t<double>>(
                    p.batch_fitness(pygmo::ndarr_to_vector<pg::vector_double>(dvs)));
            },
            pygmo::problem_batch_fitness_docstring().c_str(), py::arg("dvs"))
        .def("has_batch_fitness", &pg::problem::has_batch_fitness, pygmo::problem_has_batch_fitness_docstring().c_str())
        .def(
            "gradient",
            [](const pg::problem &p, const py::array_t<double> &dv) {
                return pygmo::vector_to_ndarr<py::array_t<double>>(
                    p.gradient(pygmo::ndarr_to_vector<pg::vector_double>(dv)));
            },
            pygmo::problem_gradient_docstring().c_str(), py::arg("dv"))
        .def("has_gradient", &pg::problem::has_gradient, pygmo::problem_has_gradient_docstring().c_str())
        .def(
            "gradient_sparsity", [](const pg::problem &p) { return pygmo::sp_to_ndarr(p.gradient_sparsity()); },
            pygmo::problem_gradient_sparsity_docstring().c_str())
        .def("has_gradient_sparsity", &pg::problem::has_gradient_sparsity,
             pygmo::problem_has_gradient_sparsity_docstring().c_str())
        .def(
            "hessians",
            [](const pg::problem &p, const py::array_t<double> &dv) -> py::list {
                py::list retval;
                for (const auto &v : p.hessians(pygmo::ndarr_to_vector<pg::vector_double>(dv))) {
                    retval.append(pygmo::vector_to_ndarr<py::array_t<double>>(v));
                }
                return retval;
            },
            pygmo::problem_hessians_docstring().c_str(), py::arg("dv"))
        .def("has_hessians", &pg::problem::has_hessians, pygmo::problem_has_hessians_docstring().c_str())
        .def(
            "hessians_sparsity",
            [](const pg::problem &p) -> py::list {
                py::list retval;
                for (const auto &sp : p.hessians_sparsity()) {
                    retval.append(pygmo::sp_to_ndarr(sp));
                }
                return retval;
            },
            pygmo::problem_hessians_sparsity_docstring().c_str())
        .def("has_hessians_sparsity", &pg::problem::has_hessians_sparsity,
             pygmo::problem_has_hessians_sparsity_docstring().c_str())
        .def("get_nobj", &pg::problem::get_nobj, pygmo::problem_get_nobj_docstring().c_str())
        .def("get_nx", &pg::problem::get_nx, pygmo::problem_get_nx_docstring().c_str())
        .def("get_nix", &pg::problem::get_nix, pygmo::problem_get_nix_docstring().c_str())
        .def("get_ncx", &pg::problem::get_ncx, pygmo::problem_get_ncx_docstring().c_str())
        .def("get_nf", &pg::problem::get_nf, pygmo::problem_get_nf_docstring().c_str())
        .def("get_nec", &pg::problem::get_nec, pygmo::problem_get_nec_docstring().c_str())
        .def("get_nic", &pg::problem::get_nic, pygmo::problem_get_nic_docstring().c_str())
        .def("get_nc", &pg::problem::get_nc, pygmo::problem_get_nc_docstring().c_str())
        .def("get_fevals", &pg::problem::get_fevals, pygmo::problem_get_fevals_docstring().c_str())
        .def("increment_fevals", &pg::problem::increment_fevals, pygmo::problem_increment_fevals_docstring().c_str(),
             py::arg("n"))
        .def("get_gevals", &pg::problem::get_gevals, pygmo::problem_get_gevals_docstring().c_str())
        .def("get_hevals", &pg::problem::get_hevals, pygmo::problem_get_hevals_docstring().c_str())
        .def("set_seed", &pg::problem::set_seed, pygmo::problem_set_seed_docstring().c_str(), py::arg("seed"))
        .def("has_set_seed", &pg::problem::has_set_seed, pygmo::problem_has_set_seed_docstring().c_str())
        .def("is_stochastic", &pg::problem::is_stochastic,
             "is_stochastic()\n\nAlias for :func:`~pygmo.problem.has_set_seed()`.\n")
        .def(
            "feasibility_x",
            [](const pg::problem &p, const py::array_t<double> &x) {
                return p.feasibility_x(pygmo::ndarr_to_vector<pg::vector_double>(x));
            },
            pygmo::problem_feasibility_x_docstring().c_str(), py::arg("x"))
        .def(
            "feasibility_f",
            [](const pg::problem &p, const py::array_t<double> &f) {
                return p.feasibility_f(pygmo::ndarr_to_vector<pg::vector_double>(f));
            },
            pygmo::problem_feasibility_f_docstring().c_str(), py::arg("f"))
        .def("get_name", &pg::problem::get_name, pygmo::problem_get_name_docstring().c_str())
        .def("get_extra_info", &pg::problem::get_extra_info, pygmo::problem_get_extra_info_docstring().c_str())
        .def("get_thread_safety", &pg::problem::get_thread_safety, pygmo::problem_get_thread_safety_docstring().c_str())
        .def_property(
            "c_tol",
            [](const pg::problem &prob) { return pygmo::vector_to_ndarr<py::array_t<double>>(prob.get_c_tol()); },
            [](pg::problem &prob, const py::object &c_tol) {
                try {
                    prob.set_c_tol(py::cast<double>(c_tol));
                } catch (const py::cast_error &) {
                    prob.set_c_tol(pygmo::ndarr_to_vector<pg::vector_double>(py::cast<py::array_t<double>>(c_tol)));
                }
            },
            pygmo::problem_c_tol_docstring().c_str());

    // Expose the C++ problems.
    pygmo::expose_problems_0(m, problem_class, problems_module);
    pygmo::expose_problems_1(m, problem_class, problems_module);

    // Finalize.
    problem_class.def(py::init<const py::object &>(), py::arg("udp"));

    // Algorithm class.
    py::class_<pg::algorithm> algorithm_class(m, "algorithm", pygmo::algorithm_docstring().c_str());
    algorithm_class
        // Def ctor.
        .def(py::init<>())
        // repr().
        .def("__repr__", &pygmo::ostream_repr<pg::algorithm>)
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<pg::algorithm>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<pg::algorithm>, "memo"_a)
        // Pickle support.
        .def(py::pickle(&pygmo::pickle_getstate_wrapper<pg::algorithm>, &pygmo::pickle_setstate_wrapper<pg::algorithm>))
        // UDA extraction.
        .def("_py_extract", &pygmo::generic_py_extract<pg::algorithm>)
        // Algorithm methods.
        .def("evolve", &pg::algorithm::evolve, pygmo::algorithm_evolve_docstring().c_str(), py::arg("pop"))
        .def("set_seed", &pg::algorithm::set_seed, pygmo::algorithm_set_seed_docstring().c_str(), py::arg("seed"))
        .def("has_set_seed", &pg::algorithm::has_set_seed, pygmo::algorithm_has_set_seed_docstring().c_str())
        .def("set_verbosity", &pg::algorithm::set_verbosity, pygmo::algorithm_set_verbosity_docstring().c_str(),
             py::arg("level"))
        .def("has_set_verbosity", &pg::algorithm::has_set_verbosity,
             pygmo::algorithm_has_set_verbosity_docstring().c_str())
        .def("is_stochastic", &pg::algorithm::is_stochastic,
             "is_stochastic()\n\nAlias for :func:`~pygmo.algorithm.has_set_seed()`.\n")
        .def("get_name", &pg::algorithm::get_name, pygmo::algorithm_get_name_docstring().c_str())
        .def("get_extra_info", &pg::algorithm::get_extra_info, pygmo::algorithm_get_extra_info_docstring().c_str())
        .def("get_thread_safety", &pg::algorithm::get_thread_safety,
             pygmo::algorithm_get_thread_safety_docstring().c_str());

    // Expose the C++ algos.
    pygmo::expose_algorithms_0(m, algorithm_class, algorithms_module);
    pygmo::expose_algorithms_1(m, algorithm_class, algorithms_module);

    // Finalize.
    algorithm_class.def(py::init<const py::object &>(), py::arg("uda"));

    // bfe class.
    py::class_<pg::bfe> bfe_class(m, "bfe", pygmo::bfe_docstring().c_str());
    bfe_class
        // Def ctor.
        .def(py::init<>())
        // repr().
        .def("__repr__", &pygmo::ostream_repr<pg::bfe>)
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<pg::bfe>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<pg::bfe>, "memo"_a)
        // Pickle support.
        .def(py::pickle(&pygmo::pickle_getstate_wrapper<pg::bfe>, &pygmo::pickle_setstate_wrapper<pg::bfe>))
        // UDBFE extraction.
        .def("_py_extract", &pygmo::generic_py_extract<pg::bfe>)
        // Bfe methods.
        .def("_call_impl",
             [](const pg::bfe &b, const pg::problem &prob, const py::array_t<double> &dvs) {
                 return pygmo::vector_to_ndarr<py::array_t<double>>(
                     b(prob, pygmo::ndarr_to_vector<pg::vector_double>(dvs)));
             })
        .def("get_name", &pg::bfe::get_name, pygmo::bfe_get_name_docstring().c_str())
        .def("get_extra_info", &pg::bfe::get_extra_info, pygmo::bfe_get_extra_info_docstring().c_str())
        .def("get_thread_safety", &pg::bfe::get_thread_safety, pygmo::bfe_get_thread_safety_docstring().c_str());

    // Expose the C++ bfes.
    pygmo::expose_bfes(m, bfe_class, batch_evaluators_module);

    // Finalize.
    bfe_class.def(py::init<const py::object &>(), py::arg("udbfe"));

    // Island class.
    py::class_<pg::island> island_class(m, "island", pygmo::island_docstring().c_str());
    island_class
        // Def ctor.
        .def(py::init<>())
        // Ctor from algo, pop, and policies.
        .def(py::init<const pg::algorithm &, const pg::population &, const pg::r_policy &, const pg::s_policy &>())
        // repr().
        .def("__repr__", &pygmo::ostream_repr<pg::island>)
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<pg::island>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<pg::island>, "memo"_a)
        // Pickle support.
        .def(py::pickle(&pygmo::pickle_getstate_wrapper<pg::island>, &pygmo::pickle_setstate_wrapper<pg::island>))
        // UDI extraction.
        .def("_py_extract", &pygmo::generic_py_extract<pg::island>)
        .def(
            "evolve", [](pg::island &isl, unsigned n) { isl.evolve(n); }, pygmo::island_evolve_docstring().c_str(),
            py::arg("n") = 1u)
        .def("wait", &pg::island::wait, pygmo::island_wait_docstring().c_str())
        .def("wait_check", &pg::island::wait_check, pygmo::island_wait_check_docstring().c_str())
        .def("get_population", &pg::island::get_population, pygmo::island_get_population_docstring().c_str())
        .def("get_algorithm", &pg::island::get_algorithm, pygmo::island_get_algorithm_docstring().c_str())
        .def("set_population", &pg::island::set_population, pygmo::island_set_population_docstring().c_str(),
             py::arg("pop"))
        .def("set_algorithm", &pg::island::set_algorithm, pygmo::island_set_algorithm_docstring().c_str(),
             py::arg("algo"))
        .def("get_name", &pg::island::get_name, pygmo::island_get_name_docstring().c_str())
        .def("get_extra_info", &pg::island::get_extra_info, pygmo::island_get_extra_info_docstring().c_str())
        .def("get_r_policy", &pg::island::get_r_policy, pygmo::island_get_r_policy_docstring().c_str())
        .def("get_s_policy", &pg::island::get_s_policy, pygmo::island_get_s_policy_docstring().c_str())
        .def_property_readonly("status", &pg::island::status, pygmo::island_status_docstring().c_str());

    // Expose the C++ islands.
    pygmo::expose_islands(m, island_class, islands_module);

    // Finalize.
    island_class.def(py::init<const py::object &, const pg::algorithm &, const pg::population &, const pg::r_policy &,
                              const pg::s_policy &>());

    // Replacement policy class.
    py::class_<pg::r_policy> r_policy_class(m, "r_policy", pygmo::r_policy_docstring().c_str());
    r_policy_class
        // Def ctor.
        .def(py::init<>())
        // repr().
        .def("__repr__", &pygmo::ostream_repr<pg::r_policy>)
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<pg::r_policy>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<pg::r_policy>, "memo"_a)
        // Pickle support.
        .def(py::pickle(&pygmo::pickle_getstate_wrapper<pg::r_policy>, &pygmo::pickle_setstate_wrapper<pg::r_policy>))
        // UDRP extraction.
        .def("_py_extract", &pygmo::generic_py_extract<pg::r_policy>)
        // r_policy methods.
        .def(
            "replace",
            [](const pg::r_policy &r, const py::iterable &inds, const pg::vector_double::size_type &nx,
               const pg::vector_double::size_type &nix, const pg::vector_double::size_type &nobj,
               const pg::vector_double::size_type &nec, const pg::vector_double::size_type &nic,
               const py::array_t<double> &tol, const py::iterable &mig) {
                return pygmo::inds_to_tuple(r.replace(pygmo::iterable_to_inds(inds), nx, nix, nobj, nec, nic,
                                                      pygmo::ndarr_to_vector<pg::vector_double>(tol),
                                                      pygmo::iterable_to_inds(mig)));
            },
            pygmo::r_policy_replace_docstring().c_str(), py::arg("inds"), py::arg("nx"), py::arg("nix"),
            py::arg("nobj"), py::arg("nec"), py::arg("nic"), py::arg("tol"), py::arg("mig"))
        .def("get_name", &pg::r_policy::get_name, pygmo::r_policy_get_name_docstring().c_str())
        .def("get_extra_info", &pg::r_policy::get_extra_info, pygmo::r_policy_get_extra_info_docstring().c_str());

    // Expose the C++ replacement policies.
    pygmo::expose_r_policies(m, r_policy_class, r_policies_module);

    // Finalize.
    r_policy_class.def(py::init<const py::object &>(), py::arg("udrp"));

    // Selection policy class.
    py::class_<pg::s_policy> s_policy_class(m, "s_policy", pygmo::s_policy_docstring().c_str());
    s_policy_class
        // Def ctor.
        .def(py::init<>())
        // repr().
        .def("__repr__", &pygmo::ostream_repr<pg::s_policy>)
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<pg::s_policy>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<pg::s_policy>, "memo"_a)
        // Pickle support.
        .def(py::pickle(&pygmo::pickle_getstate_wrapper<pg::s_policy>, &pygmo::pickle_setstate_wrapper<pg::s_policy>))
        // UDSP extraction.
        .def("_py_extract", &pygmo::generic_py_extract<pg::s_policy>)
        // s_policy methods.
        .def(
            "select",
            [](const pg::s_policy &s, const py::iterable &inds, const pg::vector_double::size_type &nx,
               const pg::vector_double::size_type &nix, const pg::vector_double::size_type &nobj,
               const pg::vector_double::size_type &nec, const pg::vector_double::size_type &nic,
               const py::array_t<double> &tol) {
                return pygmo::inds_to_tuple(s.select(pygmo::iterable_to_inds(inds), nx, nix, nobj, nec, nic,
                                                     pygmo::ndarr_to_vector<pg::vector_double>(tol)));
            },
            pygmo::s_policy_select_docstring().c_str(), py::arg("inds"), py::arg("nx"), py::arg("nix"), py::arg("nobj"),
            py::arg("nec"), py::arg("nic"), py::arg("tol"))
        .def("get_name", &pg::s_policy::get_name, pygmo::s_policy_get_name_docstring().c_str())
        .def("get_extra_info", &pg::s_policy::get_extra_info, pygmo::s_policy_get_extra_info_docstring().c_str());

    // Expose the C++ selection policies.
    pygmo::expose_s_policies(m, s_policy_class, s_policies_module);

    // Finalize.
    s_policy_class.def(py::init<const py::object &>(), py::arg("udsp"));

    // Topology class.
    py::class_<pg::topology> topology_class(m, "topology", pygmo::topology_docstring().c_str());
    topology_class
        // Def ctor.
        .def(py::init<>())
        // repr().
        .def("__repr__", &pygmo::ostream_repr<pg::topology>)
        // Copy and deepcopy.
        .def("__copy__", &pygmo::generic_copy_wrapper<pg::topology>)
        .def("__deepcopy__", &pygmo::generic_deepcopy_wrapper<pg::topology>, "memo"_a)
        // Pickle support.
        .def(py::pickle(&pygmo::pickle_getstate_wrapper<pg::topology>, &pygmo::pickle_setstate_wrapper<pg::topology>))
        // UDT extraction.
        .def("_py_extract", &pygmo::generic_py_extract<pg::topology>)
        // Topology methods.
        .def(
            "get_connections",
            [](const pg::topology &t, std::size_t n) -> py::tuple {
                auto ret = t.get_connections(n);
                return py::make_tuple(pygmo::vector_to_ndarr<py::array_t<std::size_t>>(ret.first),
                                      pygmo::vector_to_ndarr<py::array_t<double>>(ret.second));
            },
            pygmo::topology_get_connections_docstring().c_str(), py::arg("n"))
        .def(
            "push_back", [](pg::topology &t, unsigned n) { t.push_back(n); },
            pygmo::topology_push_back_docstring().c_str(), py::arg("n") = std::size_t(1))
        .def(
            "to_networkx", [](const pg::topology &t) { return pygmo::bgl_graph_t_to_networkx(t.to_bgl()); },
            pygmo::topology_to_networkx_docstring().c_str())
        .def("get_name", &pg::topology::get_name, pygmo::topology_get_name_docstring().c_str())
        .def("get_extra_info", &pg::topology::get_extra_info, pygmo::topology_get_extra_info_docstring().c_str());

    // Expose the C++ topologies.
    pygmo::expose_topologies(m, topology_class, topologies_module);

    // Finalize.
    topology_class.def(py::init<const py::object &>(), py::arg("udt"));
}
