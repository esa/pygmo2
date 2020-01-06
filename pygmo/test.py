# Copyright 2019 PaGMO development team
#
# This file is part of the pygmo library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class _prob(object):

    def get_bounds(self):
        return ([0, 0], [1, 1])

    def fitness(self, a):
        return [42]


class _quick_prob:

    def fitness(self, dv):
        return [sum(dv)]

    def get_bounds(self):
        return ([0] * 10, [1] * 10)


class _r_pol(object):

    def replace(self, inds, nx, nix, nobj, nec, nic, tol, mig):
        return inds


class _s_pol(object):

    def select(self, inds, nx, nix, nobj, nec, nic, tol):
        return inds


class core_test_case(_ut.TestCase):
    """Test case for core PyGMO functionality.

    """

    def runTest(self):
        from numpy import random, all, array
        from .core import _builtins, _type, _str, _callable, _deepcopy, _test_object_serialization as tos
        from . import __version__
        self.assertTrue(__version__ != "")
        import builtins as b
        self.assertEqual(b, _builtins())
        self.assertEqual(type(int), _type(int))
        self.assertEqual(str(123), _str(123))
        self.assertEqual(callable(1), _callable(1))
        self.assertEqual(callable(lambda _: None), _callable(lambda _: None))
        l = [1, 2, 3, ["abc"]]
        self.assert_(id(l) != id(_deepcopy(l)))
        self.assert_(id(l[3]) != id(_deepcopy(l)[3]))
        self.assertEqual(tos(l), l)
        self.assertEqual(tos({'a': l, 3: "Hello world"}),
                         {'a': l, 3: "Hello world"})
        a = random.rand(3, 2)
        self.assert_(all(tos(a) == a))

        # Run the tests for the selection of the serialization backend.
        self.run_s11n_test()

    def run_s11n_test(self):
        # Tests for the selection of the serialization backend.
        import cloudpickle as clpickle
        import pickle
        from . import set_serialization_backend as ssb, get_serialization_backend as gsb
        from . import problem, island, de
        has_dill = False
        try:
            import dill
            has_dill = True
        except ImportError:
            pass

        # Default s11n backend.
        self.assertTrue(gsb() == clpickle)

        # Error checking.
        with self.assertRaises(TypeError) as cm:
            ssb(1)
        err = cm.exception
        self.assertTrue(
            "The serialization backend must be specified as a string, but an object of type" in str(err))

        with self.assertRaises(ValueError) as cm:
            ssb("hello")
        err = cm.exception
        self.assertEqual(
            "The serialization backend 'hello' is not valid. The valid backends are: ['pickle', 'cloudpickle', 'dill']", str(err))

        if not has_dill:
            with self.assertRaises(ImportError) as cm:
                ssb("dill")
            err = cm.exception
            self.assertEqual(
                "The 'dill' serialization backend was specified, but the dill module is not installed.", str(err))

        ssb("pickle")
        self.assertTrue(gsb() == pickle)

        # Try to pickle something.
        p = problem(_prob())
        self.assertEqual(str(pickle.loads(pickle.dumps(p))), str(p))
        isl = island(prob=p, algo=de(gen=500), size=20)
        self.assertEqual(str(pickle.loads(pickle.dumps(isl))), str(isl))

        # Try with dill as well, if available.
        if has_dill:
            ssb("dill")
            self.assertTrue(gsb() == dill)

            p = problem(_prob())
            self.assertEqual(str(pickle.loads(pickle.dumps(p))), str(p))
            isl = island(prob=p, algo=de(gen=500), size=20)
            self.assertEqual(str(pickle.loads(pickle.dumps(isl))), str(isl))

        # Reset to cloudpickle before exiting.
        ssb("cloudpickle")
        self.assertTrue(gsb() == clpickle)


class population_test_case(_ut.TestCase):
    """Test case for the population class.

    """

    def runTest(self):
        self.run_init_test()
        self.run_best_worst_idx_test()
        self.run_champion_test()
        self.run_getters_test()
        self.run_problem_test()
        self.run_push_back_test()
        self.run_random_dv_test()
        self.run_set_x_xf_test()
        self.run_pickle_test()

    def run_init_test(self):
        from .core import population, null_problem, rosenbrock, problem, bfe, default_bfe, thread_bfe
        pop = population()
        self.assertTrue(len(pop) == 0)
        self.assertTrue(pop.problem.extract(null_problem) is not None)
        self.assertTrue(pop.problem.extract(rosenbrock) is None)
        pop.get_seed()
        pop = population(rosenbrock())
        self.assertTrue(len(pop) == 0)
        self.assertTrue(pop.problem.extract(null_problem) is None)
        self.assertTrue(pop.problem.extract(rosenbrock) is not None)
        pop.get_seed()
        pop = population(seed=42, size=5, prob=problem(rosenbrock()))
        self.assertTrue(len(pop) == 5)
        self.assertTrue(pop.problem.extract(null_problem) is None)
        self.assertTrue(pop.problem.extract(rosenbrock) is not None)
        self.assertEqual(pop.get_seed(), 42)

        # Tests with a bfe argument.
        p = problem(rosenbrock())
        pop = population(prob=p, size=20, b=bfe(default_bfe()))
        for x, f in zip(pop.get_x(), pop.get_f()):
            self.assertEqual(p.fitness(x), f)

        # Pass in explicit UDBFE.
        pop = population(prob=p, size=20, b=thread_bfe())
        for x, f in zip(pop.get_x(), pop.get_f()):
            self.assertEqual(p.fitness(x), f)

        # Pythonic problem.
        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

        p = problem(p())
        pop = population(prob=p, size=20, b=bfe(default_bfe()))
        for x, f in zip(pop.get_x(), pop.get_f()):
            self.assertEqual(p.fitness(x), f)

        # Pythonic problem with batch_fitness method.
        class p(object):

            def get_bounds(self):
                return ([0], [1])

            def fitness(self, a):
                return [42]

            def batch_fitness(self, dvs):
                return [43] * len(dvs)

        p = problem(p())
        pop = population(prob=p, size=20, b=bfe(default_bfe()))
        for f in pop.get_f():
            self.assertEqual(f, 43)

    def run_best_worst_idx_test(self):
        from .core import population, rosenbrock, zdt
        pop = population(rosenbrock(), size=10)
        self.assertTrue(pop.best_idx() < 10)
        self.assertTrue(pop.best_idx(0.001) < 10)
        self.assertTrue(pop.best_idx(tol=0.001) < 10)
        self.assertTrue(pop.best_idx(tol=[0.001, 0.001]) < 10)
        self.assertTrue(pop.worst_idx() < 10)
        self.assertTrue(pop.worst_idx(0.001) < 10)
        self.assertTrue(pop.worst_idx(tol=0.001) < 10)
        self.assertTrue(pop.worst_idx(tol=[0.001, 0.001]) < 10)
        pop = population(zdt(param=10), size=10)
        self.assertRaises(ValueError, lambda: pop.best_idx())
        self.assertRaises(ValueError, lambda: pop.worst_idx())

    def run_champion_test(self):
        from .core import population, null_problem, problem, zdt
        from numpy import array
        udp = null_problem()
        prob = problem(udp)
        pop = population(prob)
        self.assertEqual(len(pop.champion_f), 0)
        self.assertEqual(len(pop.champion_x), 0)
        pop.push_back([1.])
        self.assertEqual(pop.champion_f[0], 0.)
        self.assertEqual(pop.champion_x[0], 1.)
        pop = population(zdt(param=10))
        self.assertRaises(ValueError, lambda: pop.champion_x)
        self.assertRaises(ValueError, lambda: pop.champion_f)

    def run_getters_test(self):
        from .core import population
        from numpy import ndarray
        pop = population(size=100, seed=123)
        self.assertEqual(len(pop.get_ID()), 100)
        self.assertTrue(isinstance(pop.get_ID(), ndarray))
        self.assertEqual(len(pop.get_f()), 100)
        self.assertTrue(isinstance(pop.get_f(), ndarray))
        self.assertEqual(pop.get_f().shape, (100, 1))
        self.assertEqual(len(pop.get_x()), 100)
        self.assertTrue(isinstance(pop.get_x(), ndarray))
        self.assertEqual(pop.get_x().shape, (100, 1))
        self.assertEqual(pop.get_seed(), 123)

    def run_problem_test(self):
        from .core import population, rosenbrock, null_problem, problem, zdt
        import sys
        pop = population(size=10)
        rc = sys.getrefcount(pop)
        prob = pop.problem
        self.assertTrue(sys.getrefcount(pop) == rc + 1)
        del prob
        self.assertTrue(sys.getrefcount(pop) == rc)
        self.assertTrue(pop.problem.extract(null_problem) is not None)
        self.assertTrue(pop.problem.extract(rosenbrock) is None)
        pop = population(rosenbrock(), size=10)
        self.assertTrue(pop.problem.extract(null_problem) is None)
        self.assertTrue(pop.problem.extract(rosenbrock) is not None)

        def prob_setter():
            pop.problem = problem(zdt(param=10))
        self.assertRaises(AttributeError, prob_setter)

    def run_push_back_test(self):
        from .core import population, rosenbrock
        from numpy import array
        pop = population(rosenbrock(), size=5)
        self.assertEqual(len(pop), 5)
        self.assertEqual(pop.problem.get_fevals(), 5)
        pop.push_back(x=[.1, .1])
        self.assertEqual(len(pop), 6)
        self.assertEqual(pop.problem.get_fevals(), 6)
        pop.push_back(x=[.1, .1], f=array([1]))
        self.assertEqual(len(pop), 7)
        self.assertEqual(pop.problem.get_fevals(), 6)
        pop.push_back(x=[.1, .1], f=array([0.0]))
        self.assertEqual(len(pop), 8)
        self.assertEqual(pop.problem.get_fevals(), 6)
        self.assertEqual(pop.best_idx(), 7)
        pop.push_back(x=[.1, .1], f=None)
        self.assertEqual(len(pop), 9)
        self.assertEqual(pop.problem.get_fevals(), 7)
        self.assertEqual(pop.best_idx(), 7)
        # Test bogus x, f dimensions.
        pop = population(rosenbrock(5), size=5)
        self.assertRaises(ValueError, lambda: pop.push_back([]))
        self.assertRaises(ValueError, lambda: pop.push_back([], []))
        self.assertRaises(ValueError, lambda: pop.push_back([1] * 5, []))
        self.assertRaises(ValueError, lambda: pop.push_back([1] * 5, [1, 2]))

    def run_random_dv_test(self):
        from .core import population, rosenbrock
        from numpy import ndarray
        pop = population(rosenbrock())
        self.assertTrue(isinstance(pop.random_decision_vector(), ndarray))
        self.assertTrue(pop.random_decision_vector().shape == (2,))
        self.assertTrue(pop.random_decision_vector()[0] >= -5)
        self.assertTrue(pop.random_decision_vector()[0] <= 10)
        self.assertTrue(pop.random_decision_vector()[1] >= -5)
        self.assertTrue(pop.random_decision_vector()[1] <= 10)

    def run_set_x_xf_test(self):
        from .core import population, rosenbrock
        from numpy import array
        pop = population(rosenbrock())
        self.assertRaises(ValueError, lambda: pop.set_x(0, [1, 1]))
        self.assertRaises(ValueError, lambda: pop.set_xf(0, (1, 1), [1]))
        pop = population(rosenbrock(), size=10)
        self.assertRaises(ValueError, lambda: pop.set_x(0, array([1, 1, 1])))
        self.assertRaises(ValueError, lambda: pop.set_xf(0, [1, 1], [1, 1]))
        self.assertRaises(ValueError, lambda: pop.set_xf(
            0, array([1, 1, 1]), [1, 1]))
        pop.set_x(0, array([1.1, 1.1]))
        self.assertTrue(all(pop.get_x()[0] == array([1.1, 1.1])))
        self.assertTrue(
            all(pop.get_f()[0] == pop.problem.fitness(array([1.1, 1.1]))))
        pop.set_x(4, array([1.1, 1.1]))
        self.assertTrue(all(pop.get_x()[4] == array([1.1, 1.1])))
        self.assertTrue(
            all(pop.get_f()[4] == pop.problem.fitness(array([1.1, 1.1]))))
        pop.set_xf(5, array([1.1, 1.1]), [1.25])
        self.assertTrue(all(pop.get_x()[5] == array([1.1, 1.1])))
        self.assertTrue(all(pop.get_f()[5] == array([1.25])))
        pop.set_xf(6, array([1.1, 1.1]), [0.])
        self.assertTrue(all(pop.get_x()[6] == array([1.1, 1.1])))
        self.assertTrue(all(pop.get_f()[6] == array([0])))
        self.assertEqual(pop.best_idx(), 6)

    def run_pickle_test(self):
        from .core import population, rosenbrock, translate
        from pickle import dumps, loads
        pop = population(rosenbrock(), size=12, seed=42)
        p = loads(dumps(pop))
        self.assertEqual(repr(pop), repr(p))
        pop = population(translate(rosenbrock(2), 2 * [.1]), size=12, seed=42)
        p = loads(dumps(pop))
        self.assertEqual(repr(pop), repr(p))
        pop = population(_prob(), size=12, seed=42)
        p = loads(dumps(pop))
        self.assertEqual(repr(pop), repr(p))
        pop = population(translate(_prob(), 2 * [.1]), size=12, seed=42)
        p = loads(dumps(pop))
        self.assertEqual(repr(pop), repr(p))


class mbh_test_case(_ut.TestCase):
    """Test case for the mbh meta-algorithm

    """

    def runTest(self):
        from . import mbh, de, compass_search, algorithm, thread_safety as ts, null_algorithm
        from numpy import array

        class algo(object):

            def evolve(pop):
                return pop

        # Def ctor.
        a = mbh()
        self.assertFalse(a.inner_algorithm.extract(compass_search) is None)
        self.assertTrue(a.inner_algorithm.is_(compass_search))
        self.assertTrue(a.inner_algorithm.extract(de) is None)
        self.assertFalse(a.inner_algorithm.is_(de))
        self.assertEqual(a.get_log(), [])
        self.assertTrue(all(a.get_perturb() == array([0.01])))
        seed = a.get_seed()
        self.assertEqual(a.get_verbosity(), 0)
        a.set_perturb([.2])
        self.assertTrue(all(a.get_perturb() == array([0.2])))
        al = algorithm(a)
        self.assertTrue(al.get_thread_safety() == ts.basic)
        self.assertTrue(al.extract(mbh).inner_algorithm.extract(
            compass_search) is not None)
        self.assertTrue(al.extract(mbh).inner_algorithm.extract(de) is None)
        self.assertTrue(str(seed) in str(al))
        al.set_verbosity(4)
        self.assertEqual(al.extract(mbh).get_verbosity(), 4)

        # From C++ algo.
        seed = 123321
        a = mbh(algo=de(), stop=5, perturb=.4)
        a = mbh(stop=5, perturb=(.4, .2), algo=de())
        a = mbh(algo=de(), stop=5, seed=seed, perturb=(.4, .2))
        self.assertTrue(a.inner_algorithm.extract(compass_search) is None)
        self.assertFalse(a.inner_algorithm.is_(compass_search))
        self.assertFalse(a.inner_algorithm.extract(de) is None)
        self.assertTrue(a.inner_algorithm.is_(de))
        self.assertEqual(a.get_log(), [])
        self.assertTrue(all(a.get_perturb() == array([.4, .2])))
        self.assertEqual(a.get_seed(), seed)
        self.assertEqual(a.get_verbosity(), 0)
        a.set_perturb([.2])
        self.assertTrue(all(a.get_perturb() == array([0.2])))
        al = algorithm(a)
        self.assertTrue(al.get_thread_safety() == ts.basic)
        self.assertTrue(al.extract(mbh).inner_algorithm.extract(
            compass_search) is None)
        self.assertTrue(al.extract(
            mbh).inner_algorithm.extract(de) is not None)
        self.assertTrue(str(seed) in str(al))
        al.set_verbosity(4)
        self.assertEqual(al.extract(mbh).get_verbosity(), 4)

        # From Python algo.
        class algo(object):

            def evolve(self, pop):
                return pop

        seed = 123321
        a = mbh(algo=algo(), stop=5, perturb=.4)
        a = mbh(stop=5, perturb=(.4, .2), algo=algo())
        a = mbh(algo=algo(), stop=5, seed=seed, perturb=(.4, .2))
        self.assertTrue(a.inner_algorithm.extract(compass_search) is None)
        self.assertFalse(a.inner_algorithm.is_(compass_search))
        self.assertFalse(a.inner_algorithm.extract(algo) is None)
        self.assertTrue(a.inner_algorithm.is_(algo))
        self.assertEqual(a.get_log(), [])
        self.assertTrue(all(a.get_perturb() == array([.4, .2])))
        self.assertEqual(a.get_seed(), seed)
        self.assertEqual(a.get_verbosity(), 0)
        a.set_perturb([.2])
        self.assertTrue(all(a.get_perturb() == array([0.2])))
        al = algorithm(a)
        self.assertTrue(al.get_thread_safety() == ts.none)
        self.assertTrue(al.extract(mbh).inner_algorithm.extract(
            compass_search) is None)
        self.assertTrue(al.extract(
            mbh).inner_algorithm.extract(algo) is not None)
        self.assertTrue(str(seed) in str(al))
        al.set_verbosity(4)
        self.assertEqual(al.extract(mbh).get_verbosity(), 4)

        # Construction from algorithm is allowed.
        mbh(algorithm(null_algorithm()), stop=5, perturb=.4)


class fair_replace_test_case(_ut.TestCase):
    """Test case for the fair replace UDRP

    """

    def runTest(self):
        from .core import fair_replace, r_policy
        udrp = fair_replace()
        r_pol = r_policy(udrp=udrp)
        self.assertEqual(r_pol.get_name(), "Fair replace")
        self.assertTrue("Absolute migration rate: 1" in repr(r_pol))
        r_pol = r_policy(udrp=fair_replace(rate=0))
        self.assertTrue("Absolute migration rate: 0" in repr(r_pol))
        r_pol = r_policy(udrp=fair_replace(2))
        self.assertTrue("Absolute migration rate: 2" in repr(r_pol))
        r_pol = r_policy(udrp=fair_replace(rate=.5))
        self.assertTrue("Fractional migration rate: 0.5" in repr(r_pol))

        with self.assertRaises(ValueError) as cm:
            r_policy(udrp=fair_replace(-1.2))
        err = cm.exception
        self.assertTrue(
            "Invalid fractional migration rate " in str(err))

        with self.assertRaises(TypeError) as cm:
            r_policy(udrp=fair_replace(rate="dsadasdas"))
        err = cm.exception
        self.assertTrue(
            "the migration rate must be an integral or floating-point value" in str(err))


class select_best_test_case(_ut.TestCase):
    """Test case for the select best UDSP

    """

    def runTest(self):
        from .core import select_best, s_policy
        udsp = select_best()
        s_pol = s_policy(udsp=udsp)
        self.assertEqual(s_pol.get_name(), "Select best")
        self.assertTrue("Absolute migration rate: 1" in repr(s_pol))
        s_pol = s_policy(udsp=select_best(rate=0))
        self.assertTrue("Absolute migration rate: 0" in repr(s_pol))
        s_pol = s_policy(udsp=select_best(2))
        self.assertTrue("Absolute migration rate: 2" in repr(s_pol))
        s_pol = s_policy(udsp=select_best(rate=.5))
        self.assertTrue("Fractional migration rate: 0.5" in repr(s_pol))

        with self.assertRaises(ValueError) as cm:
            s_policy(udsp=select_best(-1.2))
        err = cm.exception
        self.assertTrue(
            "Invalid fractional migration rate " in str(err))

        with self.assertRaises(TypeError) as cm:
            s_policy(udsp=select_best(rate="dsadasdas"))
        err = cm.exception
        self.assertTrue(
            "the migration rate must be an integral or floating-point value" in str(err))


class unconnected_test_case(_ut.TestCase):
    """Test case for the unconnected UDT

    """

    def runTest(self):
        from .core import unconnected, topology
        udt = unconnected()
        topo = topology(udt=udt)
        self.assertTrue(len(topo.get_connections(100)[0]) == 0)
        self.assertTrue(len(topo.get_connections(100)[1]) == 0)
        topo.push_back()
        topo.push_back()
        topo.push_back()
        self.assertTrue(len(topo.get_connections(100)[0]) == 0)
        self.assertTrue(len(topo.get_connections(100)[1]) == 0)
        self.assertEqual(topo.get_name(), "Unconnected")


class ring_test_case(_ut.TestCase):
    """Test case for the ring UDT

    """

    def runTest(self):
        from .core import ring, topology

        udt = ring()
        self.assertTrue(udt.num_vertices() == 0)
        self.assertTrue(udt.get_weight() == 1.)

        udt = ring(w=.5)
        self.assertTrue(udt.num_vertices() == 0)
        self.assertTrue(udt.get_weight() == .5)

        udt = ring(n=10, w=.1)
        self.assertTrue(udt.num_vertices() == 10)
        self.assertTrue(udt.get_weight() == .1)

        with self.assertRaises(ValueError) as cm:
            ring(n=10, w=2.)
        err = cm.exception
        self.assertTrue(
            "invalid weight for the edge of a topology: the value " in str(err))

        with self.assertRaises(TypeError) as cm:
            ring(n=-10, w=.5)

        udt = ring(n=5)
        self.assertTrue(udt.are_adjacent(4, 0))
        self.assertTrue(udt.are_adjacent(0, 4))
        self.assertTrue(not udt.are_adjacent(4, 1))
        self.assertTrue(not udt.are_adjacent(1, 4))
        udt.add_edge(1, 4)
        self.assertTrue(not udt.are_adjacent(4, 1))
        self.assertTrue(udt.are_adjacent(1, 4))
        udt.remove_edge(1, 4)
        self.assertTrue(not udt.are_adjacent(1, 4))
        udt.add_vertex()
        self.assertTrue(not udt.are_adjacent(5, 1))
        self.assertTrue(not udt.are_adjacent(1, 5))
        self.assertEqual(udt.num_vertices(), 6)
        udt.set_weight(0, 4, .5)
        self.assertTrue("0.5" in repr(topology(udt)))
        udt.set_all_weights(0.25)
        self.assertTrue("0.25" in repr(topology(udt)))
        self.assertEqual(topology(udt).get_name(), "Ring")

        with self.assertRaises(ValueError) as cm:
            udt.are_adjacent(100, 101)
        err = cm.exception
        self.assertTrue(
            "invalid vertex index in a BGL topology: the index is 100, but the number of vertices is only 6" in str(err))

        with self.assertRaises(TypeError) as cm:
            udt.are_adjacent(-1, -1)

        with self.assertRaises(ValueError) as cm:
            udt.add_edge(100, 101)
        err = cm.exception
        self.assertTrue(
            "invalid vertex index in a BGL topology: the index is 100, but the number of vertices is only 6" in str(err))

        with self.assertRaises(TypeError) as cm:
            udt.add_edge(-1, -1)

        with self.assertRaises(ValueError) as cm:
            udt.add_edge(1, 4, -1.)

        with self.assertRaises(ValueError) as cm:
            udt.remove_edge(100, 101)
        err = cm.exception
        self.assertTrue(
            "invalid vertex index in a BGL topology: the index is 100, but the number of vertices is only 6" in str(err))

        with self.assertRaises(TypeError) as cm:
            udt.remove_edge(-1, -1)

        with self.assertRaises(ValueError) as cm:
            udt.set_weight(100, 101, .5)
        err = cm.exception
        self.assertTrue(
            "invalid vertex index in a BGL topology: the index is 100, but the number of vertices is only 6" in str(err))

        with self.assertRaises(TypeError) as cm:
            udt.set_weight(-1, -1, .5)

        with self.assertRaises(ValueError) as cm:
            udt.set_weight(2, 3, -.5)

        with self.assertRaises(ValueError) as cm:
            udt.set_all_weights(-.5)

        topo = topology(udt=ring(3))
        self.assertTrue(len(topo.get_connections(0)[0]) == 2)
        self.assertTrue(len(topo.get_connections(0)[1]) == 2)
        topo.push_back()
        topo.push_back()
        topo.push_back()
        self.assertTrue(len(topo.get_connections(3)[0]) == 2)
        self.assertTrue(len(topo.get_connections(3)[1]) == 2)
        self.assertEqual(topo.get_name(), "Ring")


class fully_connected_test_case(_ut.TestCase):
    """Test case for the fully_connected UDT

    """

    def runTest(self):
        from .core import fully_connected, topology

        udt = fully_connected()
        self.assertEqual(udt.num_vertices(), 0)
        self.assertEqual(udt.get_weight(), 1.)

        udt = fully_connected(w=.5)
        self.assertEqual(udt.num_vertices(), 0)
        self.assertEqual(udt.get_weight(), .5)

        udt = fully_connected(w=.5, n=10)
        self.assertEqual(udt.num_vertices(), 10)
        self.assertEqual(udt.get_weight(), .5)

        topo = topology(udt=fully_connected(10))
        self.assertTrue(len(topo.get_connections(1)[0]) == 9)
        self.assertTrue(len(topo.get_connections(1)[1]) == 9)
        topo.push_back()
        topo.push_back()
        topo.push_back()
        self.assertTrue(len(topo.get_connections(5)[0]) == 12)
        self.assertTrue(len(topo.get_connections(5)[1]) == 12)
        self.assertEqual(topo.get_name(), "Fully connected")


class thread_island_torture_test_case(_ut.TestCase):
    """Stress test for thread_island

    """

    def runTest(self):
        from .core import thread_island, population, algorithm, island
        from . import evolve_status

        pop = population(prob=_prob(), size=5)
        algo = algorithm()
        udi = thread_island()

        for i in range(50):
            l = []
            for _ in range(100):
                l.append(island(udi=udi, pop=pop, algo=algo))
                l[-1].evolve()
                l[-1].get_algorithm()
                l[-1].get_population()

            for i in l:
                i.get_algorithm()
                i.get_population()
                i.wait()
                self.assertTrue(i.status == evolve_status.idle_error)

        class my_algo(object):
            def evolve(self, pop):
                return pop

        algo = algorithm(my_algo())

        for i in range(50):
            l = []
            for _ in range(100):
                l.append(island(udi=udi, pop=pop, algo=algo))
                l[-1].evolve()
                l[-1].get_algorithm()
                l[-1].get_population()

            for i in l:
                i.get_algorithm()
                i.get_population()
                i.wait()
                self.assertTrue(i.status == evolve_status.idle_error)


class archipelago_test_case(_ut.TestCase):
    """Test case for the archipelago class.

    """

    def __init__(self, level):
        _ut.TestCase.__init__(self)
        self._level = level

    def runTest(self):
        self.run_init_tests()
        self.run_evolve_tests()
        self.run_access_tests()
        self.run_push_back_tests()
        self.run_io_tests()
        self.run_pickle_tests()
        self.run_champions_tests()
        self.run_status_tests()
        self.run_mig_log_db_tests()
        self.run_get_set_topo_tests()
        self.run_mt_mh_tests()
        if self._level > 0:
            self.run_torture_test_0()
            # NOTE: skip this test for the time being.
            # It was copy-pasted from an interactive
            # python session ages ago, and it's not clear
            # whether we can control its internal randomness
            # or not, with the result that it seems to
            # fail rarely (in the sense that it won't
            # raise an expected exception, not that it crashes
            # or anything like that). We'll have to
            # investigate further if we ever want to
            # turn it back on.
            # self.run_torture_test_1()
            self.run_migration_torture_test()
            # NOTE: temporarily disable this test due to the mingw failures.
            # self.run_mo_migration_bug_test()

    def run_init_tests(self):
        from . import (archipelago, de, rosenbrock, population, null_problem, thread_island,
                       mp_island, topology, unconnected, ring, r_policy, s_policy, fair_replace, select_best, bfe, default_bfe, thread_bfe, problem)
        a = archipelago()
        self.assertEqual(len(a), 0)
        self.assertRaises(IndexError, lambda: a[0])
        a = archipelago(5, algo=de(), prob=rosenbrock(), pop_size=10)
        self.assertEqual(len(a), 5)
        self.assertTrue(a[0].get_algorithm().is_(de))
        self.assertTrue(a[0].get_population().problem.is_(rosenbrock))
        self.assertEqual(len(a[0].get_population()), 10)
        a = archipelago(5, pop=population(), algo=de())
        self.assertEqual(len(a), 5)
        self.assertTrue(a[0].get_algorithm().is_(de))
        self.assertTrue(a[0].get_population().problem.is_(null_problem))
        self.assertEqual(len(a[0].get_population()), 0)
        a = archipelago(5, algo=de(), prob=rosenbrock(),
                        pop_size=10, udi=thread_island(), seed=5)
        self.assertEqual(len(a), 5)
        self.assertTrue(a[0].get_algorithm().is_(de))
        self.assertTrue(a[0].get_population().problem.is_(rosenbrock))
        self.assertEqual(len(a[0].get_population()), 10)
        # Check unique seeds.
        seeds = list([_.get_population().get_seed() for _ in a])
        self.assertEqual(len(seeds), len(set(seeds)))
        # Check seeding is deterministic.
        a2 = archipelago(5, algo=de(), prob=rosenbrock(),
                         pop_size=10, seed=5)
        seeds2 = list([_.get_population().get_seed() for _ in a2])
        self.assertEqual(seeds2, seeds)
        self.assertTrue(all([(t[0].get_population().get_x() == t[
                        1].get_population().get_x()).all() for t in zip(a, a2)]))
        self.assertTrue(all([(t[0].get_population().get_f() == t[
                        1].get_population().get_f()).all() for t in zip(a, a2)]))
        self.assertTrue(all([(t[0].get_population().get_ID() == t[
                        1].get_population().get_ID()).all() for t in zip(a, a2)]))
        # Check the 'size' keyword is not accepted.
        self.assertRaises(KeyError, lambda: archipelago(5, algo=de(), prob=rosenbrock(),
                                                        size=10, udi=thread_island(), seed=5))
        # Check without seed argument, seeding is non-deterministic.
        a = archipelago(5, algo=de(), prob=rosenbrock(),
                        pop_size=10, udi=thread_island())
        a2 = archipelago(5, algo=de(), prob=rosenbrock(),
                         pop_size=10, udi=thread_island())
        seeds = sorted(list([_.get_population().get_seed() for _ in a]))
        seeds2 = sorted(list([_.get_population().get_seed() for _ in a2]))
        self.assertTrue(all([t[0] != t[1] for t in zip(seeds, seeds2)]))
        self.assertTrue(all([(t[0].get_population().get_x() != t[
                        1].get_population().get_x()).all() for t in zip(a, a2)]))
        self.assertTrue(all([(t[0].get_population().get_f() != t[
                        1].get_population().get_f()).all() for t in zip(a, a2)]))
        self.assertTrue(all([(t[0].get_population().get_ID() != t[
                        1].get_population().get_ID()).all() for t in zip(a, a2)]))
        a = archipelago(5, algo=de(), prob=rosenbrock(),
                        pop_size=10, udi=mp_island(), seed=5)
        self.assertEqual(len(a), 5)
        self.assertTrue(a[0].get_algorithm().is_(de))
        self.assertTrue(a[0].get_population().problem.is_(rosenbrock))
        self.assertEqual(len(a[0].get_population()), 10)
        seeds = list([_.get_population().get_seed() for _ in a])
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertRaises(KeyError, lambda: archipelago(
            5, pop=population(), algo=de(), seed=1))

        # Constructors from topology and custom r_pol, s_pol.
        a = archipelago(5, t=topology(), algo=de(), prob=rosenbrock(),
                        pop_size=10, udi=mp_island(), seed=5)
        self.assertTrue(a.get_topology().is_(unconnected))
        self.assertTrue(
            all([isl.get_r_policy().is_(fair_replace) for isl in a]))
        self.assertTrue(
            all([isl.get_s_policy().is_(select_best) for isl in a]))

        a = archipelago(5, t=topology(ring()), algo=de(), prob=rosenbrock(),
                        pop_size=10, udi=mp_island(), seed=5)
        self.assertTrue(a.get_topology().is_(ring))

        a = archipelago(5, t=ring(), algo=de(), prob=rosenbrock(),
                        pop_size=10, udi=mp_island(), seed=5)
        self.assertTrue(a.get_topology().is_(ring))

        a = archipelago(5, t=ring(), algo=de(), prob=rosenbrock(),
                        pop_size=10, udi=mp_island(), seed=5, r_pol=r_policy())
        self.assertTrue(
            all([isl.get_r_policy().is_(fair_replace) for isl in a]))
        self.assertTrue(
            all([isl.get_s_policy().is_(select_best) for isl in a]))

        a = archipelago(5, t=ring(), algo=de(), prob=rosenbrock(),
                        pop_size=10, udi=mp_island(), seed=5, r_pol=r_policy(), s_pol=s_policy())
        self.assertTrue(
            all([isl.get_r_policy().is_(fair_replace) for isl in a]))
        self.assertTrue(
            all([isl.get_s_policy().is_(select_best) for isl in a]))

        a = archipelago(5, t=ring(), algo=de(), prob=rosenbrock(),
                        pop_size=10, udi=mp_island(), seed=5, r_pol=_r_pol(), s_pol=_s_pol())
        self.assertTrue(all([isl.get_r_policy().is_(_r_pol) for isl in a]))
        self.assertTrue(all([isl.get_s_policy().is_(_s_pol) for isl in a]))

        # Ctors with bfe.
        p = problem(rosenbrock())
        a = archipelago(5, t=topology(ring()), algo=de(), prob=p,
                        pop_size=10, udi=mp_island(), seed=5, b=bfe(default_bfe()))
        for isl in a:
            for x, f in zip(isl.get_population().get_x(), isl.get_population().get_f()):
                self.assertEqual(p.fitness(x), f)

        a = archipelago(5, t=topology(ring()), algo=de(), prob=p,
                        pop_size=10, udi=mp_island(), seed=5, b=thread_bfe())
        for isl in a:
            for x, f in zip(isl.get_population().get_x(), isl.get_population().get_f()):
                self.assertEqual(p.fitness(x), f)

        # Pythonic problem.
        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

        p = problem(p())
        a = archipelago(5, t=topology(ring()), algo=de(), prob=p,
                        pop_size=10, udi=mp_island(), seed=5, b=bfe(default_bfe()))
        for isl in a:
            for x, f in zip(isl.get_population().get_x(), isl.get_population().get_f()):
                self.assertEqual(p.fitness(x), f)

        # Pythonic problem with batch_fitness method.
        class p(object):

            def get_bounds(self):
                return ([0], [1])

            def fitness(self, a):
                return [42]

            def batch_fitness(self, dvs):
                return [43] * len(dvs)

        p = problem(p())
        a = archipelago(5, t=topology(ring()), algo=de(), prob=p,
                        pop_size=10, udi=mp_island(), seed=5, b=bfe(default_bfe()))
        for isl in a:
            for f in isl.get_population().get_f():
                self.assertEqual(f, 43)

    def run_mig_log_db_tests(self):
        from . import archipelago, de, rosenbrock, ring
        a = archipelago(5, t=ring(), algo=de(), prob=rosenbrock(), pop_size=10)
        a.evolve(10)
        a.wait_check()

        db = a.get_migrants_db()
        self.assertEqual(len(db), 5)
        for g in db:
            self.assertEqual(len(g), 3)
            self.assertEqual(len(g[0]), 1)
            self.assertEqual(g[1].shape, (1, 2))
            self.assertEqual(g[2].shape, (1, 1))

        log = a.get_migration_log()
        for e in log:
            self.assertEqual(len(e), 6)
            self.assertEqual(e[2].shape, (2,))
            self.assertEqual(e[3].shape, (1,))
            self.assertTrue(e[4] < 5)
            self.assertTrue(e[5] < 5)

    def run_get_set_topo_tests(self):
        from . import archipelago, de, rosenbrock, ring, topology, fully_connected
        a = archipelago(5, t=ring(), algo=de(), prob=rosenbrock(), pop_size=10)
        self.assertTrue(a.get_topology().is_(ring))

        # Evolve, then set topology (will trigger wait()).
        a.evolve(100)
        a.set_topology(topology(fully_connected(5)))
        self.assertTrue(a.get_topology().is_(fully_connected))
        a.set_topology(ring(5))
        self.assertTrue(a.get_topology().is_(ring))

    def run_mt_mh_tests(self):
        from . import archipelago, de, rosenbrock, migration_type, ring, migrant_handling

        a = archipelago(5, t=ring(), algo=de(), prob=rosenbrock(), pop_size=10)
        self.assertEqual(a.get_migration_type(), migration_type.p2p)
        self.assertEqual(a.get_migrant_handling(), migrant_handling.preserve)
        a.evolve(100)
        a.set_migration_type(migration_type.broadcast)
        self.assertEqual(a.get_migration_type(), migration_type.broadcast)
        a.set_migrant_handling(migrant_handling.evict)
        self.assertEqual(a.get_migrant_handling(), migrant_handling.evict)
        a.wait_check()

    def run_mo_migration_bug_test(self):
        from . import dtlz, nsga2, ring, archipelago

        udp = dtlz(2, dim=50)
        uda = nsga2(gen=100)
        topo = ring()

        archi = archipelago(n=10, t=topo, prob=udp, algo=uda, pop_size=100)
        archi.evolve(4)
        try:
            archi.wait_check()
        except:
            self.fail("The MO migration bug test failed")

    def run_evolve_tests(self):
        from . import archipelago, de, rosenbrock, mp_island, evolve_status
        from copy import deepcopy
        a = archipelago()
        self.assertTrue(a.status == evolve_status.idle)
        a = archipelago(5, algo=de(), prob=rosenbrock(), pop_size=10)
        a.evolve(10)
        a.evolve(10)
        str(a)
        a.wait()
        a.evolve(10)
        a.evolve(10)
        str(a)
        a.wait_check()
        # Copy while evolving.
        a.evolve(10)
        a.evolve(10)
        a2 = deepcopy(a)
        a.wait_check()
        a = archipelago(5, udi=mp_island(), algo=de(),
                        prob=rosenbrock(), pop_size=10)
        a.evolve(10)
        a.evolve(10)
        str(a)
        a.wait()
        a.evolve(10)
        a.evolve(10)
        str(a)
        a.wait_check()
        # Copy while evolving.
        a.evolve(10)
        a.evolve(10)
        a2 = deepcopy(a)
        a.wait_check()
        # Throws on wait_check().
        a = archipelago(5, algo=de(), prob=rosenbrock(), pop_size=3)
        a.evolve()
        self.assertRaises(ValueError, lambda: a.wait_check())

    def run_access_tests(self):
        from . import archipelago, de, rosenbrock
        import sys
        a = archipelago(5, algo=de(), prob=rosenbrock(), pop_size=10)
        i0, i1, i2 = a[0], a[1], a[2]
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        for isl in (i0, i1, i2):
            self.assertTrue(isl.get_algorithm().is_(de))
            self.assertTrue(isl.get_population().problem.is_(rosenbrock))
            self.assertEqual(len(isl.get_population()), 10)
        # Check refcount when returning internal ref.
        a = archipelago(5, algo=de(), prob=rosenbrock(), pop_size=10)
        old_rc = sys.getrefcount(a)
        i0, i1, i2, i3 = a[0], a[1], a[2], a[3]
        self.assertEqual(sys.getrefcount(a) - 4, old_rc)
        del a
        self.assertTrue(str(i0) != "")
        self.assertTrue(str(i1) != "")
        self.assertTrue(str(i2) != "")
        self.assertTrue(str(i3) != "")
        del i0, i1, i2, i3

    def run_push_back_tests(self):
        from . import (archipelago, de, rosenbrock, r_policy,
                       s_policy, fair_replace, select_best, island)
        a = archipelago(5, algo=de(), prob=rosenbrock(), pop_size=10)
        # Push back while evolving.
        a.evolve(10)
        a.evolve(10)
        a.evolve(10)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.push_back(algo=de(), prob=rosenbrock(), size=11)
        a.wait_check()
        self.assertEqual(len(a), 18)
        for i in range(5):
            self.assertTrue(a[i].get_algorithm().is_(de))
            self.assertTrue(a[i].get_population().problem.is_(rosenbrock))
            self.assertEqual(len(a[i].get_population()), 10)
        for i in range(5, 18):
            self.assertTrue(a[i].get_algorithm().is_(de))
            self.assertTrue(a[i].get_population().problem.is_(rosenbrock))
            self.assertEqual(len(a[i].get_population()), 11)

        # Push back with custom policies.
        a = archipelago(5, algo=de(), prob=rosenbrock(), pop_size=10)
        self.assertTrue(
            all([isl.get_r_policy().is_(fair_replace) for isl in a]))
        self.assertTrue(
            all([isl.get_s_policy().is_(select_best) for isl in a]))

        a.push_back(algo=de(), prob=rosenbrock(), size=11, r_pol=r_policy())
        a.push_back(algo=de(), prob=rosenbrock(), size=11,
                    r_pol=r_policy(), s_pol=s_policy())
        self.assertTrue(
            all([isl.get_r_policy().is_(fair_replace) for isl in a]))
        self.assertTrue(
            all([isl.get_s_policy().is_(select_best) for isl in a]))

        a.push_back(algo=de(), prob=rosenbrock(), size=11,
                    r_pol=_r_pol(), s_pol=_s_pol())
        a.push_back(algo=de(), prob=rosenbrock(), size=11,
                    r_pol=_r_pol(), s_pol=_s_pol())
        self.assertTrue(all([a[i].get_r_policy().is_(_r_pol)
                             for i in range(7, 9)]))
        self.assertTrue(all([a[i].get_s_policy().is_(_s_pol)
                             for i in range(7, 9)]))

        # push_back() with positional argument.
        a = archipelago()
        a.push_back(island(algo=de(), prob=rosenbrock(), size=10))
        a.push_back(island(algo=de(), prob=rosenbrock(), size=10))
        a.push_back(island(algo=de(), prob=rosenbrock(), size=10))

        self.assertEqual(len(a), 3)
        for i in range(3):
            self.assertTrue(a[i].get_algorithm().is_(de))
            self.assertTrue(a[i].get_population().problem.is_(rosenbrock))
            self.assertEqual(len(a[i].get_population()), 10)

        # Error handling.
        with self.assertRaises(ValueError) as cm:
            a.push_back(
                island(algo=de(), prob=rosenbrock(), size=10), a=5, b=6)
        err = cm.exception
        self.assertTrue(
            "if a positional argument is passed to this method, then no keyword arguments must be passed, but 2 keyword arguments were passed instead" in str(err))

        with self.assertRaises(ValueError) as cm:
            a.push_back(island(algo=de(), prob=rosenbrock(), size=10), 1)
        err = cm.exception
        self.assertTrue(
            "2 positional arguments were provided, but this method accepts only a single positional argument" in str(err))

        with self.assertRaises(TypeError) as cm:
            a.push_back(42)
        err = cm.exception
        self.assertTrue(
            "the positional argument passed to this method must be an island, but the type of the argument is '{}' instead".format(type(42)) in str(err))

    def run_io_tests(self):
        from . import archipelago, de, rosenbrock
        a = archipelago(5, algo=de(), prob=rosenbrock(), pop_size=10)
        self.assertFalse(repr(a) == "")

    def run_pickle_tests(self):
        from . import archipelago, de, rosenbrock, mp_island, ring, migration_type, migrant_handling
        from pickle import dumps, loads
        a = archipelago(5, algo=de(), prob=rosenbrock(), pop_size=10)
        self.assertEqual(repr(a), repr(loads(dumps(a))))
        a = archipelago(5, algo=de(), prob=_prob(),
                        pop_size=10, udi=mp_island())
        self.assertEqual(repr(a), repr(loads(dumps(a))))

        # Test also with custom topology, mh and mt.
        a = archipelago(n=5, t=ring(), algo=de(),
                        prob=rosenbrock(), pop_size=10)
        a.set_migration_type(migration_type.broadcast)
        a.set_migrant_handling(migrant_handling.evict)
        self.assertEqual(repr(a), repr(loads(dumps(a))))
        self.assertTrue(loads(dumps(a)).get_topology().is_(ring))
        self.assertEqual(loads(dumps(a)).get_migration_type(),
                         migration_type.broadcast)
        self.assertEqual(loads(dumps(a)).get_migrant_handling(),
                         migrant_handling.evict)

    def run_champions_tests(self):
        from . import archipelago, de, rosenbrock, zdt
        from numpy import ndarray
        a = archipelago(5, algo=de(), prob=rosenbrock(), pop_size=10)
        cf = a.get_champions_f()
        self.assertEqual(type(cf), list)
        self.assertEqual(len(cf), 5)
        self.assertEqual(type(cf[0]), ndarray)
        cx = a.get_champions_x()
        self.assertEqual(type(cx), list)
        self.assertEqual(len(cx), 5)
        self.assertEqual(type(cx[0]), ndarray)
        a.push_back(algo=de(), prob=rosenbrock(10), size=20)
        cx = a.get_champions_x()
        self.assertEqual(len(cx[4]), 2)
        self.assertEqual(len(cx[5]), 10)
        a.push_back(algo=de(), prob=zdt(), size=20)
        self.assertRaises(ValueError, lambda: a.get_champions_x())
        self.assertRaises(ValueError, lambda: a.get_champions_f())

    def run_status_tests(self):
        from . import archipelago, de, rosenbrock, evolve_status
        a = archipelago(5, algo=de(), prob=rosenbrock(), pop_size=3)
        self.assertTrue(a.status == evolve_status.idle)
        a.evolve()
        a.wait()
        self.assertTrue(a.status == evolve_status.idle_error)
        self.assertRaises(ValueError, lambda: a.wait_check())
        self.assertTrue(a.status == evolve_status.idle)

    def run_torture_test_0(self):
        from . import archipelago, de, ackley

        # pure C++
        archi = archipelago(n=1000, algo=de(
            10), prob=ackley(5), pop_size=10, seed=32)
        archi.evolve()
        archi.wait_check()

        # python prob
        archi2 = archipelago(n=1000, algo=de(
            10), prob=_quick_prob(), pop_size=10, seed=32)
        archi2.evolve()
        archi2.wait_check()

        # python prob with exceptions (will throw in osx as too many threads
        # will be opened)
        def _():
            archi3 = archipelago(n=1000, algo=simulated_annealing(
                10, 1, 50), prob=_raise_exception(throw_at=1001), pop_size=1, seed=32)
            archi3.evolve()
            archi3.wait_check()

        self.assertRaises(BaseException, _)

    def run_torture_test_1(self):
        # A torture test inspired by the heisenbug detected by Dario on OSX.

        from . import archipelago, sade, ackley

        archi = archipelago(n=5, algo=sade(
            50, ftol=0, xtol=0), prob=_raise_exception_2(), pop_size=20)
        archi.evolve()
        self.assertRaises(BaseException, lambda: archi.wait_check())

        archi = archipelago(n=5, algo=sade(
            50), prob=_raise_exception_2(), pop_size=20)
        archi.evolve()
        self.assertRaises(BaseException, lambda: archi.wait_check())
        archi.wait_check()

        archi = archipelago(n=1100, algo=sade(
            500), prob=ackley(50), pop_size=50)
        archi = archipelago(n=5, algo=sade(
            50), prob=_raise_exception_2(), pop_size=20)
        archi.evolve()
        archi = archipelago(n=1100, algo=sade(
            500), prob=ackley(50), pop_size=50)
        archi.evolve()

    def run_migration_torture_test(self):
        from . import archipelago, de, rosenbrock, fair_replace, select_best, r_policy, s_policy
        import threading

        # Use custom UDT, UDRP and UDSP for the torture test.

        # NOTE: re-implementation of a fully_connected
        # topology.
        class udt(object):
            def __init__(self, n=0):
                self._n = n
                self._lock = threading.Lock()

            def get_n(self):
                with self._lock:
                    n = self._n
                return n

            def __copy__(self):
                return udt(self.get_n())

            def __deepcopy__(self, d):
                return self.__copy__()

            def push_back(self):
                with self._lock:
                    self._n = self._n + 1

            def get_connections(self, i):
                with self._lock:
                    n = self._n
                return (list(range(0, i)) + list(range(i+1, n)), [1.]*(n-1))

        # NOTE: these two will just re-use fair_replace/select_best internally.
        class udrp(object):

            def replace(self, inds, nx, nix, nobj, nec, nic, tol, mig):
                return r_policy(fair_replace()).replace(inds, nx, nix, nobj, nec, nic, tol, mig)

        class udsp(object):

            def select(self, inds, nx, nix, nobj, nec, nic, tol):
                return s_policy(select_best()).select(inds, nx, nix, nobj, nec, nic, tol)

        archi = archipelago(n=100, t=udt(), algo=de(
            1), prob=rosenbrock(), pop_size=10, seed=32, r_pol=udrp(), s_pol=udsp())

        archi.evolve(100)
        archi.wait_check()


def run_test_suite(level=0):
    """Run the full test suite.

    This function will raise an exception if at least one test fails.

    Args:
        level(``int``): the test level (higher values run longer tests)

    """
    #from . import _problem_test, _algorithm_test, _island_test, _topology_test, _r_policy_test, _s_policy_test, _bfe_test, set_global_rng_seed
    from . import _problem_test, set_global_rng_seed, _algorithm_test, _bfe_test, _island_test, _topology_test, _r_policy_test, _s_policy_test

    # Make test runs deterministic.
    # NOTE: we'll need to place the async/migration tests at the end, so that at
    # least the first N tests are really deterministic.
    set_global_rng_seed(42)

    retval = 0
    suite = _ut.TestLoader().loadTestsFromTestCase(core_test_case)
    suite.addTest(_bfe_test.bfe_test_case())
    suite.addTest(_bfe_test.thread_bfe_test_case())
    suite.addTest(_bfe_test.member_bfe_test_case())
    suite.addTest(_bfe_test.mp_bfe_test_case())
    suite.addTest(_bfe_test.ipyparallel_bfe_test_case())
    suite.addTest(_bfe_test.default_bfe_test_case())
    suite.addTest(archipelago_test_case(level))
    suite.addTest(_island_test.island_test_case())
    suite.addTest(_s_policy_test.s_policy_test_case())
    suite.addTest(_r_policy_test.r_policy_test_case())
    suite.addTest(_topology_test.topology_test_case())
    suite.addTest(fair_replace_test_case())
    suite.addTest(select_best_test_case())
    suite.addTest(unconnected_test_case())
    suite.addTest(ring_test_case())
    suite.addTest(fully_connected_test_case())
    suite.addTest(thread_island_torture_test_case())
    suite.addTest(_problem_test.problem_test_case())
    suite.addTest(_algorithm_test.algorithm_test_case())
    suite.addTest(_island_test.mp_island_test_case(level))
    suite.addTest(_island_test.ipyparallel_island_test_case(level))

    suite.addTest(mbh_test_case())

    test_result = _ut.TextTestRunner(verbosity=2).run(suite)

    # Re-seed to random just in case anyone ever uses this function
    # in an interactive session or something.
    import random
    set_global_rng_seed(random.randint(0, 2**30))

    if len(test_result.failures) > 0 or len(test_result.errors) > 0:
        retval = 1
    if retval != 0:
        raise RuntimeError('One or more tests failed.')
