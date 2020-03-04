# Copyright 2020 PaGMO development team
#
# This file is part of the pygmo library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class _r_pol(object):

    def replace(self, inds, nx, nix, nobj, nec, nic, tol, mig):
        return inds


class _s_pol(object):

    def select(self, inds, nx, nix, nobj, nec, nic, tol):
        return inds


class _udi_01(object):

    def run_evolve(self, algo, pop):
        newpop = algo.evolve(pop)
        return algo, newpop

    def get_name(self):
        return "udi_01"

    def get_extra_info(self):
        return "extra bits"


class _udi_02(object):
    # UDI without the necessary method(s).
    pass


class _udi_03(object):
    # UDI with run_evolve() returning wrong stuff, #1.
    def run_evolve(self, algo, pop):
        return algo, pop, 25


class _prob(object):

    def __init__(self, data):
        self.data = data

    def fitness(self, x):
        return [0.]

    def get_bounds(self):
        return ([0.], [1.])


class _stateful_algo(object):

    def __init__(self):
        self._n = 0

    def evolve(self, pop):
        self._n = self._n + 1
        return pop


class island_test_case(_ut.TestCase):
    """Test case for the :class:`~pygmo.island` class.

    """

    def runTest(self):
        self.run_basic_tests()
        self.run_extract_tests()
        self.run_concurrent_access_tests()
        self.run_evolve_tests()
        self.run_get_busy_wait_tests()
        self.run_io_tests()
        self.run_status_tests()
        self.run_stateful_algo_tests()
        self.run_mo_sto_repr_bug()

    def run_basic_tests(self):
        from .core import island, thread_island, null_algorithm, null_problem, de, rosenbrock, r_policy, s_policy, fair_replace, select_best, population, bfe, thread_bfe, default_bfe, problem
        isl = island()
        self.assertTrue("Fair replace" in repr(isl))
        self.assertTrue("Select best" in repr(isl))
        self.assertTrue(isl.get_algorithm().is_(null_algorithm))
        self.assertTrue(isl.get_population().problem.is_(null_problem))
        self.assertTrue(isl.extract(thread_island) is not None)
        self.assertTrue(isl.extract(_udi_01) is None)
        self.assertTrue(isl.extract(int) is None)
        self.assertEqual(len(isl.get_population()), 0)
        isl = island(algo=de(), prob=rosenbrock(), size=10)
        self.assertTrue("Fair replace" in repr(isl))
        self.assertTrue("Select best" in repr(isl))
        self.assertTrue(isl.get_algorithm().is_(de))
        self.assertTrue(isl.get_population().problem.is_(rosenbrock))
        self.assertEqual(len(isl.get_population()), 10)
        isl = island(prob=rosenbrock(), udi=thread_island(),
                     size=11, algo=de(), seed=15)
        self.assertTrue("Fair replace" in repr(isl))
        self.assertTrue("Select best" in repr(isl))
        self.assertTrue(isl.get_algorithm().is_(de))
        self.assertTrue(isl.get_population().problem.is_(rosenbrock))
        self.assertEqual(len(isl.get_population()), 11)
        self.assertEqual(isl.get_population().get_seed(), 15)
        isl = island(udi=thread_island(),
                     algo=de(), pop=population())
        self.assertTrue("Fair replace" in repr(isl))
        self.assertTrue("Select best" in repr(isl))
        self.assertTrue(isl.get_algorithm().is_(de))
        self.assertTrue(isl.get_population().problem.is_(null_problem))
        self.assertEqual(len(isl.get_population()), 0)
        isl = island(prob=rosenbrock(), udi=_udi_01(),
                     size=11, algo=de(), seed=15)
        self.assertTrue("Fair replace" in repr(isl))
        self.assertTrue("Select best" in repr(isl))
        self.assertEqual(isl.get_name(), "udi_01")
        self.assertEqual(isl.get_extra_info(), "extra bits")
        self.assertTrue(isl.get_algorithm().is_(de))
        self.assertTrue(isl.get_population().problem.is_(rosenbrock))
        self.assertEqual(len(isl.get_population()), 11)
        self.assertRaises(NotImplementedError, lambda: island(prob=rosenbrock(), udi=_udi_02(),
                                                              size=11, algo=de(), seed=15))

        # Verify that the constructor copies the UDI instance.
        udi_01_inst = _udi_01()
        isl = island(udi=udi_01_inst, algo=de(), prob=rosenbrock(), size=10)
        self.assertTrue(id(isl.extract(thread_island)) != id(udi_01_inst))

        # Local island using local variable.
        glob = []

        class loc_01(object):
            def __init__(self, g):
                self.g = g

            def run_evolve(self, algo, pop):
                self.g.append(1)
                return algo, pop

        loc_inst = loc_01(glob)
        isl = island(udi=loc_inst, algo=de(), prob=rosenbrock(), size=20)
        isl.evolve(10)
        isl.wait_check()
        # Assert that loc_inst was deep-copied into isl:
        # the instance in isl will have its own copy of glob
        # and it will not be a reference the outside object.
        self.assertEqual(len(glob), 0)
        self.assertEqual(len(isl.extract(loc_01).g), 10)

        isl = island(prob=rosenbrock(), udi=_udi_03(),
                     size=11, algo=de(), seed=15)
        isl.evolve()
        with self.assertRaises(RuntimeError) as cm:
            isl.wait_check()
        err = cm.exception
        self.assertTrue(
            "the tuple returned by the 'run_evolve()' method of a user-defined island must have 2 elements, but instead it has 3 element(s)" in str(err))

        # Test that construction from another pygmo.island fails.
        with self.assertRaises(NotImplementedError) as cm:
            island(prob=rosenbrock(), udi=isl, size=11, algo=de(), seed=15)

        # Constructors with r/s_pol arguments.
        isl = island(prob=rosenbrock(), udi=thread_island(),
                     size=11, algo=de(), seed=15, r_pol=r_policy())
        self.assertTrue("Fair replace" in repr(isl))
        self.assertTrue("Select best" in repr(isl))
        self.assertTrue(isl.get_algorithm().is_(de))
        self.assertTrue(isl.get_population().problem.is_(rosenbrock))
        self.assertEqual(len(isl.get_population()), 11)
        self.assertEqual(isl.get_population().get_seed(), 15)

        isl = island(prob=rosenbrock(), udi=thread_island(),
                     size=11, algo=de(), seed=15, r_pol=_r_pol())
        self.assertFalse("Fair replace" in repr(isl))
        self.assertTrue("Select best" in repr(isl))
        self.assertTrue(isl.get_algorithm().is_(de))
        self.assertTrue(isl.get_population().problem.is_(rosenbrock))
        self.assertEqual(len(isl.get_population()), 11)
        self.assertEqual(isl.get_population().get_seed(), 15)

        isl = island(prob=rosenbrock(), udi=thread_island(),
                     size=11, algo=de(), seed=15, s_pol=s_policy())
        self.assertTrue("Fair replace" in repr(isl))
        self.assertTrue("Select best" in repr(isl))
        self.assertTrue(isl.get_algorithm().is_(de))
        self.assertTrue(isl.get_population().problem.is_(rosenbrock))
        self.assertEqual(len(isl.get_population()), 11)
        self.assertEqual(isl.get_population().get_seed(), 15)

        isl = island(prob=rosenbrock(), udi=thread_island(),
                     size=11, algo=de(), seed=15, s_pol=_s_pol())
        self.assertTrue("Fair replace" in repr(isl))
        self.assertFalse("Select best" in repr(isl))
        self.assertTrue(isl.get_algorithm().is_(de))
        self.assertTrue(isl.get_population().problem.is_(rosenbrock))
        self.assertEqual(len(isl.get_population()), 11)
        self.assertEqual(isl.get_population().get_seed(), 15)

        # Test the r/s_policy getters.
        isl = island(prob=rosenbrock(), udi=thread_island(),
                     size=11, algo=de(), seed=15, r_pol=_r_pol(), s_pol=_s_pol())

        self.assertTrue(isl.get_r_policy().is_(_r_pol))
        self.assertTrue(isl.get_s_policy().is_(_s_pol))

        # Ctors from bfe.
        p = problem(rosenbrock())
        isl = island(prob=p, size=20, b=bfe(default_bfe()), algo=de())
        for x, f in zip(isl.get_population().get_x(), isl.get_population().get_f()):
            self.assertEqual(p.fitness(x), f)

        # Pass in explicit UDBFE.
        isl = island(prob=p, size=20, b=thread_bfe(), algo=de())
        for x, f in zip(isl.get_population().get_x(), isl.get_population().get_f()):
            self.assertEqual(p.fitness(x), f)

        # Pythonic problem.
        class p(object):

            def get_bounds(self):
                return ([0, 0], [1, 1])

            def fitness(self, a):
                return [42]

        p = problem(p())
        isl = island(prob=p, size=20, b=bfe(default_bfe()), algo=de())
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
        isl = island(prob=p, size=20, b=bfe(default_bfe()), algo=de())
        for f in isl.get_population().get_f():
            self.assertEqual(f, 43)

    def run_concurrent_access_tests(self):
        import threading as thr
        from .core import island, de, rosenbrock
        isl = island(algo=de(), prob=rosenbrock(), size=10)

        def thread_func():
            for i in range(100):
                pop = isl.get_population()
                isl.set_population(pop)
                algo = isl.get_algorithm()
                isl.set_algorithm(algo)

        thr_list = [thr.Thread(target=thread_func) for i in range(4)]
        [_.start() for _ in thr_list]
        [_.join() for _ in thr_list]

    def run_evolve_tests(self):
        from .core import island, de, rosenbrock
        from copy import deepcopy
        isl = island(algo=de(), prob=rosenbrock(), size=25)
        isl.evolve(0)
        isl.wait_check()
        isl.evolve()
        isl.evolve()
        isl.wait_check()
        isl.evolve(20)
        isl.wait_check()
        for i in range(10):
            isl.evolve(20)
        isl2 = deepcopy(isl)
        isl2.wait_check()
        isl.wait_check()

    def run_status_tests(self):
        from . import island, de, rosenbrock, evolve_status
        isl = island(algo=de(), prob=rosenbrock(), size=3)
        isl.evolve(20)
        isl.wait()
        self.assertTrue(isl.status == evolve_status.idle_error)
        self.assertRaises(BaseException, lambda: isl.wait_check())
        self.assertTrue(isl.status == evolve_status.idle)

    def run_get_busy_wait_tests(self):
        from . import island, de, rosenbrock, evolve_status
        isl = island(algo=de(), prob=rosenbrock(), size=25)
        self.assertTrue(isl.status == evolve_status.idle)
        isl = island(algo=de(), prob=rosenbrock(), size=3)
        isl.evolve(20)
        self.assertRaises(BaseException, lambda: isl.wait_check())
        isl.evolve(20)
        isl.wait()

    def run_io_tests(self):
        from .core import island, de, rosenbrock
        isl = island(algo=de(), prob=rosenbrock(), size=25)
        self.assertTrue(repr(isl) != "")
        self.assertTrue(isl.get_name() == "Thread island")
        self.assertTrue(isl.get_extra_info() == "")
        isl = island(algo=de(), prob=rosenbrock(), size=25, udi=_udi_01())
        self.assertTrue(repr(isl) != "")
        self.assertTrue(isl.get_name() == "udi_01")
        self.assertTrue(isl.get_extra_info() == "extra bits")

    def run_serialization_tests(self):
        from .core import island, de, rosenbrock
        from pickle import dumps, loads
        isl = island(algo=de(), prob=rosenbrock(), size=25)
        tmp = repr(isl)
        isl = loads(dumps(isl))
        self.assertEqual(tmp, repr(isl))

        # Check with custom policies as well.
        isl = island(algo=de(), prob=rosenbrock(), size=25,
                     r_pol=_r_pol(), s_pol=_s_pol())
        tmp = repr(isl)
        isl = loads(dumps(isl))
        self.assertEqual(tmp, repr(isl))

    def run_stateful_algo_tests(self):
        from .core import island, rosenbrock
        isl = island(algo=_stateful_algo(), prob=rosenbrock(), size=25)
        isl.evolve(20)
        isl.wait_check()
        self.assertTrue(isl.get_algorithm().extract(_stateful_algo)._n == 20)

    def run_extract_tests(self):
        from .core import island, _test_island, null_problem, null_algorithm, thread_island
        import sys

        # First we try with a C++ test island.
        isl = island(udi=_test_island(), algo=null_algorithm(),
                     prob=null_problem(), size=1)
        # Verify the refcount of p is increased after extract().
        rc = sys.getrefcount(isl)
        tisl = isl.extract(_test_island)
        self.assertFalse(tisl is None)
        self.assertTrue(isl.is_(_test_island))
        self.assertEqual(sys.getrefcount(isl), rc + 1)
        del tisl
        self.assertEqual(sys.getrefcount(isl), rc)
        # Verify we are modifying the inner object.
        isl.extract(_test_island).set_n(5)
        self.assertEqual(isl.extract(_test_island).get_n(), 5)
        # Try to extract the wrong C++ island type.
        self.assertTrue(isl.extract(thread_island) is None)
        self.assertFalse(isl.is_(thread_island))

        class tisland(object):

            def __init__(self):
                self._n = 1

            def get_n(self):
                return self._n

            def set_n(self, n):
                self._n = n

            def run_evolve(self, algo, pop):
                return algo, pop

        # Test with Python problem.
        isl = island(udi=tisland(), algo=null_algorithm(),
                     prob=null_problem(), size=1)
        rc = sys.getrefcount(isl)
        tisl = isl.extract(tisland)
        self.assertFalse(tisl is None)
        self.assertTrue(isl.is_(tisland))
        # Reference count does not increase because
        # tisland is stored as a proper Python object
        # with its own refcount.
        self.assertEqual(sys.getrefcount(isl), rc)
        self.assertEqual(tisl.get_n(), 1)
        tisl.set_n(12)
        self.assertEqual(isl.extract(tisland).get_n(), 12)
        # Try to extract the wrong Python island type.
        self.assertTrue(isl.extract(_udi_01) is None)
        self.assertFalse(isl.is_(_udi_01))

        # Check that we can extract Python UDIs also via Python's object type.
        isl = island(udi=tisland(), algo=null_algorithm(),
                     prob=null_problem(), size=1)
        self.assertTrue(not isl.extract(object) is None)
        # Check we are referring to the same object.
        self.assertEqual(id(isl.extract(object)), id(isl.extract(tisland)))
        # Check that it will not work with exposed C++ islands.
        isl = island(udi=thread_island(), algo=null_algorithm(),
                     prob=null_problem(), size=1)
        self.assertTrue(isl.extract(object) is None)
        self.assertTrue(not isl.extract(thread_island) is None)

    def run_mo_sto_repr_bug(self):
        # Old bug: printing islands containing MO/sto
        # problems would throw due to an error being raised
        # when accessing the champion.
        from .core import island, de, rosenbrock, zdt, inventory

        isl = island(algo=de(), prob=rosenbrock(), size=25)
        self.assertTrue("Champion decision vector" in repr(isl))
        self.assertTrue("Champion fitness" in repr(isl))

        isl = island(algo=de(), prob=zdt(), size=25)
        self.assertFalse("Champion decision vector" in repr(isl))
        self.assertFalse("Champion fitness" in repr(isl))

        isl = island(algo=de(), prob=inventory(), size=25)
        self.assertFalse("Champion decision vector" in repr(isl))
        self.assertFalse("Champion fitness" in repr(isl))


class mp_island_test_case(_ut.TestCase):
    """Test case for the :class:`~pygmo.mp_island` class.

    """

    def __init__(self, level):
        _ut.TestCase.__init__(self)
        self._level = level

    def runTest(self):
        self.run_basic_tests()

    def run_basic_tests(self):
        from .core import island, de, rosenbrock
        from . import mp_island
        from copy import copy, deepcopy
        from pickle import dumps, loads
        # Try shutting down a few times, to confirm that the second
        # and third shutdowns don't do anything.
        mp_island.shutdown_pool()
        mp_island.shutdown_pool()
        mp_island.shutdown_pool()
        isl = island(algo=de(), prob=rosenbrock(), size=25, udi=mp_island())
        self.assertTrue("Using a process pool: yes" in str(isl))
        self.assertEqual(isl.get_name(), "Multiprocessing island")
        self.assertTrue(isl.get_extra_info() != "")
        self.assertTrue(mp_island.get_pool_size() > 0)
        self.assertTrue(isl.extract(object).use_pool)
        with self.assertRaises(ValueError) as cm:
            isl.extract(object).pid
        err = cm.exception
        self.assertTrue(
            "The 'pid' property is available only when the island is configured to spawn" in str(err))

        # Init a few times, to confirm that the second
        # and third inits don't do anything.
        mp_island.init_pool()
        mp_island.init_pool()
        mp_island.init_pool()
        mp_island.shutdown_pool()
        self.assertRaises(TypeError, lambda: mp_island.init_pool("dasda"))
        self.assertRaises(ValueError, lambda: mp_island.init_pool(0))
        self.assertRaises(ValueError, lambda: mp_island.init_pool(-1))
        mp_island.init_pool()
        mp_island.resize_pool(6)
        isl.evolve(20)
        isl.wait_check()
        mp_island.resize_pool(4)
        isl.wait_check()
        isl.evolve(20)
        isl.evolve(20)
        isl.wait()
        self.assertRaises(ValueError, lambda: mp_island.resize_pool(-1))
        self.assertRaises(TypeError, lambda: mp_island.resize_pool("dasda"))

        # Shutdown and verify that evolve() throws.
        mp_island.shutdown_pool()
        isl.evolve(20)
        with self.assertRaises(RuntimeError) as cm:
            isl.wait_check()
        err = cm.exception
        self.assertTrue(
            "The multiprocessing island pool was stopped. Please restart it via mp_island.init_pool()." in str(err))

        # Verify that asking for the pool size triggers the creation of a new pool.
        self.assertTrue(mp_island.get_pool_size() > 0)
        mp_island.resize_pool(4)

        # Check the picklability of a problem storing a lambda.
        isl = island(algo=de(), prob=_prob(
            lambda x, y: x + y), size=25, udi=mp_island())
        isl.evolve()
        isl.wait_check()

        # Copy/deepcopy.
        isl2 = copy(isl)
        isl3 = deepcopy(isl)
        self.assertEqual(str(isl2), str(isl))
        self.assertEqual(str(isl3), str(isl))
        self.assertTrue(isl2.extract(object).use_pool)
        self.assertTrue(isl3.extract(object).use_pool)
        # Do some copying while the island evolves.
        isl.evolve(20)
        isl2 = copy(isl)
        isl3 = deepcopy(isl)
        self.assertTrue(isl2.extract(object).use_pool)
        self.assertTrue(isl3.extract(object).use_pool)
        isl.wait_check()

        # Pickle.
        self.assertEqual(str(loads(dumps(isl))), str(isl))
        self.assertTrue(loads(dumps(isl)).extract(object).use_pool)
        self.assertTrue("Using a process pool: yes" in str(loads(dumps(isl))))
        # Pickle during evolution.
        isl.evolve(20)
        self.assertTrue("Using a process pool: yes" in str(loads(dumps(isl))))
        isl.wait_check()

        # Tests when not using the pool.
        with self.assertRaises(TypeError) as cm:
            island(algo=de(), prob=rosenbrock(),
                   size=25, udi=mp_island(use_pool=None))
        err = cm.exception
        self.assertTrue(
            "The 'use_pool' parameter in the mp_island constructor must be a boolean" in str(err))

        # Island properties, copy/deepcopy, pickle.
        isl = island(algo=de(), prob=rosenbrock(), size=25,
                     udi=mp_island(use_pool=False))
        self.assertTrue("Using a process pool: no" in str(isl))
        self.assertFalse(isl.extract(object).use_pool)
        self.assertTrue(isl.extract(object).pid is None)
        isl2 = copy(isl)
        isl3 = deepcopy(isl)
        self.assertFalse(isl2.extract(object).use_pool)
        self.assertFalse(isl3.extract(object).use_pool)
        self.assertFalse(loads(dumps(isl)).extract(object).use_pool)
        self.assertTrue("Using a process pool: no" in str(loads(dumps(isl))))
        # Do some copying/pickling while the island evolves.
        isl.evolve(20)
        self.assertTrue("Using a process pool: no" in str(loads(dumps(isl))))
        isl2 = copy(isl)
        isl3 = deepcopy(isl)
        self.assertFalse(isl2.extract(object).use_pool)
        self.assertFalse(isl3.extract(object).use_pool)
        isl.wait_check()

        # Run some evolutions in a separate process.
        isl.evolve(20)
        isl.evolve(20)
        self.assertTrue("Using a process pool: no" in str(isl))
        isl.wait()
        self.assertTrue(isl.extract(object).pid is None)
        isl.evolve(20)
        isl.evolve(20)
        self.assertTrue("Using a process pool: no" in str(isl))
        isl.wait_check()
        self.assertTrue(isl.extract(object).pid is None)

        # Error transport when not using a pool.
        isl = island(algo=de(), prob=_prob(
            lambda x, y: x + y), size=2, udi=mp_island(use_pool=False))
        isl.evolve()
        isl.wait()
        self.assertTrue("**error occurred**" in repr(isl))
        with self.assertRaises(RuntimeError) as cm:
            isl.wait_check()
        err = cm.exception
        self.assertTrue(
            "An exception was raised in the evolution of a multiprocessing island. The full error message is:" in str(err))
        self.assertTrue(isl.extract(object).pid is None)

        if self._level == 0:
            return

        # Check exception transport.
        for _ in range(1000):
            isl = island(algo=de(), prob=_prob(
                lambda x, y: x + y), size=2, udi=mp_island(use_pool=True))
            isl.evolve()
            isl.wait()
            self.assertTrue("**error occurred**" in repr(isl))
            self.assertRaises(RuntimeError, lambda: isl.wait_check())


class ipyparallel_island_test_case(_ut.TestCase):
    """Test case for the :class:`~pygmo.ipyparallel` class.

    """

    def __init__(self, level):
        _ut.TestCase.__init__(self)
        self._level = level

    def runTest(self):
        try:
            import ipyparallel
        except ImportError:
            return

        self.run_basic_tests()

    def run_basic_tests(self):
        from .core import island, de, rosenbrock
        from . import ipyparallel_island
        from copy import copy, deepcopy
        from pickle import dumps, loads
        import ipyparallel

        ipyparallel_island.shutdown_view()
        ipyparallel_island.shutdown_view()
        ipyparallel_island.shutdown_view()

        to = .5
        isl = island(algo=de(), prob=rosenbrock(),
                     size=25, udi=ipyparallel_island())
        ipyparallel_island.shutdown_view()
        try:
            # Try with kwargs for the client.
            ipyparallel_island.init_view(client_kwargs={'timeout': to})
        except OSError:
            return
        isl = island(algo=de(), prob=rosenbrock(),
                     size=25, udi=ipyparallel_island())
        self.assertEqual(isl.get_name(), "Ipyparallel island")
        self.assertTrue(isl.get_extra_info() != "")
        ipyparallel_island.shutdown_view()
        isl.evolve(20)
        isl.wait_check()
        isl.evolve(20)
        isl.evolve(20)
        isl.wait_check()

        # Try kwargs for the view.
        ipyparallel_island.init_view(view_kwargs={'targets': [1]})
        isl.evolve(20)
        isl.evolve(20)
        isl.wait_check()

        # Check the picklability of a problem storing a lambda.
        isl = island(algo=de(), prob=_prob(lambda x, y: x + y),
                     size=25, udi=ipyparallel_island())
        isl.evolve()
        isl.wait_check()

        # Copy/deepcopy.
        isl2 = copy(isl)
        isl3 = deepcopy(isl)
        self.assertEqual(str(isl2.get_population()), str(isl.get_population()))
        self.assertEqual(str(isl2.get_algorithm()), str(isl.get_algorithm()))
        self.assertEqual(str(isl2.get_name()), str(isl.get_name()))
        self.assertEqual(str(isl3.get_population()), str(isl.get_population()))
        self.assertEqual(str(isl3.get_algorithm()), str(isl.get_algorithm()))
        self.assertEqual(str(isl3.get_name()), str(isl.get_name()))
        # Do some copying while the island evolves.
        isl.evolve(20)
        isl2 = copy(isl)
        isl3 = deepcopy(isl)
        self.assertEqual(str(isl2.get_name()), str(isl.get_name()))
        self.assertEqual(str(isl3.get_name()), str(isl.get_name()))
        isl.wait_check()

        # Pickle.
        pisl = loads(dumps(isl))
        self.assertEqual(str(pisl.get_population()), str(isl.get_population()))
        self.assertEqual(str(pisl.get_algorithm()), str(isl.get_algorithm()))
        self.assertEqual(str(pisl.get_name()), str(isl.get_name()))
        # Pickle during evolution.
        isl.evolve(20)
        pisl = loads(dumps(isl))
        self.assertEqual(str(pisl.get_name()), str(isl.get_name()))
        isl.wait_check()

        if self._level == 0:
            return

        ipyparallel_island.shutdown_view()

        # Check exception transport.
        for _ in range(10):
            isl = island(algo=de(), prob=_prob(
                lambda x, y: x + y), size=2, udi=ipyparallel_island())
            isl.evolve()
            isl.wait()
            self.assertTrue("**error occurred**" in repr(isl))
            self.assertRaises(RuntimeError,
                              lambda: isl.wait_check())
