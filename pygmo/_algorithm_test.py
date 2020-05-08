# Copyright 2020 PaGMO development team
#
# This file is part of the pygmo library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class _algo(object):
    def evolve(self, pop):
        return pop


class algorithm_test_case(_ut.TestCase):
    """Test case for the :class:`~pygmo.algorithm` class.

    """

    def runTest(self):
        self.run_basic_tests()
        self.run_extract_tests()
        self.run_seed_tests()
        self.run_verbosity_tests()
        self.run_name_info_tests()
        self.run_thread_safety_tests()
        self.run_pickle_tests()
        self.run_scipy_wrapper_tests()

    def run_basic_tests(self):
        # Tests for minimal algorithm, and mandatory methods.
        from .core import algorithm, de, population, null_problem, null_algorithm
        from . import thread_safety as ts

        # Def construction.
        a = algorithm()
        self.assertTrue(a.extract(null_algorithm) is not None)
        self.assertTrue(a.extract(de) is None)

        # First a few non-algos.
        self.assertRaises(NotImplementedError, lambda: algorithm(1))
        self.assertRaises(NotImplementedError, lambda: algorithm("hello world"))
        self.assertRaises(NotImplementedError, lambda: algorithm([]))
        self.assertRaises(TypeError, lambda: algorithm(int))
        # Some algorithms missing methods, wrong arity, etc.

        class na0(object):
            pass

        self.assertRaises(NotImplementedError, lambda: algorithm(na0()))

        class na1(object):

            evolve = 45

        self.assertRaises(NotImplementedError, lambda: algorithm(na1()))

        # The minimal good citizen.
        glob = []

        class a(object):

            def __init__(self, g):
                self.g = g

            def evolve(self, pop):
                self.g.append(1)
                return pop

        a_inst = a(glob)
        algo = algorithm(a_inst)

        # Test the keyword arg.
        algo = algorithm(uda=de())
        algo = algorithm(uda=a_inst)

        # Check a few algo properties.
        self.assertEqual(algo.is_stochastic(), False)
        self.assertEqual(algo.has_set_seed(), False)
        self.assertEqual(algo.has_set_verbosity(), False)
        self.assertEqual(algo.get_thread_safety(), ts.none)
        self.assertEqual(algo.get_extra_info(), "")
        self.assertRaises(NotImplementedError, lambda: algo.set_seed(123))
        self.assertRaises(NotImplementedError, lambda: algo.set_verbosity(1))
        self.assertTrue(algo.extract(int) is None)
        self.assertTrue(algo.extract(de) is None)
        self.assertFalse(algo.extract(a) is None)
        self.assertTrue(algo.is_(a))
        self.assertTrue(isinstance(algo.evolve(population()), population))
        # Assert that a_inst was deep-copied into algo:
        # the instance in algo will have its own copy of glob
        # and it will not be a reference the outside object.
        self.assertEqual(len(glob), 0)
        self.assertEqual(len(algo.extract(a).g), 1)
        algo = algorithm(de())
        self.assertEqual(algo.is_stochastic(), True)
        self.assertEqual(algo.has_set_seed(), True)
        self.assertEqual(algo.has_set_verbosity(), True)
        self.assertEqual(algo.get_thread_safety(), ts.basic)
        self.assertTrue(algo.get_extra_info() != "")
        self.assertTrue(algo.extract(int) is None)
        self.assertTrue(algo.extract(a) is None)
        self.assertFalse(algo.extract(de) is None)
        self.assertTrue(algo.is_(de))
        algo.set_seed(123)
        algo.set_verbosity(0)
        self.assertTrue(isinstance(algo.evolve(
            population(null_problem(), 5)), population))
        # Wrong retval for evolve().

        class a(object):

            def evolve(self, pop):
                return 3

        algo = algorithm(a())
        self.assertRaises(RuntimeError, lambda: algo.evolve(
            population(null_problem(), 5)))

        # Test that construction from another pygmo.algorithm fails.
        with self.assertRaises(TypeError) as cm:
            algorithm(algo)
        err = cm.exception
        self.assertTrue(
            "a pygmo.algorithm cannot be used as a UDA for another pygmo.algorithm (if you need to copy an algorithm please use the standard Python copy()/deepcopy() functions)" in str(err))

    def run_extract_tests(self):
        from .core import algorithm, _test_algorithm, mbh, de
        import sys

        # First we try with a C++ test algo.
        p = algorithm(_test_algorithm())
        # Verify the refcount of p is increased after extract().
        rc = sys.getrefcount(p)
        tprob = p.extract(_test_algorithm)
        self.assertTrue(sys.getrefcount(p) == rc + 1)
        del tprob
        self.assertTrue(sys.getrefcount(p) == rc)
        # Verify we are modifying the inner object.
        p.extract(_test_algorithm).set_n(5)
        self.assertTrue(p.extract(_test_algorithm).get_n() == 5)
        # Chain extracts.
        t = mbh(_test_algorithm(), stop=5, perturb=[0.4])
        pt = algorithm(t)
        rc = sys.getrefcount(pt)
        talgo = pt.extract(mbh)
        # Verify that extraction of mbh from the algo
        # increases the refecount of pt.
        self.assertTrue(sys.getrefcount(pt) == rc + 1)
        # Extract the _test_algorithm from mbh.
        rc2 = sys.getrefcount(talgo)
        ttalgo = talgo.inner_algorithm.extract(_test_algorithm)
        # The refcount of pt is not affected.
        self.assertTrue(sys.getrefcount(pt) == rc + 1)
        # The refcount of talgo has increased.
        self.assertTrue(sys.getrefcount(talgo) == rc2 + 1)
        del talgo
        # We can still access ttalgo.
        self.assertTrue(ttalgo.get_n() == 1)
        self.assertTrue(sys.getrefcount(pt) == rc + 1)
        del ttalgo
        # Now the refcount of pt decreases, because deleting
        # ttalgo eliminates the last ref to talgo, which in turn
        # decreases the refcount of pt.
        self.assertTrue(sys.getrefcount(pt) == rc)

        class talgorithm(object):

            def __init__(self):
                self._n = 1

            def get_n(self):
                return self._n

            def set_n(self, n):
                self._n = n

            def evolve(self, pop):
                return pop

        # Test with Python algo.
        p = algorithm(talgorithm())
        rc = sys.getrefcount(p)
        talgo = p.extract(talgorithm)
        # Reference count does not increase because
        # talgorithm is stored as a proper Python object
        # with its own refcount.
        self.assertTrue(sys.getrefcount(p) == rc)
        self.assertTrue(talgo.get_n() == 1)
        talgo.set_n(12)
        self.assertTrue(p.extract(talgorithm).get_n() == 12)

        # Check that we can extract Python UDAs also via Python's object type.
        a = algorithm(talgorithm())
        self.assertTrue(not a.extract(object) is None)
        # Check we are referring to the same object.
        self.assertEqual(id(a.extract(object)), id(a.extract(talgorithm)))
        # Check that it will not work with exposed C++ algorithms.
        a = algorithm(de())
        self.assertTrue(a.extract(object) is None)
        self.assertTrue(not a.extract(de) is None)

    def run_seed_tests(self):
        from .core import algorithm

        class a(object):

            def evolve(self, pop):
                return pop

        self.assertTrue(not algorithm(a()).has_set_seed())
        self.assertRaises(NotImplementedError,
                          lambda: algorithm(a()).set_seed(12))

        class a(object):

            def evolve(self, pop):
                return pop

            def has_set_seed(self):
                return True

        self.assertTrue(not algorithm(a()).has_set_seed())
        self.assertRaises(NotImplementedError,
                          lambda: algorithm(a()).set_seed(12))

        class a(object):

            def evolve(self, pop):
                return pop

            def set_seed(self, seed):
                pass

        self.assertTrue(algorithm(a()).has_set_seed())
        algorithm(a()).set_seed(87)

        class a(object):

            def evolve(self, pop):
                return pop

            def set_seed(self, seed):
                pass

            def has_set_seed(self):
                return False

        self.assertTrue(not algorithm(a()).has_set_seed())

        class a(object):

            def evolve(self, pop):
                return pop

            def set_seed(self, seed):
                pass

            def has_set_seed(self):
                return True

        self.assertTrue(algorithm(a()).has_set_seed())
        algorithm(a()).set_seed(0)
        algorithm(a()).set_seed(87)
        self.assertRaises(TypeError, lambda: algorithm(a()).set_seed(-1))

    def run_verbosity_tests(self):
        from .core import algorithm

        class a(object):

            def evolve(self, pop):
                return pop

        self.assertTrue(not algorithm(a()).has_set_verbosity())
        self.assertRaises(NotImplementedError,
                          lambda: algorithm(a()).set_verbosity(12))

        class a(object):

            def evolve(self, pop):
                return pop

            def has_set_verbosity(self):
                return True

        self.assertTrue(not algorithm(a()).has_set_verbosity())
        self.assertRaises(NotImplementedError,
                          lambda: algorithm(a()).set_verbosity(12))

        class a(object):

            def evolve(self, pop):
                return pop

            def set_verbosity(self, level):
                pass

        self.assertTrue(algorithm(a()).has_set_verbosity())
        algorithm(a()).set_verbosity(87)

        class a(object):

            def evolve(self, pop):
                return pop

            def set_verbosity(self, level):
                pass

            def has_set_verbosity(self):
                return False

        self.assertTrue(not algorithm(a()).has_set_verbosity())

        class a(object):

            def evolve(self, pop):
                return pop

            def set_verbosity(self, level):
                pass

            def has_set_verbosity(self):
                return True

        self.assertTrue(algorithm(a()).has_set_verbosity())
        algorithm(a()).set_verbosity(0)
        algorithm(a()).set_verbosity(87)
        self.assertRaises(TypeError, lambda: algorithm(a()).set_verbosity(-1))

    def run_name_info_tests(self):
        from .core import algorithm

        class a(object):

            def evolve(self, pop):
                return pop

        algo = algorithm(a())
        self.assertTrue(algo.get_name() != '')
        self.assertTrue(algo.get_extra_info() == '')

        class a(object):

            def evolve(self, pop):
                return pop

            def get_name(self):
                return 'pippo'

        algo = algorithm(a())
        self.assertTrue(algo.get_name() == 'pippo')
        self.assertTrue(algo.get_extra_info() == '')

        class a(object):

            def evolve(self, pop):
                return pop

            def get_extra_info(self):
                return 'pluto'

        algo = algorithm(a())
        self.assertTrue(algo.get_name() != '')
        self.assertTrue(algo.get_extra_info() == 'pluto')

        class a(object):

            def evolve(self, pop):
                return pop

            def get_name(self):
                return 'pippo'

            def get_extra_info(self):
                return 'pluto'

        algo = algorithm(a())
        self.assertTrue(algo.get_name() == 'pippo')
        self.assertTrue(algo.get_extra_info() == 'pluto')

    def run_thread_safety_tests(self):
        from .core import algorithm, de, _tu_test_algorithm, mbh
        from . import thread_safety as ts

        class a(object):

            def evolve(self, pop):
                return pop

        self.assertTrue(algorithm(a()).get_thread_safety() == ts.none)
        self.assertTrue(algorithm(de()).get_thread_safety() == ts.basic)
        self.assertTrue(
            algorithm(_tu_test_algorithm()).get_thread_safety() == ts.none)
        self.assertTrue(
            algorithm(mbh(_tu_test_algorithm(), stop=5, perturb=.4)).get_thread_safety() == ts.none)
        self.assertTrue(
            algorithm(mbh(a(), stop=5, perturb=.4)).get_thread_safety() == ts.none)
        self.assertTrue(
            algorithm(mbh(de(), stop=5, perturb=.4)).get_thread_safety() == ts.basic)

    def run_pickle_tests(self):
        from .core import algorithm, de, mbh
        from pickle import dumps, loads

        a_ = algorithm(de())
        a = loads(dumps(a_))
        self.assertEqual(repr(a), repr(a_))
        self.assertTrue(a.is_(de))
        a_ = algorithm(mbh(de(), 10, 0.1))
        a = loads(dumps(a_))
        self.assertEqual(repr(a), repr(a_))
        self.assertTrue(a.is_(mbh))
        self.assertTrue(a.extract(mbh).inner_algorithm.is_(de))

        a_ = algorithm(_algo())
        a = loads(dumps(a_))
        self.assertEqual(repr(a), repr(a_))
        self.assertTrue(a.is_(_algo))
        a_ = algorithm(mbh(_algo(), 10, 0.1))
        a = loads(dumps(a_))
        self.assertEqual(repr(a), repr(a_))
        self.assertTrue(a.is_(mbh))
        self.assertTrue(a.extract(mbh).inner_algorithm.is_(_algo))

    def run_scipy_wrapper_tests(self):
        from . import (
            ackley,
            algorithm,
            golomb_ruler,
            hock_schittkowsky_71,
            luksan_vlcek1,
            minlp_rastrigin,
            population,
            problem,
            rastrigin,
            rosenbrock,
            s_policy,
            select_best,
            scipy_optimize,
        )
        from copy import deepcopy

        # testing invalid method
        self.assertRaises(ValueError, lambda: scipy_optimize(method="foo"))

        # simple test with ackley, a problem without gradients or constraints
        methods = ["L-BFGS-B", "TNC", "SLSQP", None]
        prob = problem(ackley(10))
        pop = population(prob=prob, size=1, seed=0)
        init = pop.champion_f

        for m in methods:
            popc = deepcopy(pop)
            scp = algorithm(scipy_optimize(method=m))
            result = scp.evolve(popc).champion_f
            self.assertTrue(result[0] <= init[0])
            self.assertTrue(popc.problem.get_fevals() > 1)

        # simple test with rosenbrock, a problem with a gradient
        methods = ["L-BFGS-B", "TNC", "SLSQP", "trust-constr", None]
        prob = problem(rosenbrock(10))
        pop = population(prob=prob, size=1, seed=0)
        init = pop.champion_f

        for m in methods:
            popc = deepcopy(pop)
            scp = algorithm(scipy_optimize(method=m))
            result = scp.evolve(popc).champion_f
            self.assertTrue(result[0] <= init[0])
            self.assertTrue(popc.problem.get_fevals() > 1)
            self.assertTrue(popc.problem.get_gevals() > 1)

        # testing Hessian and Hessian sparsity
        methods = ["trust-constr", "trust-exact", "trust-krylov", None]
        problems = [problem(rastrigin(10)), problem(minlp_rastrigin(10))]

        for inst in problems:
            pop = population(prob=inst, size=1, seed=0)
            init = pop.champion_f

            for m in methods:
                popc = deepcopy(pop)
                scp = algorithm(scipy_optimize(method=m))
                result = scp.evolve(popc).champion_f
                self.assertTrue(result[0] <= init[0])
                self.assertTrue(popc.problem.get_fevals() > 1)
                self.assertTrue(popc.problem.get_gevals() > 0)
                if m is not None:
                    self.assertTrue(popc.problem.get_hevals() > 0)

        # testing constraints without Hessians
        methods = ["SLSQP", "trust-constr", None]
        raw_probs = [luksan_vlcek1(10), golomb_ruler(2, 10)]
        instances = [problem(prob) for prob in raw_probs]

        for inst in instances:
            pop = population(prob=inst, size=1, seed=0)
            init = pop.champion_f

            for m in methods:
                popc = deepcopy(pop)
                # print(m, ": ", end="")
                scp = algorithm(scipy_optimize(method=m))
                result = scp.evolve(popc).champion_f
                self.assertTrue(result[0] <= init[0])
                self.assertTrue(popc.problem.get_fevals() > 1)
                # TODO: test that result fulfills constraints

        # testing constraints with gradients and Hessians
        methods = ["trust-constr", None]
        prob = problem(hock_schittkowsky_71())
        pop = population(prob=prob, size=1, seed=0)
        init = pop.champion_f

        for m in methods:
            popc = deepcopy(pop)
            scp = algorithm(scipy_optimize(method=m))
            result = scp.evolve(popc).champion_f
            self.assertTrue(result[0] <= init[0])
            self.assertTrue(popc.problem.get_fevals() > 1)
            self.assertTrue(popc.problem.get_gevals() > 0)
            if m is not None:
                self.assertTrue(popc.problem.get_hevals() > 0)

        # testing verbosity
        method_list = [
            "Nelder-Mead",
            "Powell",
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "SLSQP",
            "trust-constr",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
            None,
        ]
        for m in method_list:
            scp = algorithm(scipy_optimize(method=m))
            scp.set_verbosity(1)
            scp.get_name()
            scp.set_verbosity(0)


        # testing constrained problem on incompatible methods
        prob = problem(luksan_vlcek1(10))
        pop = population(prob=prob, size=1, seed=0)

        methods = [
            "Nelder-Mead",
            "Powell",
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
        ]

        for m in methods:
            popc = deepcopy(pop)
            scp = algorithm(scipy_optimize(method=m))
            self.assertRaises(ValueError, lambda: scp.evolve(popc))

        # testing invalid selection policy
        prob = problem(luksan_vlcek1(10))
        pop = population(prob=prob, size=10, seed=0)
        scp = algorithm(scipy_optimize(selection = s_policy(select_best(rate=2))))
        self.assertRaises(ValueError, lambda: scp.evolve(pop))

        # testing callback
        class callback_counter:

            value = 0

            def increment(self, *args, **kwargs):
                callback_counter.value += 1

        prob = problem(luksan_vlcek1(10))
        pop = population(prob=prob, size=10, seed=0)
        counter = callback_counter()
        scp = algorithm(scipy_optimize(callback = counter.increment))
        scp.evolve(pop)
        self.assertTrue(counter.value > 0)

        # testing gradient wrapper generator
        from numpy import array

        prob = problem(luksan_vlcek1(10))
        prob.gradient([0] * prob.get_nx())
        wrapper = scipy_optimize._fitness_wrapper(prob)
        for i in range(prob.get_nobj() + prob.get_nc()):
            f = wrapper._generate_gradient_sparsity_wrapper(i)
            self.assertEqual(len(f(array([0] * prob.get_nx()))), prob.get_nx())

        # testing invalid index for gradient wrapper
        self.assertRaises(
            ValueError, lambda: wrapper._generate_gradient_sparsity_wrapper(9)
        )

        # testing gradient function of wrong dimension
        smallerProb = problem(luksan_vlcek1(8))
        wrapped_gradient = scipy_optimize._fitness_wrapper(
            smallerProb
        )._generate_gradient_sparsity_wrapper(0)
        self.assertRaises(
            ValueError, lambda: wrapped_gradient(array([0] * prob.get_nx()))
        )

        # testing hessian wrapper generator
        prob = problem(rastrigin(10))
        f = scipy_optimize._fitness_wrapper(prob)._generate_hessian_sparsity_wrapper(0)
        hessian = f(array([0] * prob.get_nx()))
        self.assertEqual(len(hessian), prob.get_nx())
        self.assertEqual(len(hessian[0]), prob.get_nx())

        # testing invalid index for hessian wrapper
        self.assertRaises(
            ValueError,
            lambda: scipy_optimize._fitness_wrapper(prob)
            ._generate_hessian_sparsity_wrapper(5),
        )
