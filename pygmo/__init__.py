# Copyright 2020 PaGMO development team
#
# This file is part of the pygmo library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from ._check_deps import *
# Version setup.
from ._version import __version__
# We import the sub-modules into the root namespace
from .core import *
from .core import _pagmo_version_major, _pagmo_version_minor, _pagmo_version_patch
from .plotting import *
from ._py_islands import *
from ._py_problems import *
from ._py_bfes import *
from ._py_algorithms import *
# Patch the problem class.
from . import _patch_problem
# Patch the algorithm class.
from . import _patch_algorithm
# Patch the bfe class.
from . import _patch_bfe
# Patch the island class.
from . import _patch_island
# Patch the policies.
from . import _patch_r_policy
from . import _patch_s_policy
# Patch the topology.
from . import _patch_topology
import cloudpickle as _cloudpickle
# Explicitly import the test submodule
from . import test
import atexit as _atexit


# We move into the problems, algorithms, etc. namespaces
# all the pure python UDAs, UDPs and UDIs.
for _item in dir(_py_islands):
    if not _item.startswith("_"):
        setattr(islands, _item, getattr(_py_islands, _item))
del _py_islands

for _item in dir(_py_problems):
    if not _item.startswith("_"):
        setattr(problems, _item, getattr(_py_problems, _item))
del _py_problems

for _item in dir(_py_bfes):
    if not _item.startswith("_"):
        setattr(batch_evaluators, _item, getattr(_py_bfes, _item))
del _py_bfes

for _item in dir(_py_algorithms):
    if not _item.startswith("_"):
        setattr(algorithms, _item, getattr(_py_algorithms, _item))
del _py_algorithms

del _item

# Machinery for the setup of the serialization backend.
_serialization_backend = _cloudpickle

# Override of the translate meta-problem constructor.
__original_translate_init = translate.__init__

# NOTE: the idea of having the translate init here instead of exposed from C++ is to allow the use
# of the syntax translate(udp, translation) for all udps


def _translate_init(self, prob=None, translation=[0.]):
    """
    Args:
        prob: a user-defined problem (either Python or C++), or an instance of :class:`~pygmo.problem`
            (if *prob* is :data:`None`, a :class:`~pygmo.null_problem` will be used in its stead)
        translation (array-like object): an array containing the translation to be applied

    Raises:
        ValueError: if the length of *translation* is not equal to the dimension of *prob*

        unspecified: any exception thrown by:

           * the constructor of :class:`pygmo.problem`,
           * the constructor of the underlying C++ class,
           * failures at the intersection between C++ and Python (e.g., type conversion errors, mismatched function
             signatures, etc.)
    """
    if prob is None:
        # Use the null problem for default init.
        prob = null_problem()
    if type(prob) == problem:
        # If prob is a pygmo problem, we will pass it as-is to the
        # original init.
        prob_arg = prob
    else:
        # Otherwise, we attempt to create a problem from it. This will
        # work if prob is an exposed C++ problem or a Python UDP.
        prob_arg = problem(prob)
    __original_translate_init(self, prob_arg, translation)


setattr(translate, "__init__", _translate_init)

# Override of the decompose meta-problem constructor.
__original_decompose_init = decompose.__init__

# NOTE: the idea of having the translate init here instead of exposed from C++ is to allow the use
# of the syntax decompose(udp, ..., ) for all udps


def _decompose_init(self, prob=None, weight=[0.5, 0.5], z=[0., 0.], method='weighted', adapt_ideal=False):
    """
    Args:
        prob: a user-defined problem (either Python or C++), or an instance of :class:`~pygmo.problem`
            (if *prob* is :data:`None`, a :class:`~pygmo.null_problem` will be used in its stead)
        weight (array-like object): the vector of weights :math:`\\boldsymbol \lambda`
        z (array-like object): the reference point :math:`\mathbf z^*`
        method (str): a string containing the decomposition method chosen
        adapt_ideal (bool): when :data:`True`, the reference point is adapted at each fitness evaluation
            to be the ideal point

    Raises:
        ValueError: if either:

           * *prob* is single objective or constrained,
           * *method* is not one of [``'weighted'``, ``'tchebycheff'``, ``'bi'``],
           * *weight* is not of size :math:`n`,
           * *z* is not of size :math:`n`
           * *weight* is not such that :math:`\\lambda_i > 0, \\forall i=1..n`,
           * *weight* is not such that :math:`\\sum_i \\lambda_i = 1`

        unspecified: any exception thrown by:

           * the constructor of :class:`pygmo.problem`,
           * the constructor of the underlying C++ class,
           * failures at the intersection between C++ and Python (e.g., type conversion errors, mismatched function
             signatures, etc.)

    """
    if prob is None:
        # Use the null problem for default init.
        prob = null_problem(nobj=2)
    if type(prob) == problem:
        # If prob is a pygmo problem, we will pass it as-is to the
        # original init.
        prob_arg = prob
    else:
        # Otherwise, we attempt to create a problem from it. This will
        # work if prob is an exposed C++ problem or a Python UDP.
        prob_arg = problem(prob)
    __original_decompose_init(self, prob_arg, weight, z, method, adapt_ideal)


setattr(decompose, "__init__", _decompose_init)

# Override of the unconstrain meta-problem constructor.
__original_unconstrain_init = unconstrain.__init__

# NOTE: the idea of having the unconstrain init here instead of exposed from C++ is to allow the use
# of the syntax unconstrain(udp, ... ) for all udps


def _unconstrain_init(self, prob=None, method="death penalty", weights=[]):
    """
    Args:
        prob: a :class:`~pygmo.problem` or a user-defined problem, either C++ or Python (if
              *prob* is :data:`None`, a :class:`~pygmo.null_problem` will be used in its stead)
        method (str): a string containing the unconstrain method chosen, one of [``'death penalty'``, ``'kuri'``, ``'weighted'``, ``'ignore_c'``, ``'ignore_o'``]
        weights (array-like object): the vector of weights to be used if the method chosen is ``'weighted'``

    Raises:
        ValueError: if either:

           * *prob* is unconstrained,
           * *method* is not one of [``'death penalty'``, ``'kuri'``, ``'weighted'``, ``'ignore_c'``, ``'ignore_o'``],
           * *weight* is not of the same size as the problem constraints (if the method ``'weighted'`` is selected), or not empty otherwise.

        unspecified: any exception thrown by:

           * the constructor of :class:`pygmo.problem`,
           * the constructor of the underlying C++ class,
           * failures at the intersection between C++ and Python (e.g., type conversion errors, mismatched function
             signatures, etc.)

    """
    if prob is None:
        # Use the null problem for default init.
        prob = null_problem(nobj=2, nec=3, nic=4)
    if type(prob) == problem:
        # If prob is a pygmo problem, we will pass it as-is to the
        # original init.
        prob_arg = prob
    else:
        # Otherwise, we attempt to create a problem from it. This will
        # work if prob is an exposed C++ problem or a Python UDP.
        prob_arg = problem(prob)
    __original_unconstrain_init(self, prob_arg, method, weights)


setattr(unconstrain, "__init__", _unconstrain_init)

# Override of the cstrs_self_adaptive meta-algorithm constructor.
__original_cstrs_self_adaptive_init = cstrs_self_adaptive.__init__
# NOTE: the idea of having the cstrs_self_adaptive init here instead of exposed from C++ is to allow the use
# of the syntax cstrs_self_adaptive(uda, ...) for all udas


def _cstrs_self_adaptive_init(self, iters=1, algo=None, seed=None):
    """
    Args:
        iter (int): number of iterations (i.e., calls to the inner algorithm evolve)
        algo: an :class:`~pygmo.algorithm` or a user-defined algorithm, either C++ or Python (if
             *algo* is :data:`None`, a :class:`~pygmo.de` algorithm will be used in its stead)
        seed (int): seed used by the internal random number generator (if *seed* is :data:`None`, a
             randomly-generated value will be used in its stead)

    Raises:
        ValueError: if *iters* is negative or greater than an implementation-defined value
        unspecified: any exception thrown by the constructor of :class:`pygmo.algorithm`, or by
             failures at the intersection between C++ and Python (e.g., type conversion errors, mismatched function
             signatures, etc.)

    """
    if algo is None:
        # Use the null problem for default init.
        algo = de()
    if type(algo) == algorithm:
        # If algo is a pygmo algorithm, we will pass it as-is to the
        # original init.
        algo_arg = algo
    else:
        # Otherwise, we attempt to create an algorithm from it. This will
        # work if algo is an exposed C++ algorithm or a Python UDA.
        algo_arg = algorithm(algo)
    if seed is None:
        __original_cstrs_self_adaptive_init(self, iters, algo_arg)
    else:
        __original_cstrs_self_adaptive_init(self, iters, algo_arg, seed)


setattr(cstrs_self_adaptive, "__init__", _cstrs_self_adaptive_init)


# Override of the population constructor.
__original_population_init = population.__init__


def _population_init(self, prob=None, size=0, b=None, seed=None):
    # NOTE: the idea of having the pop init here instead of exposed from C++ is that like this we don't need
    # to expose a new pop ctor each time we expose a new problem: in this method we will use the problem ctor
    # from a C++ problem, and on the C++ exposition side we need only to
    # expose the ctor of pop from pagmo::problem.
    """
    Args:
        prob: a user-defined problem (either Python or C++), or an instance of :class:`~pygmo.problem`
             (if *prob* is :data:`None`, a default-constructed :class:`~pygmo.problem` will be used
             in its stead)
        size (:class:`int`): the number of individuals
        b: a user-defined batch fitness evaluator (either Python or C++), or an instance of :class:`~pygmo.bfe`
             (if *b* is :data:`None`, the evaluation of the population's individuals will be performed
             in sequential mode)
        seed (:class:`int`): the random seed (if *seed* is :data:`None`, a randomly-generated value will be used
             in its stead)

    Raises:
        TypeError: if *size* is not an :class:`int` or *seed* is not :data:`None` and not an :class:`int`
        OverflowError:  is *size* or *seed* are negative
        unspecified: any exception thrown by the invoked C++ constructors, by the constructor of
            :class:`~pygmo.problem`, or the constructor of :class:`~pygmo.bfe`, or by failures at
            the intersection between C++ and
            Python (e.g., type conversion errors, mismatched function signatures, etc.)

    """
    from .core import _random_device_next
    # Check input params.
    if not isinstance(size, int):
        raise TypeError("the 'size' parameter must be an integer")
    if not seed is None and not isinstance(seed, int):
        raise TypeError("the 'seed' parameter must be None or an integer")
    if prob is None:
        # Problem not specified, def-construct it.
        prob = problem()
    elif type(prob) != problem:
        # If prob is not a problem, we attempt to create a problem from it. This will
        # work if prob is an exposed C++ problem or a Python UDP.
        prob = problem(prob)

    if seed is None:
        # Seed not specified, randomly generate it
        # with the global pagmo rng.
        seed = _random_device_next()

    if b is None:
        # No bfe specified, init in sequential mode.
        __original_population_init(self, prob, size, seed)
    else:
        # A bfe was specified. Same as above with the problem.
        __original_population_init(self, prob, b if type(
            b) == bfe else bfe(b), size, seed)


setattr(population, "__init__", _population_init)


# Override of the island constructor.
__original_island_init = island.__init__


def _island_init(self, **kwargs):
    """
    Keyword Args:
        udi: a user-defined island, either Python or C++
        algo: a user-defined algorithm (either Python or C++), or an instance of :class:`~pygmo.algorithm`
        pop (:class:`~pygmo.population`): a population
        prob: a user-defined problem (either Python or C++), or an instance of :class:`~pygmo.problem`
        b: a user-defined batch fitness evaluator (either Python or C++), or an instance of :class:`~pygmo.bfe`
        size (:class:`int`): the number of individuals
        r_pol: a user-defined replacement policy (either Python or C++), or an instance of :class:`~pygmo.r_policy`
        s_pol: a user-defined selection policy (either Python or C++), or an instance of :class:`~pygmo.s_policy`
        seed (:class:`int`): the random seed (if not specified, it will be randomly-generated)

    Raises:
        KeyError: if the set of keyword arguments is invalid
        unspecified: any exception thrown by the invoked C++ constructors,
          the deep copy of the UDI, the constructors of :class:`~pygmo.algorithm` and :class:`~pygmo.population`,
          failures at the intersection between C++ and Python (e.g., type conversion errors, mismatched function
          signatures, etc.)

    """
    if len(kwargs) == 0:
        # Default constructor.
        __original_island_init(self)
        return

    # If we are not dealing with a def ctor, we always need the algo argument.
    if not 'algo' in kwargs:
        raise KeyError(
            "the mandatory 'algo' parameter is missing from the list of arguments "
            "of the island constructor")
    algo = kwargs.pop('algo')
    algo = algo if type(algo) == algorithm else algorithm(algo)

    # Population setup. We either need an input pop, or the prob and size,
    # plus optionally seed and b.
    if 'pop' in kwargs and ('prob' in kwargs or 'size' in kwargs or 'seed' in kwargs or 'b' in kwargs):
        raise KeyError(
            "if the 'pop' argument is provided, the 'prob', 'size', 'seed' and 'b' "
            "arguments must not be provided")
    elif 'pop' in kwargs:
        pop = kwargs.pop('pop')
    elif 'prob' in kwargs and 'size' in kwargs:
        pop = population(prob=kwargs.pop('prob'),
                         size=kwargs.pop('size'), seed=kwargs.pop('seed') if 'seed' in kwargs else None,
                         b=kwargs.pop('b') if 'b' in kwargs else None)
    else:
        raise KeyError(
            "unable to construct a population from the arguments of "
            "the island constructor: you must either pass a population "
            "('pop') or a set of arguments that can be used to build one "
            "('prob', 'size' and, optionally, 'seed' and 'b')")

    # UDI, if any.
    if 'udi' in kwargs:
        args = [kwargs.pop('udi'), algo, pop]
    else:
        args = [algo, pop]

    # Replace/selection policies, if any.
    if 'r_pol' in kwargs:
        r_pol = kwargs.pop('r_pol')
        r_pol = r_pol if type(r_pol) == r_policy else r_policy(r_pol)
        args.append(r_pol)
    else:
        args.append(r_policy())

    if 's_pol' in kwargs:
        s_pol = kwargs.pop('s_pol')
        s_pol = s_pol if type(s_pol) == s_policy else s_policy(s_pol)
        args.append(s_pol)
    else:
        args.append(s_policy())

    if len(kwargs) != 0:
        raise KeyError(
            'unrecognised keyword arguments: {}'.format(list(kwargs.keys())))

    __original_island_init(self, *args)


setattr(island, "__init__", _island_init)


# Override of the mbh meta-algorithm constructor.
__original_mbh_init = mbh.__init__
# NOTE: the idea of having the mbh init here instead of exposed from C++ is to allow the use
# of the syntax mbh(uda, ...) for all udas


def _mbh_init(self, algo=None, stop=5, perturb=1e-2, seed=None):
    """
    Args:
        algo: an :class:`~pygmo.algorithm` or a user-defined algorithm, either C++ or Python (if
              *algo* is :data:`None`, a :class:`~pygmo.compass_search` algorithm will be used in its stead)
        stop (int): consecutive runs of the inner algorithm that need to result in no improvement for
             :class:`~pygmo.mbh` to stop
        perturb (float or array-like object): the perturbation to be applied to each component
        seed (int): seed used by the internal random number generator (if *seed* is :data:`None`, a
             randomly-generated value will be used in its stead)

    Raises:
        ValueError: if *perturb* (or one of its components, if *perturb* is an array) is not in the
             (0,1] range
        unspecified: any exception thrown by the constructor of :class:`pygmo.algorithm`, or by
             failures at the intersection between C++ and Python (e.g., type conversion errors, mismatched function
             signatures, etc.)

    """
    import numbers
    if algo is None:
        # Use the compass search algo for default init.
        algo = compass_search()
    if type(algo) == algorithm:
        # If algo is a pygmo algorithm, we will pass it as-is to the
        # original init.
        algo_arg = algo
    else:
        # Otherwise, we attempt to create an algorithm from it. This will
        # work if algo is an exposed C++ algorithm or a Python UDA.
        algo_arg = algorithm(algo)
    if isinstance(perturb, numbers.Number):
        perturb = [perturb]
    if seed is None:
        __original_mbh_init(self, algo_arg, stop, perturb)
    else:
        __original_mbh_init(self, algo_arg, stop, perturb, seed)


setattr(mbh, "__init__", _mbh_init)


# Override of the archi constructor.
__original_archi_init = archipelago.__init__


def _archi_init(self, n=0, t=topology(), **kwargs):
    """__init__(self, n=0, t=topology(), **kwargs)

    The constructor will initialise an archipelago with a topology *t* and
    *n* islands built from *kwargs*.
    The keyword arguments accept the same format as explained in the constructor of
    :class:`~pygmo.island`, with the following differences:

    * *size* is replaced by *pop_size*, for clarity,
    * the *seed* argument, if present, is used to initialise a random number generator
      that, in turn, is used to generate random seeds for each island population. In other
      words, the *seed* argument allows to generate randomly (but deterministically)
      the seeds of the populations in the archipelago. If *seed* is not provided, the seeds
      of the populations will be random and non-deterministic.

    This class is the Python counterpart of the C++ class :cpp:class:`pagmo::archipelago`.

    Args:
        n (:class:`int`): the number of islands in the archipelago
        t: a user-defined topology (either Python or C++), or an instance of :class:`~pygmo.topology`

    Keyword Args:
        udi: a user-defined island, either Python or C++
        algo: a user-defined algorithm (either Python or C++), or an instance of :class:`~pygmo.algorithm`
        pop (:class:`~pygmo.population`): a population
        prob: a user-defined problem (either Python or C++), or an instance of :class:`~pygmo.problem`
        b: a user-defined batch fitness evaluator (either Python or C++), or an instance of :class:`~pygmo.bfe`
        pop_size (:class:`int`): the number of individuals for each island
        r_pol: a user-defined replacement policy (either Python or C++), or an instance of :class:`~pygmo.r_policy`
        s_pol: a user-defined selection policy (either Python or C++), or an instance of :class:`~pygmo.s_policy`
        seed (:class:`int`): the random seed

    Raises:
        TypeError: if *n* is not an integral type
        ValueError: if *n* is negative
        unspecified: any exception thrown by the constructor of :class:`~pygmo.island`,
          by the underlying C++ constructor, :func:`~pygmo.archipelago.push_back()` or
          by the public interface of :class:`~pygmo.topology`

    Examples:
        >>> from pygmo import *
        >>> archi = archipelago(n = 16, algo = de(), prob = rosenbrock(10), pop_size = 20, seed = 32)
        >>> archi.evolve()
        >>> archi #doctest: +SKIP
        Number of islands: 16
        Status: busy
        <BLANKLINE>
        Islands summaries:
        <BLANKLINE>
                #   Type           Algo                    Prob                                  Size  Status
                -----------------------------------------------------------------------------------------------
                0   Thread island  Differential Evolution  Multidimensional Rosenbrock Function  20    idle
                1   Thread island  Differential Evolution  Multidimensional Rosenbrock Function  20    idle
                2   Thread island  Differential Evolution  Multidimensional Rosenbrock Function  20    idle
                3   Thread island  Differential Evolution  Multidimensional Rosenbrock Function  20    idle
                4   Thread island  Differential Evolution  Multidimensional Rosenbrock Function  20    busy
                5   Thread island  Differential Evolution  Multidimensional Rosenbrock Function  20    idle
                6   Thread island  Differential Evolution  Multidimensional Rosenbrock Function  20    busy
                7   Thread island  Differential Evolution  Multidimensional Rosenbrock Function  20    busy
                8   Thread island  Differential Evolution  Multidimensional Rosenbrock Function  20    idle
                9   Thread island  Differential Evolution  Multidimensional Rosenbrock Function  20    idle
                10  Thread island  Differential Evolution  Multidimensional Rosenbrock Function  20    idle
                11  Thread island  Differential Evolution  Multidimensional Rosenbrock Function  20    idle
                12  Thread island  Differential Evolution  Multidimensional Rosenbrock Function  20    idle
                13  Thread island  Differential Evolution  Multidimensional Rosenbrock Function  20    busy
                14  Thread island  Differential Evolution  Multidimensional Rosenbrock Function  20    busy
                15  Thread island  Differential Evolution  Multidimensional Rosenbrock Function  20    idle
        <BLANKLINE>
        >>> archi.wait()
        >>> res = archi.get_champions_f()
        >>> res #doctest: +SKIP
        [array([ 125441.77328885]),
        array([ 144025.63904164]),
        array([ 25387.38989711]),
        array([ 56029.44160232]),
        array([ 47760.99082202]),
        array([ 118552.94891993]),
        array([ 118405.29575447]),
        array([ 101866.81846325]),
        array([ 166106.12039851]),
        array([ 167408.00058506]),
        array([ 148815.47953885]),
        array([ 165186.74476375]),
        array([ 326615.28881936]),
        array([ 167301.16445135]),
        array([ 166871.42760503]),
        array([ 75133.38815736])]


    """
    # Check n.
    if not isinstance(n, int):
        raise TypeError("the 'n' parameter must be an integer")
    if n < 0:
        raise ValueError(
            "the 'n' parameter must be non-negative, but it is {} instead".format(n))

    # Replace the 'pop_size' kw arg with just 'size', for later use in the
    # island ctor.

    if 'size' in kwargs:
        raise KeyError(
            "the 'size' argument cannot appear among the named arguments of the archipelago constructor")

    if 'pop_size' in kwargs:
        # Extract 'pop_size', replace with just 'size'.
        ps_val = kwargs.pop('pop_size')
        kwargs['size'] = ps_val

    # Call the original init, which constructs an empty archi from a topology.
    t = t if type(t) == topology else topology(t)
    __original_archi_init(self, t)

    if 'seed' in kwargs:
        # Special handling of the 'seed' argument.
        from random import Random
        from .core import _max_unsigned
        # Create a random engine with own state.
        RND = Random()
        # Get the seed from kwargs.
        seed = kwargs.pop('seed')
        if not isinstance(seed, int):
            raise TypeError("the 'seed' parameter must be an integer")
        # Seed the rng.
        RND.seed(seed)
        u_max = _max_unsigned()
        # Push back the islands with different seed.
        for _ in range(n):
            kwargs['seed'] = RND.randint(0, u_max)
            self.push_back(**kwargs)

    else:
        # Push back islands.
        for _ in range(n):
            self.push_back(**kwargs)


setattr(archipelago, "__init__", _archi_init)


def _archi_push_back(self, *args, **kwargs):
    """Add an island.

    This method will construct an island from the supplied arguments and add it to the archipelago.
    Islands are added at the end of the archipelago (that is, the new island will have an index
    equal to the size of the archipelago before the call to this method). :func:`pygmo.topology.push_back()`
    will also be called on the :class:`~pygmo.topology` associated to this archipelago, so that
    the addition of a new island to the archipelago is mirrored by the addition of a new vertex
    to the topology.

    This method accepts either a single positional argument, or a set of
    keyword arguments:

    * if no positional arguments are provided, then the keyword arguments will
      be used to construct a :class:`~pygmo.island` which will then be added
      to the archipelago; otherwise,
    * if a single positional argument is provided and no keyword arguments are
      provided, then the positional argument is interpreted as an :class:`~pygmo.island`
      object to be added to the archipelago.

    Any other combination of positional/keyword arguments will result in an error.

    Raises:
        ValueError: if, when using positional arguments, there are more than 1 positional arguments,
          or if keyword arguments are also used at the same time
        TypeError: if, when using a single positional argument, the type of that argument
          is not :class:`~pygmo.island`
        unspecified: any exception thrown by the constructor of :class:`~pygmo.island`,
          :func:`pygmo.topology.push_back()` or by the underlying C++ method

    """
    from . import island

    if len(args) == 0:
        self._push_back(island(**kwargs))
    else:
        if len(args) != 1:
            raise ValueError(
                "{} positional arguments were provided, but this method accepts only a single positional argument".format(len(args)))
        if len(kwargs) != 0:
            raise ValueError(
                "if a positional argument is passed to this method, then no keyword arguments must be passed, but {} keyword arguments were passed instead".format(len(kwargs)))
        if type(args[0]) != island:
            raise TypeError(
                "the positional argument passed to this method must be an island, but the type of the argument is '{}' instead".format(type(args[0])))
        self._push_back(args[0])


setattr(archipelago, "push_back", _archi_push_back)


def _archi_set_topology(self, t):
    """This method will wait for any ongoing evolution in the archipelago to finish,
    and it will then set the topology of the archipelago to *t*.

    Note that it is the user's responsibility to ensure that the new topology is
    consistent with the archipelago's properties.

    Args:
        t: a user-defined topology (either Python or C++), or an instance of :class:`~pygmo.topology`

    Raises:
        unspecified: any exception thrown by copying the topology

    """
    t = t if type(t) == topology else topology(t)
    self._set_topology(t)


setattr(archipelago, "set_topology", _archi_set_topology)


if _pagmo_version_major > 2 or (_pagmo_version_major == 2 and _pagmo_version_minor >= 15):
    # Override of the free_form constructor.
    __original_free_form_init = free_form.__init__

    def _free_form_init(self, t=None):
        """
        Args:
            t: the object that will be used for construction

        Raises:
            ValueError: if the edges of the input :class:`networkx.DiGraph`
              do not all have a ``weight`` attribute, or if any edge weight
              is outside the :math:`\\left[ 0, 1 \\right]` range
            unspecified: any exception thrown by :func:`pygmo.topology.to_networkx()`, or by
              the construction of a :class:`~pygmo.topology` from a UDT

        """
        import networkx as nx

        if t is None:
            # Default ctor.
            __original_free_form_init(self)
        elif type(t) == topology or isinstance(t, nx.DiGraph):
            # Ctors from topology or DiGraph are exposed
            # directly.
            __original_free_form_init(self, t)
        else:
            # If t is neither a topology, nor a DiGraph,
            # assume that it is a UDT.
            __original_free_form_init(self, topology(t))

    setattr(free_form, "__init__", _free_form_init)


def set_serialization_backend(name):
    """Set pygmo's serialization backend.

    This function allows to specify the serialization backend that is used internally by pygmo
    for the (de)serialization of pythonic user-defined entities (e.g., user-defined pythonic
    problems, algorithms, etc.).

    By default, pygmo uses the `cloudpickle <https://github.com/cloudpipe/cloudpickle>`__
    module, which extends the capabilities of the standard :mod:`pickle` module with support
    for lambdas, functions and classes defined interactively in the ``__main__`` module, etc.

    In some specific cases, however, different serialization backends might work better than cloudpickle,
    and thus pygmo provides the possibility for the cognizant user to switch to another
    serialization backend.

    The valid backends are:

    * ``'pickle'`` (i.e., the standard Python :mod:`pickle` module),
    * ``'cloudpickle'``,
    * ``'dill'`` (from the `dill <https://pypi.org/project/dill/>`__ library).

    .. warning::

       Setting the serialization backend is not thread-safe: do **not** set
       the serialization backend while concurrently setting/getting it from another thread,
       or while asynchronous evolutions/optimisations are ongoing.

    Args:
        name (str): the name of the desired backend

    Raises:
        TypeError: if *name* is not a :class:`str`
        ValueError: if *name* is not one of ``['pickle', 'cloudpickle', 'dill']``
        ImportError: if *name* is ``'dill'`` but the dill module is not installed

    """
    if not isinstance(name, str):
        raise TypeError(
            "The serialization backend must be specified as a string, but an object of type {} was provided instead".format(type(name)))
    global _serialization_backend
    if name == "pickle":
        import pickle
        _serialization_backend = pickle
    elif name == "cloudpickle":
        _serialization_backend = _cloudpickle
    elif name == "dill":
        try:
            import dill
            _serialization_backend = dill
        except ImportError:
            raise ImportError(
                "The 'dill' serialization backend was specified, but the dill module is not installed.")
    else:
        raise ValueError(
            "The serialization backend '{}' is not valid. The valid backends are: ['pickle', 'cloudpickle', 'dill']".format(name))


def get_serialization_backend():
    """Get pygmo's serialization backend.

    This function will return pygmo's current serialization backend (see
    :func:`~pygmo.set_serialization_backend()` for an explanation of the
    available backends).

    Returns:
       types.ModuleType: the current serialization backend (as a Python module)

    """
    return _serialization_backend


def _cleanup():
    mp_island.shutdown_pool()
    mp_bfe.shutdown_pool()
    ipyparallel_island.shutdown_view()
    ipyparallel_bfe.shutdown_view()


_atexit.register(_cleanup)
