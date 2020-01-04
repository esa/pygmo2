# Copyright 2019 PaGMO development team
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
#from .plotting import *
#from ._py_islands import *
#from ._py_problems import *
from ._py_bfes import *
# Patch the problem class.
from . import _patch_problem
# Patch the algorithm class.
from . import _patch_algorithm
# Patch the bfe class.
from . import _patch_bfe
# Patch the island class.
# from . import _patch_island
import cloudpickle as _cloudpickle
# Explicitly import the test submodule
from . import test


# We move into the problems, algorithms, etc. namespaces
# all the pure python UDAs, UDPs and UDIs.
# for _item in dir(_py_islands):
#     if not _item.startswith("_"):
#         setattr(islands, _item, getattr(_py_islands, _item))
# del _py_islands

# for _item in dir(_py_problems):
#     if not _item.startswith("_"):
#         setattr(problems, _item, getattr(_py_problems, _item))
# del _py_problems

for _item in dir(_py_bfes):
    if not _item.startswith("_"):
        setattr(batch_evaluators, _item, getattr(_py_bfes, _item))
del _py_bfes

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
       the serialization backend while concurrently setting/getting it from another thread.

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
