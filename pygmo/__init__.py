# Copyright 2019 PaGMO development team
#
# This file is part of the pygmo library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from ._check_deps import *
from .core import *
# Patch the problem class.
from . import _patch_problem
import cloudpickle as _cloudpickle
# Explicitly import the test submodule
from . import test

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
