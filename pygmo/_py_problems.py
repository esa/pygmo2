# Copyright 2020, 2021 PaGMO development team
#
# This file is part of the pygmo library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Import the unconstrain meta-problem so that we can re-use
# the docstring of its inner_problem property in the documentation
# of the inner_problem property of decorator_problem.
from .core import unconstrain as _unconstrain
from typing import List, Union


def _with_decorator(f):
    # A decorator that will decorate the input method f of a decorator_problem
    # with one of the decorators stored inside the problem itself, in the _decors
    # dictionary.
    from functools import wraps

    @wraps(f)
    def wrapper(self, *args, **kwds):
        dec = self._decors.get(f.__name__)
        if dec is None:
            return f(self, *args, **kwds)
        else:
            return dec(f)(self, *args, **kwds)

    return wrapper


def _add_doc(value):
    # Small decorator for changing the docstring
    # of a function to 'value'. See:
    # https://stackoverflow.com/questions/4056983/how-do-i-programmatically-set-the-docstring
    def _doc(func):
        func.__doc__ = value
        return func

    return _doc


class decorator_problem(object):
    """Decorator meta-problem.

    .. versionadded:: 2.9

    This meta-problem allows to apply arbitrary transformations to the functions
    of a PyGMO :class:`~pygmo.problem` via Python decorators.

    The decorators are passed as keyword arguments during initialisation, and they
    must be named after the function they are meant to decorate plus the
    ``_decorator`` suffix. For instance, we can define a minimal decorator
    for the fitness function as follows:

    >>> def f_decor(orig_fitness_function):
    ...     def new_fitness_function(self, dv):
    ...         print("Evaluating dv: {}".format(dv))
    ...         return orig_fitness_function(self, dv)
    ...     return new_fitness_function

    This decorator will print the input decision vector *dv* before invoking the
    original fitness function. We can then construct a decorated Rosenbrock problem
    as follows:

    >>> from pygmo import decorator_problem, problem, rosenbrock
    >>> dprob = problem(decorator_problem(rosenbrock(), fitness_decorator=f_decor))

    We can then verify that calling the fitness function of *dprob* will print
    the decision vector before returning the fitness value:

    >>> fv = dprob.fitness([1, 2])
    Evaluating dv: [1. 2.]
    >>> print(fv)
    [100.]

    An extended :ref:`tutorial <py_tutorial_udp_meta_decorator>` on the usage of this class is available
    in PyGMO's documentation.

    All the functions in the public API of a UDP can be decorated (see the documentation
    of :class:`pygmo.problem` for the full list). Note that the public API of :class:`~pygmo.decorator_problem`
    includes the UDP public API: there is a ``fitness()`` method, methods to query the problem's properties,
    sparsity-related methods, etc. In order to avoid duplication, we do not repeat here the documentation of
    the UDP API and we document instead only the few methods which are specific to :class:`~pygmo.decorator_problem`.
    Users can refer to the documentation of :class:`pygmo.problem` for detailed information on the UDP API.

    Both *prob* and the decorators will be deep-copied inside the instance upon construction. As
    usually done in meta-problems, this class will store as an internal data member a :class:`~pygmo.problem`
    containing a copy of *prob* (this is commonly referred to as the *inner problem* of the
    meta-problem). The inner problem is accessible via the :attr:`~pygmo.decorator_problem.inner_problem`
    read-only property.

    """

    def __init__(self, prob=None, **kwargs):
        """
        Args:

           prob: a :class:`~pygmo.problem` or a user-defined problem, either C++ or Python (if
              *prob* is :data:`None`, a :class:`~pygmo.null_problem` will be used in its stead)
           kwargs: the dictionary of decorators to be applied to the functions of the input problem

        Raises:

           TypeError: if at least one of the values in *kwargs* is not callable
           unspecified: any exception thrown by the constructor of :class:`~pygmo.problem` or the deep copy
              of *prob* or *kwargs*

        """
        from . import problem, null_problem
        from warnings import warn
        from copy import deepcopy

        if prob is None:
            prob = null_problem()
        if type(prob) == problem:
            # If prob is a pygmo problem, we will make a copy
            # and store it. The copy is to ensure consistent behaviour
            # with the other meta problems and with the constructor
            # from a UDP (which will end up making a deep copy of
            # the input object).
            self._prob = deepcopy(prob)
        else:
            # Otherwise, we attempt to create a problem from it. This will
            # work if prob is an exposed C++ problem or a Python UDP.
            self._prob = problem(prob)
        self._decors = {}
        for k in kwargs:
            if k.endswith("_decorator"):
                if not callable(kwargs[k]):
                    raise TypeError(
                        "Cannot register the decorator for the '{}' method: the supplied object "
                        "'{}' is not callable.".format(k[:-10], kwargs[k])
                    )
                self._decors[k[:-10]] = deepcopy(kwargs[k])
            else:
                warn(
                    "A keyword argument without the '_decorator' suffix, '{}', was used in the "
                    "construction of a decorator problem. This keyword argument will be ignored.".format(
                        k
                    )
                )

    @_with_decorator
    def fitness(self, dv):
        return self._prob.fitness(dv)

    @_with_decorator
    def batch_fitness(self, dvs):
        return self._prob.batch_fitness(dvs)

    @_with_decorator
    def has_batch_fitness(self):
        return self._prob.has_batch_fitness()

    @_with_decorator
    def get_bounds(self):
        return self._prob.get_bounds()

    @_with_decorator
    def get_nobj(self):
        return self._prob.get_nobj()

    @_with_decorator
    def get_nec(self):
        return self._prob.get_nec()

    @_with_decorator
    def get_nic(self):
        return self._prob.get_nic()

    @_with_decorator
    def get_nix(self):
        return self._prob.get_nix()

    @_with_decorator
    def has_gradient(self):
        return self._prob.has_gradient()

    @_with_decorator
    def gradient(self, dv):
        return self._prob.gradient(dv)

    @_with_decorator
    def has_gradient_sparsity(self):
        return self._prob.has_gradient_sparsity()

    @_with_decorator
    def gradient_sparsity(self):
        return self._prob.gradient_sparsity()

    @_with_decorator
    def has_hessians(self):
        return self._prob.has_hessians()

    @_with_decorator
    def hessians(self, dv):
        return self._prob.hessians(dv)

    @_with_decorator
    def has_hessians_sparsity(self):
        return self._prob.has_hessians_sparsity()

    @_with_decorator
    def hessians_sparsity(self):
        return self._prob.hessians_sparsity()

    @_with_decorator
    def has_set_seed(self):
        return self._prob.has_set_seed()

    @_with_decorator
    def set_seed(self, s):
        return self._prob.set_seed(s)

    @_with_decorator
    def get_name(self):
        return self._prob.get_name() + " [decorated]"

    @_with_decorator
    def get_extra_info(self):
        retval = self._prob.get_extra_info()
        if len(self._decors) == 0:
            retval += "\tNo registered decorators.\n"
        else:
            retval += "\tRegistered decorators:\n"
            for i, k in enumerate(self._decors):
                retval += "\t\t" + k + (",\n" if i < len(self._decors) - 1 else "")
            retval += "\n"
        return retval

    @property
    @_add_doc(_unconstrain.inner_problem.__doc__)
    def inner_problem(self):
        return self._prob

    def get_decorator(self, fname):
        """Get the decorator for the function called *fname*.

        This method will return a copy of the decorator that has been registered upon construction
        for the function called *fname*. If no decorator for *fname* has been specified during
        construction, :data:`None` will be returned.

        >>> from pygmo import decorator_problem, problem, rosenbrock
        >>> def f_decor(orig_fitness_function):
        ...     def new_fitness_function(self, dv):
        ...         print("Evaluating dv: {}".format(dv))
        ...         return orig_fitness_function(self, dv)
        ...     return new_fitness_function
        >>> dprob = decorator_problem(rosenbrock(), fitness_decorator=f_decor)
        >>> dprob.get_decorator("fitness") # doctest: +ELLIPSIS
        <function ...>
        >>> dprob.get_decorator("gradient") is None
        True

        Args:

           fname(str): the name of the function whose decorator will be returned

        Returns:

            a copy of the decorator registered for *fname*, or :data:`None` if no decorator for *fname* has been registered

        Raises:

           TypeError: if *fname* is not a string
           unspecified: any exception thrown by the deep copying of the decorator for *fname*

        """
        if not isinstance(fname, str):
            raise TypeError(
                "The input parameter 'fname' must be a string, but it is of type '{}' instead.".format(
                    type(fname)
                )
            )
        from copy import deepcopy

        return deepcopy(self._decors.get(fname))


class constant_arguments:
    """Meta problem that sets some arguments of the original problem to constants

    .. versionadded:: 2.19

    If good values for some of the dimensions of a problem are known, this
    wrapper allows to reduce the dimensions of the search space and is an alternative
    to restricting a value using identical lower and upper bounds.

    We can construct an instance of this problem by passing the original problem,
    and a list of containing one entry for each argument, either the fixed argument
    for this dimension or None if the argument should remain free:

    >>> from pygmo import constant_arguments, problem, rosenbrock
    >>> cprob = problem(constant_arguments(rosenbrock(dim=3), fixed_arguments=[1, None, None]))

    We now see that the new problem has two dimensions, since the original problem had three and we fixed one:

    >>> cprob.get_nx()
    2

    """

    def __init__(self, prob, fixed_arguments: List[Union[float, None]]):
        """
        Args:

           prob: a :class:`~pygmo.problem` or a user-defined problem, either C++ or Python (if
              *prob* is :data:`None`, a :class:`~pygmo.null_problem` will be used in its stead)
           fixed_arguments: a list of values, one for each dimension of the wrapped problem.
               Each value should be either a float, if the argument should be fixed to this value,
               or None, if it should remain free

        Raises:

           ValueError: if the lengths of fixed_arguments differs from the number of dimensions of the wrapped problem
           ValueError: if any of the fixed arguments violate the bounds of the wrapped problem
           ValueError: if a problem with nix() > 0 is passed
           unspecified: any exception thrown by the constructor of :class:`~pygmo.problem` or the deep copy
              of *prob*
        """

        from . import problem, null_problem
        from copy import deepcopy

        if prob is None:
            prob = null_problem()
        if type(prob) == problem:
            # If prob is a pygmo problem, we will make a copy
            # and store it. The copy is to ensure consistent behaviour
            # with the other meta problems and with the constructor
            # from a UDP (which will end up making a deep copy of
            # the input object).
            self._problem = deepcopy(prob)
        else:
            # Otherwise, we attempt to create a problem from it. This will
            # work if prob is an exposed C++ problem or a Python UDP.
            self._problem = problem(prob)

        minBound, maxBound = self._problem.get_bounds()

        self.full_dim = self._problem.get_nx()

        if len(fixed_arguments) != self.full_dim:
            raise ValueError(
                "Got {} argument array for problem of dimension {}".format(
                    len(fixed_arguments), self.full_dim
                )
            )

        if self._problem.get_nix() > 0:
            raise ValueError("Mixed integer-problems not yet supported.")

        self.minBound = []
        self.maxBound = []

        for i in range(self.full_dim):
            arg = fixed_arguments[i]

            if arg is None:
                # free variable
                self.minBound.append(minBound[i])
                self.maxBound.append(maxBound[i])
            else:
                # fixed variable
                if not arg >= minBound[i]:
                    raise ValueError(
                        "Fixed argument {} violates min bound {}".format(
                            arg, minBound[i]
                        )
                    )
                if not arg <= maxBound[i]:
                    raise ValueError(
                        "Fixed argument {} violates max bound {}".format(
                            arg, maxBound[i]
                        )
                    )

        # converting to internal format
        self.fixed_arguments = [elem for elem in fixed_arguments if elem is not None]
        self.fixed_flags = [elem is not None for elem in fixed_arguments]

        assert len(self.minBound) + sum(self.fixed_flags) == self.full_dim

    def get_bounds(self):
        return (self.minBound, self.maxBound)

    def get_nobj(self):
        return self._problem.get_nobj()

    def get_nec(self):
        return self._problem.get_nec()

    def get_nic(self):
        return self._problem.get_nic()

    def get_nc(self):
        return self._problem.get_nc()

    def get_nx(self):
        return len(self.minBound)

    def get_nix(self):
        return 0

    def fitness(self, x) -> List[float]:
        return self._problem.fitness(self.get_full_x(x))

    def has_batch_fitness(self):
        return self._problem.has_batch_fitness()

    def batch_fitness(self, dvs):
        contiguous_x = []
        if len(dvs) % self.get_nx() != 0:
            raise ValueError(
                "Expected multiple of {} but got {}".format(self.get_nx(), len(dvs))
            )
        num_dvs = len(dvs) // self.get_nx()
        for i in range(num_dvs):
            begin_index = i * self.get_nx()
            end_index = (i + 1) * self.get_nx()
            contiguous_x.extend(self.get_full_x(dvs[begin_index:end_index]))
        return self._problem.batch_fitness(contiguous_x)

    def get_full_x(self, x) -> List[float]:
        """Get the full x for a given x of lower dimension"""

        if len(x) != len(self.minBound):
            raise ValueError(
                "Got x of length {} but expected {}".format(len(x), self.get_nx())
            )

        fullx = [None for i in range(self.full_dim)]

        j = 0
        k = 0
        for i in range(self.full_dim):
            if self.fixed_flags[i]:
                fullx[i] = self.fixed_arguments[j]
                j += 1
            else:
                fullx[i] = x[k]
                k += 1

        assert j + k == self.full_dim
        assert all(elem is not None for elem in fullx)
        return fullx
