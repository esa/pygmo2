# Copyright 2020 PaGMO development team
#
# This file is part of the pygmo library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import typing

from . import s_policy, select_best


class scipy_optimize:
    """
    This class is a user defined algorithm (UDA) providing a wrapper around the function :func:`scipy.optimize.minimize`.

    This wraps several well-known local optimization algorithms:

     - Nelder-Mead
     - Powell
     - CG
     - BFGS
     - Newton-CG
     - L-BFGS-B
     - TNC
     - COBYLA
     - SLSQP
     - trust-constr
     - dogleg
     - trust-ncg
     - trust-exact
     - trust-krylov

    These methods are mostly variants of gradient descent. Some of them require a gradient and will throw
    an error if invoked on a problem that does not offer one.
    Constraints are only supported by methods COBYLA, SLSQP and trust-constr.

    Example:

    >>> import pygmo as pg
    >>> prob = pg.problem(pg.rosenbrock(10))
    >>> pop = pg.population(prob=prob, size=1, seed=0)
    >>> pop.champion_f[0]
    929975.7994682974
    >>> scp = pg.algorithm(pg.scipy_optimize(method="L-BFGS-B"))
    >>> result = scp.evolve(pop).champion_f
    >>> result[0] # doctest: +ELLIPSIS
    1.13770...
    >>> pop.problem.get_fevals()
    55
    >>> pop.problem.get_gevals()
    54
    """

    class _fitness_wrapper:
        """
        A helper class to prevent redundant evaluations of the fitness function.

        In pygmo, the constraints of a problem are part of the fitness function, while scipy requires a separate
        callable for each constraint. This class provides separate callables which use a common cache.
        """

        def __init__(self, problem) -> None:
            self.problem = problem
            self.last_x = None
            self.last_fitness = None
            self.last_gradient_x = None
            self.last_gradient_result = None

        def _update_cache(self, x, *args, **kwargs) -> None:
            if self.last_x is None or not all(self.last_x == x):
                self.last_x = x.copy()
                self.last_fitness = self.problem.fitness(x, *args, **kwargs)

        def _update_gradient_cache(self, x, *args, **kwargs) -> None:
            if self.last_gradient_x is None or not all(self.last_gradient_x == x):
                self.last_gradient_x = x.copy()
                self.last_gradient_result = self.problem.gradient(
                    x, *args, **kwargs)

        def get_fitness_func(self):
            if self.problem.get_nc() == 0:
                return self.problem.fitness
            else:

                def fitness(x, *args, **kwargs):
                    self._update_cache(x, *args, **kwargs)
                    result = self.last_fitness[: self.problem.get_nobj()]
                    return result

                return fitness

        def get_eq_func(self, idx: int):
            def eq_func(x, *args, **kwargs):
                self._update_cache(x, *args, **kwargs)
                result = self.last_fitness[self.problem.get_nobj() + idx]
                return result

            return eq_func

        def get_neq_func(self, idx: int):
            def neq_func(x, *args, **kwargs):
                self._update_cache(x, *args, **kwargs)
                # In pagmo, inequality constraints have to be negative, in scipy they have to be non-negative.
                result = -self.last_fitness[
                    self.problem.get_nobj() + self.problem.get_nec() + idx
                ]
                return result

            return neq_func

        def get_gradient_func(self):
            if self.problem.get_nc() == 0:
                return self.problem.gradient
            else:

                def gradient_func(x, *args, **kwargs):
                    self._update_gradient_cache(x, *args, **kwargs)
                    result = self.last_gradient_result
                    return result

                return gradient_func

        @staticmethod
        def _unpack_sparse_gradient(
            sparse_values: typing.Mapping[int, float],
            idx: int,
            shape: typing.Tuple[int],
            sparsity_pattern,
            invert_sign: bool = False,
        ):
            import numpy

            nnz = len(sparse_values)
            sign = 1
            if invert_sign:
                sign = -1

            result = numpy.zeros(shape)
            for i in range(nnz):
                # filter for just the dimension we need
                if sparsity_pattern[i][0] == idx:
                    result[sparsity_pattern[i][1]] = sign * sparse_values[i]

            return result

        def _generate_gradient_sparsity_wrapper(self, idx: int):
            """
            A function to extract a sparse gradient from a pygmo problem to a dense gradient expectecd by scipy.

            Pygmo convention is to include problem constraints into its fitness function. The same applies to the gradient.
            The scipy.optimize.minimize function expects a separate callable for each constraint, this function creates a wrapper that extracts a requested dimension.
            It also transforms the sparse gradient into a dense representation.

            Args:

                idx: the requested dimension.

            Returns:

                a callable that dense gradient at dimension idx

            Raises:

                unspecified: any exception thrown by self.problem.gradient_sparsity()


            """
            import numpy

            sparsity_pattern = self.problem.gradient_sparsity()
            func = self.get_gradient_func()
            dim: int = len(self.problem.get_bounds()[0])
            invert_sign: bool = (
                idx >= self.problem.get_nobj() + self.problem.get_nec()
            )

            if idx < 0 or idx >= self.problem.get_nf():
                raise ValueError(
                    "Invalid dimensions index "
                    + str(idx)
                    + " for problem of fitness dimension "
                    + str(self.problem.get_nf())
                )

            def wrapper(*args, **kwargs) -> numpy.ndarray:
                """
                Calls the gradient callable and returns dense representation along a fixed dimension

                Args:

                    args: arguments for callable
                    kwargs: keyword arguments for callable

                Returns:

                    dense representation of gradient

                Raises:

                    ValueError: If number of non-zeros in gradient and sparsity pattern disagree
                    unspecified: any exception thrown by wrapped callable

                """
                sparse_values = func(*args, **kwargs)
                nnz = len(sparse_values)
                if nnz != len(sparsity_pattern):
                    raise ValueError(
                        "Sparse gradient has "
                        + str(nnz)
                        + " non-zeros, but sparsity pattern has "
                        + str(len(sparsity_pattern))
                    )
                return scipy_optimize._fitness_wrapper._unpack_sparse_gradient(
                    sparse_values, idx, dim, sparsity_pattern, invert_sign
                )

            return wrapper

        @staticmethod
        def _unpack_sparse_hessian(
            sparse_values: typing.Mapping[int, float],
            idx: int,
            shape: typing.Tuple[int, int],
            sparsity_pattern,
            invert_sign: bool = False,
        ):

            import numpy

            nnz = len(sparse_values)
            sign = 1
            if invert_sign:
                sign = -1

            result = numpy.zeros(shape)
            for i in range(nnz):
                result[sparsity_pattern[i][0]][sparsity_pattern[i][1]] = (
                    sign * sparse_values[i]
                )
                # symmetrize matrix. Decided against a check for redundancy,
                # since branching within the loop is too expensive
                result[sparsity_pattern[i][1]][sparsity_pattern[i][0]] = (
                    sign * sparse_values[i]
                )

            return result

        def _generate_hessian_sparsity_wrapper(self, idx: int):
            """
            A function to extract a hessian gradient from a pygmo problem to a dense hessian expectecd by scipy.

            Pygmo convention is to include problem constraints into its fitness function. The same applies to the hessian
            The scipy.optimize.minimize function expects separate callables for the fitness function and each constraint.
            This function creates a wrapper that extracts a requested dimension and also transforms the sparse hessian into a dense representation.

            Keyword args:

                idx: the requested dimension.

            Returns:

                a callable that passes all arguments to the hessian callable and returns the dense hessian at dimension idx

            Raises:

                unspecified: any exception thrown by self.problem.hessian_sparsity()


            """
            import numpy

            if idx < 0 or idx >= self.problem.get_nf():
                raise ValueError(
                    "Invalid dimensions index "
                    + str(idx)
                    + " for problem fitness dimension "
                    + str(self.problem.get_nf())
                )

            sparsity_pattern = self.problem.hessians_sparsity()[idx]
            func = self.problem.hessians
            dim: int = len(self.problem.get_bounds()[0])
            invert_sign: bool = (
                idx >= self.problem.get_nobj() + self.problem.get_nec()
            )
            shape: typing.Tuple[int, int] = (dim, dim)

            def wrapper(*args, **kwargs) -> numpy.ndarray:
                """
                Calls the hessian callable and returns dense representation along a fixed dimension

                Args:

                    args: arguments for callable
                    kwargs: keyword arguments for callable

                Returns:

                    dense representation of hessian

                Raises:

                    ValueError: If number of non-zeros in hessian and sparsity pattern disagree
                    unspecified: any exception thrown by wrapped callable

                """
                sparse_values = func(*args, **kwargs)[idx]
                nnz = len(sparse_values)
                if nnz != len(sparsity_pattern):
                    raise ValueError(
                        "Sparse hessian has "
                        + str(nnz)
                        + " non-zeros, but sparsity pattern has "
                        + str(len(sparsity_pattern))
                    )

                return scipy_optimize._fitness_wrapper._unpack_sparse_hessian(
                    sparse_values, idx, shape, sparsity_pattern, invert_sign
                )

            return wrapper

    def __init__(
        self,
        args=(),
        method: str = None,
        tol: float = None,
        callback: typing.Optional[typing.Callable[[
            typing.Any], typing.Any]] = None,
        options: typing.Optional[typing.MutableMapping[str,
                                                       typing.Any]] = None,
        selection: s_policy = s_policy(select_best(rate=1)),
    ) -> None:
        """
            The constructor initializes a wrapper instance for a specific algorithm.
            Construction arguments are those options of :func:`scipy.optimize.minimize` that are not problem-specific.
            Problem-specific options, for example the bounds, constraints and the existence of a gradient and hessian,
            are deduced from the problem in the population given to the evolve function.

            Args:

                args: optional - extra arguments for fitness callable
                method: optional - string specifying the method to be used by scipy. From scipy docs: "If not given, chosen to be one of BFGS, L-BFGS-B, SLSQP, depending if the problem has constraints or bounds."
                tol: optional - tolerance for termination
                callback: optional - callable that is called in each iteration, independent from the fitness function
                options: optional - dict of solver-specific options
                selection: optional - s_policy to select candidate for local optimization

            Raises:

                ValueError: If method is not one of Nelder-Mead Powell, CG, BFGS, Newton-CG, L-BFGS-B, TNC, COBYLA, SLSQP, trust-constr, dogleg, trust-ncg, trust-exact, trust-krylov or None.

        """

        try:
            from scipy.optimize import minimize, NonlinearConstraint

        except ImportError as e:
            raise ImportError(
                "from scipy.optimize import minimize raised an exception, please make sure scipy is installed and reachable. Error: "
                + str(e)
            )

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
        ]
        if method in method_list + [None]:
            self.method = method
        else:
            raise ValueError(
                "Method "
                + str(method)
                + " not supported, only the following "
                + str(method_list)
            )

        self.args = args
        self.tol = tol
        self.callback = callback
        self.options = options
        self.selection = selection

    def evolve(self, population):
        """
        Call scipy.optimize.minimize with a random member of the population as start value.

        The problem is extracted from the population and its fitness function gives the objective value for the optimization process.

        Args:

            population: The population containing the problem and a set of initial solutions.

        Returns:

            The changed population.

        Raises:

            ValueError: If the problem has constraints, but during construction a method was selected that cannot deal with them.
            ValueError: If the problem contains multiple objectives
            ValueError: If the problem is stochastic
            unspecified: any exception thrown by the member functions of the problem
        """

        from scipy.optimize import minimize, NonlinearConstraint

        problem = population.problem

        if problem.get_nc() > 0 and self.method not in [
            "COBYLA",
            "SLSQP",
            "trust-constr",
            None,
        ]:
            raise ValueError(
                "Problem "
                + problem.get_name()
                + " has constraints. Constraints are not implemented for method "
                + str(self.method)
                + ", they are only implemented for methods COBYLA, SLSQP and trust-constr."
            )

        if problem.get_nobj() > 1:
            raise ValueError(
                "Multiple objectives detected in "
                + problem.get_name()
                + " instance. The wrapped scipy.optimize.minimize cannot deal with them"
            )

        if problem.is_stochastic():
            raise ValueError(
                problem.get_name()
                + " appears to be stochastic, the wrapped scipy.optimize.minimize cannot deal with it"
            )

        if problem.get_nec() > 0 and self.method == "COBYLA":
            raise ValueError(
                problem.get_name()
                + " has equality constraints, but selected method COBYLA only supports inequality constraints."
            )

        bounds = problem.get_bounds()
        dim = len(bounds[0])
        bounds_seq = [(bounds[0][d], bounds[1][d]) for d in range(dim)]

        fitness_wrapper = scipy_optimize._fitness_wrapper(problem)

        jac = None
        hess = None
        if problem.has_gradient():
            jac = fitness_wrapper._generate_gradient_sparsity_wrapper(0)

        if problem.has_hessians():
            hess = fitness_wrapper._generate_hessian_sparsity_wrapper(0)

        constraints = ()  # default argument, implying an unconstrained problem

        selected = self.selection.select(
            (population.get_ID(), population.get_x(), population.get_f()),
            problem.get_nx(),
            problem.get_nix(),
            problem.get_nobj(),
            problem.get_nec(),
            problem.get_nic(),
            problem.c_tol,
        )

        if len(selected[0]) != 1:
            raise ValueError(
                "Selection policy returned "
                + str(len(selected[0]))
                + " elements, but 1 was needed."
            )

        idx = list(population.get_ID()).index(selected[0][0])

        if problem.get_nc() > 0:
            # translate constraints into right format
            constraints = []
            if self.method in ["COBYLA", "SLSQP", None]:
                # COBYLYA and SLSQP
                for i in range(problem.get_nec() + problem.get_nic()):
                    constraint = dict()
                    if i < problem.get_nec():
                        constraint["type"] = "eq"
                        constraint["fun"] = fitness_wrapper.get_eq_func(i)
                    else:
                        constraint["type"] = "ineq"
                        constraint["fun"] = fitness_wrapper.get_neq_func(
                            i - problem.get_nec()
                        )

                    if problem.has_gradient():
                        # extract gradient of constraint
                        constraint[
                            "jac"
                        ] = fitness_wrapper._generate_gradient_sparsity_wrapper(
                            problem.get_nobj() + i
                        )

                    constraints.append(constraint)
            else:
                # this should be method trust-constr
                if not self.method == "trust-constr":
                    raise ValueError(
                        "Unexpected method with constraints: " + self.method
                    )

                if problem.has_hessians():
                    import warnings

                    warnings.warn(
                        "Problem "
                        + problem.get_name()
                        + " has constraints and hessians, but trust-constr requires the callable to also accept lagrange multipliers. Thus, hessians of constraints are ignored."
                    )

                for i in range(problem.get_nc()):
                    func = None
                    ub = 0

                    if i < problem.get_nec():
                        # Equality constraint
                        func = fitness_wrapper.get_eq_func(i)
                        ub = 0
                    else:
                        # Inequality constraint
                        func = fitness_wrapper.get_neq_func(
                            i - problem.get_nec())
                        ub = float("inf")

                    # Constructing the actual constraint objects. All constraints in pygmo are treated as nonlinear.
                    if problem.has_gradient():
                        conGrad = fitness_wrapper._generate_gradient_sparsity_wrapper(
                            problem.get_nobj() + i,
                        )
                        constraint = NonlinearConstraint(
                            func, 0, ub, jac=conGrad)
                    else:
                        constraint = NonlinearConstraint(func, 0, 0)

                    constraints.append(constraint)

        # Call scipy minimizer
        result = minimize(
            fitness_wrapper.get_fitness_func(),
            population.get_x()[idx],
            args=self.args,
            method=self.method,
            jac=jac,
            hess=hess,
            bounds=bounds_seq,
            constraints=constraints,
            tol=self.tol,
            callback=self.callback,
            options=self.options,
        )

        # wrap result in array if necessary
        fun = result.fun
        try:
            iter(fun)
        except TypeError:
            fun = [fun]

        if problem.get_nc() > 0:
            # the constraint values are not reported, so we cannot set them
            population.set_x(idx, result.x)
        else:
            population.set_xf(idx, result.x, fun)
        return population

    def get_name(self) -> str:
        """
        Returns the method name if one was selected, scipy.optimize.minimize otherwise
        """
        if self.method is not None:
            return self.method + ", provided by SciPy"
        else:
            return "scipy.optimize.minimize, method unspecified."

    def set_verbosity(self, level: int) -> None:
        """
        Modifies the 'disp' parameter in the options dict, which prints out a final convergence message.

        Args:

            level: Every verbosity level above zero prints out a convergence message.

        Raises:

            ValueError: If options dict was given in instance constructor and has options conflicting with verbosity level

        """
        if level > 0:
            if self.options is None:
                self.options = dict()

            if "disp" in self.options and self.options["disp"] is False:
                raise ValueError(
                    "Conflicting options: Verbosity set to "
                    + str(level)
                    + ", but disp to False"
                )

            self.options["disp"] = True

        if level <= 0:
            if self.options is not None:
                self.options.pop("disp", None)
