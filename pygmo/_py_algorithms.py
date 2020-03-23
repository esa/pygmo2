# Copyright 2020 PaGMO development team
#
# This file is part of the pygmo library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import random
import warnings

import numpy
from scipy.optimize import NonlinearConstraint, minimize


class scipy:
    """
    This class is a user defined algorithm (UDA) providing a wrapper around the function scipy.optimize.minimize.

    Construction arguments are those options of scipy.optimize.minimize that are not problem-specific. 
    The problem-specific ones, for example the bounds, constraints and the existence of a gradient and hessian,
     are deduced from the problem in the population given to evolve.

    Example:

    >>> import pygmo as pg
    >>> prob = pg.problem(pg.rosenbrock(10))
    >>> pop = pg.population(prob=prob, size=1, seed=0)
    >>> pop.champion_f[0]
    929975.7994682974
    >>> scp = pg.algorithm(pg.scipy(method="L-BFGS-B"))
    >>> result = scp.evolve(pop).champion_f
    >>> result[0]
    1.1377074469794883e-11
    >>> pop.problem.get_fevals()
    55
    >>> pop.problem.get_gevals()
    54
    """

    try:
        from scipy.optimize import minimize

    except ImportError as e:
        raise ImportError(
            "from scipy.optimize import minimize raised an exception, please make sure scipy is installed and reachable. Error: "
            + str(e)
        )

    def _maybe_jit(func):
        try:
            from numba import jit

            return jit(nopython=True)(func)
        except ModuleNotFoundError:
            return func

    @_maybe_jit
    def _unpack_sparse_gradient(
        sparse_values, idx: int, shape, sparsity_pattern, invert_sign: bool = False
    ):
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

    @_maybe_jit
    def _unpack_sparse_hessian(
        sparse_values, idx: int, shape, sparsity_pattern, invert_sign: bool = False
    ):
        nnz = len(sparse_values)
        sign = 1
        if invert_sign:
            sign = -1

        result = numpy.zeros(shape)
        for i in range(nnz):
            result[sparsity_pattern[i][0]][sparsity_pattern[i][1]] = (
                sign * sparse_values[i]
            )

        return result

    def _generate_gradient_sparsity_wrapper(
        func, idx, shape, sparsity_func, invert_sign=False
    ):
        """
        A function to extract a sparse gradient from a pygmo problem to a dense gradient expectecd by scipy.

        Pygmo convention is to include problem constraints into its fitness function. The same applies to the gradient.
        The scipy.optimize.minimize function expects a separate callable for each constraint, this function creates a wrapper that extracts a requested dimension.
        It also transforms the sparse gradient into a dense representation.

        Args:

            func: the gradient callable
            idx: the requested dimension.
            shape: the shape of the result as interpreted by numpy. Should be (dim) for a problem of input dimension dim.
            sparsity_func: a callable giving the sparsity pattern. Use problem.gradient_sparsity.
            invert_sign: whether all values of the gradient should be multiplied with -1. This is necessary for inequality constraints, where the feasible side is interpreted the opposite way by scipy and pygmo.

        Returns:

            a callable that passes all arguments to the gradient callable func and returns the dense gradient at dimension idx

        Raises:

            unspecified: any exception thrown by sparsity_func


        """
        sparsity_pattern = sparsity_func()

        def wrapper(*args, **kwargs):
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
            return scipy._unpack_sparse_gradient(
                sparse_values, idx, shape, sparsity_pattern, invert_sign
            )

        return wrapper

    def _generate_hessian_sparsity_wrapper(
        func, idx, shape, sparsity_func, invert_sign=False
    ):
        """
        A function to extract a hessian gradient from a pygmo problem to a dense hessian expectecd by scipy.

        Pygmo convention is to include problem constraints into its fitness function. The same applies to the hessian
        The scipy.optimize.minimize function expects separate callables for the fitness function and each constraint.
        This function creates a wrapper that extracts a requested dimension and also transforms the sparse hessian into a dense representation.

        Keyword args:

            func: the hessian callable
            idx: the requested dimension.
            shape: the shape of the result as interpreted by numpy. Should be (dim,dim) for a problem of input dimension dim.
            sparsity_func: a callable giving the sparsity pattern. Use problem.hessians_sparsity.
            invert_sign: whether all values of the hessian should be multiplied with -1. This is necessary for inequality constraints, where the feasible side is interpreted the opposite way by scipy and pygmo.

        Returns:

            a callable that passes all arguments to the hessian callable func and returns the dense hessian at dimension idx

        Raises:

            unspecified: any exception thrown by sparsity_func


        """
        sparsity_pattern = sparsity_func()[idx]
        sign = 1
        if invert_sign:
            sign = -1

        def wrapper(*args, **kwargs):
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

            return scipy._unpack_sparse_hessian(
                sparse_values, idx, shape, sparsity_pattern, invert_sign
            )

        return wrapper

    def _generate_eq_constraint(problem, i):
        def eqFunc(*args, **kwargs):
            return problem.fitness(*args, **kwargs)[problem.get_nobj() + i]

        return eqFunc

    def _generate_neq_constraint(problem, i):
        def neqFunc(*args, **kwargs):
            # In pagmo, inequality constraints have to be negative, in scipy they have to be non-negative.
            return -problem.fitness(*args, **kwargs)[
                problem.get_nobj() + problem.get_nec() + i
            ]

        return neqFunc

    def __init__(
        self,
        args=(),
        method: str = None,
        tol: float = None,
        callback=None,
        options: dict = None,
    ) -> None:
        """
            Args:

                args: optional - extra arguments for fitness callable
                method: optional - string specifying the method to be used by scipy. From scipy docs: "If not given, chosen to be one of BFGS, L-BFGS-B, SLSQP, depending if the problem has constraints or bounds."
                tol: optional - tolerance for termination
                callback: optional - callable that is called in each iteration, independent from the fitness function
                options: optional - dict of solver-specific options

            Raises:

                ValueError: If method is not one of Nelder-Mead Powell, CG, BFGS, Newton-CG, L-BFGS-B, TNC, COBYLA, SLSQP, trust-constr, dogleg, trust-ncg, trust-exact, trust-krylov or None.

        """
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
            unspecified: any exception thrown the member functions of the problem
        """
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

        bounds = problem.get_bounds()
        dim = len(bounds[0])
        bounds_seq = [(bounds[0][d], bounds[1][d]) for d in range(dim)]

        jac = None
        hess = None
        if problem.has_gradient():
            jac = scipy._generate_gradient_sparsity_wrapper(
                problem.gradient, 0, dim, problem.gradient_sparsity
            )

        if problem.has_hessians():
            hess = scipy._generate_hessian_sparsity_wrapper(
                problem.hessians, 0, (dim, dim), problem.hessians_sparsity
            )

        idx = random.randint(0, len(population) - 1)
        if problem.get_nc() > 0:
            constraints = []
            if self.method in ["COBYLA", "SLSQP", None]:
                # COBYLYA and SLSQP
                for i in range(problem.get_nec()):
                    constraint = {
                        "type": "eq",
                        "fun": scipy._generate_eq_constraint(problem, i),
                    }

                    if problem.has_gradient():
                        constraint["jac"] = scipy._generate_gradient_sparsity_wrapper(
                            problem.gradient,
                            problem.get_nobj() + i,
                            dim,
                            problem.gradient_sparsity,
                        )

                    constraints.append(constraint)

                for i in range(problem.get_nic()):
                    constraint = {
                        "type": "ineq",
                        "fun": scipy._generate_neq_constraint(problem, i),
                    }

                    if problem.has_gradient():
                        constraint["jac"] = scipy._generate_gradient_sparsity_wrapper(
                            problem.gradient,
                            problem.get_nobj() + problem.get_nec() + i,
                            dim,
                            problem.gradient_sparsity,
                            invert_sign=True,
                        )

                    constraints.append(constraint)
            else:
                # this should be method trust-constr
                if not self.method == "trust-constr":
                    raise ValueError(
                        "Unexpected method with constraints: " + self.method
                    )

                if problem.has_hessians():
                    warnings.warn(
                        "Problem "
                        + problem.get_name()
                        + " has constraints and hessians, but trust-constr requires the callable to also accept lagrange multipliers. Thus, hessians of constraints are ignored."
                    )

                for i in range(problem.get_nc()):
                    func = None
                    ub = 0
                    invert_sign = i >= problem.get_nec()

                    if i < problem.get_nec():
                        # Equality constraint
                        func = scipy._generate_eq_constraint(problem, i)
                        ub = 0
                    else:
                        # Inequality constraint, have to negate the sign
                        func = scipy._generate_neq_constraint(
                            problem, i - problem.get_nec()
                        )
                        ub = float("inf")

                    conGrad = None
                    if problem.has_gradient():
                        conGrad = scipy._generate_gradient_sparsity_wrapper(
                            problem.gradient,
                            problem.get_nobj() + i,
                            dim,
                            problem.gradient_sparsity,
                            invert_sign=invert_sign,
                        )

                    # Constructing the actual constraint objects. All constraints in pygmo are treated as nonlinear.
                    if problem.has_gradient():
                        constraint = NonlinearConstraint(func, 0, ub, jac=conGrad)
                    else:
                        constraint = NonlinearConstraint(func, 0, 0)

                    constraints.append(constraint)

            result = minimize(
                lambda *args, **kwargs: problem.fitness(*args, **kwargs)[
                    : problem.get_nobj()
                ],
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
        else:
            # Case without constraints
            result = minimize(
                problem.fitness,
                population.get_x()[idx],
                args=self.args,
                method=self.method,
                jac=jac,
                hess=hess,
                bounds=bounds_seq,
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
