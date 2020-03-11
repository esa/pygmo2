import random
import warnings

import numpy
from scipy.optimize import NonlinearConstraint, minimize


def _generate_gradient_sparsity_wrapper(
    func, idx, shape, sparsity_func, invert_sign=False
):

    sparsity = sparsity_func()
    sign = 1
    if invert_sign:
        sign = -1

    def wrapper(*args, **kwargs):
        # Here we get back just one one-dimensional gradient, including all dimensions and constraints
        sparse_values = func(*args, **kwargs)
        nnz = len(sparse_values)
        if nnz != len(sparsity):
            raise ValueError(
                "Sparse gradient/hessian has "
                + str(nnz)
                + " non-zeros, but sparsity pattern has "
                + str(len(sparsity))
            )

        result = numpy.zeros(shape)
        for i in range(nnz):
            # filter for just the dimension we need
            if sparsity[i][0] == idx:
                result[sparsity[i][1]] = sign * sparse_values[i]

        return result

    return wrapper


def _generate_hessian_sparsity_wrapper(
    func, idx, shape, sparsity_func, invert_sign=False
):

    sparsity = sparsity_func()[idx]
    sign = 1
    if invert_sign:
        sign = -1

    def wrapper(*args, **kwargs):
        sparse_values = func(*args, **kwargs)[idx]
        nnz = len(sparse_values)
        if nnz != len(sparsity):
            raise ValueError(
                "Sparse gradient/hessian has "
                + str(nnz)
                + " non-zeros, but sparsity pattern has "
                + str(len(sparsity))
            )

        result = numpy.zeros(shape)
        for i in range(nnz):
            result[sparsity[i][0]][sparsity[i][1]] = sign * sparse_values[i]

        return result

    return wrapper


class _fitness_cache:
    def __init__(self, problem):
        self.problem = problem
        self.args = None
        self.kwargs = None
        self.result = None

    def update_cache(self, *args, **kwargs):
        if True or not (self.args == args and self.kwargs == kwargs):
            # print("Updating fitness")
            self.args = args
            self.kwargs = kwargs
            self.result = self.problem.fitness(*args, **kwargs)
        else:
            # print("Keeping cached fitness")
            pass

    def fitness(self, *args, **kwargs):
        self.update_cache(*args, **kwargs)
        return self.result[: self.problem.get_nobj()]

    def generate_eq_constraint(self, i):
        def eqFunc(*args, **kwargs):
            self.update_cache(*args, **kwargs)
            return self.result[self.problem.get_nobj() + i]

        return eqFunc

    def generate_neq_constraint(self, i):
        def neqFunc(*args, **kwargs):
            self.update_cache(*args, **kwargs)
            # In pagmo, inequality constraints have to be negative, in scipy they have to be non-negative.
            return -self.result[self.problem.get_nobj() + self.problem.get_nec() + i]

        return neqFunc


class scipy:
    """
    This class is a user defined algorithm (UDA) providing a wrapper around the function scipy.optimize.minimize.
    The constructor accepts those arguments that are specific to the algorithm:
    - args
    - method
    - tol - the tolerance
    - callback
    - options

    The problem bounds and the existence of a gradient and hessian are deduced calling the relevant problem methods (problem.has_gradient(), etc..)
    """

    def __init__(self, args=(), method=None, tol=None, callback=None, options=None):
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
            raise ValueError("Method not supported: " + method)

        self.args = args
        self.tol = tol
        self.callback = callback
        self.options = options

    def evolve(self, population):
        """
        Take a random member of the population, use it as initial guess
        for calling scipy.optimize.minimize and replace it with the final result.

        Modifies the given population and returns it.
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
            jac = _generate_gradient_sparsity_wrapper(
                problem.gradient, 0, dim, problem.gradient_sparsity
            )

        if problem.has_hessians():
            hess = _generate_hessian_sparsity_wrapper(
                problem.hessians, 0, (dim, dim), problem.hessians_sparsity
            )

        idx = random.randint(0, len(population) - 1)
        if problem.get_nc() > 0:
            # Need to handle constraints, put them in a wrapper to avoid multiple fitness evaluations.
            fitness_wrapper = _fitness_cache(problem)
            constraints = []
            if self.method in ["COBYLA", "SLSQP", None]:
                for i in range(problem.get_nec()):
                    constraint = {
                        "type": "eq",
                        "fun": fitness_wrapper.generate_eq_constraint(i),
                    }

                    if problem.has_gradient():
                        constraint["jac"] = _generate_gradient_sparsity_wrapper(
                            problem.gradient,
                            problem.get_nobj() + i,
                            dim,
                            problem.gradient_sparsity,
                        )

                    constraints.append(constraint)

                for i in range(problem.get_nic()):
                    constraint = {
                        "type": "ineq",
                        "fun": fitness_wrapper.generate_neq_constraint(i),
                    }

                    if problem.has_gradient():
                        constraint["jac"] = _generate_gradient_sparsity_wrapper(
                            problem.gradient,
                            problem.get_nobj() + problem.get_nec() + i,
                            dim,
                            problem.gradient_sparsity,
                            invert_sign=True,
                        )

                    constraints.append(constraint)
            else:
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
                        func = fitness_wrapper.generate_eq_constraint(i)
                        ub = 0
                    else:
                        # Inequality constraint, have to negate the sign
                        func = fitness_wrapper.generate_neq_constraint(
                            i - problem.get_nec()
                        )
                        ub = float("inf")

                    conGrad = None
                    if problem.has_gradient():
                        conGrad = _generate_gradient_sparsity_wrapper(
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
                fitness_wrapper.fitness,
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
        Modifies the 'disp' parameter in the options dict. Every verbosity level above zero sets it to true.
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
