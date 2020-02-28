import random

try:
    from scipy.optimize import minimize

    class scipy:
        """
        This class is a user defined algorithm (UDA) providing a wrapper around the function scipy.optimize.minimize.
        The constructor accepts those arguments that are specific to the algorithm:
        - args
        - method
        - tol - the tolerance
        - callback
        - options

        Other aspects, like bounds or the existence of a gradient and hessian, are taken from the problem.
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

            if problem.get_nc() > 0:
                raise NotImplementedError(
                    "Constraints not yet supported in SciPy wrapper, as they differ among methods."
                )

            if problem.get_nobj() > 1:
                raise NotImplementedError("Multiple objectives not supported.")

            jac = None
            hess = None
            if problem.has_gradient():
                jac = problem.gradient

            if problem.has_hessians():
                hess = problem.hessians

            bounds = problem.get_bounds()
            dim = len(bounds[0])
            bounds_seq = [(bounds[0][d], bounds[1][d]) for d in range(dim)]

            idx = random.randint(0, len(population) - 1)
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


except ImportError:
    pass
