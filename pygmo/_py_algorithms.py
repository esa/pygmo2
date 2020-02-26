import random

try:
    from scipy.optimize import minimize

    class scipy:
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
            problem = population.problem

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
            population.set_xf(idx, result.x, result.fun)
            return population

        def get_name(self):
            if self.method is not None:
                return "SciPy implementation of " + self.method
            else:
                return "Wrapper around scipy.optimize.minimize, method unspecified."


except ImportError:
    pass
