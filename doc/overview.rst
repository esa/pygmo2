Capabilities
============

Overview
--------

* Support for a wide array of types of
  optimisation problems (continuous, integer, single
  and multi-objective, constrained and unconstrained,
  with or without derivatives, stochastic, etc.).
* A comprehensive library of algorithms,
  including global and local solvers, meta-heuristics,
  single and multi-objective algorithms,
  wrappers for third-party solvers (e.g.,
  `NLopt <https://nlopt.readthedocs.io/en/latest/>`__,
  `Ipopt <https://coin-or.github.io/Ipopt/>`__,
  `SciPy <https://www.scipy.org/>`__, etc.).
* Comprehensive support for coarse-grained
  parallelisation via the
  `generalised island model <https://link.springer.com/chapter/10.1007/978-3-642-28789-3_7>`__.
  In the island model, multiple optimisation instances
  run in parallel (possibly on different machines) and
  exchange information as the optimisation proceeds,
  improving the overall time-to-solution and allowing
  to harness the computational power of modern computer
  architectures (including massively-parallel
  high-performance clusters).
* Support for fine-grained parallelisation
  (i.e., at the level of single objective function
  evaluations) in selected algorithms via the batch
  fitness evaluation framework. This allows to
  speed-up single optimisations via parallel
  processing (e.g., multithreading, high-performance
  clusters, GPUs, SIMD vectorization, etc.).
* A library of ready-to-use optimisation problems
  for algorithmic testing and performance evaluation
  (Rosenbrock, Rastrigin, Lennard-Jones, etc.).
* A library of optimisation-oriented utilities
  (e.g., hypervolume computation, non-dominated
  sorting, plotting, etc.).

.. _available_algorithms:

List of algorithms
------------------

This is the list of user defined algorithms (UDAs) currently
provided with pygmo. These are classes that
can be used to construct a :class:`pygmo.algorithm`, which will
then provide a unified interface to access the algorithm's functionalities.

Generally speaking, algorithms can solve only specific problem classes.
In the tables below, we use the following
flags to signal which problem types an algorithm can solve:

* S = Single-objective
* M = Multi-objective
* C = Constrained
* U = Unconstrained
* I = Integer programming
* sto = Stochastic

Note that algorithms that do not directly support integer
programming will still work on integer problems
(i.e., they will optimise the relaxed problem).
Note also that it is possible to use :ref:`meta-problems <available_meta_problems>`
to turn constrained problems into unconstrained ones,
and multi-objective problems into single-objective ones.

Heuristic Global Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
============================================================ ========================================= =========================
Common Name                                                  Docs of the python class                  Capabilities
============================================================ ========================================= =========================
Extended Ant Colony Optimization (GACO)                      :class:`pygmo.gaco`                       S-CU-I
Differential Evolution (DE)                                  :class:`pygmo.de`                         S-U
Self-adaptive DE (jDE and iDE)                               :class:`pygmo.sade`                       S-U
Self-adaptive DE (de_1220 aka pDE)                           :class:`pygmo.de1220`                     S-U
Grey wolf optimizer (GWO)                                    :class:`pygmo.gwo`                        S-U
Improved Harmony Search                                      :class:`pygmo.ihs`                        SM-CU-I
Particle Swarm Optimization (PSO)                            :class:`pygmo.pso`                        S-U
Particle Swarm Optimization Generational (GPSO)              :class:`pygmo.pso_gen`                    S-U-sto
(N+1)-ES Simple Evolutionary Algorithm                       :class:`pygmo.sea`                        S-U-sto
Simple Genetic Algorithm                                     :class:`pygmo.sga`                        S-U-I-sto
Corana's Simulated Annealing (SA)                            :class:`pygmo.simulated_annealing`        S-U
Artificial Bee Colony (ABC)                                  :class:`pygmo.bee_colony`                 S-U
Covariance Matrix Adaptation Evo. Strategy (CMA-ES)          :class:`pygmo.cmaes`                      S-U-sto
Exponential Evolution Strategies (xNES)                      :class:`pygmo.xnes`                       S-U-sto
Non-dominated Sorting GA (NSGA2)                             :class:`pygmo.nsga2`                      M-U-I
Multi-objective EA with Decomposition (MOEA/D)               :class:`pygmo.moead`                      M-U
Multi-objective EA with Decomposition Generational (GMOEA/D) :class:`pygmo.moead_gen`                  M-U
Multi-objective Hypervolume-based ACO (MHACO)                :class:`pygmo.maco`                       M-U-I
Non-dominated Sorting PSO (NSPSO)                            :class:`pygmo.nspso`                      M-U
============================================================ ========================================= =========================

Local optimization
^^^^^^^^^^^^^^^^^^
====================================================== ========================================================================================= ===============
Common Name                                            Docs of the python class                                                                  Capabilities
====================================================== ========================================================================================= ===============
Compass Search (CS)                                    :class:`pygmo.compass_search`                                                             S-CU
COBYLA (from NLopt)                                    :class:`pygmo.nlopt`                                                                      S-CU
BOBYQA (from NLopt)                                    :class:`pygmo.nlopt`                                                                      S-U
NEWUOA + bound constraints (from NLopt)                :class:`pygmo.nlopt`                                                                      S-U
PRAXIS (from NLopt)                                    :class:`pygmo.nlopt`                                                                      S-U
Nelder-Mead simplex (from NLopt)                       :class:`pygmo.nlopt`                                                                      S-U
Subplex (from NLopt)                                   :class:`pygmo.nlopt`                                                                      S-U
MMA (Method of Moving Asymptotes) (from NLopt)         :class:`pygmo.nlopt`                                                                      S-CU
CCSA (from NLopt)                                      :class:`pygmo.nlopt`                                                                      S-CU
SLSQP (from NLopt)                                     :class:`pygmo.nlopt`                                                                      S-CU
Low-storage BFGS (from NLopt)                          :class:`pygmo.nlopt`                                                                      S-U
Preconditioned truncated Newton (from NLopt)           :class:`pygmo.nlopt`                                                                      S-U
Shifted limited-memory variable-metric (from NLopt)    :class:`pygmo.nlopt`                                                                      S-U
Nelder-Mead simplex (from SciPy)                       :class:`pygmo.scipy_optimize`                                                             S-U
Powell (from SciPy)                                    :class:`pygmo.scipy_optimize`                                                             S-U
CG (from SciPy)                                        :class:`pygmo.scipy_optimize`                                                             S-U
BFGS (from SciPy)                                      :class:`pygmo.scipy_optimize`                                                             S-U
Low-storage BFGS (from SciPy)                          :class:`pygmo.scipy_optimize`                                                             S-U
COBYLA (from SciPy)                                    :class:`pygmo.scipy_optimize`                                                             S-CU
SLSQP (from SciPy)                                     :class:`pygmo.scipy_optimize`                                                             S-CU
Trust Constr (from SciPy)                              :class:`pygmo.scipy_optimize`                                                             S-CU
Dogleg (from SciPy)                                    :class:`pygmo.scipy_optimize`                                                             S-U
Trust Ncg (from SciPy)                                 :class:`pygmo.scipy_optimize`                                                             S-U
Trust exact (from SciPy)                               :class:`pygmo.scipy_optimize`                                                             S-U
Trust Krylov (from SciPy)                              :class:`pygmo.scipy_optimize`                                                             S-U
Ipopt                                                  :class:`pygmo.ipopt`                                                                      S-CU
SNOPT (in pagmo_plugins_non_free affiliated package)   `pygmo.snopt7 <https://esa.github.io/pagmo_plugins_nonfree/py_snopt7.html>`__             S-CU
WORHP (in pagmo_plugins_non_free affiliated package)   `pygmo.wohrp <https://esa.github.io/pagmo_plugins_nonfree/py_worhp.html>`__               S-CU
====================================================== ========================================================================================= ===============

Meta-algorithms
^^^^^^^^^^^^^^^

====================================================== ============================================ ==========================
Common Name                                            Docs of the python class                     Capabilities [#meta_capa]_
====================================================== ============================================ ==========================
Monotonic Basin Hopping (MBH)                          :class:`pygmo.mbh`                           S-CU
Cstrs Self-Adaptive                                    :class:`pygmo.cstrs_self_adaptive`           S-C
Augmented Lagrangian algorithm (from NLopt) [#auglag]_ :class:`pygmo.nlopt`                         S-CU
====================================================== ============================================ ==========================

.. rubric:: Footnotes

.. [#meta_capa] The capabilities of the meta-algorithms depend
   also on the capabilities of the algorithms they wrap. If, for instance,
   a meta-algorithm supporting constrained problems is constructed from an
   algorithm which does *not* support constrained problems, the
   resulting meta-algorithms will *not* be able to solve constrained problems.

.. [#auglag] The Augmented Lagrangian algorithm can be used only
   in conjunction with other NLopt algorithms.

.. _available_problems:

List of problems
----------------

This is the list of user defined problems (UDPs) currently provided with pygmo.
These are classes that can be used to construct a :class:`pygmo.problem`,
which will
then provide a unified interface to access the problem's functionalities.

In the tables below, we classify optimisation problems
according to the following flags:

* S = Single-objective
* M = Multi-objective
* C = Constrained
* U = Unconstrained
* I = Integer programming
* sto = Stochastic

Scalable problems
^^^^^^^^^^^^^^^^^
========================================================== ========================================= ===============
Common Name                                                Docs of the python class                  Type
========================================================== ========================================= ===============
Ackley                                                     :class:`pygmo.ackley`                     S-U
Golomb Ruler                                               :class:`pygmo.golomb_ruler`               S-C-I
Griewank                                                   :class:`pygmo.griewank`                   S-U
Hock schittkowski 71                                       :class:`pygmo.hock_schittkowski_71`       S-C
Inventory                                                  :class:`pygmo.inventory`                  S-U-sto
Lennard Jones                                              :class:`pygmo.lennard_jones`              S-U
Luksan Vlcek 1                                             :class:`pygmo.luksan_vlcek1`              S-C
Rastrigin                                                  :class:`pygmo.rastrigin`                  S-U
MINLP Rastrigin                                            :class:`pygmo.minlp_rastrigin`            S-U-I
Rosenbrock                                                 :class:`pygmo.rosenbrock`                 S-U
Schwefel                                                   :class:`pygmo.schwefel`                   S-U
========================================================== ========================================= ===============

Problem suites
^^^^^^^^^^^^^^^
================================== ============================================ ===============
Common Name                        Docs of the python class                     Type
================================== ============================================ ===============
CEC2006                            :class:`pygmo.cec2006`                       S-C
CEC2009                            :class:`pygmo.cec2009`                       S-C
CEC2013                            :class:`pygmo.cec2013`                       S-U
CEC2014                            :class:`pygmo.cec2014`                       S-U
ZDT                                :class:`pygmo.zdt`                           M-U
DTLZ                               :class:`pygmo.dtlz`                          M-U
WFG                                :class:`pygmo.wfg`                           M-U
================================== ============================================ =============== 

.. _available_meta_problems:

Meta-problems
^^^^^^^^^^^^^

Meta-problems are UDPs that take another UDP as input, yielding a new UDP
which modifies the behaviour and/or the properties of the original
problem in a variety of ways.

========================================================== =========================================
Common Name                                                Docs of the python class
========================================================== =========================================
Decompose                                                  :class:`pygmo.decompose`
Translate                                                  :class:`pygmo.translate`
Unconstrain                                                :class:`pygmo.unconstrain`
Decorator                                                  :class:`pygmo.decorator_problem`
Constant Arguments                                         :class:`pygmo.constant_arguments`
========================================================== =========================================

.. _available_islands:

List of islands
---------------

This is the list of user defined islands (UDIs)
currently provided with pygmo. These are classes that
can be used to construct a :class:`pygmo.island`,
which will then
provide a unified interface to access the island's functionalities.

In the pygmo jargon, an island is an entity tasked with
managing the asynchronous evolution of a population via
an algorithm in the generalised island model.
Different UDIs enable different parallelisation
strategies (e.g., multithreading, multiprocessing,
cluster architectures, etc.).

========================================================== =========================================
Common Name                                                Docs of the python class                 
========================================================== =========================================
Thread island                                              :class:`pygmo.thread_island`
Multiprocessing island                                     :class:`pygmo.mp_island`
Ipyparallel island                                         :class:`pygmo.ipyparallel_island`
========================================================== =========================================

List of batch fitness evaluators
--------------------------------

This is the list of user defined batch fitness
evaluators (UDBFEs)
currently provided with pygmo. These are classes that
can be used to construct a :class:`pygmo.bfe`,
which will then
provide a unified interface to access the evaluator's
functionalities.

In the pygmo jargon, a batch fitness evaluator
implements the capability of evaluating a group
of decision vectors in a parallel and/or vectorised
fashion. Batch fitness evaluators are used to implement
fine-grained parallelisation in pygmo (e.g., parallel
initialisation of populations, or parallel
fitness evaluations within the inner loop of an algorithm).

========================================================== =========================================
Common Name                                                Docs of the python class                 
========================================================== =========================================
Default BFE                                                :class:`pygmo.default_bfe`
Thread BFE                                                 :class:`pygmo.thread_bfe`
Member BFE                                                 :class:`pygmo.member_bfe`
Multiprocessing BFE                                        :class:`pygmo.mp_bfe`
Ipyparallel BFE                                            :class:`pygmo.ipyparallel_bfe`
========================================================== =========================================
