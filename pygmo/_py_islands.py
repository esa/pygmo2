# Copyright 2020, 2021 PaGMO development team
#
# This file is part of the pygmo library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from threading import Lock as _Lock


def _evolve_func_mp_pool(ser_algo_pop):
    # The evolve function that is actually run from the separate processes
    # in mp_island (when using the pool).
    has_dill = False
    try:
        import dill

        has_dill = True
    except ImportError:
        pass
    if has_dill:
        from dill import dumps, loads
    else:
        from pickle import dumps, loads
    algo, pop = loads(ser_algo_pop)
    new_pop = algo.evolve(pop)
    return dumps((algo, new_pop))


def _evolve_func_mp_pipe(conn, ser_algo_pop):
    # The evolve function that is actually run from the separate processes
    # in mp_island (when *not* using the pool). Communication with the
    # parent process happens through the conn pipe.
    from ._mp_utils import _temp_disable_sigint

    # NOTE: disable SIGINT with the goal of preventing the user from accidentally
    # interrupting the evolution via hitting Ctrl+C in an interactive session
    # in the parent process. Note that this disables the signal only during
    # evolution, but the signal is still enabled when the process is bootstrapping
    # (so the user can still cause troubles in the child process with a well-timed
    # Ctrl+C). There's nothing we can do about it: the only way would be to disable
    # SIGINT before creating the child process, but unfortunately the creation
    # of a child process happens in a separate thread and Python disallows messing
    # with signal handlers from a thread different from the main one :(
    with _temp_disable_sigint():
        has_dill = False
        try:
            import dill

            has_dill = True
        except ImportError:
            pass
        if has_dill:
            from dill import dumps, loads
        else:
            from pickle import dumps, loads
        try:
            algo, pop = loads(ser_algo_pop)
            new_pop = algo.evolve(pop)
            conn.send(dumps((algo, new_pop)))
        except Exception as e:
            conn.send(
                RuntimeError(
                    "An exception was raised in the evolution of a multiprocessing island. The full error message is:\n{}".format(
                        e
                    )
                )
            )
        finally:
            conn.close()


class mp_island(object):
    """Multiprocessing island.

    .. versionadded:: 2.10

       The *use_pool* parameter (in previous versions, :class:`~pygmo.mp_island` always used a process pool).

    This user-defined island (UDI) will dispatch evolution tasks to an external Python process
    using the facilities provided by the standard Python :mod:`multiprocessing` module.

    If the construction argument *use_pool* is :data:`True`, then a process from a global
    :class:`pool <multiprocessing.pool.Pool>` shared between different instances of
    :class:`~pygmo.mp_island` will be used. The pool is created either implicitly by the construction
    of the first :class:`~pygmo.mp_island` object or explicitly via the :func:`~pygmo.mp_island.init_pool()`
    static method. The default number of processes in the pool is equal to the number of logical CPUs on the
    current machine. The pool's size can be queried via :func:`~pygmo.mp_island.get_pool_size()`,
    and changed via :func:`~pygmo.mp_island.resize_pool()`. The pool can be stopped via
    :func:`~pygmo.mp_island.shutdown_pool()`.

    If *use_pool* is :data:`False`, each evolution launched by an :class:`~pygmo.mp_island` will be offloaded
    to a new :class:`process <multiprocessing.Process>` which will then be terminated at the end of the evolution.

    Generally speaking, a process pool will be faster (and will use fewer resources) than spawning a new process
    for every evolution. A process pool, however, by its very nature limits the number of evolutions that can
    be run simultaneously on the system, and it introduces a serializing behaviour that might not be desirable
    in certain situations (e.g., when studying parallel evolution with migration in an :class:`~pygmo.archipelago`).

    .. note::

       Due to certain implementation details of CPython, it is not possible to initialise, resize or shutdown the pool
       from a thread different from the main one. Normally this is not a problem, but, for instance, if the first
       :class:`~pygmo.mp_island` instance is created in a thread different from the main one, an error
       will be raised. In such a situation, the user should ensure to call :func:`~pygmo.mp_island.init_pool()`
       from the main thread before spawning the secondary thread.

    .. warning::

       Due to internal limitations of CPython, sending an interrupt signal (e.g., by pressing ``Ctrl+C`` in an interactive
       Python session) while an :class:`~pygmo.mp_island` is evolving might end up sending an interrupt signal also to the
       external evolution process(es). This can lead to unpredictable runtime behaviour (e.g., the session may hang). Although
       pygmo tries hard to limit as much as possible the chances of this occurrence, it cannot eliminate them completely. Users
       are thus advised to tread carefully with interrupt signals (especially in interactive sessions) when using
       :class:`~pygmo.mp_island`.

    .. warning::

       Due to an `upstream bug <https://bugs.python.org/issue38501>`_, when using Python 3.8 the multiprocessing
       machinery may lead to a hangup when exiting a Python session. As a workaround until the bug is resolved, users
       are advised to explicitly call :func:`~pygmo.mp_island.shutdown_pool()` before exiting a Python session.

    """

    # Static variables for the pool.
    _pool_lock = _Lock()
    _pool = None
    _pool_size = None

    def __init__(self, use_pool=True):
        """
        Args:

           use_pool(:class:`bool`): if :data:`True`, a process from a global pool will be used to run the evolution, otherwise a new
              process will be spawned for each evolution

        Raises:

           TypeError: if *use_pool* is not of type :class:`bool`
           unspecified: any exception thrown by :func:`~pygmo.mp_island.init_pool()` if *use_pool* is :data:`True`

        """
        self._init(use_pool)

    def _init(self, use_pool):
        # Implementation of the ctor. Factored out
        # because it's re-used in the pickling support.
        if not isinstance(use_pool, bool):
            raise TypeError(
                "The 'use_pool' parameter in the mp_island constructor must be a boolean, but it is of type {} instead.".format(
                    type(use_pool)
                )
            )
        self._use_pool = use_pool
        if self._use_pool:
            # Init the process pool, if necessary.
            mp_island.init_pool()
        else:
            # Init the pid member and associated lock.
            self._pid_lock = _Lock()
            self._pid = None

    @property
    def use_pool(self):
        """Pool usage flag (read-only).

        Returns:

           :class:`bool`: :data:`True` if this island uses a process pool, :data:`False` otherwise

        """
        return self._use_pool

    def __copy__(self):
        # For copy/deepcopy, construct a new instance
        # with the same arguments used to construct self.
        # NOTE: no need for locking, as _use_pool is set
        # on construction and never touched again.
        return mp_island(self._use_pool)

    def __deepcopy__(self, d):
        return self.__copy__()

    def __getstate__(self):
        # For pickle/unpickle, we employ the construction
        # argument, which will be used to re-init the class
        # during unpickle.
        return self._use_pool

    def __setstate__(self, state):
        # NOTE: we need to do a full init of the object,
        # in order to set the use_pool flag and, if necessary,
        # construct the _pid and _pid_lock objects.
        self._init(state)

    def run_evolve(self, algo, pop):
        """Evolve population.

        This method will evolve the input :class:`~pygmo.population` *pop* using the input
        :class:`~pygmo.algorithm` *algo*, and return *algo* and the evolved population. The evolution
        is run either on one of the processes of the pool backing :class:`~pygmo.mp_island`, or in
        a new separate process. If this island is using a pool, and the pool was previously
        shut down via :func:`~pygmo.mp_island.shutdown_pool()`, an exception will be raised.

        Args:

           algo(:class:`~pygmo.algorithm`): the input algorithm
           pop(:class:`~pygmo.population`): the input population

        Returns:

           :class:`tuple`: a tuple of 2 elements containing *algo* (i.e., the :class:`~pygmo.algorithm` object that was used for the evolution) and the evolved :class:`~pygmo.population`

        Raises:

           RuntimeError: if the pool was manually shut down via :func:`~pygmo.mp_island.shutdown_pool()`
           unspecified: any exception thrown by the evolution, by the (de)serialization
             of the input arguments or of the return value, or by the public interface of the
             process pool


        """
        # NOTE: the idea here is that we pass the *already serialized*
        # arguments to the mp machinery, instead of letting the multiprocessing
        # module do the serialization. The advantage of doing so is
        # that if there are serialization errors, we catch them early here rather
        # than failing in the bootstrap phase of the remote process, which
        # can lead to hangups.
        has_dill = False
        try:
            import dill

            has_dill = True
        except ImportError:
            pass
        if has_dill:
            from dill import dumps, loads
        else:
            from pickle import dumps, loads
        ser_algo_pop = dumps((algo, pop))

        if self._use_pool:
            with mp_island._pool_lock:
                # NOTE: run this while the pool is locked. We have
                # functions to modify the pool (e.g., resize()) and
                # we need to make sure we are not trying to touch
                # the pool while we are sending tasks to it.
                if mp_island._pool is None:
                    raise RuntimeError(
                        "The multiprocessing island pool was stopped. Please restart it via mp_island.init_pool()."
                    )
                res = mp_island._pool.apply_async(_evolve_func_mp_pool, (ser_algo_pop,))
            # NOTE: there might be a bug in need of a workaround lurking in here:
            # http://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python
            # Just keep it in mind.
            return loads(res.get())
        else:
            from ._mp_utils import _get_spawn_context

            # Get the context for spawning the process.
            mp_ctx = _get_spawn_context()

            parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
            p = mp_ctx.Process(
                target=_evolve_func_mp_pipe, args=(child_conn, ser_algo_pop)
            )
            p.start()
            with self._pid_lock:
                self._pid = p.pid
            # NOTE: after setting the pid, wrap everything
            # in a try block with a finally clause for
            # resetting the pid to None. This way, even
            # if there are exceptions, we are sure the pid
            # is set back to None.
            try:
                res = parent_conn.recv()
                p.join()
            finally:
                with self._pid_lock:
                    self._pid = None
            if isinstance(res, RuntimeError):
                raise res
            return loads(res)

    @property
    def pid(self):
        """ID of the evolution process (read-only).

        This property is available only if the island is *not* using a process pool.

        Returns:

           :class:`int`: the ID of the process running the current evolution, or :data:`None` if no evolution is ongoing

        Raises:

           ValueError: if the island is using a process pool

        """
        if self._use_pool:
            raise ValueError(
                "The 'pid' property is available only when the island is configured to spawn new processes, but this mp_island is using a process pool instead."
            )
        with self._pid_lock:
            pid = self._pid
        return pid

    def get_name(self):
        """Island's name.

        Returns:

           :class:`str`: ``"Multiprocessing island"``

        """
        return "Multiprocessing island"

    def get_extra_info(self):
        """Island's extra info.

        If the island uses a process pool and the pool was previously shut down via :func:`~pygmo.mp_island.shutdown_pool()`,
        invoking this function will trigger the creation of a new pool.

        Returns:

           :class:`str`: a string containing information about the state of the island (e.g., number of processes in the pool, ID of the evolution process, etc.)

        Raises:

           unspecified: any exception thrown by :func:`~pygmo.mp_island.get_pool_size()`

        """
        retval = "\tUsing a process pool: {}\n".format(
            "yes" if self._use_pool else "no"
        )
        if self._use_pool:
            retval += "\tNumber of processes in the pool: {}".format(
                mp_island.get_pool_size()
            )
        else:
            with self._pid_lock:
                pid = self._pid
            if pid is None:
                retval += "\tNo active evolution process"
            else:
                retval += "\tEvolution process ID: {}".format(pid)
        return retval

    @staticmethod
    def _init_pool_impl(processes):
        # Implementation method for initing
        # the pool. This will *not* do any locking.
        from ._mp_utils import _make_pool

        if mp_island._pool is None:
            mp_island._pool, mp_island._pool_size = _make_pool(processes)

    @staticmethod
    def init_pool(processes=None):
        """Initialise the process pool.

        This method will initialise the process pool backing :class:`~pygmo.mp_island`, if the pool
        has not been initialised yet or if the pool was previously shut down via :func:`~pygmo.mp_island.shutdown_pool()`.
        Otherwise, this method will have no effects.

        Args:

           processes(:data:`None` or an :class:`int`): the size of the pool (if :data:`None`, the size of the pool will be
             equal to the number of logical CPUs on the system)

        Raises:

           ValueError: if the pool does not exist yet and the function is being called from a thread different
             from the main one, or if *processes* is a non-positive value
           TypeError: if *processes* is not :data:`None` and not an :class:`int`

        """
        with mp_island._pool_lock:
            mp_island._init_pool_impl(processes)

    @staticmethod
    def get_pool_size():
        """Get the size of the process pool.

        If the process pool was previously shut down via :func:`~pygmo.mp_island.shutdown_pool()`, invoking this
        function will trigger the creation of a new pool.

        Returns:

           :class:`int`: the current size of the pool

        Raises:

           unspecified: any exception thrown by :func:`~pygmo.mp_island.init_pool()`

        """
        with mp_island._pool_lock:
            mp_island._init_pool_impl(None)
            return mp_island._pool_size

    @staticmethod
    def resize_pool(processes):
        """Resize pool.

        This method will resize the process pool backing :class:`~pygmo.mp_island`.

        If the process pool was previously shut down via :func:`~pygmo.mp_island.shutdown_pool()`, invoking this
        function will trigger the creation of a new pool.

        Args:

           processes(:class:`int`): the desired number of processes in the pool

        Raises:

           TypeError: if the *processes* argument is not an :class:`int`
           ValueError: if the *processes* argument is not strictly positive
           unspecified: any exception thrown by :func:`~pygmo.mp_island.init_pool()`

        """
        from ._mp_utils import _make_pool

        if not isinstance(processes, int):
            raise TypeError("The 'processes' argument must be an int")
        if processes <= 0:
            raise ValueError("The 'processes' argument must be strictly positive")

        with mp_island._pool_lock:
            # NOTE: this will either init a new pool
            # with the requested number of processes,
            # or do nothing if the pool exists already.
            mp_island._init_pool_impl(processes)
            if processes == mp_island._pool_size:
                # Don't do anything if we are not changing
                # the size of the pool.
                return
            # Create new pool.
            new_pool, new_size = _make_pool(processes)
            # Stop the current pool.
            mp_island._pool.close()
            mp_island._pool.join()
            # Assign the new pool.
            mp_island._pool = new_pool
            mp_island._pool_size = new_size

    @staticmethod
    def shutdown_pool():
        """Shutdown pool.

        .. versionadded:: 2.8

        This method will shut down the process pool backing :class:`~pygmo.mp_island`, after
        all pending tasks in the pool have completed.

        After the process pool has been shut down, attempting to run an evolution on the island
        will raise an error. A new process pool can be created via an explicit call to
        :func:`~pygmo.mp_island.init_pool()` or one of the methods of the public API of
        :class:`~pygmo.mp_island` which trigger the creation of a new process pool.

        """
        with mp_island._pool_lock:
            if mp_island._pool is not None:
                mp_island._pool.close()
                mp_island._pool.join()
                mp_island._pool = None
                mp_island._pool_size = None


def _evolve_func_ipy(ser_algo_pop):
    # The evolve function that is actually run from the separate processes
    # in ipyparallel_island.
    has_dill = False
    try:
        import dill

        has_dill = True
    except ImportError:
        pass
    if has_dill:
        from dill import dumps, loads
    else:
        from pickle import dumps, loads
    algo, pop = loads(ser_algo_pop)
    new_pop = algo.evolve(pop)
    return dumps((algo, new_pop))


class ipyparallel_island(object):
    """Ipyparallel island.

    This user-defined island (UDI) will dispatch evolution tasks to an ipyparallel cluster.
    The communication with the cluster is managed via an :class:`ipyparallel.LoadBalancedView`
    instance which is created either implicitly when the first evolution is run, or explicitly
    via the :func:`~pygmo.ipyparallel_island.init_view()` method. The
    :class:`~ipyparallel.LoadBalancedView` instance is a global object shared among all the
    ipyparallel islands.

    .. seealso::

       https://ipyparallel.readthedocs.io/en/latest/

    """

    # Static variables for the view.
    _view_lock = _Lock()
    _view = None

    @staticmethod
    def init_view(client_args=[], client_kwargs={}, view_args=[], view_kwargs={}):
        """Init the ipyparallel view.

        .. versionadded:: 2.12

        This method will initialise the :class:`ipyparallel.LoadBalancedView`
        which is used by all ipyparallel islands to submit the evolution tasks
        to an ipyparallel cluster. If the :class:`ipyparallel.LoadBalancedView`
        has already been created, this method will perform no action.

        The input arguments *client_args* and *client_kwargs* are forwarded
        as positional and keyword arguments to the construction of an
        :class:`ipyparallel.Client` instance. From the constructed client,
        an :class:`ipyparallel.LoadBalancedView` instance is then created
        via the :func:`ipyparallel.Client.load_balanced_view()` method, to
        which the positional and keyword arguments *view_args* and
        *view_kwargs* are passed.

        Note that usually it is not necessary to explicitly invoke this
        method: an :class:`ipyparallel.LoadBalancedView` is automatically
        constructed with default settings the first time an evolution task
        is submitted to an ipyparallel island. This method should be used
        only if it is necessary to pass custom arguments to the construction
        of the :class:`ipyparallel.Client` or :class:`ipyparallel.LoadBalancedView`
        objects.

        Args:

            client_args(:class:`list`): the positional arguments used for the
              construction of the client
            client_kwargs(:class:`dict`): the keyword arguments used for the
              construction of the client
            view_args(:class:`list`): the positional arguments used for the
              construction of the view
            view_kwargs(:class:`dict`): the keyword arguments used for the
              construction of the view

        Raises:

           unspecified: any exception thrown by the constructor of :class:`ipyparallel.Client`
             or by the :func:`ipyparallel.Client.load_balanced_view()` method

        """
        from ._ipyparallel_utils import _make_ipyparallel_view

        with ipyparallel_island._view_lock:
            if ipyparallel_island._view is None:
                # Create the new view.
                ipyparallel_island._view = _make_ipyparallel_view(
                    client_args, client_kwargs, view_args, view_kwargs
                )

    @staticmethod
    def shutdown_view():
        """Destroy the ipyparallel view.

        .. versionadded:: 2.12

        This method will destroy the :class:`ipyparallel.LoadBalancedView`
        currently being used by the ipyparallel islands for submitting
        evolution tasks to an ipyparallel cluster. The view can be re-inited
        implicitly by submitting a new evolution task, or by invoking
        the :func:`~pygmo.ipyparallel_island.init_view()` method.

        """
        import gc

        with ipyparallel_island._view_lock:
            if ipyparallel_island._view is None:
                return

            old_view = ipyparallel_island._view
            ipyparallel_island._view = None
            del old_view
            gc.collect()

    def run_evolve(self, algo, pop):
        """Evolve population.

        This method will evolve the input :class:`~pygmo.population` *pop* using the input
        :class:`~pygmo.algorithm` *algo*, and return *algo* and the evolved population. The evolution
        task is submitted to the ipyparallel cluster via a global :class:`ipyparallel.LoadBalancedView`
        instance initialised either implicitly by the first invocation of this method,
        or by an explicit call to the :func:`~pygmo.ipyparallel_island.init_view()` method.

        Args:

            pop(:class:`~pygmo.population`): the input population
            algo(:class:`~pygmo.algorithm`): the input algorithm

        Returns:

            :class:`tuple`: a tuple of 2 elements containing *algo* (i.e., the :class:`~pygmo.algorithm` object that was used for the evolution) and the evolved :class:`~pygmo.population`

        Raises:

            unspecified: any exception thrown by the evolution, by the creation of a
              :class:`ipyparallel.LoadBalancedView`, or by the sumission of the evolution task
              to the ipyparallel cluster

        """
        # NOTE: as in the mp_island, we pre-serialize
        # the algo and pop, so that we can catch
        # serialization errors early.
        from ._ipyparallel_utils import _make_ipyparallel_view

        has_dill = False
        try:
            import dill

            has_dill = True
        except ImportError:
            pass
        if has_dill:
            from dill import dumps, loads
        else:
            from pickle import dumps, loads

        ser_algo_pop = dumps((algo, pop))
        with ipyparallel_island._view_lock:
            if ipyparallel_island._view is None:
                ipyparallel_island._view = _make_ipyparallel_view([], {}, [], {})
            ret = ipyparallel_island._view.apply_async(_evolve_func_ipy, ser_algo_pop)

        return loads(ret.get())

    def get_name(self):
        """Island's name.

        Returns:
            :class:`str`: ``"Ipyparallel island"``

        """
        return "Ipyparallel island"

    def get_extra_info(self):
        """Island's extra info.

        Returns:
            :class:`str`: a string with extra information about the status of the island

        """
        from copy import deepcopy

        with ipyparallel_island._view_lock:
            if ipyparallel_island._view is None:
                return "\tNo cluster view has been created yet"
            else:
                d = deepcopy(ipyparallel_island._view.queue_status())
        return "\tQueue status:\n\t\n\t" + "\n\t".join(
            ["(" + str(k) + ", " + str(d[k]) + ")" for k in d]
        )
