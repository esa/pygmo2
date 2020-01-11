# Copyright 2020 PaGMO development team
#
# This file is part of the pygmo library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


def _get_spawn_context():
    # Small utlity to get a context that will use the 'spawn' method to
    # create new processes with the multiprocessing module. We want to enforce
    # a uniform way of creating new processes across platforms in
    # order to prevent users from implicitly relying on platform-specific
    # behaviour (e.g., fork()), only to discover later that their
    # code is not portable across platforms. See:
    # https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods

    import multiprocessing as mp

    return mp.get_context('spawn')


class _temp_disable_sigint(object):
    # A small helper context class to disable CTRL+C temporarily.

    def __enter__(self):
        import signal
        # Store the previous sigint handler and assign the new sig handler
        # (i.e., ignore SIGINT).
        self._prev_signal = signal.signal(signal.SIGINT, signal.SIG_IGN)

    def __exit__(self, type, value, traceback):
        import signal
        # Restore the previous sighandler.
        signal.signal(signal.SIGINT, self._prev_signal)


def _make_pool(processes):
    # A small factory function to create a process pool.
    # It accomplishes the tasks of selecting the correct method for
    # starting the processes ("spawn") and making sure that the
    # created processes will ignore the SIGINT signal (this prevents
    # troubles when the user issues an interruption with ctrl+c from
    # the main process).

    if processes is not None and not isinstance(processes, int):
        raise TypeError("The 'processes' argument must be None or an int")

    if processes is not None and processes <= 0:
        raise ValueError(
            "The 'processes' argument, if not None, must be strictly positive")

    # Get the context for spawning the process.
    mp_ctx = _get_spawn_context()

    # NOTE: we temporarily disable sigint while creating the pool.
    # This ensures that the processes created in the pool will ignore
    # interruptions issued via ctrl+c (only the main process will
    # be affected by them).
    with _temp_disable_sigint():
        pool = mp_ctx.Pool(processes=processes)

    pool_size = mp_ctx.cpu_count() if processes is None else processes

    # Return the created pool and its size.
    return pool, pool_size
