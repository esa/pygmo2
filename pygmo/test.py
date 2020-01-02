# Copyright 2019 PaGMO development team
#
# This file is part of the pygmo library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


def run_test_suite(level=0):
    """Run the full test suite.

    This function will raise an exception if at least one test fails.

    Args:
        level(``int``): the test level (higher values run longer tests)

    """
    #from . import _problem_test, _algorithm_test, _island_test, _topology_test, _r_policy_test, _s_policy_test, _bfe_test, set_global_rng_seed
    from . import _problem_test, set_global_rng_seed

    # Make test runs deterministic.
    # NOTE: we'll need to place the async/migration tests at the end, so that at
    # least the first N tests are really deterministic.
    set_global_rng_seed(42)

    retval = 0
    suite = _ut.TestLoader().loadTestsFromTestCase(_problem_test.problem_test_case)

    test_result = _ut.TextTestRunner(verbosity=2).run(suite)

    # Re-seed to random just in case anyone ever uses this function
    # in an interactive session or something.
    import random
    set_global_rng_seed(random.randint(0, 2**30))

    if len(test_result.failures) > 0 or len(test_result.errors) > 0:
        retval = 1
    if retval != 0:
        raise RuntimeError('One or more tests failed.')
