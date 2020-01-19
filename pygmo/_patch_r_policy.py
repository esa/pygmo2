# Copyright 2020 PaGMO development team
#
# This file is part of the pygmo library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from .core import r_policy


def _r_policy_extract(self, t):
    """Extract the user-defined replacement policy.

    This method allows to extract a reference to the user-defined replacement policy (UDRP) stored within this
    :class:`~pygmo.r_policy` instance. The behaviour of this function depends on the value
    of *t* (which must be a :class:`type`) and on the type of the internal UDRP:

    * if the type of the UDRP is *t*, then a reference to the UDRP will be returned
      (this mirrors the behaviour of the corresponding C++ method
      :cpp:func:`pagmo::r_policy::extract()`),
    * if *t* is :class:`object` and the UDRP is a Python object (as opposed to an
      :ref:`exposed C++ replacement policy <r_policies_cpp>`), then a reference to the
      UDRP will be returned (this allows to extract a Python UDRP without knowing its type),
    * otherwise, :data:`None` will be returned.

    Args:
        t (:class:`type`): the type of the user-defined replacement policy to extract

    Returns:
        a reference to the internal user-defined replacement policy, or :data:`None` if the extraction fails

    Raises:
        TypeError: if *t* is not a :class:`type`

    """
    if not isinstance(t, type):
        raise TypeError("the 't' parameter must be a type")
    if hasattr(t, "_pygmo_cpp_r_policy"):
        return self._cpp_extract(t())
    return self._py_extract(t)


def _r_policy_is(self, t):
    """Check the type of the user-defined replacement policy.

    This method returns :data:`False` if :func:`extract(t) <pygmo.r_policy.extract>` returns
    :data:`None`, and :data:`True` otherwise.

    Args:
        t (:class:`type`): the type that will be compared to the type of the UDRP

    Returns:
        bool: whether the UDRP is of type *t* or not

    Raises:
        unspecified: any exception thrown by :func:`~pygmo.r_policy.extract()`

    """
    return not self.extract(t) is None


# Do the actual patching.
setattr(r_policy, "extract", _r_policy_extract)
setattr(r_policy, "is_", _r_policy_is)
