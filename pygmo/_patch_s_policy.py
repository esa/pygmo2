# Copyright 2020 PaGMO development team
#
# This file is part of the pygmo library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from .core import s_policy


def _s_policy_extract(self, t):
    """Extract the user-defined selection policy.

    This method allows to extract a reference to the user-defined selection policy (UDSP) stored within this
    :class:`~pygmo.s_policy` instance. The behaviour of this function depends on the value
    of *t* (which must be a :class:`type`) and on the type of the internal UDSP:

    * if the type of the UDSP is *t*, then a reference to the UDSP will be returned
      (this mirrors the behaviour of the corresponding C++ method
      :cpp:func:`pagmo::s_policy::extract()`),
    * if *t* is :class:`object` and the UDSP is a Python object (as opposed to an
      :ref:`exposed C++ selection policy <s_policies_cpp>`), then a reference to the
      UDSP will be returned (this allows to extract a Python UDSP without knowing its type),
    * otherwise, :data:`None` will be returned.

    Args:
        t (:class:`type`): the type of the user-defined selection policy to extract

    Returns:
        a reference to the internal user-defined selection policy, or :data:`None` if the extraction fails

    Raises:
        TypeError: if *t* is not a :class:`type`

    """
    if not isinstance(t, type):
        raise TypeError("the 't' parameter must be a type")
    if hasattr(t, "_pygmo_cpp_s_policy"):
        return self._cpp_extract(t())
    return self._py_extract(t)


def _s_policy_is(self, t):
    """Check the type of the user-defined selection policy.

    This method returns :data:`False` if :func:`extract(t) <pygmo.s_policy.extract>` returns
    :data:`None`, and :data:`True` otherwise.

    Args:
        t (:class:`type`): the type that will be compared to the type of the UDSP

    Returns:
        bool: whether the UDSP is of type *t* or not

    Raises:
        unspecified: any exception thrown by :func:`~pygmo.s_policy.extract()`

    """
    return not self.extract(t) is None


# Do the actual patching.
setattr(s_policy, "extract", _s_policy_extract)
setattr(s_policy, "is_", _s_policy_is)
