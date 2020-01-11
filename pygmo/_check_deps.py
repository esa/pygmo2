# Copyright 2020 PaGMO development team
#
# This file is part of the pygmo library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

try:
    import numpy as _np
except ImportError:
    raise ImportError(
        "The 'numpy' module could not be imported. Please make sure that numpy has been correctly installed, since pygmo depends on it.")
del(_np)

try:
    import cloudpickle as _cp
except ImportError:
    raise ImportError(
        "The 'cloudpickle' module could not be imported. Please make sure that cloudpickle has been correctly installed, since pygmo depends on it.")
del(_cp)
