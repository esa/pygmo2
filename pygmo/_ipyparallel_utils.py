# Copyright 2020 PaGMO development team
#
# This file is part of the pygmo library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


def _make_ipyparallel_view(client_args, client_kwargs, view_args, view_kwargs):
    # Small helper to create an ipyparallel view.
    from ipyparallel import Client

    rc = Client(*client_args, **client_kwargs)
    rc[:].use_cloudpickle()
    return rc.load_balanced_view(*view_args, **view_kwargs)
