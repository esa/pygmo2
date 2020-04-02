.. _changelog:

Changelog
=========

2.15.0 (2020-04-02)
-------------------

New
~~~

- The topologies implemented on top of the Boost Graph Library
  now provide a ``get_edge_weight()``
  function to fetch the weight of an edge
  (`#34 <https://github.com/esa/pygmo2/pull/34>`__).

- Add the :class:`~pygmo.free_form` topology
  (`#34 <https://github.com/esa/pygmo2/pull/34>`__).

- User-defined topologies can now (optionally) implement
  a conversion function to a NetworkX graph object
  (`#34 <https://github.com/esa/pygmo2/pull/34>`__).

Fix
~~~

- Build system fixes for recent CMake versions
  (`#35 <https://github.com/esa/pygmo2/pull/35>`__).

- Various doc fixes
  (`#35 <https://github.com/esa/pygmo2/pull/35>`__,
  `#34 <https://github.com/esa/pygmo2/pull/34>`__).

2.14.1 (2020-03-06)
-------------------

Fix
~~~

- Fix the upload of the binary wheels to pypi.

2.14.0 (2020-03-04)
-------------------

New
~~~

- Initial stand-alone version of pygmo. See
  `the pagmo changelog <https://esa.github.io/pagmo2/changelog.html>`__
  for the changelog of previous pygmo
  versions.
- Implement a setter for the migration database
  of an archipelago
  (`#25 <https://github.com/esa/pygmo2/pull/25>`__).

Fix
~~~

- Fix a serialization issue when using ipyparallel
  functionalities in Python 3.8
  (`#23 <https://github.com/esa/pygmo2/pull/23>`__).
