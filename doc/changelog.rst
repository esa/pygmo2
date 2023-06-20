.. _changelog:

Changelog
=========

2.19.0 (2023-01-19)
-------------------

New
~~~

- Added fixed arguments meta problem
  (`#95 <https://github.com/esa/pygmo2/pull/95>`__,
  `#87 <https://github.com/esa/pygmo2/pull/87>`__).

- Added a generational version of MOEA-D with batch fitness evaluation support.
  (`#112 <https://github.com/esa/pygmo2/pull/112>`__).

Changes
~~~~~~~

- pygmo now requires pybind11 >= 2.10
  (`#104 <https://github.com/esa/pygmo2/pull/104>`__).

- pygmo now requires cmake >= 3.18.0
  (`#117 <https://github.com/esa/pygmo2/pull/117>`__).

Fix
~~~

- Fix failing tests with recent versions of pybind11
  (`#104 <https://github.com/esa/pygmo2/pull/104>`__).

- Fix linux wheels creation and upload to PyPi
  (`#117 <https://github.com/esa/pygmo2/pull/117>`__).

2.18.0 (2021-08-03)
-------------------

New
~~~

- pygmo now officially supports 64-bit ARM and PowerPC processors
  (`#82 <https://github.com/esa/pygmo2/pull/82>`__).

Changes
~~~~~~~

- pygmo now requires CMake >= 3.17
  (`#81 <https://github.com/esa/pygmo2/pull/81>`__).
- Various internal changes to the pickling implementation
  (`#79 <https://github.com/esa/pygmo2/pull/79>`__).
- pygmo now requires pagmo >= 2.18
  (`#79 <https://github.com/esa/pygmo2/pull/79>`__).

Fix
~~~

- Fix build in debug mode with Python >= 3.9
  (`#79 <https://github.com/esa/pygmo2/pull/79>`__).
- Various doc and build system fixes
  (`#79 <https://github.com/esa/pygmo2/pull/79>`__).

2.16.1 (2020-12-22)
-------------------

Changes
~~~~~~~

- pygmo now requires pybind11 >= 2.6 when compiling
  from source
  (`#66 <https://github.com/esa/pygmo2/pull/66>`__).

Fix
~~~

- Various doc and build system fixes
  (`#66 <https://github.com/esa/pygmo2/pull/66>`__,
  `#60 <https://github.com/esa/pygmo2/pull/60>`__).


2.16.0 (2020-09-25)
-------------------

New
~~~

- The genetic operators from pagmo are now available in pygmo
  (`#51 <https://github.com/esa/pygmo2/pull/51>`__).

- Add :class:`~pygmo.scipy_optimize`, a wrapper
  for SciPy's local optimisation algorithms
  (`#31 <https://github.com/esa/pygmo2/pull/31>`__).

Changes
~~~~~~~

- :class:`~pygmo.thread_island` can now use a thread pool
  (`#47 <https://github.com/esa/pygmo2/pull/47>`__).

- pygmo now requires a C++17 capable compiler when building
  from source
  (`#46 <https://github.com/esa/pygmo2/pull/46>`__,
  `#44 <https://github.com/esa/pygmo2/pull/44>`__).

- The CEC2013/CEC2014 problem suites are now available on all platforms
  (`#40 <https://github.com/esa/pygmo2/pull/40>`__).

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
