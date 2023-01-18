Installation
============

Dependencies
------------

pygmo has the following **mandatory** runtime dependencies:

* `Python <https://www.python.org/>`__ 3.4 or later (Python 2.x is
  **not** supported),
* the `pagmo C++ library <https://esa.github.io/pagmo2/>`__, version 2.18 or later,
* the `Boost serialization library <https://www.boost.org/doc/libs/release/libs/serialization/doc/index.html>`__,
  version 1.60 or later,
* `NumPy <https://numpy.org/>`__,
* `cloudpickle <https://github.com/cloudpipe/cloudpickle>`__.

Additionally, pygmo has the following **optional** runtime
dependencies:

* `dill <https://dill.readthedocs.io>`__, which can be used as an
  alternative serialization backend,
* `Matplotlib <https://matplotlib.org/>`__, which is used by a few
  plotting utilities,
* `NetworkX <https://networkx.github.io/>`__, which is used for
  importing/exporting topologies as graphs,
* `SciPy <https://www.scipy.org/>`__, which is used in the implementation
  of the :class:`~pygmo.scipy_optimize` algorithm wrapper.

Packages
--------

pygmo packages are available from a variety
of package managers on several platforms.

Conda
^^^^^

pygmo is available via the `conda <https://docs.conda.io/en/latest/>`__
package manager for Linux, OSX and Windows
thanks to the infrastructure provided by `conda-forge <https://conda-forge.org/>`__.
In order to install pygmo via conda, you just need to add ``conda-forge``
to the channels, and then we can immediately install pygmo:

.. code-block:: console

   $ conda config --add channels conda-forge
   $ conda config --set channel_priority strict
   $ conda install pygmo

The conda packages for pygmo are maintained by the core development team,
and they are regularly updated when new pygmo versions are released.

Please refer to the `conda documentation <https://docs.conda.io/en/latest/>`__
for instructions on how to setup and manage
your conda installation.

pip
^^^

pygmo is also available on Linux via the `pip <https://pip.pypa.io/en/stable/>`__
package installer. The installation of pygmo with pip is straightforward:

.. code-block:: console

   $ pip install pygmo

If you want to install pygmo for a single user instead of
system-wide, which is in general a good idea, you can do:

.. code-block:: console

   $ pip install --user pygmo

An even better idea is to make use of Python's
`virtual environments <https://virtualenv.pypa.io/en/latest/>`__.

The pip packages for pygmo are maintained by the core development team,
and they are updated when new pygmo versions are released.

.. warning::

   Note however that we **strongly** encourage users to install pygmo with conda
   rather than with pip. The reason is that pygmo is built on a
   moderately complicated
   stack of C++ libraries, which have to be bundled together with pygmo
   in the pip package.
   This is a problem if one uses pygmo together with other Python
   packages sharing dependencies with pygmo, because multiple incompatible
   versions of the same C++ library might end up being loaded at the
   same time, leading to crashes and erratic runtime behaviour.
   The conda packages do not suffer from this issue.

.. note::

   Any bug reported against the pip build and not reproduced with other installation methods
   (e.g., conda package, installation from source) will be automatically closed as invalid.

.. note::

   Due to a lack of manpower, we are currently unable to provide
   pip packages for Windows or OSX. If you are willing to help us
   out, please get in contact with us on the
   `gitter channel <https://gitter.im/pagmo2/Lobby>`__ or (even better)
   open a PR on `github <https://github.com/esa/pygmo2/pulls>`__.

Arch Linux
^^^^^^^^^^

pygmo is available on the `Arch User Repository
<https://aur.archlinux.org>`__ (AUR) in Arch Linux. It is
recommended to use an AUR helper like
`yay <https://aur.archlinux.org/packages/yay/>`__ or
`pikaur <https://aur.archlinux.org/packages/pikaur/>`__ for ease of installation.
See the `AUR helpers <https://wiki.archlinux.org/index.php/AUR_helpers>`__ page on
the Arch Linux wiki for more info.

Install ``python-pygmo``:

.. code-block:: console

   $ yay -S python-pygmo

Installation from source
------------------------

In order to install pygmo from source, you will need:

* a C++17 capable compiler (recent versions of GCC,
  Clang or MSVC should do),
* a `Python <https://www.python.org/>`__ installation,
* `pybind11 <https://github.com/pybind/pybind11>`__ (version >= 2.10),
* the `pagmo C++ library <https://esa.github.io/pagmo2/>`__, (version >=2.19)
* the `Boost libraries <https://www.boost.org/>`__,
* `CMake <https://cmake.org/>`__, version 3.18 or later.

After making sure the dependencies are installed on your system, you can
download the pygmo source code from the
`GitHub release page <https://github.com/esa/pygmo/releases>`__. Alternatively,
and if you like living on the bleeding edge, you can get the very latest
version of pygmo via ``git``:

.. code-block:: console

   $ git clone https://github.com/esa/pygmo2.git

We follow the usual PR-based development workflow, thus pygmo's ``master``
branch is normally kept in a working state.

After downloading and/or unpacking pygmo's
source code, go to pygmo's
source tree, create a ``build`` directory and ``cd`` into it. E.g.,
on a Unix-like system:

.. code-block:: console

   $ cd /path/to/pygmo
   $ mkdir build
   $ cd build

Once you are in the ``build`` directory, you must configure your build
using ``cmake``. There are various useful CMake variables you can set,
such as:

* ``CMAKE_BUILD_TYPE``: the build type (``Release``, ``Debug``, etc.),
  defaults to ``Release``.
* ``CMAKE_PREFIX_PATH``: additional paths that will be searched by CMake
  when looking for dependencies.
* ``PYGMO_INSTALL_PATH``: the path into which the pygmo module
  will be installed. If left empty (the default), pygmo will be installed
  in the global modules directory of your Python installation.
* ``PYGMO_ENABLE_IPO``: set this flag to ``ON`` to compile pygmo
  with link-time optimisations. Requires compiler support,
  defaults to ``OFF``.

Please consult `CMake's documentation <https://cmake.org/cmake/help/latest/>`_
for more details about CMake's variables and options.

The ``PYGMO_INSTALL_PATH`` option is particularly important. If you
want to install pygmo locally instead of globally (which is in general
a good idea), you can set this variable to the output of
``python -m site --user-site``.

After configuring the build with CMake, we can then proceed to actually
building pygmo:

.. code-block:: console

   $ cmake --build .

Finally, we can install pygmo with the command:

.. code-block:: console

   $ cmake  --build . --target install

Verifying the installation
--------------------------

You can verify that pygmo was successfully compiled and
installed by running the test suite. From a
Python session, run the following commands:

.. code-block:: python

   >>> import pygmo
   >>> pygmo.test.run_test_suite()

If these commands execute without any error, then
your pygmo installation is ready for use.

Getting help
------------

If you run into troubles installing pygmo, please do not hesitate
to contact us either through our `gitter channel <https://gitter.im/pagmo2/Lobby>`__
or by opening an issue report on `github <https://github.com/esa/pygmo2/issues>`__.
