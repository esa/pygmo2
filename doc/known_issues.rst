Known issues
============

* Due to an `upstream bug <https://bugs.python.org/issue38501>`__,
  when using Python 3.8 pygmo might hang when exiting the Python
  interpreter. In order to work around this issue, you can execute the
  following lines

  .. code-block:: python

     pygmo.mp_island.shutdown_pool()
     pygmo.mp_bfe.shutdown_pool()

  at the end of your script/interactive session.
