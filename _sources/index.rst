.. pygmo documentation master file, created by
   sphinx-quickstart on Thu Jan 16 13:44:25 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pygmo
=====

pygmo is a scientific Python library for massively parallel optimization.
It is built
around the idea of providing a unified interface to optimization
algorithms and problems,
and to make their deployment in massively parallel environments easy.

Efficient implementantions of bio-inspired and evolutionary algorithms are
sided to state-of-the-art optimization algorithms
(Simplex Methods, SQP methods, interior points methods, ...)
and can be easily mixed (also with your newly-invented algorithms)
to build a super-algorithm exploiting algorithmic cooperation via the
asynchronous, generalized island model.

pygmo can be used to solve constrained, unconstrained, single objective,
multiple objective, continuous and integer optimization
problems, stochastic and deterministic problems,
as well as to perform research on novel algorithms and paradigms, and
easily compare them to state-of-the-art implementations of established ones.

If you are using pygmo as part of your research, teaching, or other activities, we would be grateful if you could star
the repository and/or cite our work. For citation purposes, you can use the following BibTex entry, which refers
to the `pygmo paper <https://doi.org/10.21105/joss.02338>`__ in the Journal of Open Source Software:

.. code-block:: bibtex

   @article{Biscani2020,
     doi = {10.21105/joss.02338},
     url = {https://doi.org/10.21105/joss.02338},
     year = {2020},
     publisher = {The Open Journal},
     volume = {5},
     number = {53},
     pages = {2338},
     author = {Francesco Biscani and Dario Izzo},
     title = {A parallel global multiobjective framework for optimization: pagmo},
     journal = {Journal of Open Source Software}
   }

The DOI of the latest version of the software is available at
`this link <https://doi.org/10.5281/zenodo.1045336>`__.

pygmo is based on the `pagmo C++ library <https://github.com/esa/pagmo2>`__.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   overview
   tutorials/tutorials
   api_reference
   credits
   changelog
   known_issues
