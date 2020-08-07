# pygmo

[![Build Status](https://img.shields.io/circleci/project/github/esa/pygmo2/master.svg?style=for-the-badge)](https://circleci.com/gh/esa/pygmo2)
[![Build Status](https://img.shields.io/travis/esa/pygmo2/master.svg?logo=travis&style=for-the-badge)](https://travis-ci.org/esa/pygmo2)
[![Build Status](https://img.shields.io/azure-devops/build/bluescarni/00914570-450b-4bd6-a575-d8c64fc25d1e/5?style=for-the-badge)](https://dev.azure.com/bluescarni/pygmo/_build)

[![Anaconda-Server Badge](https://img.shields.io/conda/vn/conda-forge/pygmo.svg?style=for-the-badge)](https://anaconda.org/conda-forge/pygmo)
[![PyPI](https://img.shields.io/pypi/v/pygmo.svg?style=for-the-badge)](https://pypi.python.org/pypi/pygmo)

[![Join the chat at https://gitter.im/pagmo2/Lobby](https://img.shields.io/badge/gitter-join--chat-green.svg?logo=gitter-white&style=for-the-badge)](https://gitter.im/pagmo2/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1045337.svg)](https://doi.org/10.5281/zenodo.1045336)

pygmo is a scientific Python library for massively parallel optimization. It is built around the idea
of providing a unified interface to optimization algorithms and to optimization problems and to make their
deployment in massively parallel environments easy.

If you are using pygmo as part of your research, teaching, or other activities, we would be grateful if you could star
the repository and/or cite our work. The DOI of the latest version and other citation resources are available
at [this link](https://doi.org/10.5281/zenodo.1045336).

The full documentation can be found [here](https://esa.github.io/pygmo2/).

## Building and installing from source

```
pip3 install git+https://github.com/esa/pygmo2.git
```

## Quick testing

```
python3 -c "import pygmo; pygmo.test.run_test_suite(1); pygmo.mp_island.shutdown_pool(); pygmo.mp_bfe.shutdown_pool()"
```

## Upgrading from pygmo 1.x.x

If you were using the old pygmo, have a look here on some technical data on what and why a completely new API
and code was developed: https://github.com/esa/pagmo2/wiki/From-1.x-to-2.x

You will find many tutorials in the documentation, we suggest to skim through them to realize the differences.
The new pygmo (version 2) should be considered (and is) as an entirely different code.

