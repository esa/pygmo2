#!/usr/bin/env bash

set -x
set -e

: "${PYGMO_PYTHON_VERSION:?PYGMO_PYTHON_VERSION must be set}"

# Ensure conda-forge is used consistently across all non-manylinux Unix jobs.
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install -y -q \
    c-compiler cxx-compiler cmake 'eigen' nlopt ipopt libboost-devel tbb tbb-devel \
    "python=${PYGMO_PYTHON_VERSION}" \
    numpy cloudpickle networkx numba pybind11 scipy

deps_dir="${CONDA_PREFIX}"

# Install pagmo.
git clone https://github.com/esa/pagmo2.git
cd pagmo2
mkdir build
cd build
cmake ../ \
    -DCMAKE_BUILD_TYPE=Release \
    -DBoost_NO_BOOST_CMAKE=ON \
    -DPAGMO_WITH_EIGEN3=ON \
    -DPAGMO_WITH_IPOPT=ON \
    -DPAGMO_WITH_NLOPT=ON \
    -DCMAKE_PREFIX_PATH="${deps_dir}" \
    -DCMAKE_INSTALL_PREFIX="${deps_dir}" \
    -DPAGMO_ENABLE_IPO=ON
make -j4 install VERBOSE=1
cd ../..

# Build and install pygmo.
mkdir build
cd build
cmake ../ \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBoost_NO_BOOST_CMAKE=ON \
    -DCMAKE_PREFIX_PATH="${deps_dir}" \
    -DCMAKE_INSTALL_PREFIX="${deps_dir}" \
    -DPYGMO_ENABLE_IPO=ON
make -j2 install VERBOSE=1
cd

# Run the test suite.
python -c "import pygmo; pygmo.test.run_test_suite(1); pygmo.mp_island.shutdown_pool(); pygmo.mp_bfe.shutdown_pool()"

set +e
set +x