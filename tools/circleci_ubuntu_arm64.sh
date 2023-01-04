#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install wget

# Install conda+deps.
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O miniconda.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
conda create -y -q -p $deps_dir c-compiler cxx-compiler cmake eigen nlopt boost-cpp tbb tbb-devel python=3.8 numpy cloudpickle networkx dill=0.3.5.1 numba pybind11 scipy
source activate $deps_dir

# Install pagmo.
git clone https://github.com/esa/pagmo2.git
cd pagmo2
mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=Debug -DBoost_NO_BOOST_CMAKE=ON -DPAGMO_WITH_EIGEN3=ON -DPAGMO_WITH_NLOPT=ON -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_INSTALL_PREFIX=$deps_dir -DPAGMO_ENABLE_IPO=ON -DPAGMO_INSTALL_LIBDIR=lib
make -j4 install VERBOSE=1
cd ..
cd ..

# Create the build dir and cd into it.
mkdir build
cd build

# Build pygmo.
cmake ../ -DCMAKE_BUILD_TYPE=Debug -DBoost_NO_BOOST_CMAKE=ON -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_INSTALL_PREFIX=$deps_dir -DPYGMO_ENABLE_IPO=ON
make -j2 install VERBOSE=1
cd

# Run the test suite.
python -c "import pygmo; pygmo.test.run_test_suite(1); pygmo.mp_island.shutdown_pool(); pygmo.mp_bfe.shutdown_pool()"

# Run the additional tests.
cd ~/project/tools
python circleci_additional_tests.py

set +e
set +x
