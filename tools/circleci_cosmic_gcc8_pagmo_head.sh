#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install build-essential wget

# Install conda+deps.
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
export PATH="$deps_dir/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
conda config --add channels conda-forge --force
conda_pkgs="cmake eigen nlopt ipopt boost-cpp tbb tbb-devel python=3.7 numpy cloudpickle dill numba pip pybind11 ipyparallel"
conda create -q -p $deps_dir -y
source activate $deps_dir
conda install $conda_pkgs -y

# Install pagmo.
git clone https://github.com/esa/pagmo2.git
cd pagmo2
mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=Debug -DPAGMO_WITH_EIGEN3=ON -DPAGMO_WITH_IPOPT=ON -DPAGMO_WITH_NLOPT=ON -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_CXX_STANDARD=17
make -j4 install VERBOSE=1
cd ..
cd ..

# Create the build dir and cd into it.
mkdir build
cd build

# GCC build with address sanitizer.
cmake ../ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_CXX_STANDARD=17
make -j2 install VERBOSE=1
cd

# Start ipcluster.
ipcluster  start --debug --n=1 &
sleep 20

python -c "import pygmo; pygmo.test.run_test_suite(1)"

ipcluster stop

set +e
set +x

