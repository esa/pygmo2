#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install wget

# Install conda+deps.
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create -y -q -p $deps_dir c-compiler cxx-compiler cmake eigen nlopt ipopt boost-cpp tbb tbb-devel python=3.10 numpy cloudpickle networkx dill=0.3.5.1 numba pybind11 sphinx myst-nb sphinx-book-theme scipy
source activate $deps_dir

# Install pagmo.
git clone https://github.com/esa/pagmo2.git
cd pagmo2
mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=Debug -DBoost_NO_BOOST_CMAKE=ON -DPAGMO_WITH_EIGEN3=ON -DPAGMO_WITH_IPOPT=ON -DPAGMO_WITH_NLOPT=ON -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_INSTALL_PREFIX=$deps_dir -DPAGMO_ENABLE_IPO=ON -DPAGMO_INSTALL_LIBDIR=lib
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

# Build the documentation.
cd ~/project/doc
make html linkcheck

# Run the doctests.
make doctest;

if [[ ! -z "${CI_PULL_REQUEST}" ]]; then
    echo "Testing a pull request, the generated documentation will not be uploaded.";
    exit 0;
fi

if [[ "${CIRCLE_BRANCH}" != "master" ]]; then
    echo "Branch is not master, the generated documentation will not be uploaded.";
    exit 0;
fi

# Check out the gh_pages branch in a separate dir.
cd ../..
git config --global push.default simple
git config --global user.name "CircleCI"
git config --global user.email "bluescarni@gmail.com"
set +x
git clone "https://${GH_ACCESS_TOKEN}@github.com/esa/pygmo2.git" pygmo2_gh_pages -q
set -x
cd pygmo2_gh_pages
git checkout -b gh-pages --track origin/gh-pages;
git rm -fr *;
mv ../project/doc/_build/html/* .;
git add *;
# We assume here that a failure in commit means that there's nothing
# to commit.
git commit -m "Update Sphinx documentation, commit ${CIRCLE_SHA1} [skip ci]." || exit 0
PUSH_COUNTER=0
until git push -q
do
    git pull -q
    PUSH_COUNTER=$((PUSH_COUNTER + 1))
    if [ "$PUSH_COUNTER" -gt 3 ]; then
        echo "Push failed, aborting.";
        exit 1;
    fi
done

set +e
set +x
