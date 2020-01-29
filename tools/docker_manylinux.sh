#!/usr/bin/env bash

# Echo each command.
set -x

# Exit on error.
set -e

PAGMO_LATEST="2.13.0"
PYBIND11_VERSION="2.4.3"

if [[ ${PYGMO_BUILD_TYPE} == *38* ]]; then
	PYTHON_DIR="cp38-cp38"
	PYTHON_VERSION="3.8"
elif [[ ${PYGMO_BUILD_TYPE} == *37* ]]; then
	PYTHON_DIR="cp37-cp37m"
	PYTHON_VERSION="3.7"
elif [[ ${PYGMO_BUILD_TYPE} == *36* ]]; then
	PYTHON_DIR="cp36-cp36m"
	PYTHON_VERSION="3.6"
else
	echo "Invalid build type: ${PYGMO_BUILD_TYPE}"
	exit 1
fi

cd
cd install

# Install conda+deps.
curl -L https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh > miniconda.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
conda config --add channels conda-forge --force
conda_pkgs="cmake eigen nlopt ipopt boost-cpp tbb tbb-devel python=${PYTHON_VERSION} numpy cloudpickle dill numba pip pybind11"
conda create -q -p $deps_dir -y
source activate $deps_dir
conda install $conda_pkgs -y

# Install git (-y avoids a user prompt)
yum -y install git

# Install pagmo
if [[ ${PYGMO_BUILD_TYPE} == *latest ]]; then
	curl -L https://github.com/esa/pagmo2/archive/v${PAGMO_LATEST}.tar.gz > v${PAGMO_LATEST}
	tar xvf v${PAGMO_LATEST} > /dev/null 2>&1
	cd pagmo2-${PAGMO_LATEST}
elif [[ ${PYGMO_BUILD_TYPE} == *head ]]; then
	git clone https://github.com/esa/pagmo2.git
	cd pagmo2
else
	echo "Invalid build type: ${PYGMO_BUILD_TYPE}"
	exit 1
fi
mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=Release \
	-DBoost_NO_BOOST_CMAKE=ON \
	-DPAGMO_WITH_EIGEN3=ON \
	-DPAGMO_WITH_IPOPT=ON \
	-DPAGMO_WITH_NLOPT=ON \
	-DCMAKE_PREFIX_PATH=$deps_dir \
	-DCMAKE_INSTALL_PREFIX=$deps_dir \
	-DCMAKE_CXX_STANDARD=17 ../
make -j2 install

# pygmo
cd /pygmo2
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
	-DBoost_NO_BOOST_CMAKE=ON \
	-DCMAKE_PREFIX_PATH=$deps_dir \
	-DCMAKE_INSTALL_PREFIX=$deps_dir \
	-DCMAKE_CXX_STANDARD=17 ../;
make -j2 install

# Making the wheel and isntalling it
cd wheel
# Copy the installed pygmo files, wherever they might be in /usr/local,
# into the current dir.
cp -a `find /usr/local/lib -type d -iname 'pygmo'` ./
# Create the wheel and repair it.
/opt/python/${PYTHON_DIR}/bin/python setup.py bdist_wheel
auditwheel repair dist/pygmo* -w ./dist2
# Try to install it and run the tests.
cd /
/opt/python/${PYTHON_DIR}/bin/pip install /pygmo2/build/wheel/dist2/pygmo*
/opt/python/${PYTHON_DIR}/bin/python -c "import pygmo; pygmo.test.run_test_suite(1); pygmo.mp_island.shutdown_pool(); pygmo.mp_bfe.shutdown_pool()"

# Upload to pypi. This variable will contain something if this is a tagged build (vx.y.z), otherwise it will be empty.
export PYGMO_RELEASE_VERSION=`echo "${TRAVIS_TAG}"|grep -E 'v[0-9]+\.[0-9]+.*'|cut -c 2-`
if [[ "${PYGMO_RELEASE_VERSION}" != "" ]]; then
    echo "Release build detected, uploading to PyPi."
    /opt/python/${PYTHON_DIR}/bin/pip install twine
	/opt/python/${PYTHON_DIR}/bin/twine upload -u ci4esa /pygmo2/build/wheel/dist2/pygmo*
fi
