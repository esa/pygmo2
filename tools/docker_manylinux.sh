#!/usr/bin/env bash

# Echo each command.
set -x

# Exit on error.
set -e

PAGMO_LATEST="2.16.1"
PYBIND11_VERSION="2.6.1"

if [[ ${PYGMO_BUILD_TYPE} == *38* ]]; then
	PYTHON_DIR="cp38-cp38"
elif [[ ${PYGMO_BUILD_TYPE} == *37* ]]; then
	PYTHON_DIR="cp37-cp37m"
elif [[ ${PYGMO_BUILD_TYPE} == *36* ]]; then
	PYTHON_DIR="cp36-cp36m"
else
	echo "Invalid build type: ${PYGMO_BUILD_TYPE}"
	exit 1
fi

cd
cd install

# Python mandatory deps.
/opt/python/${PYTHON_DIR}/bin/pip install cloudpickle numpy
# Python optional deps.
/opt/python/${PYTHON_DIR}/bin/pip install dill networkx ipyparallel scipy
/opt/python/${PYTHON_DIR}/bin/ipcluster start --daemonize=True

# Install git (-y avoids a user prompt)
yum -y install git

# Install pybind11
curl -L https://github.com/pybind/pybind11/archive/v${PYBIND11_VERSION}.tar.gz > v${PYBIND11_VERSION}
tar xvf v${PYBIND11_VERSION} > /dev/null 2>&1
cd pybind11-${PYBIND11_VERSION}
mkdir build
cd build
cmake ../ -DPYBIND11_TEST=OFF > /dev/null
make install > /dev/null 2>&1
cd ..
cd ..

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
cmake -DBoost_NO_BOOST_CMAKE=ON \
	-DPAGMO_WITH_EIGEN3=yes \
	-DPAGMO_WITH_NLOPT=yes \
	-DPAGMO_WITH_IPOPT=yes \
	-DPAGMO_ENABLE_IPO=ON \
	-DCMAKE_BUILD_TYPE=Release ../;
make -j4 install


# pygmo
cd /pygmo2
mkdir build
cd build
cmake -DBoost_NO_BOOST_CMAKE=ON \
	-DCMAKE_BUILD_TYPE=Release \
	-DPYGMO_ENABLE_IPO=ON \
	-DPYTHON_EXECUTABLE=/opt/python/${PYTHON_DIR}/bin/python ../;
make -j2 install

# Making the wheel and installing it
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
if [[ "${PYGMO_RELEASE_VERSION}" != "" ]] && [[ ${PYGMO_BUILD_TYPE} == *latest ]]; then
	echo "Release build detected, uploading to PyPi."
	/opt/python/${PYTHON_DIR}/bin/pip install twine
	/opt/python/${PYTHON_DIR}/bin/twine upload -u ci4esa /pygmo2/build/wheel/dist2/pygmo*
fi
