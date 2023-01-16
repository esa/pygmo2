#!/usr/bin/env bash

# Echo each command.
set -x

# Exit on error.
set -e

# Report on the environrnt variables used for this build
echo "PYGMO_BUILD_TYPE: ${PYGMO_BUILD_TYPE}"
echo "GITHUB_REF: ${GITHUB_REF}"
echo "GITHUB_WORKSPACE: ${GITHUB_WORKSPACE}"

# 1 - We read for what python wheels have to be built
if [[ ${PYGMO_BUILD_TYPE} == *38* ]]; then
	PYTHON_DIR="cp38-cp38"
elif [[ ${PYGMO_BUILD_TYPE} == *39* ]]; then
	PYTHON_DIR="cp39-cp39"
elif [[ ${PYGMO_BUILD_TYPE} == *310* ]]; then
	PYTHON_DIR="cp310-cp310"
elif [[ ${PYGMO_BUILD_TYPE} == *311* ]]; then
	PYTHON_DIR="cp311-cp311"
else
	echo "Invalid build type: ${PYGMO_BUILD_TYPE}"
	exit 1
fi

# Report the inferred directory whwere python is found
echo "PYTHON_DIR: ${PYTHON_DIR}"

# Python mandatory deps.
/opt/python/${PYTHON_DIR}/bin/pip install cloudpickle numpy
# Python optional deps.
/opt/python/${PYTHON_DIR}/bin/pip install dill==0.3.5.1 networkx ipyparallel scipy

# In the pagmo2/manylinux228_x86_64_with_deps:latest image in dockerhub
# the working directory is /root/install, we will install pagmo there
cd /root/install

# Install pagmo
git clone https://github.com/esa/pagmo2.git
cd pagmo2

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
cd ${GITHUB_WORKSPACE}
mkdir build
cd build
cmake -DBoost_NO_BOOST_CMAKE=ON \
	-DCMAKE_BUILD_TYPE=Release \
	-DPYGMO_ENABLE_IPO=ON \
	-DPython3_EXECUTABLE=/opt/python/${PYTHON_DIR}/bin/python ../;
make -j2 install

# Making the wheel and installing it
cd wheel
# Copy the installed pygmo files, wherever they might be in /usr/local,
# into the current dir.
cp -r ../pygmo ./
# Create the wheel and repair it.
/opt/python/${PYTHON_DIR}/bin/python setup.py bdist_wheel
auditwheel repair dist/pygmo* -w ./dist2
# Try to install it and run the tests.
cd /
/opt/python/${PYTHON_DIR}/bin/pip install ${GITHUB_WORKSPACE}/build/wheel/dist2/pygmo*
/opt/python/${PYTHON_DIR}/bin/ipcluster start --daemonize=True
sleep 20

/opt/python/${PYTHON_DIR}/bin/python -c "import pygmo; pygmo.test.run_test_suite(1); pygmo.mp_island.shutdown_pool(); pygmo.mp_bfe.shutdown_pool()"

# Upload to pypi. This variable will contain something if this is a tagged build (vx.y.z), otherwise it will be empty.
export PYGMO_RELEASE_VERSION=`echo "${GITHUB_REF}"|grep -E 'v[0-9]+\.[0-9]+.*'|cut -c 2-`
if [[ "${PYGMO_RELEASE_VERSION}" != "" ]]; then
	echo "Release build detected, uploading to PyPi."
	/opt/python/${PYTHON_DIR}/bin/pip install twine
	/opt/python/${PYTHON_DIR}/bin/twine upload -u ci4esa /pygmo2/build/wheel/dist2/pygmo*
fi
