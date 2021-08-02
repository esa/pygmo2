
# Powershell script
# Install conda environment
conda config --set always_yes yes
conda create --name pygmo cmake eigen nlopt ipopt boost-cpp tbb tbb-devel numpy cloudpickle networkx dill numba pybind11 scipy
conda activate pygmo

# Install pagmo.
git clone https://github.com/esa/pagmo2.git
cd pagmo2
mkdir build
cd build

cmake `
    -G "Visual Studio 15 2017 Win64" `
    -DCMAKE_PREFIX_PATH=C:\Miniconda\envs\pygmo `
    -DCMAKE_INSTALL_PREFIX=C:\Miniconda\envs\pygmo `
    -DBoost_NO_BOOST_CMAKE=ON `
    -DPAGMO_WITH_EIGEN3=ON `
    -DPAGMO_WITH_IPOPT=ON `
    -DPAGMO_WITH_NLOPT=ON `
    -DPAGMO_ENABLE_IPO=ON `
    ..

cmake --build . --config Release --target install
cd ../
cd ../

mkdir build
cd build
cmake `
    -G "Visual Studio 15 2017 Win64" `
    -DCMAKE_PREFIX_PATH=C:\Miniconda\envs\pygmo `
    -DCMAKE_INSTALL_PREFIX=C:\Miniconda\envs\pygmo `
    -DBoost_NO_BOOST_CMAKE=ON `
    -DPYGMO_ENABLE_IPO=yes `
    ..

cmake --build . --config Release --target install

cd c:\
python -c "import pygmo; pygmo.test.run_test_suite(1); pygmo.mp_island.shutdown_pool(); pygmo.mp_bfe.shutdown_pool()"
