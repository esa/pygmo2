# Powershell script
$ErrorActionPreference = "Stop"


$prefixRoot = $env:CONDA_PREFIX
$prefixLibrary = Join-Path $prefixRoot "Library"
$pagmoConfigDir = Join-Path $prefixLibrary "lib\cmake\pagmo"

Write-Host "CONDA_PREFIX: $prefixRoot"
Write-Host "Library prefix: $prefixLibrary"

conda config --set always_yes yes
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install cmake "eigen" nlopt ipopt libboost-devel tbb tbb-devel numpy cloudpickle networkx numba pybind11 scipy

# Install pagmo.
git clone https://github.com/esa/pagmo2.git
cd pagmo2
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 `
    -DCMAKE_PREFIX_PATH="$prefixRoot;$prefixLibrary" `
    -DCMAKE_INSTALL_PREFIX="$prefixLibrary" `
    -DBoost_NO_BOOST_CMAKE=ON `
    -DPAGMO_WITH_EIGEN3=ON `
    -DPAGMO_WITH_IPOPT=ON `
    -DPAGMO_WITH_NLOPT=ON `
    -DPAGMO_ENABLE_IPO=ON `
    ..

cmake --build . --config Release --target install
cd ../
cd ../

if (Test-Path build) { Remove-Item -Recurse -Force build }
New-Item -ItemType Directory -Path build | Out-Null
cd build

if (-not (Test-Path $pagmoConfigDir)) {
    throw "pagmo config directory not found: $pagmoConfigDir"
}

cmake -G "Visual Studio 17 2022" -A x64 `
    -DCMAKE_PREFIX_PATH="$prefixRoot;$prefixLibrary" `
    -DCMAKE_INSTALL_PREFIX="$prefixRoot" `
    -Dpagmo_DIR="$pagmoConfigDir" `
    -DPython3_EXECUTABLE="$prefixRoot\\python.exe" `
    -DBoost_NO_BOOST_CMAKE=ON `
    -DPYGMO_ENABLE_IPO=yes `
    ..

cmake --build . --config Release --target install

cd $prefixRoot
& "$prefixRoot\python.exe" -c "import pygmo; pygmo.test.run_test_suite(1); pygmo.mp_island.shutdown_pool(); pygmo.mp_bfe.shutdown_pool()"
