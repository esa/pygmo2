# Configure the version file.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/_version.py.in" "${CMAKE_CURRENT_BINARY_DIR}/_version.py" @ONLY)

# Configure the files needed to make the python wheels (for PyPi packages)
if(MINGW OR ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        message(STATUS "Creating the files for the generation of a binary wheel.")
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../tools/wheel_setup.py" "${CMAKE_CURRENT_BINARY_DIR}/../wheel/setup.py" @ONLY)
        if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
            # NOTE: this is necessary on linux but harmful on mingw.
            configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../tools/wheel_setup.cfg" "${CMAKE_CURRENT_BINARY_DIR}/../wheel/setup.cfg" @ONLY)
        endif()
        if(MINGW)
            configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../tools/mingw_wheel_libs_python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}.txt" "${CMAKE_CURRENT_BINARY_DIR}/wheel/mingw_wheel_libs_python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}.txt" @ONLY)
        endif()
endif()


# The list of pygmo's Python files.
set(PYGMO_PYTHON_FILES
    __init__.py
    _check_deps.py
    test.py
    _patch_problem.py
    _problem_test.py
    _patch_algorithm.py
    _patch_bfe.py
    _patch_island.py
    _py_bfes.py
    _mp_utils.py
    _py_islands.py
    _py_algorithms.py
    _algorithm_test.py
    _bfe_test.py
    _island_test.py
    _patch_r_policy.py
    _patch_s_policy.py
    _patch_topology.py
    _s_policy_test.py
    _r_policy_test.py
    _topology_test.py
    _py_problems.py
    _ipyparallel_utils.py
)

# Copy the python files in the current binary dir,
# so that we can import pygmo from the build dir.
# NOTE: importing from the build dir will work
# only on single-configuration generators.
foreach(PYGMO_PYTHON_FILE ${PYGMO_PYTHON_FILES})
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${PYGMO_PYTHON_FILE}"
        "${CMAKE_CURRENT_BINARY_DIR}/${PYGMO_PYTHON_FILE}" COPYONLY)
endforeach()

# Core module.
Python3_add_library(core MODULE WITH_SOABI
    core.cpp
    common_utils.cpp
    common_base.cpp
    island.cpp
    docstrings.cpp
    problem.cpp
    expose_problems_0.cpp
    expose_problems_1.cpp
    algorithm.cpp
    bfe.cpp
    expose_algorithms_0.cpp
    expose_algorithms_1.cpp
    expose_bfes.cpp
    expose_islands.cpp
    r_policy.cpp
    expose_r_policies.cpp
    s_policy.cpp
    expose_s_policies.cpp
    topology.cpp
    expose_topologies.cpp
    handle_thread_py_exception.cpp
)

target_link_libraries(core PRIVATE Pagmo::pagmo Boost::boost Boost::serialization)
# NOTE: quench warnings from Boost when building the library.
target_compile_definitions(core PRIVATE BOOST_ALLOW_DEPRECATED_HEADERS)
target_include_directories(core SYSTEM PRIVATE "${pybind11_INCLUDE_DIR}")
target_compile_definitions(core PRIVATE "${pybind11_DEFINITIONS}")
target_compile_options(core PRIVATE
    "$<$<CONFIG:Debug>:${PYGMO_CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${PYGMO_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:RelWithDebInfo>:${PYGMO_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:MinSizeRel>:${PYGMO_CXX_FLAGS_RELEASE}>"
)
set_target_properties(core PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(core PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE)
# Set the minimum C++ standard to C++17 when building
# the core module.
target_compile_features(core PRIVATE cxx_std_17)
# Enforce vanilla C++ when compiling.
set_property(TARGET core PROPERTY CXX_EXTENSIONS NO)

if (PYGMO_ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _PYGMO_IPO_RESULT OUTPUT _PYGMO_IPO_OUTPUT)
    if (_PYGMO_IPO_RESULT)
        message(STATUS "IPO requested and supported, enabling.")
        set_property(TARGET core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "IPO requested, but it is not supported by the compiler:\n${_PYGMO_IPO_OUTPUT}")
    endif()
    unset(_PYGMO_IPO_RESULT)
    unset(_PYGMO_IPO_OUTPUT)
endif()

# Installation setup.
if(PYGMO_INSTALL_PATH STREQUAL "")
    message(STATUS "pygmo will be installed in the default location: ${Python3_SITEARCH}")
    set(_PYGMO_INSTALL_DIR "${Python3_SITEARCH}/pygmo")
else()
    message(STATUS "pygmo will be installed in the custom location: ${PYGMO_INSTALL_PATH}")
    set(_PYGMO_INSTALL_DIR "${PYGMO_INSTALL_PATH}/pygmo")
endif()

# Add submodules directories
add_subdirectory(plotting)

# Install the core module.
install(TARGETS core
    RUNTIME DESTINATION ${_PYGMO_INSTALL_DIR}
    LIBRARY DESTINATION ${_PYGMO_INSTALL_DIR}
)

# Install the Python files.
install(FILES ${PYGMO_PYTHON_FILES} "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
    DESTINATION ${_PYGMO_INSTALL_DIR})

unset(_PYGMO_INSTALL_DIR)
