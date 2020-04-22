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
YACMA_PYTHON_MODULE(core
    core.cpp
    common_utils.cpp
    common_base.cpp
    object_serialization.cpp
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
if(NOT CMAKE_CXX_STANDARD)
    # The user did not provide the CMAKE_CXX_STANDARD variable,
    # go with the default (C++17).
    message(STATUS "Setting the C++ standard version to the default value (17).")
    set_property(TARGET core PROPERTY CXX_STANDARD 17)
else()
    message(STATUS "Using the manually-specified value for the C++ standard version (${CMAKE_CXX_STANDARD}).")
endif()
set_property(TARGET core PROPERTY CXX_STANDARD_REQUIRED YES)
set_property(TARGET core PROPERTY CXX_EXTENSIONS NO)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.9.0")
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
endif()

# Setup the installation path.
set(PYGMO_INSTALL_PATH "${YACMA_PYTHON_MODULES_INSTALL_PATH}/pygmo")

# Add submodules directories
add_subdirectory(plotting)

# Install the core module.
install(TARGETS core
    RUNTIME DESTINATION ${PYGMO_INSTALL_PATH}
    LIBRARY DESTINATION ${PYGMO_INSTALL_PATH}
)

# Install the Python files.
install(FILES ${PYGMO_PYTHON_FILES} "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
    DESTINATION ${PYGMO_INSTALL_PATH})
