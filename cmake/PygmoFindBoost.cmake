set(_PYGMO_BOOST_MINIMUM_VERSION 1.60.0)
find_package(Boost ${_PYGMO_BOOST_MINIMUM_VERSION} REQUIRED COMPONENTS serialization)

message(STATUS "Detected Boost version: ${Boost_VERSION}")
message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")

# Might need to recreate targets if they are missing (e.g., older CMake versions).
if(NOT TARGET Boost::boost)
    message(STATUS "The 'Boost::boost' target is missing, creating it.")
    add_library(Boost::boost INTERFACE IMPORTED)
    set_target_properties(Boost::boost PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}")
endif()

if(NOT TARGET Boost::serialization)
    message(STATUS "The 'Boost::serialization' imported target is missing, creating it.")
    add_library(Boost::serialization UNKNOWN IMPORTED)
    set_target_properties(Boost::serialization PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}"
    )
    set_target_properties(Boost::serialization PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
        IMPORTED_LOCATION "${Boost_SERIALIZATION_LIBRARY}"
    )
endif()

if(NOT TARGET Boost::disable_autolinking)
    message(STATUS "The 'Boost::disable_autolinking' target is missing, creating it.")
    add_library(Boost::disable_autolinking INTERFACE IMPORTED)
    if(WIN32)
        set_target_properties(Boost::disable_autolinking PROPERTIES INTERFACE_COMPILE_DEFINITIONS "BOOST_ALL_NO_LIB")
    endif()
endif()

unset(_PYGMO_BOOST_MINIMUM_VERSION)
