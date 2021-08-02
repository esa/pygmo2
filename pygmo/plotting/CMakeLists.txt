# Copy the python files in the current binary dir,
# so that we can import pygmo from the build dir.
# NOTE: importing from the build dir will work
# only on single-configuration generators.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/__init__.py"
    "${CMAKE_CURRENT_BINARY_DIR}/__init__.py" COPYONLY)

# Add the Python files.
install(FILES __init__.py DESTINATION ${_PYGMO_INSTALL_DIR}/plotting)
