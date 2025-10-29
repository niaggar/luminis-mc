cmake_minimum_required(VERSION 3.15...4.0)


project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# Core headers
add_library(_core_headers INTERFACE)
target_include_directories(_core_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)


# Core library
add_library(_core_c
    src/detector.cpp
    src/laser.cpp
    src/medium.cpp
    src/phase.cpp
    src/photon.cpp
    src/rng.cpp
    src/simulation.cpp
    src/table.cpp
    src/absortion.cpp
    src/meanfreepath.cpp
    src/vec.cpp
)
target_link_libraries(_core_c PRIVATE _core_headers)
set_target_properties(_core_c PROPERTIES POSITION_INDEPENDENT_CODE ON)


# Python bindings
find_package(pybind11 CONFIG REQUIRED)
python_add_library(_core python/python_module.cpp)
target_link_libraries(_core PRIVATE pybind11::headers _core_headers _core_c)
set_target_properties(_core PROPERTIES POSITION_INDEPENDENT_CODE ON)


# Install targets
install(TARGETS _core DESTINATION luminis_mc)
