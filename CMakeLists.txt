cmake_minimum_required(VERSION 3.15...4.0)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Mie Fortran library ---
add_library(mie_fortran STATIC
    src/mie/MIEV0noP.f
    src/mie/ErrPack.f
    src/mie/dmiev.cpp
)
target_include_directories(mie_fortran PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(mie_fortran PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    target_compile_options(mie_fortran PRIVATE
        $<$<COMPILE_LANGUAGE:Fortran>:-fdefault-real-8>
        $<$<COMPILE_LANGUAGE:Fortran>:-fdefault-double-8>
    )
endif()

# --- Core headers ---
add_library(_core_headers INTERFACE)
target_include_directories(_core_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- Core library ---
add_library(_core_c
    src/absortion.cpp
    src/core.cpp
    src/detector.cpp
    src/laser.cpp
    src/meanfreepath.cpp
    src/medium.cpp
    src/phase.cpp
    src/photon.cpp
    src/results.cpp
    src/rng.cpp
    src/sample.cpp
    src/simulation.cpp
    src/table.cpp
    src/utils.cpp
    src/vec.cpp
)
target_link_libraries(_core_c PRIVATE _core_headers mie_fortran)
set_target_properties(_core_c PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- Python bindings ---
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_core python/python_module.cpp)
target_link_libraries(_core PRIVATE pybind11::headers _core_headers _core_c)
set_target_properties(_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS _core DESTINATION luminis_mc)
