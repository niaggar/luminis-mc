cmake_minimum_required(VERSION 3.15...4.0)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Build type & optimization options ---
# Default to Release: the per-photon transport kernel must be compiled with
# optimizations (-O3 -DNDEBUG). A Debug build runs the hot loop unoptimized.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "No build type specified; defaulting to Release")
endif()

# -march=native: vectorize for the host CPU (still IEEE-strict; NO -ffast-math).
# The resulting binary is not portable across different CPU microarchitectures.
option(LUMINIS_NATIVE "Optimize for the host CPU (-march=native)" OFF)
# Link-time optimization across translation units.
option(LUMINIS_LTO "Enable interprocedural/link-time optimization" OFF)

set(LUMINIS_CXX_OPT_FLAGS "")
if(LUMINIS_NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    list(APPEND LUMINIS_CXX_OPT_FLAGS -march=native)
endif()

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
target_compile_options(_core_c PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${LUMINIS_CXX_OPT_FLAGS}>)

# --- Python bindings ---
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_core python/python_module.cpp)
target_link_libraries(_core PRIVATE pybind11::headers _core_headers _core_c)
set_target_properties(_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${LUMINIS_CXX_OPT_FLAGS}>)

# --- Optional link-time optimization ---
if(LUMINIS_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_msg)
    if(_ipo_ok)
        set_target_properties(_core_c _core PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
        message(STATUS "Link-time optimization (IPO/LTO) enabled")
    else()
        message(WARNING "IPO/LTO requested but not supported: ${_ipo_msg}")
    endif()
endif()

install(TARGETS _core DESTINATION luminis_mc)

# --- Type stubs ---
install(FILES
    python/luminis_mc/_core.pyi
    python/luminis_mc/py.typed
    DESTINATION luminis_mc
)
