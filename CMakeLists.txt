cmake_minimum_required(VERSION 3.15...4.0)

# [NUEVO] Agregamos 'Fortran' a los lenguajes del proyecto
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX Fortran)

# Nota: C++23 es muy nuevo. Si tienes problemas, baja a 17 o 20.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- [NUEVO] Compilación de la Librería de Mie (Fortran) ---
# Creamos una librería estática para el código Fortran
add_library(mie_fortran STATIC
    src/mie/MIEV0noP.f
    src/mie/ErrPack.f
    src/mie/dmiev.cpp
)

target_include_directories(mie_fortran PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- AGREGA ESTO ---
# Obliga a gfortran a usar 8 bytes para REAL (igual que C++ double)
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    target_compile_options(mie_fortran PRIVATE
        $<$<COMPILE_LANGUAGE:Fortran>:-fdefault-real-8>
        $<$<COMPILE_LANGUAGE:Fortran>:-fdefault-double-8>
    )
endif()

# [CRÍTICO] Esto es OBLIGATORIO para módulos de Python.
# El código Fortran debe ser compilado con -fPIC para poder ser
# linkeado dentro de la librería compartida de Python.
set_target_properties(mie_fortran PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------

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
    src/core.cpp
)

# [NUEVO] Linkeamos la librería de Mie (_core_c ahora puede llamar a Fortran)
target_link_libraries(_core_c PRIVATE _core_headers mie_fortran)

# [CRÍTICO] Habilitar PIC también para el core C++
set_target_properties(_core_c PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Python bindings
find_package(pybind11 CONFIG REQUIRED)
python_add_library(_core python/python_module.cpp)

# Linkeamos todo junto
target_link_libraries(_core PRIVATE pybind11::headers _core_headers _core_c)
set_target_properties(_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Install targets
install(TARGETS _core DESTINATION luminis_mc)
