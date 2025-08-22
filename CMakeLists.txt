cmake_minimum_required(VERSION 4.0)
project(luminis_mc)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(luminis
    src/laser.cpp
    src/phase.cpp
    src/photon.cpp
    src/rng.cpp
    src/table.cpp
)
target_include_directories(luminis PUBLIC include)



# CLI app
add_executable(luminis_app apps/main.cpp)
target_link_libraries(luminis_app PRIVATE luminis)



# Python bindings
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(luminis_mc bindings/python_module.cpp)
target_link_libraries(luminis_mc PRIVATE luminis)
target_compile_features(luminis_mc PRIVATE cxx_std_23)
install(TARGETS luminis_mc LIBRARY DESTINATION .)
