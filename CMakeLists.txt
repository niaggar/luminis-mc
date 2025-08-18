cmake_minimum_required(VERSION 4.0)
project(luminis_mc)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(luminis
    src/photon.cpp
)

target_include_directories(luminis PUBLIC include)


add_executable(luminis_app apps/main.cpp)
target_link_libraries(luminis_app PRIVATE luminis)
