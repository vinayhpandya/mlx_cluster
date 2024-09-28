cmake_minimum_required(VERSION 3.27)
project(_ext LANGUAGES CXX)

# ----- Setup required -----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

# ----- Dependencies required ----
find_package(MLX CONFIG REQUIRED)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# ------ Adding extensions to the library -----

# Add library
add_library(mlx_cluster)

target_sources(mlx_cluster
PUBLIC
${CMAKE_CURRENT_LIST_DIR}/random_walks/RandomWalk.cpp
${CMAKE_CURRENT_LIST_DIR}/random_walks/BiasedRandomWalk.cpp
)

target_include_directories(mlx_cluster
PUBLIC
${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(mlx_cluster PUBLIC mlx)


if(MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET mlx_cluster_metallib
    TITLE mlx_cluster
    SOURCES ${CMAKE_CURRENT_LIST_DIR}/random_walks/random_walk.metal
    INCLUDE_DIRS ${PROJECT_SOURCE_DIR} ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
  )

  add_dependencies(
    mlx_cluster
    mlx_cluster_metallib
  )

endif()
# ----- Nanobind module -----
nanobind_add_module(
  _ext
  NB_STATIC STABLE_ABI LTO NOMINSIZE
  NB_DOMAIN mlx 
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp
)

target_link_libraries(_ext PRIVATE mlx_cluster)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
