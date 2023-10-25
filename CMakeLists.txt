cmake_minimum_required(VERSION 3.16.2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
project(mandelbrot_avx VERSION 0.1.2 LANGUAGES CXX)
include(cmake/general.cmake)
include(cmake/functions.cmake)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                                 Conan                                  #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
set(CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR} ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR} ${CMAKE_PREFIX_PATH})

find_package(fmt REQUIRED)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                                 SPL                                  #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
add_library(spl SHARED IMPORTED GLOBAL)
set_target_properties(spl PROPERTIES
        IMPORTED_LOCATION /home/momo/git/spl/build/libspl.a
        INTERFACE_INCLUDE_DIRECTORIES /home/momo/git/spl/include/
        )
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                            Enable warnings                             #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
add_library(project_warnings INTERFACE)
include(cmake/compiler_warnings.cmake)
set_project_warnings(project_warnings)
find_package(Threads REQUIRED)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                                Executable                              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
add_executable(mandelbrot_avx)
target_sources(mandelbrot_avx PRIVATE src/main.cpp)
target_compile_features(mandelbrot_avx PUBLIC cxx_std_20)
target_compile_options(mandelbrot_avx PUBLIC -fcoroutines -march=native)
target_link_options(mandelbrot_avx PRIVATE)
target_link_libraries(mandelbrot_avx
    PRIVATE
        fmt::fmt
        spl
        project_warnings
        sfml-graphics
        sfml-window
        sfml-system
)
target_include_directories(mandelbrot_avx PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include")

get_property(incdirs TARGET mandelbrot_avx PROPERTY INCLUDE_DIRECTORIES)
enable_lto(mandelbrot_avx)
#enable_sanitizers(mandelbrot_avx)
