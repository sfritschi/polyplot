cmake_minimum_required(VERSION 3.0)
project("Polynomial")

set(CMAKE_C_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")

find_package(LAPACK REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Make sure Python version is >= 3.0
if(${Python_VERSION} VERSION_LESS "3.0")
    message(FATAL_ERROR "Expected Python version >= 3.0")
endif()

# Note: m is the standard math library, i.e. libm.so
set(LIBS m ${LAPACK_LIBRARIES} ${Python_LIBRARIES})

include_directories(${Python_INCLUDE_DIRS})

add_executable(polynomial
    polynomial.c
)

# Add warning flags to build
target_compile_options(polynomial PRIVATE -Wall -Wextra -Wpedantic -Werror)

target_link_libraries(polynomial ${LIBS})
