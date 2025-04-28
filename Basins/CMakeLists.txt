cmake_minimum_required(VERSION 3.20)
project(Basins LANGUAGES CXX CUDA)

# Укажите архитектуры CUDA
set(CMAKE_CUDA_ARCHITECTURES 80)

# Установите стандарт языка
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
# Укажите исходные файлы проекта
set(SOURCES
    src/basinsCUDA.cu
    src/basinsHOST.cpp
)

set(HEADERS
    include/basinsHOST.h
    include/basinsCUDA.cuh
)

# Укажите заголовочные файлы для компиляции
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/debri
    ${CMAKE_SOURCE_DIR}/systems
    )

add_library(Basins STATIC ${SOURCES})

target_link_libraries(Basins PRIVATE ${CUDA_LIBRARIES})

set_target_properties(Basins PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
) 