cmake_minimum_required(VERSION 3.20)
project(debri LANGUAGES CXX CUDA)

# Укажите архитектуры CUDA
set(CMAKE_CUDA_ARCHITECTURES 60 70 80)

# Установите стандарт языка
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Укажите исходные файлы проекта
set(SOURCES
    cudaLibrary.cu
    cudaMacros.cu
    hostLibrary.cu
)

set(HEADERS
    cudaLibrary.cuh
    cudaMacros.cuh
    hostLibrary.cuh
    )

# Укажите заголовочные файлы для компиляции
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR})

add_library(debri STATIC ${SOURCES})

target_link_libraries(debri PRIVATE  ${CUDA_LIBRARIES})
#target_link_options(debri PRIVATE /NODEFAULTLIB:LIBCMT)
set_target_properties(CudaBifurcation PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib # Для статических библиотек
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib # Для динамических библиотек (если понадобится SHARED)
)


