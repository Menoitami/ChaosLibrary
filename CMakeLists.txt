cmake_minimum_required(VERSION 3.20)
project(ChaosLib LANGUAGES CXX CUDA)

# Укажите архитектуры CUDA
set(CMAKE_CUDA_ARCHITECTURES 60 70 80)

# Установите стандарт языка
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Выведите информацию о компиляторах
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Bifurcation)


set(SOURCES
    main.cpp
)

# Укажите заголовочные файлы для компиляции
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Bifurcation/include
    )

# Создайте исполняемый файл
add_executable(ChaosLib ${SOURCES})

target_link_libraries(ChaosLib PUBLIC CudaBifurcation)
set_target_properties(ChaosLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)