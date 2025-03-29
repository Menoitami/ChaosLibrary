cmake_minimum_required(VERSION 3.20)
project(ChaosLib LANGUAGES CXX CUDA)
#find_package(CUDA REQUIRED)

# Укажите архитектуры CUDA
set(CMAKE_CUDA_ARCHITECTURES 60 70 80)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(OUTPUT_FILE_PATH "{CMAKE_CURRENT_SOURCE_DIR}/workspace")

# Установите стандарт языка
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Выведите информацию о компиляторах
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Bifurcation)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/debri)


set(SOURCES
    src/main.cpp
)

# Укажите заголовочные файлы для компиляции
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/Bifurcation/include
    ${CMAKE_CURRENT_SOURCE_DIR}/debri
    )

# Создайте исполняемый файл
add_executable(ChaosLib ${SOURCES})


target_link_libraries(ChaosLib PRIVATE  ${CUDA_LIBRARIES})
target_link_libraries(ChaosLib PUBLIC CudaBifurcation debri)
target_compile_definitions(ChaosLib PRIVATE OUTPUT_FILE_PATH="${OUTPUT_FILE_PATH}")
set_target_properties(ChaosLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)