cmake_minimum_required(VERSION 3.20)
project(ChaosLib LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(OUTPUT_FILE_PATH "{CMAKE_CURRENT_SOURCE_DIR}/workspace")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Bifurcation)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/debri)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/LLE)


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
    ${CMAKE_CURRENT_SOURCE_DIR}/LLE/include
    )

# Создайте исполняемый файл
add_executable(ChaosLib ${SOURCES})

target_link_libraries(ChaosLib PRIVATE  ${CUDA_LIBRARIES})
target_link_libraries(ChaosLib PUBLIC CudaBifurcation debri LLE)
target_compile_definitions(ChaosLib PRIVATE OUTPUT_FILE_PATH="${OUTPUT_FILE_PATH}")
set_target_properties(ChaosLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)