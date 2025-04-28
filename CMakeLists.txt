cmake_minimum_required(VERSION 3.20)
project(ChaosLib LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Создаем директории для результатов каждой программы
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/workspace/bifurcation)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/workspace/lle)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/workspace/debri)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/workspace/basins)

# Устанавливаем пути для каждой программы
set(BIFURCATION_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/workspace/bifurcation")
set(LLE_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/workspace/lle")
set(DEBRI_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/workspace/debri")
set(BASINS_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/workspace/basins")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")

# Укажите заголовочные файлы для компиляции
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/Bifurcation/include
    ${CMAKE_CURRENT_SOURCE_DIR}/Basins/include
    ${CMAKE_CURRENT_SOURCE_DIR}/debri
    ${CMAKE_CURRENT_SOURCE_DIR}/LLE/include
    ${CMAKE_CURRENT_SOURCE_DIR}/systems
    )

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Bifurcation)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Basins)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/debri)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/LLE)

set(SOURCES
    src/main.cu
)

# Создайте исполняемый файл
add_executable(ChaosLib ${SOURCES})

target_link_libraries(ChaosLib PRIVATE  ${CUDA_LIBRARIES})
target_link_libraries(ChaosLib PUBLIC Bifurcation debri LLE Basins)

# Передаем пути для каждой программы
target_compile_definitions(ChaosLib PRIVATE 
    BIFURCATION_OUTPUT_PATH="${BIFURCATION_OUTPUT_PATH}"
    LLE_OUTPUT_PATH="${LLE_OUTPUT_PATH}"
    DEBRI_OUTPUT_PATH="${DEBRI_OUTPUT_PATH}"
    BASINS_OUTPUT_PATH="${BASINS_OUTPUT_PATH}"
)

set_target_properties(ChaosLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)