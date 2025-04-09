cmake_minimum_required(VERSION 3.20)
project(Systems LANGUAGES CUDA)

# Collect all system files
set(SOURCES
    systems.cu
)

set(HEADERS
    systems.cuh
)

# Create a static library for all systems
add_library(Systems STATIC ${SOURCES})

# Set include directories
target_include_directories(Systems PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set CUDA properties
set_target_properties(Systems PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Add device code linking flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -dc")

# Make sure the system files are installed
install(FILES ${HEADERS}
    DESTINATION include/systems
) 