cmake_minimum_required(VERSION 3.18)

project(TSDApp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit)

set(SOURCES
        src/tensor.cpp
        src/logger.cpp
        src/main.cpp
)

if(CUDAToolkit_FOUND)
    message(STATUS "CUDA toolkit found, building with GPU support")

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    list(APPEND SOURCES src/tensor_gpu.cu)

    add_executable(TSDApp ${SOURCES})

    target_compile_definitions(TSDApp PRIVATE USE_CUDA=1)

    target_include_directories(TSDApp PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(TSDApp PRIVATE CUDA::cudart)

    set_target_properties(TSDApp PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )

else()
    message(WARNING "CUDA not found, building CPU-only")
    add_executable(TSDApp ${SOURCES})

    target_include_directories(TSDApp PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
endif()