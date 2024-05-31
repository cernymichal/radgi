# CMakeList.txt : CMake project for radgi, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.28)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project("radgi" LANGUAGES CXX CUDA)
add_executable(${PROJECT_NAME})

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

# CXX
set_target_properties(${PROJECT_NAME} PROPERTIES 
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

if (MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            /W4 /external:W0
            /MP # Multi-processor compilation
            /permissive-
            /openmp:llvm # OpenMP support
         >)
else()
	target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            -Wall -Wextra -pedantic
            -fopenmp # OpenMP support
        >)
endif()

# CUDA
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES 75
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_RUNTIME_LIBRARY Static
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --relocatable-device-code=true
        --use_fast_math
        --expt-relaxed-constexpr
        -Xcudafe "--diag_suppress=esa_on_defaulted_function_ignored"
    >)

target_include_directories(${PROJECT_NAME} PRIVATE "src")

# Add source to this project's executable.
target_sources (${PROJECT_NAME} PRIVATE
    src/main.cpp
    src/pch.cpp
    src/pch.h

    src/BVH.cpp
    src/BVH.h
    src/Mesh.cpp
    src/Mesh.h
    src/Scene.cpp
    src/Scene.h
    src/Texture.cpp
    src/Texture.h

    src/Utils/Log.h
    src/Utils/Math.h
    src/Utils/Ptr.h
    src/Utils/Random.h
    src/Utils/Scalars.h

    src/GISolver/CUDASolver/CUDASolver.cpp
    src/GISolver/CUDASolver/CUDASolver.h
    src/GISolver/CUDASolver/CUDAStructs.h
    src/GISolver/CUDASolver/radiosity.cu
    src/GISolver/CUDASolver/utils.cuh
    src/GISolver/GatheringSolver.cpp
    src/GISolver/GatheringSolver.h
    src/GISolver/IGISolver.cpp
    src/GISolver/IGISolver.h
    src/GISolver/ProgressiveSolver.cpp
    src/GISolver/ProgressiveSolver.h
)

target_precompile_headers(${PROJECT_NAME}
    PUBLIC src/pch.h
)

find_package(OpenMP)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

find_package(glm CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)

find_package(miniz CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE miniz::miniz)

find_package(tinyexr CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE unofficial::tinyexr::tinyexr)

find_package(tinyobjloader CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE tinyobjloader::tinyobjloader)

find_package(argh CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE argh)
