cmake_minimum_required(VERSION 3.18)
project(paddlevino CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    add_definitions(-DUNICODE -D_UNICODE -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS)
endif ()

# ---------------------------------------------------------------------------
# OpenCV
#
# Resolved via vcpkg (pass -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
# and -DVCPKG_TARGET_TRIPLET=x64-windows on the cmake configure line). vcpkg
# ships with GitHub Actions' windows-latest image, so no extra download step
# is required for OpenCV.
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED)

# ---------------------------------------------------------------------------
# ONNX Runtime
#
# Resolved from an official prebuilt package (see
# https://github.com/microsoft/onnxruntime/releases, asset
# onnxruntime-win-x64-<version>.zip) extracted to ONNXRUNTIME_ROOTDIR.
# Pass -DONNXRUNTIME_ROOTDIR=<path> on the cmake configure line; the CI
# workflow (.github/workflows/build-windows.yml) does this automatically.
# ---------------------------------------------------------------------------
if (NOT DEFINED ONNXRUNTIME_ROOTDIR)
    set(ONNXRUNTIME_ROOTDIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")
endif ()
message(STATUS "ONNXRUNTIME_ROOTDIR: ${ONNXRUNTIME_ROOTDIR}")

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        PATHS "${ONNXRUNTIME_ROOTDIR}/include" "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime")
if (NOT ONNXRUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR "onnxruntime_cxx_api.h not found under ${ONNXRUNTIME_ROOTDIR}/include. "
            "Download an onnxruntime-win-x64 release and pass -DONNXRUNTIME_ROOTDIR=<path>.")
endif ()

find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS "${ONNXRUNTIME_ROOTDIR}/lib")
if (NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "onnxruntime.lib not found under ${ONNXRUNTIME_ROOTDIR}/lib.")
endif ()

# project include
# The base RapidOcrOnnx code includes onnxruntime headers as
# <onnxruntime/core/session/onnxruntime_cxx_api.h>, so expose the include
# root both directly and via an "onnxruntime/" alias directory.
set(ONNXRUNTIME_ALIAS_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_include_alias")
file(MAKE_DIRECTORY "${ONNXRUNTIME_ALIAS_DIR}/onnxruntime")
file(GLOB _ort_headers "${ONNXRUNTIME_INCLUDE_DIR}/*.h")
foreach (_hdr ${_ort_headers})
    get_filename_component(_hdr_name "${_hdr}" NAME)
    configure_file("${_hdr}" "${ONNXRUNTIME_ALIAS_DIR}/onnxruntime/core/session/${_hdr_name}" COPYONLY)
endforeach ()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
include_directories(${ONNXRUNTIME_ALIAS_DIR})

file(GLOB OCR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(paddlevino ${OCR_SRC})
target_link_libraries(paddlevino ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})

if (WIN32)
    # Copy the ONNX Runtime shared library next to the built executable so it
    # runs without needing to be on PATH.
    file(GLOB ONNXRUNTIME_DLLS "${ONNXRUNTIME_ROOTDIR}/lib/*.dll")
    foreach (_dll ${ONNXRUNTIME_DLLS})
        add_custom_command(TARGET paddlevino POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" "$<TARGET_FILE_DIR:paddlevino>"
                COMMENT "Copying ${_dll} next to paddlevino executable")
    endforeach ()
endif ()

install(TARGETS paddlevino RUNTIME DESTINATION bin)
