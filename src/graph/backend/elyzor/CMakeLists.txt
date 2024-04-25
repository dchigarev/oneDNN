#===============================================================================
# Copyright 2021-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

if(NOT ONEDNN_EXPERIMENTAL_ELYZOR_BACKEND)
    message(STATUS "Elyzor backend is disabled.")
    return()
endif()

message(STATUS "Elyzor backend is enabled.")

if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    set(CCXX_NOWARN_FLAGS "")
    append(CCXX_NOWARN_FLAGS "/wd4200")
    # allow usage of "deprecated" functions
    append(CCXX_NOWARN_FLAGS "/wd4996")
    # inherits via dominance
    append(CCXX_NOWARN_FLAGS "/wd4250")
    # conversion from 'size_t' to 'uint16_t'
    append(CCXX_NOWARN_FLAGS "/wd4267")
    # function assumed not to throw an exception but does
    append(CCXX_NOWARN_FLAGS "/wd4297")
    #  format string '%lu' requires an argument of type 'unsigned long'
    append(CCXX_NOWARN_FLAGS "/wd4477")
    # not enough arguments for function-like macro
    append(CCXX_NOWARN_FLAGS "/wd4003")
    # 
    append(CCXX_NOWARN_FLAGS "/wd4624")
    # 'elem_type': unreferenced local variable
    append(CCXX_NOWARN_FLAGS "/wd4101")
    # unary minus operator applied to unsigned type
    append(CCXX_NOWARN_FLAGS "/wd4146")
    # destructor never returns, potential memory leak
    append(CCXX_NOWARN_FLAGS "/wd4722")
    # needs to have dll-interface to be used by clients of struct
    append(CCXX_NOWARN_FLAGS "/wd4251")
    
    append(CMAKE_CCXX_NOWARN_FLAGS ${CCXX_NOWARN_FLAGS})
    set_property(GLOBAL PROPERTY ELYZOR_CCXX_NOWARN_FLAGS "${CCXX_NOWARN_FLAGS}")
endif()

append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")
append_host_compiler_options(CMAKE_CXX_FLAGS "${DPCPP_CXX_NOWARN_FLAGS}")

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp
    )

set(OBJ_LIB dnnl_graph_backend_elyzor)
add_library(${OBJ_LIB} OBJECT ${SOURCES})

set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
