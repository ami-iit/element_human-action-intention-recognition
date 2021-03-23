# Copyright (C) 2020 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 14)

set(PACKAGE_VERSION 1.0.0) 
project(HumanActionIntetionRecognition
  VERSION ${PACKAGE_VERSION})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Python REQUIRED)
find_program(PYTHON "Python3")

## install python scripts
set(SETUP_PY    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/__init__.py")
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}")

message("CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}")
message("CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}")

add_custom_command(OUTPUT ${OUTPUT}
                      COMMAND ${PYTHON} ${SETUP_PY} build
                      COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                      DEPENDS ${DEPS})

add_custom_target(target ALL DEPENDS ${OUTPUT})
install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install --prefix ${OUTPUT})")

## install the modules and application files
add_subdirectory(modules)
add_subdirectory(app)
