# Copyright (C) 2021 Fondazione Istituto Italiano di Tecnologia (IIT)
# All Rights Reserved.
# Authors: Kourosh Darvish <Kourosh.Darvish@iit.it>

# set target name
set(EXE_TARGET_NAME SequencePrediction_DataGenerated)


find_package(YARP REQUIRED)

# set cpp files
set(${EXE_TARGET_NAME}_SRC
  src/main.cpp
  src/SequencePrediction.cpp)

# set hpp files
set(${EXE_TARGET_NAME}_HDR
  include/SequencePrediction.hpp)

# add include directories to the build.
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# add an executable to the project using the specified source files.
add_executable(${EXE_TARGET_NAME} ${${EXE_TARGET_NAME}_SRC} ${${EXE_TARGET_NAME}_HDR} )

target_link_libraries(${EXE_TARGET_NAME} LINK_PUBLIC
  ${YARP_LIBRARIES}
  TensorflowCC::TensorflowCC
  )

install(TARGETS ${EXE_TARGET_NAME} DESTINATION bin)
