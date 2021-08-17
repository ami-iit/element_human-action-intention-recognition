# Copyright (C) 2021 Fondazione Istituto Italiano di Tecnologia (IIT)
# All Rights Reserved.
# Authors: Kourosh Darvish <Kourosh.Darvish@iit.it>

# List the subdirectory
# http://stackoverflow.com/questions/7787823/cmake-how-to-get-the-name-of-all-subdirectories-of-a-directory
macro(SUBDIRLIST result curdir)
  file(GLOB children RELATIVE ${curdir} ${curdir}/*)
  set(dirlist "")
  foreach(child ${children})
    if(IS_DIRECTORY ${curdir}/${child})
      list(APPEND dirlist ${child})
    endif()
  endforeach()
  set(${result} ${dirlist})
endmacro()

# Get list of models
subdirlist(subdirs ${CMAKE_CURRENT_SOURCE_DIR}/robots/)

# Install each model
foreach (dir ${subdirs})
  file(GLOB scripts config/*.ini)
  yarp_install(FILES ${scripts} DESTINATION bin)
endforeach ()

file(GLOB scripts scripts/*.xml)
yarp_install(FILES ${scripts} DESTINATION bin)

if (MSVC)
  yarp_install(DIRECTORY virtualizerRetargeting DESTINATION bin})
endif (MSVC)
