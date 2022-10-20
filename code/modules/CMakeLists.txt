# Copyright (C) 2021 Fondazione Istituto Italiano di Tecnologia (IIT)
# All Rights Reserved.
# Authors: Kourosh Darvish <Kourosh.Darvish@iit.it>

#if(UNIX)
#    find_package(TensorflowCC QUIET)
#endif()

#if(NOT TensorflowCC_FOUND)
#	message(FATAL_ERROR "TensorflowCC is not found" )
#endif()

#add_subdirectory(PredictSequenceDataGenerated_module)


add_subdirectory(humanMotionDataAcquisition)
add_subdirectory(humanPredictionVisualizer)

