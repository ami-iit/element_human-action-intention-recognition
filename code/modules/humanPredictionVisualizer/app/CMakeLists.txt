# Copyright (C) 2020 Fondazione Istituto Italiano di Tecnologia (IIT)
# All Rights Reserved.
# Authors: Kourosh Darvish <kourosh.darvish@iit.it>

file(GLOB scripts robots/*.ini)
#yarp_install(FILES ${scripts} DESTINATION ${YARP_ROBOTS_INSTALL_DIR})
yarp_install(FILES ${scripts} DESTINATION bin)


file(GLOB scripts scripts/*.xml)
yarp_install(FILES ${scripts} DESTINATION bin})

