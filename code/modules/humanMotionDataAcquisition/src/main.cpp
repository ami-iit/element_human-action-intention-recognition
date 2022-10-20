/**
 * @file main.cpp
 * @authors Giulio Romualdi <giulio.romualdi@iit.it>
 * @copyright 2018 iCub Facility - Istituto Italiano di Tecnologia
 *            Released under the terms of the LGPLv2.1 or later, see LGPL.TXT
 * @date 2018
 */

// YARP
#include <module.hpp>
#include <thread>
#include <yarp/os/LogStream.h>
#include <yarp/os/Network.h>
#include <yarp/os/RFModule.h>

int main(int argc, char *argv[]) {
  // initialise yarp network
  yarp::os::Network yarp;
  if (!yarp.checkNetwork()) {
    yError() << "[main] Unable to find YARP network";
    return EXIT_FAILURE;
  }

  // prepare and configure the resource finder
  yarp::os::ResourceFinder &rf =
      yarp::os::ResourceFinder::getResourceFinderSingleton();
  
  /*
  const yarp::os::Searchable &config = 
      yarp::os::Searchable::check();
  */

  rf.setDefaultConfigFile("HumanDataAcquisition.ini");

  rf.configure(argc, argv);

  // create the module
  HumanDataAcquisitionModule module;

  if (!module.configure(rf)) {
    yError() << "[main] cannot configure the module";
    return 1;
  }
  try {

    std::thread run_thread(&HumanDataAcquisitionModule::updateModule, &module);

    std::thread keyboard_thread(&HumanDataAcquisitionModule::keyboardHandler,
                                &module);

    keyboard_thread.join();
    run_thread.join();

  } catch (std::exception &e) {
    std::cerr << "Unhandled Exception: " << e.what() << std::endl;
  }

  return 0;
}
