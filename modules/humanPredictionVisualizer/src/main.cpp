/*
 * Copyright (C) 2021 Fondazione Istituto Italiano di Tecnologia
 *
 * Licensed under either the GNU Lesser General Public License v3.0 :
 * https://www.gnu.org/licenses/lgpl-3.0.html
 * or the GNU Lesser General Public License v2.1 :
 * https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html
 * at your option.
 */

#include <iDynTree/ModelIO/ModelLoader.h>
#include <iDynTree/Visualizer.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Network.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/sig/Vector.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <thread>
#include <unordered_map>

const std::string ModuleName = "HumanPredictionVisualizer";
const std::string LogPrefix = ModuleName + " :";

std::atomic<bool> isClosing{false};

typedef struct {
  std::string urdfFile;
  std::vector<std::string> jointList;
  std::string basePosePortName;
  std::string jointPositionPortName;
  iDynTree::Model model;
  yarp::os::BufferedPort<yarp::sig::Vector> basePosePort;
  yarp::os::BufferedPort<yarp::sig::Vector> jointPositionPort;
  iDynTree::VectorDynSize joints;
  iDynTree::Transform wHb;
  iDynTree::Vector4 baseOrientationQuaternion;
  iDynTree::Position basePosition;
  iDynTree::Position basePositionOld;
  bool changeModelColor;
  iDynTree::ColorViz modelColor;
  bool visualizeWrenches;
  double forceScalingFactor;
  std::vector<std::string> wrenchSourceLinks;
  std::vector<iDynTree::LinkIndex> wrenchSourceLinkIndices;
  std::string wrenchPortName;
  yarp::os::BufferedPort<yarp::sig::Vector> wrenchPort;
} modelConfiguration_t;

void my_handler(int signal) { isClosing = true; }

#ifdef WIN32

#include <windows.h>

BOOL WINAPI CtrlHandler(DWORD fdwCtrlType) {
  switch (fdwCtrlType) {
    // Handle the CTRL-C signal.
  case CTRL_C_EVENT:
  case CTRL_CLOSE_EVENT:
  case CTRL_SHUTDOWN_EVENT:
    my_handler(0);
    return TRUE;

  // Handle all other events
  default:
    return FALSE;
  }
}
#endif

void handleSigInt() {
#ifdef WIN32
  SetConsoleCtrlHandler(CtrlHandler, TRUE);
#else
  struct sigaction action;
  memset(&action, 0, sizeof(action));
  action.sa_handler = &my_handler;
  sigaction(SIGINT, &action, NULL);
  sigaction(SIGTERM, &action, NULL);
  sigaction(SIGABRT, &action, NULL);
#endif
}

int main(int argc, char *argv[]) {

  // Listen to signals for closing in a clean way the application
  handleSigInt();

  // parse the configuraiton options
  yarp::os::ResourceFinder &rf =
      yarp::os::ResourceFinder::getResourceFinderSingleton();
  rf.setDefaultConfigFile("HumanPredictionVisualizer.ini");
  rf.configure(argc, argv);

  if (rf.isNull()) {
    yError() << LogPrefix << "Empty configuration file.";
    return EXIT_FAILURE;
  }

  iDynTree::Position cameraDeltaPosition;
  if (!(rf.check("cameraDeltaPosition") &&
        rf.find("cameraDeltaPosition").isList() &&
        rf.find("cameraDeltaPosition").asList()->size() == 3)) {
    yError() << LogPrefix
             << "'cameraDeltaPosition' option not found or not valid.";
    return EXIT_FAILURE;
  }
  for (size_t idx = 0; idx < 3; idx++) {
    if (!(rf.find("cameraDeltaPosition").asList()->get(idx).isFloat64())) {
      yError() << LogPrefix << "'cameraDeltaPosition' entry [ " << idx
               << " ] is not valid.";
      return EXIT_FAILURE;
    }
    cameraDeltaPosition.setVal(
        idx, rf.find("cameraDeltaPosition").asList()->get(idx).asFloat64());
  }

  if (!(rf.check("useFixedCamera") && rf.find("useFixedCamera").isBool())) {
    yError() << LogPrefix << "'useFixedCamera' option not found or not valid.";
    return EXIT_FAILURE;
  }
  bool useFixedCamera = rf.find("useFixedCamera").asBool();

  // define base link for human model
  std::string baseLink;
  if (!(rf.check("baseLink") && rf.find("baseLink").isString())) {
    yError() << LogPrefix << "'baseLink' option not found or not valid.";
    return EXIT_FAILURE;
  }
  baseLink = rf.find("baseLink").asString();


  iDynTree::Position fixedCameraTarget;
  if (useFixedCamera) {
    if (!(rf.check("fixedCameraTarget") &&
          rf.find("fixedCameraTarget").isList() &&
          rf.find("fixedCameraTarget").asList()->size() == 3)) {
      yError() << LogPrefix
               << "'fixedCameraTarget' option not found or not valid.";
      return EXIT_FAILURE;
    }
    for (size_t idx = 0; idx < 3; idx++) {
      if (!(rf.find("fixedCameraTarget").asList()->get(idx).isFloat64())) {
        yError() << LogPrefix << "'fixedCameraTarget' entry [ " << idx
                 << " ] is not valid.";
        return EXIT_FAILURE;
      }
      fixedCameraTarget.setVal(
          idx, rf.find("fixedCameraTarget").asList()->get(idx).asFloat64());
    }
  }

  if (!(rf.check("maxVisualizationFPS") &&
        rf.find("maxVisualizationFPS").isInt32() &&
        rf.find("maxVisualizationFPS").asInt32() > 0)) {
    yError() << LogPrefix
             << "'maxVisualizationFPS' option not found or not valid...";
    return EXIT_FAILURE;
  }
  unsigned int maxVisualizationFPS = rf.find("maxVisualizationFPS").asInt32();
  //unsigned int maxVisualizationFPS = 40;

  if (!(rf.check("models") && rf.find("models").isList())) {
    yError() << LogPrefix << "'models' option not found or not valid.";
    return EXIT_FAILURE;
  }
  auto modelList = rf.find("models").asList();
  std::vector<std::string> modelNameList;
  for (size_t it = 0; it < modelList->size(); it++) {
    if (!modelList->get(it).isString()) {
      yError() << LogPrefix
               << "in 'models' there is a field that is not a string.";
      return EXIT_FAILURE;
    }
    modelNameList.push_back(modelList->get(it).asString());
  }

  std::string cameraFocusModel;
  if (!useFixedCamera) {
    if (!(rf.check("cameraFocusModel") &&
          rf.find("cameraFocusModel").isString())) {
      yError() << LogPrefix
               << "'cameraFocusModel' option not found or not valid.";
      return EXIT_FAILURE;
    }
    cameraFocusModel = rf.find("cameraFocusModel").asString();
    if (std::find(modelNameList.begin(), modelNameList.end(),
                  cameraFocusModel) == modelNameList.end()) {
      yError() << LogPrefix << " the choosen 'cameraFocusModel' is not found.";
      return EXIT_FAILURE;
    }
  }

  std::unordered_map<std::string, modelConfiguration_t> modelConfigurationMap;
  yarp::os::Bottle modelConfigurationGroup;

  for (auto &modelName : modelNameList) {
    if (!(rf.check(modelName) && rf.find(modelName).isList())) {
      yError() << LogPrefix << "group [" << modelNameList
               << "] not found or not valid";
      return EXIT_FAILURE;
    }
    modelConfigurationGroup = rf.findGroup(modelName);

    if (!(modelConfigurationGroup.check("modelURDFName") &&
          modelConfigurationGroup.find("modelURDFName").isString())) {
      yError() << LogPrefix
               << "'modelURDFName' option not found or not valid in "
               << modelName;
      return EXIT_FAILURE;
    }
    modelConfigurationMap[modelName];
    modelConfigurationMap[modelName].urdfFile =
        modelConfigurationGroup.find("modelURDFName").asString();

    modelConfigurationMap[modelName].changeModelColor = false;
    if (modelConfigurationGroup.check("modelColor")) {
      iDynTree::Vector4 modelColorVector;
      if (!(modelConfigurationGroup.find("modelColor").isList() &&
            modelConfigurationGroup.find("modelColor").asList()->size() == 4)) {
        yError() << LogPrefix << "'modelColor' option not valid in "
                 << modelName;
        return EXIT_FAILURE;
      }
      for (size_t idx = 0; idx < 4; idx++) {
        if (!(modelConfigurationGroup.find("modelColor")
                  .asList()
                  ->get(idx)
                  .isFloat64())) {
          yError() << LogPrefix << "'modelColor' entry [ " << idx
                   << " ] is not valid.";
          return EXIT_FAILURE;
        }
        modelColorVector.setVal(idx, modelConfigurationGroup.find("modelColor")
                                         .asList()
                                         ->get(idx)
                                         .asFloat64());
      }
      modelConfigurationMap[modelName].changeModelColor = true;
      modelConfigurationMap[modelName].modelColor =
          iDynTree::ColorViz(modelColorVector);
    }

    modelConfigurationMap[modelName].visualizeWrenches = false;
    if (modelConfigurationGroup.check("visualizeWrenches")) {
      if (!(modelConfigurationGroup.find("visualizeWrenches").isBool())) {
        yError() << LogPrefix << "'visualizeWrenches' option not valid in "
                 << modelName;
        return EXIT_FAILURE;
      }
      modelConfigurationMap[modelName].visualizeWrenches =
          modelConfigurationGroup.find("visualizeWrenches").asBool();
    }

    if (modelConfigurationMap[modelName].visualizeWrenches) {
      if (!(modelConfigurationGroup.check("forceScalingFactor") &&
            modelConfigurationGroup.find("forceScalingFactor").isFloat64())) {
        yError() << LogPrefix
                 << "'forceScalingFactor' option not found or not valid in "
                 << modelName << ". Wrench Visualization will be disabled.";
        modelConfigurationMap[modelName].visualizeWrenches = false;
      } else {
        modelConfigurationMap[modelName].forceScalingFactor =
            modelConfigurationGroup.find("forceScalingFactor").asFloat64();
      }
    }

    if (!(modelConfigurationGroup.check("jointList") &&
          modelConfigurationGroup.find("jointList").isList())) {
      yError() << LogPrefix << "'jointList' option not found or not valid in "
               << modelName;
      return EXIT_FAILURE;
    }
    auto jointListBottle = modelConfigurationGroup.find("jointList").asList();
    for (int i = 0; i < jointListBottle->size(); i++) {
      // check if the elements of the bottle are strings
      if (!jointListBottle->get(i).isString()) {
        yError() << LogPrefix
                 << "in 'jointList' there is a field that is not a string. in "
                 << modelName;
        return EXIT_FAILURE;
      }
      modelConfigurationMap[modelName].jointList.push_back(
          jointListBottle->get(i).asString());
    }

    if (modelConfigurationMap[modelName].visualizeWrenches) {
      if (!(modelConfigurationGroup.check("wrenchSourceLinks") &&
            modelConfigurationGroup.find("wrenchSourceLinks").isList())) {
        yError() << LogPrefix
                 << "'wrenchSourceLinks' option not found or not valid in "
                 << modelName << ". Wrench Visualization will be disabled";
        modelConfigurationMap[modelName].visualizeWrenches = false;
      } else {
        auto wrenchSourceLinksList =
            modelConfigurationGroup.find("wrenchSourceLinks").asList();
        for (size_t it = 0; it < wrenchSourceLinksList->size(); it++) {
          if (!wrenchSourceLinksList->get(it).isString()) {
            yError() << LogPrefix << "in 'wrenchSourceLinks' in " << modelName
                     << " there is a field that is not a string.";
            return EXIT_FAILURE;
          }
          modelConfigurationMap[modelName].wrenchSourceLinks.push_back(
              wrenchSourceLinksList->get(it).asString());
        }
      }
    }

    if (!(modelConfigurationGroup.check("basePosePortName") &&
          modelConfigurationGroup.find("basePosePortName").isString())) {
      yError() << LogPrefix
               << "'basePosePortName' option not found or not valid in "
               << modelName;
      return EXIT_FAILURE;
    }
    modelConfigurationMap[modelName].basePosePortName =
        modelConfigurationGroup.find("basePosePortName").asString();

    // if joint list is empty, skip the jointPositionPort
    if (!modelConfigurationMap[modelName].jointList.empty()) {
      if (!(modelConfigurationGroup.check("jointPositionPortName") &&
            modelConfigurationGroup.find("jointPositionPortName").isString())) {
        yError() << LogPrefix
                 << "'jointPositionPortName' option not found or not valid in "
                 << modelName;
        return EXIT_FAILURE;
      }
      modelConfigurationMap[modelName].jointPositionPortName =
          modelConfigurationGroup.find("jointPositionPortName").asString();
    }

    if (modelConfigurationMap[modelName].visualizeWrenches) {
      if (!(modelConfigurationGroup.check("wrenchPortName") &&
            modelConfigurationGroup.find("wrenchPortName").isString())) {
        yError() << LogPrefix
                 << "'wrenchPortName' option not found or not valid in "
                 << modelName << ". Wrench Visualization will be disabled";
        modelConfigurationMap[modelName].visualizeWrenches = false;
      } else {
        modelConfigurationMap[modelName].wrenchPortName =
            modelConfigurationGroup.find("wrenchPortName").asString();
      }
    }
  }

  // initialise yarp network
  yarp::os::Network yarp;
  if (!yarp.checkNetwork()) {
    yError() << LogPrefix << "[main] Unable to find YARP network";
    return EXIT_FAILURE;
  }

  // initialize buffers variables
  yarp::sig::Vector *jointValuesVector;
  yarp::sig::Vector *wrenchValuesVector;
  yarp::sig::Vector *basePoseVector;

  iDynTree::Transform linkTransform;
  iDynTree::Direction force;

  // initialise models
  for (auto &modelName : modelNameList) {
    // load model
    std::string urdfFilePath =
        rf.findFile(modelConfigurationMap[modelName].urdfFile);
    if (urdfFilePath.empty()) {
      yError() << LogPrefix << "Failed to find file"
               << modelConfigurationMap[modelName].urdfFile;
      return EXIT_FAILURE;
    }

    iDynTree::ModelLoader modelLoader;
    if (!modelLoader.loadModelFromFile(urdfFilePath) ||
        !modelLoader.isValid()) {
      yError() << LogPrefix << "Failed to load model" << urdfFilePath;
      return EXIT_FAILURE;
    }
    modelConfigurationMap[modelName].model = modelLoader.model();
    // !!Here modified: determine base link in HumanPredictionVisualizer.ini file!!
    int baseIndex = modelConfigurationMap[modelName].model.getLinkIndex(baseLink);
    modelConfigurationMap[modelName].model.setDefaultBaseLink(baseIndex);

    // check if the selected joints exist in the model
    yInfo() << LogPrefix << "Selected [ "
            << modelConfigurationMap[modelName].jointList.size() << " ] joints";
    for (auto jointName : modelConfigurationMap[modelName].jointList) {
      if (modelConfigurationMap[modelName].model.getJointIndex(jointName) ==
          iDynTree::JOINT_INVALID_INDEX) {
        yError() << LogPrefix << "joint [ " << jointName
                 << " ] not found in the visualized model.";
        return EXIT_FAILURE;
      }
    }

    // check if the selected links exist in the model
    modelConfigurationMap[modelName].wrenchSourceLinkIndices.clear();
    yInfo() << LogPrefix << "Selected [ "
            << modelConfigurationMap[modelName].wrenchSourceLinks.size()
            << " ] links for wrench measurements";
    for (auto linkName : modelConfigurationMap[modelName].wrenchSourceLinks) {
      auto frameIndex =
          modelConfigurationMap[modelName].model.getFrameIndex(linkName);
      if (frameIndex == iDynTree::FRAME_INVALID_INDEX) {
        yError() << LogPrefix << "link [ " << linkName
                 << " ] not found in the visualized model.";
        return EXIT_FAILURE;
      }
      modelConfigurationMap[modelName].wrenchSourceLinkIndices.push_back(
          frameIndex);
    }

    // Connect to the base pose port
    modelConfigurationMap[modelName].basePosePort.open(
        "/" + ModuleName + "/" + modelName + "/basePose:i");
    if (modelConfigurationMap[modelName].basePosePort.isClosed()) {
      yError() << LogPrefix << "failed to open the port "
               << modelConfigurationMap[modelName].basePosePort.getName();
      return EXIT_FAILURE;
    }
    if (!yarp.connect(
            modelConfigurationMap[modelName].basePosePortName,
            modelConfigurationMap[modelName].basePosePort.getName())) {
      yError() << LogPrefix << "failed to connect to the port"
               << modelConfigurationMap[modelName].basePosePortName;
      return EXIT_FAILURE;
    }
    basePoseVector = modelConfigurationMap[modelName].basePosePort.read(true);
    while (basePoseVector == nullptr) {
      yError() << LogPrefix << "no data coming from the port "
               << modelConfigurationMap[modelName].basePosePortName;
      return EXIT_FAILURE;
    }

    // check if the base pose data has the correct lenght (7 values: pos, quat)
    if (basePoseVector->size() != 7) {
      yError() << LogPrefix << "reading base pose data with size ["
               << basePoseVector->size() << "] expected [7]";
      return EXIT_FAILURE;
    }

    // if joint list is empty, skip the jointPositionPort
    if (!modelConfigurationMap[modelName].jointList.empty()) {
      // Connect to the joint position port
      modelConfigurationMap[modelName].jointPositionPort.open(
          "/" + ModuleName + "/" + modelName + "jointPosition:i");
      if (modelConfigurationMap[modelName].jointPositionPort.isClosed()) {
        yError()
            << LogPrefix << "failed to open the port"
            << modelConfigurationMap[modelName].jointPositionPort.getName();
        return EXIT_FAILURE;
      }
      if (!yarp.connect(
              modelConfigurationMap[modelName].jointPositionPortName,
              modelConfigurationMap[modelName].jointPositionPort.getName())) {
        yError() << LogPrefix << "failed to connect to the port "
                 << modelConfigurationMap[modelName].jointPositionPortName;
        return EXIT_FAILURE;
      }
      yInfo() << "connected to the ports: "
              << modelConfigurationMap[modelName].jointPositionPortName
              << modelConfigurationMap[modelName].jointPositionPort.getName();
      jointValuesVector =
          modelConfigurationMap[modelName].jointPositionPort.read(true);
      while (jointValuesVector == nullptr) {
        yError() << LogPrefix << "no data coming from the port "
                 << modelConfigurationMap[modelName].jointPositionPortName;
        return EXIT_FAILURE;
      }
      yInfo() << "joint values: " << jointValuesVector->toString();

      // check if the joits values read from the port correspond to the length
      // of the joint list
      if (jointValuesVector->size() !=
          modelConfigurationMap[modelName].jointList.size()) {
        yError() << LogPrefix << "reading vector of joint position with size ["
                 << jointValuesVector->size()
                 << " ] different from lenght of jointList size ["
                 << modelConfigurationMap[modelName].jointList.size() << "] ";
        return EXIT_FAILURE;
      }
    }

    // connect to the wrench port
    if (modelConfigurationMap[modelName].visualizeWrenches) {
      // Connect to the wrenches port
      modelConfigurationMap[modelName].wrenchPort.open("/" + ModuleName + "/" +
                                                       modelName + "/wrench:i");
      if (modelConfigurationMap[modelName].wrenchPort.isClosed()) {
        yError() << LogPrefix << "failed to open the port"
                 << modelConfigurationMap[modelName].wrenchPort.getName();
        return EXIT_FAILURE;
      }
      if (!yarp.connect(
              modelConfigurationMap[modelName].wrenchPortName,
              modelConfigurationMap[modelName].wrenchPort.getName())) {
        yError() << LogPrefix << "failed to connect to the port "
                 << modelConfigurationMap[modelName].wrenchPortName;
        return EXIT_FAILURE;
      }
      wrenchValuesVector =
          modelConfigurationMap[modelName].wrenchPort.read(true);
      while (wrenchValuesVector == nullptr) {
        yError() << LogPrefix << "no data coming from the port "
                 << modelConfigurationMap[modelName].wrenchPortName;
        return EXIT_FAILURE;
      }

      // check if the joits values read from the port correspond to the length
      // of the joint list
      if (wrenchValuesVector->size() !=
          6 * modelConfigurationMap[modelName].wrenchSourceLinks.size()) {
        yError() << LogPrefix << "reading vector of wrenches with size ["
                 << wrenchValuesVector->size()
                 << " ] different from expected size [ 6 * "
                 << modelConfigurationMap[modelName].wrenchSourceLinks.size()
                 << "] ";
        return EXIT_FAILURE;
      }
    }

    // initialize state variables for visualization
    modelConfigurationMap[modelName].joints.resize(
        modelConfigurationMap[modelName].model.getNrOfDOFs());
    modelConfigurationMap[modelName].joints.zero();
    modelConfigurationMap[modelName].wHb = iDynTree::Transform::Identity();
    modelConfigurationMap[modelName].basePosition.zero();
    modelConfigurationMap[modelName].basePositionOld = fixedCameraTarget;
  }

  // initialize visualization
  iDynTree::Visualizer viz;
  iDynTree::VisualizerOptions options;

  viz.init(options);

  viz.camera().setPosition(cameraDeltaPosition);
  viz.camera().setTarget(fixedCameraTarget);

  viz.camera().animator()->enableMouseControl(true);

  iDynTree::ColorViz env_color(0.85, 0.85, 0.89, 0.1);
  iDynTree::ColorViz grid_color(0.6, 0.6, 0.6, 1.0);

  if (!viz.setColorPalette("meshcat")) {
    yError() << "cannot set the color pallete.";
    return EXIT_FAILURE;
  }
  viz.enviroment().setBackgroundColor(env_color);
  //  viz.enviroment().setFloorGridColor(grid_color);

  // add models to viz
  for (auto &modelName : modelNameList) {
    viz.addModel(modelConfigurationMap[modelName].model, modelName);
    if (modelConfigurationMap[modelName].changeModelColor) {
      viz.modelViz(modelName).setModelColor(
          modelConfigurationMap[modelName].modelColor);
    }

    // add forces to viz
    if (modelConfigurationMap[modelName].visualizeWrenches) {
      for (size_t vectorIndex = 0;
           vectorIndex <
           modelConfigurationMap[modelName].wrenchSourceLinks.size();
           vectorIndex++) {
        linkTransform = viz.modelViz(modelName).getWorldLinkTransform(
            modelConfigurationMap[modelName].wrenchSourceLinkIndices.at(
                vectorIndex));
        for (size_t i = 0; i < 3; i++) {
          force.setVal(i, modelConfigurationMap[modelName].forceScalingFactor *
                              wrenchValuesVector->data()[6 * vectorIndex + i]);
        }
        force = linkTransform.getRotation() * force;
        viz.vectors().addVector(linkTransform.getPosition(), force);
      }
    }
  }
  //  yInfo() << "frame 0: " << viz.frames().getFrameLabel(0);

  // start Visualization
  std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point lastViz =
      std::chrono::steady_clock::now();

  long minimumMicroSecViz = std::round(1e6 / (double)maxVisualizationFPS);
  //long minimumMicroSecViz = std::round(1e6 / maxVisualizationFPS);

  while (viz.run() && !isClosing) {
    now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::microseconds>(now - lastViz)
            .count() < minimumMicroSecViz) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    for (auto &modelName : modelNameList) {

      // read values from base pose port
      basePoseVector =
          modelConfigurationMap[modelName].basePosePort.read(false);
      if (basePoseVector != nullptr) {
        for (size_t i = 0; i < 4; i++) {
          modelConfigurationMap[modelName].baseOrientationQuaternion.setVal(
              i, basePoseVector->data()[3 + i]);
        }

        modelConfigurationMap[modelName].basePosition = iDynTree::Position(
            basePoseVector->data()[0], basePoseVector->data()[1],
            basePoseVector->data()[2]);
        modelConfigurationMap[modelName].wHb.setRotation(
            iDynTree::Rotation::RotationFromQuaternion(
                modelConfigurationMap[modelName].baseOrientationQuaternion));
        modelConfigurationMap[modelName].wHb.setPosition(
            modelConfigurationMap[modelName].basePosition);
      }
      // if joint list is empty, skip the jointPositionPort
      if (!modelConfigurationMap[modelName].jointList.empty()) {
        // read values from joint position port
        jointValuesVector =
            modelConfigurationMap[modelName].jointPositionPort.read(false);
        if (jointValuesVector != nullptr) {
          for (size_t jointPosPortIdx = 0;
               jointPosPortIdx <
               modelConfigurationMap[modelName].jointList.size();
               jointPosPortIdx++) {
            std::string jointName =
                modelConfigurationMap[modelName].jointList.at(jointPosPortIdx);
            double jointVal = jointValuesVector->data()[jointPosPortIdx];

            iDynTree::JointIndex jointIndex =
                viz.modelViz(modelName).model().getJointIndex(jointName);
            if (jointIndex != iDynTree::JOINT_INVALID_INDEX) {
              modelConfigurationMap[modelName].joints.setVal(jointIndex,
                                                             jointVal);
            }
          }
        }
      }

      // Update the visulizer
      viz.modelViz(modelName).setPositions(
          modelConfigurationMap[modelName].wHb,
          modelConfigurationMap[modelName].joints);

      if (modelConfigurationMap[modelName].visualizeWrenches) {
        wrenchValuesVector =
            modelConfigurationMap[modelName].wrenchPort.read(false);
        if (wrenchValuesVector != nullptr) {
          for (size_t vectorIndex = 0;
               vectorIndex <
               modelConfigurationMap[modelName].wrenchSourceLinks.size();
               vectorIndex++) {
            linkTransform = viz.modelViz(modelName).getWorldLinkTransform(
                modelConfigurationMap[modelName].wrenchSourceLinkIndices.at(
                    vectorIndex));
            for (size_t i = 0; i < 3; i++) {
              force.setVal(i,
                           modelConfigurationMap[modelName].forceScalingFactor *
                               wrenchValuesVector->data()[6 * vectorIndex + i]);
            }

            force = linkTransform.getRotation() * force;
            viz.vectors().updateVector(vectorIndex, linkTransform.getPosition(),
                                       force);
          }
        }
      }
    }

    // follow the desired link with the camera
    if (!useFixedCamera) {
      auto modelConfigurationPair =
          modelConfigurationMap.find(cameraFocusModel);
      cameraDeltaPosition =
          viz.camera().getPosition() -
          modelConfigurationMap[cameraFocusModel].basePositionOld;
      viz.camera().setPosition(
          modelConfigurationMap[cameraFocusModel].basePosition +
          cameraDeltaPosition);
      viz.camera().setTarget(
          modelConfigurationMap[cameraFocusModel].basePosition);

      modelConfigurationMap[cameraFocusModel].basePositionOld =
          modelConfigurationMap[cameraFocusModel].basePosition;
    }

    viz.draw();
    lastViz = std::chrono::steady_clock::now();
  }

  // close the ports
  for (auto &modelName : modelNameList) {
    modelConfigurationMap[modelName].basePosePort.close();
    // if joint list is empty, skip the jointPositionPort
    if (!modelConfigurationMap[modelName].jointList.empty()) {
      modelConfigurationMap[modelName].jointPositionPort.close();
    }
    if (!modelConfigurationMap[modelName].wrenchSourceLinks.empty()) {
      modelConfigurationMap[modelName].wrenchPort.close();
    }
  }

  viz.close();

  return 0;
}
