//#include "yarp/ HumanState.h"
#include <chrono>
#include <fstream>
#include <thread>

#include <module.hpp>

class HumanDataAcquisitionModule::impl {
public:
  /*
   * map the joint values (order) coming from HDE to the controller order
   * @param robotJointsListNames list of the joint names used in controller
   * @param humanJointsListName list of the joint names received from the HDE
   * (human-dynamics-estimation repository)
   * @param humanToRobotMap the container for mapping of the human joints to the
   * robot ones
   * @return true in case of success and false otherwise
   */
  bool mapJointsHDE2Controller(std::vector<std::string> robotJointsListNames,
                               std::vector<std::string> humanJointsListName,
                               std::vector<unsigned> &humanToRobotMap);
};

HumanDataAcquisitionModule::HumanDataAcquisitionModule() : pImpl{new impl()} {};

HumanDataAcquisitionModule::~HumanDataAcquisitionModule(){};

bool HumanDataAcquisitionModule::configure(yarp::os::ResourceFinder &rf) {
  // check if the configuration file is empty
  m_isClosed = false;
  if (rf.isNull()) {
    yError() << "[HumanDataAcquisition::configure] Empty configuration for the "
                "OculusModule "
                "application.";
    return false;
  }

  // get the period
  m_dT = rf.check("samplingTime", yarp::os::Value(0.1)).asFloat64();

  // set the module name
  std::string name;
  name = rf.check("name", yarp::os::Value("HumanDataAcquisition")).asString();

  setName(name.c_str());

  m_logData = rf.check("logData", yarp::os::Value(true)).asBool();

  m_readDataFromFile = rf.check("readFromFile", yarp::os::Value(true)).asBool();

  m_DataLineIndex = 0;
  m_time = 0.0;

  m_useJointValues =
      rf.check("useJointValues", yarp::os::Value(false)).asBool();
  m_useJointVelocities =
      rf.check("useJointVelocities", yarp::os::Value(false)).asBool();
  /*
  m_useLeftFootWrench =
      rf.check("useLeftFootWrench", yarp::os::Value(false)).asBool();
  m_useRightFootWrench =
      rf.check("useRightFootWrench", yarp::os::Value(false)).asBool();
  */
  m_useFeetWrench = rf.check("useFeetWrench", yarp::os::Value(false)).asBool();

  m_useBaseValues = rf.check("useBaseValues", yarp::os::Value(false)).asBool();

  m_useComValues = rf.check("useCOMValues", yarp::os::Value(false)).asBool();

  m_streamData = rf.check("streamData", yarp::os::Value(true)).asBool();

  m_useForAnnotation =
      rf.check("useForAnnotation", yarp::os::Value(false)).asBool();

  // the annotation list size
  m_annotationList.resize(0);
  if (rf.check("AnnotationList")) {
    yarp::os::Value *annotationListYarp;
    if (!rf.check("AnnotationList", annotationListYarp)) {
      yError()
          << "[HumanDataAcquisition::configure] Unable to find AnnotationList"
             "into config file.";
      return false;
    }
    yarpListToStringVector(annotationListYarp, m_annotationList);
  }

  m_isPaused = false;

  m_latestAnnotation =
      rf.check("InitialAnnotation", yarp::os::Value("None")).asString();

  // initialize minimum jerk trajectory for the whole body
  yarp::os::Value *axesListYarp;

  if (!rf.check("joints_list", axesListYarp)) {
    yError() << "[HumanDataAcquisition::configure] Unable to find joints_list"
                "into config file.";
    return false;
  }

  yarpListToStringVector(axesListYarp, m_robotJointsListNames);

  m_actuatedDOFs = m_robotJointsListNames.size();

  m_jointValues.resize(m_actuatedDOFs, 0.0);
  m_jointVelocities.resize(m_actuatedDOFs, 0.0);

  yInfo() << "HumanDataAcquisition::configure:  NoOfJoints: " << m_actuatedDOFs;

  std::string portNameIn, portNameOut;

  // when reading data from yarp port:
  if (!m_readDataFromFile) {
    portNameIn = rf.check("HDEJointsPortIn",
                          yarp::os::Value("/HumanStateWrapper/state:i"))
                     .asString();

    if (!m_wholeBodyHumanJointsPort.open("/" + getName() + portNameIn)) {
      yError() << "[HumanDataAcquisition::configure] " << portNameIn
               << " port already open.";
      return false;
    }

    portNameOut = rf.check("HDEJointsPortOut",
                           yarp::os::Value("/HDE/HumanStateWrapper/state:o"))
                      .asString();

    yarp::os::Network::connect(portNameOut, "/" + getName() + portNameIn);
    
    // **********MODIFY THE CODE: START**********
    // open the input port for receiving the data of left shoe
    /*
    if (m_useLeftFootWrench) {
      portNameIn = rf.check("WearablesBothShoesPortIn",
                            yarp::os::Value("/iFeelSuit/WearableData/data:i"))
                       .asString();
      if (!m_bothShoesPort.open("/" + getName() + portNameIn)) {
        yError() << "[HumanDataAcquisition::configure] " << portNameIn
                 << " port already open.";
        return false;
      }

      // check the output port that we want to read is available
      portNameOut = rf.check("WearablesBothShoesPortOut",
                             yarp::os::Value("/iFeelSuit/WearableData/data:o"))
                        .asString();
      
      // connect the input and output ports
      yarp::os::Network::connect(portNameOut, "/" + getName() + portNameIn);
    }

    // open the input port for receiving the data of right shoe
    if (m_useRightFootWrench) {
      portNameIn = rf.check("WearablesBothShoesPortIn",
                            yarp::os::Value("/iFeelSuit/WearableData/data:i"))
                       .asString();
      if (!m_bothShoesPort.open("/" + getName() + portNameIn)) {
        yError() << "[HumanDataAcquisition::configure] " << portNameIn
                 << " port already open.";
        return false;
      }

      // check the output port is available
      portNameOut =
          rf.check("WearablesBothShoesPortOut",
                   yarp::os::Value("/iFeelSuit/WearableData/data:o"))
              .asString();
      
      // connect the input and output ports
      yarp::os::Network::connect(portNameOut, "/" + getName() + portNameIn);
    }
    */
    if (m_useFeetWrench) {
      portNameIn = rf.check("WearablesBothShoesPortIn",
                              yarp::os::Value("/iFeelSuit/WearableData/data:i")).asString();
      
      if (!m_bothShoesPort.open("/" + getName() + portNameIn)) {
        yError() << "[HumanDataAcquisition::configure] " << portNameIn
                 << "port already open.";
        return false;
      }

      portNameOut = rf.check("WearablesBothShoesPortOut",
                             yarp::os::Value("/iFeelSuit/WearableData/data:o")).asString();

      yarp::os::Network::connect(portNameOut, "/" + getName() + portNameIn);
    }
    // **********MODIFY THE CODE: END**********
  }

  if (m_readDataFromFile) {
    std::string fileNameToRead =
        rf.check("filePathToRead", yarp::os::Value("")).asString();

    m_readDataMapVector.resize(0);

    this->readDataFile(fileNameToRead);
  }

  if (m_streamData) {
    portNameIn =
        rf.check("humanJointsPort", yarp::os::Value("/jointPosition:o"))
            .asString();

    if (!m_wholeBodyJointsPort.open("/" + getName() + portNameIn)) {
      yError() << "[HumanDataAcquisition::configure] Unable to open the port "
               << portNameIn;
      return false;
    }

    portNameIn = rf.check("humanCoMPort", yarp::os::Value("/CoM:o")).asString();

    if (!m_HumanCoMPort.open("/" + getName() + portNameIn)) {
      yError() << "[HumanDataAcquisition::configure] Unable to open the port "
               << portNameIn;
      return false;
    }

    portNameIn =
        rf.check("humanBasePort", yarp::os::Value("/basePose:o")).asString();

    if (!m_basePort.open("/" + getName() + portNameIn)) {
      yError() << "[HumanDataAcquisition::configure] Unable to open the port "
               << portNameIn;
      return false;
    }

    portNameIn =
        rf.check("humanWrenchPort", yarp::os::Value("/wrenchesVector:o"))
            .asString();

    if (!m_wrenchPort.open("/" + getName() + portNameIn)) {
      yError() << "[HumanDataAcquisition::configure] Unable to open the port "
               << portNameIn;
      return false;
    }

    portNameIn = rf.check("humanKinDynPort", yarp::os::Value("/humanKinDyn:o"))
                     .asString();

    if (!m_KinDynPort.open("/" + getName() + portNameIn)) {
      yError() << "[HumanDataAcquisition::configure] Unable to open the port "
               << portNameIn;
      return false;
    }
  }

  m_firstIteration = true;

  double jointThreshold;
  jointThreshold =
      rf.check("jointDifferenceThreshold", yarp::os::Value(0.01)).asFloat64();
  m_jointDiffThreshold = jointThreshold;
  m_CoMValues.resize(3, 0.0);

  m_leftShoes.resize(6, 0.0);
  m_rightShoes.resize(6, 0.0);

  size_t wrenchSize = 0;
  /*
  if (m_useLeftFootWrench)
    wrenchSize += 6;
  if (m_useRightFootWrench)
    wrenchSize += 6;
  */
  if (m_useFeetWrench)
    wrenchSize += 6;

  m_wrenchValues.resize(wrenchSize, 0.0);

  // size:  Joint values size, joint velocities, wrench values
  size_t kinDynSize = 0;
  if (m_useJointValues)
    kinDynSize += m_robotJointsListNames.size();
  if (m_useJointVelocities)
    kinDynSize += m_robotJointsListNames.size();
  /*
  if (m_useLeftFootWrench)
    kinDynSize += 6;
  if (m_useRightFootWrench)
    kinDynSize += 6;
  */
  if (m_useFeetWrench)
    kinDynSize += 6;

  m_kinDynValues.resize(kinDynSize, 0.0);
  yInfo() << "wrenchSize: " << wrenchSize;
  yInfo() << "kinDynSize: " << kinDynSize;

  m_baseValues.resize(7, 0.0);

  //***********************

  std::string features = "time ";
  if (m_useJointValues) {
    for (size_t i = 0; i < m_robotJointsListNames.size(); i++) {
      features += (m_robotJointsListNames[i]) + "_val ";
    }
  }

  if (m_useJointVelocities) {
    for (size_t i = 0; i < m_robotJointsListNames.size(); i++) {
      features += (m_robotJointsListNames[i]) + "_vel ";
      m_jointVelocitiesFeatuersName.push_back((m_robotJointsListNames[i]) +
                                              "_vel");
    }
  }
  
  /*
  if (m_useLeftFootWrench) {
    features += "l_shoe_fx l_shoe_fy l_shoe_fz l_shoe_tx l_shoe_ty l_shoe_tz ";
  }

  if (m_useRightFootWrench) {
    features += "r_shoe_fx r_shoe_fy r_shoe_fz r_shoe_tx r_shoe_ty r_shoe_tz ";
  }
  */

  if (m_useFeetWrench) {
    features += "l_shoe_fx l_shoe_fy l_shoe_fz l_shoe_tx l_shoe_ty l_shoe_tz ";

    features += "r_shoe_fx r_shoe_fy r_shoe_fz r_shoe_tx r_shoe_ty r_shoe_tz ";
                
  }

  if (m_useBaseValues) {
    features += "base_pos_x base_pos_y base_pos_z base_quat_w base_quat_x "
                "base_quat_y base_quat_z ";
  }
  if (m_useComValues) {
    features += "com_x com_y com_z ";
  }

  if (m_useForAnnotation) {
    features += "label ";
  }
  if (features.back() == ' ')
    features.pop_back();

  yInfo() << "features: " << features;

  // Chose an unique filename
  if (m_logData) {
    m_logBuffer.resize(0);
    m_annotationBuffer.resize(0);

    m_fastBackwardSteps =
        rf.check("FastBackwardsSteps", yarp::os::Value(1)).asInt8();

    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    std::stringstream fileName;
    fileName << "Dataset_" << std::put_time(&tm, "%Y_%m_%d_%H_%M_%S") << ".txt";

    // open the log file
    m_logger.open(fileName.str().c_str());
    yInfo() << "data will save in: " << fileName.str();

    m_logger << features + "\n";
  }
  // add the names when reading the data
  m_jointValuesFeatuersName.resize(0);
  m_jointVelocitiesFeatuersName.resize(0);

  for (size_t i = 0; i < m_robotJointsListNames.size(); i++) {
    m_jointVelocitiesFeatuersName.push_back((m_robotJointsListNames[i]) +
                                            "_vel");
  }

  for (size_t i = 0; i < m_robotJointsListNames.size(); i++) {
    m_jointValuesFeatuersName.push_back((m_robotJointsListNames[i]) + "_val");
  }

  m_baseFeatuersName = {"base_pos_x",  "base_pos_y",  "base_pos_z",
                        "base_quat_w", "base_quat_x", "base_quat_y",
                        "base_quat_z"};

  m_comFeatuersName = {"com_x", "com_y", "com_z"};

  m_leftWrenchFeatuersName = {"l_shoe_fx", "l_shoe_fy", "l_shoe_fz",
                              "l_shoe_tx", "l_shoe_ty", "l_shoe_tz"};

  m_rightWrenchFeatuersName = {"r_shoe_fx", "r_shoe_fy", "r_shoe_fz",
                               "r_shoe_tx", "r_shoe_ty", "r_shoe_tz"};

  //***********************

  yInfo() << "[HumanDataAcquisition::configure]"
          << " Sampling time  : " << m_dT;
  yInfo() << "[HumanDataAcquisition::configure] m_logData: " << m_logData;
  yInfo() << "[HumanDataAcquisition::configure] m_streamData: " << m_streamData;
  yInfo() << "[HumanDataAcquisition::configure]"
          << " Joint threshold: " << m_jointDiffThreshold;

  yInfo() << " [HumanDataAcquisition::configure] done!";
  return true;
}

bool HumanDataAcquisitionModule::readDataFile(std::string fileName) {
  std::string delimiter = " ";
  std::vector<std::string> columnTags;
  bool firstLine = true; // since the column tags are there
  std::ifstream file(fileName);
  if (file.is_open()) {
    std::string line;

    while (std::getline(file, line)) { // read one by one the lines of the file
      // using printf() in all tests for consistency
      // printf("%s\n", line.c_str());
      std::unordered_map<std::string, double> lineTokens;

      // parse each line to tokens:
      size_t pos = 0;
      std::string token;
      size_t count = 0;
      while ((pos = line.find(delimiter)) != std::string::npos) {
        token = line.substr(0, pos);
        //        std::cout << token << std::endl;
        // get the column tags which is located in the first row
        if (firstLine) {
          columnTags.push_back(token);
        } else {
          // make the key and unordered map
          lineTokens.insert(
              std::make_pair(columnTags[count], std::stod(token)));
        }
        count++;
        line.erase(0, pos + delimiter.length());
      }
      if (!firstLine) {
        m_readDataMapVector.push_back(lineTokens);
      }
      firstLine = false;
    }
    file.close();
  }
  return true;
}

bool HumanDataAcquisitionModule::readDataLineAtMoment() {
  // read data from the matrix
  return true;
}
bool HumanDataAcquisitionModule::getVectorizeHumanStates() {

  if (!m_readDataFromFile) {
    hde::msgs::HumanState *desiredHumanStates =
        m_wholeBodyHumanJointsPort.read(false);

    if (desiredHumanStates == nullptr) {
      return true;
    }

    // get the new joint values
    std::vector<double> newHumanjointsValues = desiredHumanStates->positions;
    std::vector<double> newHumanjointsVelocities =
        desiredHumanStates->velocities;

    // get the new CoM positions
    hde::msgs::Vector3 CoMValues = desiredHumanStates->CoMPositionWRTGlobal;

    m_CoMValues(0) = CoMValues.x;
    m_CoMValues(1) = CoMValues.y;
    m_CoMValues(2) = CoMValues.z;

    // get base values
    hde::msgs::Vector3 newBasePosition =
        desiredHumanStates->baseOriginWRTGlobal;
    hde::msgs::Quaternion newBaseOrientation =
        desiredHumanStates->baseOrientationWRTGlobal;

    m_baseValues(0) = newBasePosition.x;
    m_baseValues(1) = newBasePosition.y;
    m_baseValues(2) = newBasePosition.z;

    m_baseValues(3) = newBaseOrientation.w;
    m_baseValues(4) = newBaseOrientation.imaginary.x;
    m_baseValues(5) = newBaseOrientation.imaginary.y;
    m_baseValues(6) = newBaseOrientation.imaginary.z;

    if (!m_firstIteration) {

      for (unsigned j = 0; j < m_actuatedDOFs; j++) {
        // check for the spikes in joint values
        if (std::abs(newHumanjointsValues[m_humanToRobotMap[j]] -
                     m_jointValues(j)) < m_jointDiffThreshold) {
          m_jointValues(j) = newHumanjointsValues[m_humanToRobotMap[j]];
          m_jointVelocities(j) = newHumanjointsVelocities[m_humanToRobotMap[j]];

        } else {
          yWarning() << "spike in data: joint : " << j << " , "
                     << m_robotJointsListNames[j]
                     << " ; old data: " << m_jointValues(j) << " ; new data:"
                     << newHumanjointsValues[m_humanToRobotMap[j]];
        }
      }
    } else {

      /* We should do a maping between two vectors here: human and robot joint
       vectors, since their order are not the same! */

      // check the human joints name list
      m_humanJointsListName = desiredHumanStates->jointNames;

      /* print human and robot joint name list */
      yInfo()
          << "Human joints name list: [human joints list] [robot joints list]"
          << m_humanJointsListName.size() << " , "
          << m_robotJointsListNames.size();

      for (size_t i = 0; i < m_humanJointsListName.size(); i++) {
        if (i < m_robotJointsListNames.size())
          yInfo() << "(" << i << "): " << m_humanJointsListName[i] << " , "
                  << m_robotJointsListNames[i];
        else {
          yInfo() << "(" << i << "): " << m_humanJointsListName[i] << " , --";
        }
      }
      /* find the map between the human and robot joint list orders*/
      if (!pImpl->mapJointsHDE2Controller(m_robotJointsListNames,
                                          m_humanJointsListName,
                                          m_humanToRobotMap)) {
        yError() << "[HumanDataAcquisition::getJointValues()] mapping is not "
                    "possible";
        return false;
      }
      if (m_humanToRobotMap.size() == 0) {
        yError() << "[HumanDataAcquisition::getJointValues()] "
                    "m_humanToRobotMap.size is zero";
      }

      /* fill the robot joint list values*/
      for (unsigned j = 0; j < m_actuatedDOFs; j++) {
        m_jointValues(j) = newHumanjointsValues[m_humanToRobotMap[j]];
        m_jointVelocities(j) = newHumanjointsVelocities[m_humanToRobotMap[j]];
        yInfo() << " robot initial joint value: (" << j
                << "): " << m_jointValues[j];
      }
    }

    //    yInfo() << "joint [0]: " << m_robotJointsListNames[0] << " : "
    //            << newHumanjointsValues[m_humanToRobotMap[0]] << " , " <<
    //            m_jointValues(0) <<" , "<<m_jointVelocities(0);
  } else {

    m_time = (m_readDataMapVector[m_DataLineIndex])["time"];
    for (size_t i = 0; i < m_jointValuesFeatuersName.size(); i++)
      m_jointValues(i) =
          (m_readDataMapVector[m_DataLineIndex])[m_jointValuesFeatuersName[i]];

    for (size_t i = 0; i < m_jointVelocitiesFeatuersName.size(); i++)
      m_jointVelocities(i) = (m_readDataMapVector[m_DataLineIndex])
          [m_jointVelocitiesFeatuersName[i]];

    for (size_t i = 0; i < 7; i++)
      m_baseValues(i) =
          (m_readDataMapVector[m_DataLineIndex])[m_baseFeatuersName[i]];

    for (size_t i = 0; i < 3; i++)
      m_CoMValues(i) =
          (m_readDataMapVector[m_DataLineIndex])[m_comFeatuersName[i]];
  }

  if (m_firstIteration && m_useForAnnotation) {
    yInfo() << "\033[1;31m Module is Running ... \033[0m";
    yInfo()
        << "\033[1;31m To do fast backward press `'` on the keyboard\033[0m";
    yInfo() << "\033[1;31m To pause press `p` or `P` on the keyboard\033[0m";
    yInfo() << "\033[1;31m To return press `r` or `R` on the keyboard\033[0m";
    yInfo() << "\033[1;31m IMPORTANT: Press `S` or `s` to exit safely\033[0m";
  }
  m_firstIteration = false;
  return true;
}

// **********MODIFY THE CODE: START**********
// read the input port that was previously connected to the output port for left shoe 
/*
bool HumanDataAcquisitionModule::getLeftShoesWrenches() {

  if (!m_readDataFromFile) {
    
    yarp::os::Bottle *leftShoeWrench = m_bothShoesPort.read(false);

    if (leftShoeWrench == NULL) {
      //    yInfo() << "[getLeftShoesWrenches()] left shoes port is empty";
      return true;
    }
    

    // parse the data in order to extract the relevant information:
    // the 6 values of force/torque
    // Here the parsing assumes that the data in the port contains onlt a single shoe data
    // NOTE: here we are not using remapper device but parsing the data in the port directly.
    // NOTE: in general this should be avoided by using IWearRemapper device. 
    // data are located in 5th element:
    
    yarp::os::Value list4 = leftShoeWrench->get(4);

    yarp::os::Bottle *tmp1 = list4.asList();
    yarp::os::Bottle *tmp2 = tmp1->get(0).asList();
    yarp::os::Bottle *tmp3 = tmp2->get(1).asList();

    for (size_t i = 0; i < 6; i++)
      m_leftShoes(i) = tmp3->get(i + 2).asFloat64();

    //    yInfo()<<"m_leftShoes: "<<m_leftShoes.toString();
  } else {

    for (size_t i = 0; i < m_leftWrenchFeatuersName.size(); i++)
      m_leftShoes(i) =
          (m_readDataMapVector[m_DataLineIndex])[m_leftWrenchFeatuersName[i]];
  }
  

    yarp::os::Value list4 = leftShoeWrench->get(4);

    yarp::os::Bottle *tmp1 = list4.asList();
    yarp::os::Bottle *tmp2 = tmp1->get(0).asList();
    yarp::os::Bottle *tmp3 = tmp2->get(1).asList();

    for (size_t i = 0; i < 6; i++)
      m_leftShoes(i) = tmp3->get(i+2).asFloat64();

  } else {

    for (size_t i = 0; i < m_leftWrenchFeatuersName.size(); i++)
      m_leftShoes(i) = 
          (m_readDataMapVector[m_DataLineIndex])[m_leftWrenchFeatuersName[i]];
  }

  return true;
}
*/

// **********MODIFY THE CODE: END**********

// **********MODIFY THE CODE: START**********
// read the input port that was previously connected to the output port fro right shoe 
/*
bool HumanDataAcquisitionModule::getRightShoesWrenches() {

  if (!m_readDataFromFile) {
    yarp::os::Bottle *rightShoeWrench = m_bothShoesPort.read(false);

    if (rightShoeWrench == NULL) {
      //    yInfo() << "[getRightShoesWrenches()] right shoes port is empty";
      return true;
    }
    //    yInfo()<<"right shoes: rightShoeWrench size:
    //    "<<rightShoeWrench->size();

    // data are located in 5th element:
    
    yarp::os::Value list4 = rightShoeWrench->get(4);

    yarp::os::Bottle *tmp1 = list4.asList();
    yarp::os::Bottle *tmp2 = tmp1->get(0).asList();
    yarp::os::Bottle *tmp3 = tmp2->get(1).asList();

    for (size_t i = 0; i < 6; i++)
      m_rightShoes(i) = tmp3->get(i + 2).asFloat64();

    //    yInfo()<<"m_rightShoes: "<<m_rightShoes.toString();
  } else {
    // to fill: read from file
    for (size_t i = 0; i < m_rightWrenchFeatuersName.size(); i++)
      m_rightShoes(i) =
          (m_readDataMapVector[m_DataLineIndex])[m_rightWrenchFeatuersName[i]];
  }
  
    yarp::os::Value list4 = rightShoeWrench->get(4);

    yarp::os::Bottle *tmp1 = list4.asList();
    yarp::os::Bottle *tmp2 = tmp1->get(1).asList();
    yarp::os::Bottle *tmp3 = tmp2->get(1).asList();

    for (size_t i = 0; i < 6; i++)
      m_rightShoes(i) = tmp3->get(i+2).asFloat64();
  } else {

    for (size_t i = 0; i < m_rightWrenchFeatuersName.size(); i++)
      m_rightShoes(i) =
          (m_readDataMapVector[m_DataLineIndex])[m_rightWrenchFeatuersName[i]];
  }

  return true;
}
*/
bool HumanDataAcquisitionModule::getShoesWrenches() {

  if (!m_readDataFromFile) {
    yarp::os::Bottle *myShoesWrench = m_bothShoesPort.read(false);

    if (myShoesWrench == NULL) {

      return true;
    }

    yarp::os::Value list4 = myShoesWrench->get(4);

    yarp::os::Bottle *tmp0 = list4.asList();

    yarp::os::Bottle *tmp1 = tmp0->get(0).asList();
    yarp::os::Bottle *tmp2 = tmp1->get(1).asList();

    yarp::os::Bottle *tmp3 = tmp0->get(1).asList();
    yarp::os::Bottle *tmp4 = tmp3->get(1).asList();

    for (size_t i = 0; i < 6; i++) {
      m_leftShoes(i) = tmp2->get(i+2).asFloat64();
      m_rightShoes(i) = tmp4->get(i+2).asFloat64();
    
    } 
  } 
  else {
      for (size_t i = 0; i < m_leftWrenchFeatuersName.size(); i++)
        m_leftShoes(i) = (m_readDataMapVector[m_DataLineIndex])[m_leftWrenchFeatuersName[i]];

      for (size_t i = 0; i< m_rightWrenchFeatuersName.size(); i++)
        m_rightShoes(i) = (m_readDataMapVector[m_DataLineIndex])[m_rightWrenchFeatuersName[i]];
  }

  return true;
}
// **********MODIFY THE CODE: END**********

double HumanDataAcquisitionModule::getPeriod() { return m_dT; }

bool HumanDataAcquisitionModule::updateModule() {

  while (!m_isClosed) {
    auto start = std::chrono::steady_clock::now();

    if (m_isPaused) {
      auto time_to_wait = seconds_to_duration(this->getPeriod());

      // wait if required
      if (time_to_wait > std::chrono::milliseconds::zero()) {
        std::this_thread::sleep_for(time_to_wait);
      }
      continue;
    }

    if (!getVectorizeHumanStates()) {
      yError() << "[updateModule] cannot get human states.";
      return false;
    }
    
    // **********MODIFY THE CODE: START**********
    // call the method for reading the port (defined above)
    /*
    if (m_useLeftFootWrench)
      getLeftShoesWrenches();
    if (m_useRightFootWrench)
      getRightShoesWrenches();
    */
    if (m_useFeetWrench)
      getShoesWrenches();

    // **********MODIFY THE CODE: END**********

    if (m_logData)
      logData();

    if (!m_readDataFromFile) {
      if (m_wholeBodyHumanJointsPort.isClosed()) {
        yError() << "[HumanDataAcquisition::updateModule] "
                    "m_wholeBodyHumanJointsPort port is closed";
        return false;
      }
      /*
      if (m_useLeftFootWrench) {
        if (m_bothShoesPort.isClosed()) {
          yError()
              << "[HumanDataAcquisition::updateModule] m_bothShoesPort port "
                 "is closed";
          return false;
        }
      }
      if (m_useRightFootWrench) {
        if (m_bothShoesPort.isClosed()) {
          yError()
              << "[HumanDataAcquisition::updateModule] m_bothShoesPort port "
                 "is closed";
          return false;
        }
      }
      */
      if (m_useFeetWrench) {
        if (m_bothShoesPort.isClosed()) {
          yError()
              << "[HumanDataAcquisition::updateModule] m_bothShoesPort port "
                 "is closed";
          return false;
        }
      }
    }

    if (m_streamData) {
      if (m_wholeBodyJointsPort.isClosed()) {
        yError()
            << "[HumanDataAcquisition::updateModule] m_wholeBodyJointsPort "
               "port is closed";
        return false;
      }

      if (m_HumanCoMPort.isClosed()) {
        yError()
            << "[HumanDataAcquisition::updateModule] m_HumanCoMPort port is "
               "closed";
        return false;
      }

      if (m_basePort.isClosed()) {
        yError() << "[HumanDataAcquisition::updateModule] m_basePort port is "
                    "closed";
        return false;
      }
      /*
      if (m_useLeftFootWrench || m_useRightFootWrench) {
        if (m_wrenchPort.isClosed()) {
          yError()
              << "[HumanDataAcquisition::updateModule] m_wrenchPort port is "
                 "closed";
          return false;
        }
      }
      */
      if (m_useFeetWrench) {
        if (m_wrenchPort.isClosed()) {
          yError()
              << "[HumanDataAcquisition::updateModule] m_wrenchPort port is "
                 "closed";
          return false;
        }
      }

      if (m_KinDynPort.isClosed()) {
        yError() << "[HumanDataAcquisition::updateModule] m_KinDynPort port is "
                    "closed";
        return false;
      }
    }

    if (!m_firstIteration && m_streamData) {
      // CoM
      yarp::sig::Vector &CoMrefValues = m_HumanCoMPort.prepare();
      CoMrefValues = m_CoMValues;
      m_HumanCoMPort.write();

      // Joints
      yarp::sig::Vector &refValues = m_wholeBodyJointsPort.prepare();

      refValues = m_jointValues;
      m_wholeBodyJointsPort.write();

      // Wrench vector

      /*
      size_t wrenchCounter = 0;
      if (m_useLeftFootWrench) {
        for (size_t i = 0; i < m_leftShoes.size(); i++) {
          m_wrenchValues(i) = m_leftShoes(i);
          wrenchCounter++;
        }
      }
      if (m_useRightFootWrench) {
        for (size_t i = 0; i < m_rightShoes.size(); i++) {
          // if uses both wrenches, wrenchCounter is 6, otherwise it will 0.
          m_wrenchValues(wrenchCounter + i) = m_rightShoes(i);
        }
      }
      */
      size_t wrenchCounter = 0;
      if (m_useFeetWrench) {
        for (size_t i = 0; i < m_leftShoes.size(); i++) {
          m_wrenchValues(i) = m_leftShoes(i);
          wrenchCounter++;
        }

        for (size_t i = 0; i < m_rightShoes.size(); i++) {
          m_wrenchValues(wrenchCounter + i) = m_rightShoes(i);
        }
      }

      yarp::sig::Vector &wrenchValues = m_wrenchPort.prepare();
      wrenchValues = m_wrenchValues;
      m_wrenchPort.write();

      // base vector
      yarp::sig::Vector &baseValues = m_basePort.prepare();
      baseValues = m_baseValues;
      m_basePort.write();

      // joint values, velocities, left and right wrenchs vector
      // size:  Joint values size, joint velocities, wrench values
      size_t count = 0;
      for (size_t i = 0; i < m_jointValues.size(); i++) {
        if (m_useJointValues) {
          m_kinDynValues(i) = m_jointValues(i);
          count++;
        }
        if (m_useJointVelocities) {
          m_kinDynValues(m_jointValues.size() + i) = m_jointVelocities(i);
          count++;
        }
      }
      for (size_t i = 0; i < m_wrenchValues.size(); i++) {
        m_kinDynValues(count + i) = m_wrenchValues(i);
      }

      yarp::sig::Vector &kinDynValues = m_KinDynPort.prepare();
      kinDynValues = m_kinDynValues;
      m_KinDynPort.write();
    }

    if (m_DataLineIndex % 100 == 0)
      yInfo() << "[update] index of the data: " << m_DataLineIndex;

    if (m_DataLineIndex == m_readDataMapVector.size() - 1) {
      yInfo() << "[update] all data in the file is read and streamed.";
      yInfo() << "[update] closing ...";
      this->close();
    }

    m_DataLineIndex++;
    // evaluate the time required for running the previous code
    auto end = std::chrono::steady_clock::now();
    auto elapsed = end - start;

    // compute the time to wait
    auto time_to_wait = seconds_to_duration(this->getPeriod()) - elapsed;

    // wait if required
    if (time_to_wait > std::chrono::milliseconds::zero()) {
      std::this_thread::sleep_for(time_to_wait);
    }
  }

  return true;
}

void HumanDataAcquisitionModule::keyboardHandler() {
  constexpr std::chrono::microseconds keyboad_input_thread_period{10};
  std::string lastAnnotation;
  while (m_useForAnnotation) {
    auto start = std::chrono::steady_clock::now();

    // get the string from terminal
    std::string input;
    std::cin >> input; // most of the time, this thread is blocked here, waiting
                       // for the input

    if (m_isClosed) {
      yInfo() << "[keyboardHandler] module is closed.";
      return;
    }

    // if the string is equal to the key the function is called
    if (input == "s" || input == "S") {
      yInfo() << "input:" << input;
      yInfo() << "[keyboardHandler] closing ... ";
      this->close();
      return;
    } else if (input == "'") {
      std::lock_guard<std::mutex> lock(m_mutex);
      yInfo() << " current time of the data: " << std::to_string(m_time);
      yInfo() << "Fast Backward by by" << m_fastBackwardSteps << " steps";
      yInfo() << "log buffer size" << m_logBuffer.size();
      if (m_annotationBuffer.size() > 0)
        m_latestAnnotation = m_annotationBuffer[0];

      m_DataLineIndex -= m_logBuffer.size();
      m_logBuffer.clear();
      m_annotationBuffer.clear();
      yInfo() << "reseting logging annotation to:"
              << ("\033[1;31m" + m_latestAnnotation + "\033[0m");

    } else if (input == "p" || input == "P") {

      m_isPaused = true;
      yInfo() << "[keyboardHandler] prgrammed is paused. ";
      yInfo() << "[keyboardHandler] latest annotation is:"
              << m_latestAnnotation;

    } else if (input == "r" || input == "R") {

      m_isPaused = false;
      yInfo() << "[keyboardHandler] prgrammed is returned. ";

    } else {
      unsigned int idx = -1; // so that by default returns None
      if (m_annotationList.size() > 0) {
        try {
          idx = std::stoul(input);
        } catch (std::exception &e) {
          std::cout << "error converting input: " << input << '\n';
        }
        if (idx > 0 && idx <= m_annotationList.size()) {
          lastAnnotation =
              m_annotationList[idx - 1]; // since the index is srating from 1
        } else {
          lastAnnotation = "None";
        }

      } else {
        lastAnnotation = input;
      }
      yInfo() << "input:" << input << ", logging annotation:"
              << ("\033[1;31m" + lastAnnotation + "\033[0m");

      {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_latestAnnotation = lastAnnotation;
      }
    }

    // evaluate the time required for running the previous code
    auto end = std::chrono::steady_clock::now();
    auto elapsed = end - start;

    // compute the time to wait
    auto time_to_wait = keyboad_input_thread_period - elapsed;

    // wait if required
    if (time_to_wait > std::chrono::milliseconds::zero()) {
      std::this_thread::sleep_for(time_to_wait);
    }
  }

  return;
}

bool HumanDataAcquisitionModule::logData() {
  if (!m_readDataFromFile) {
    m_time = yarp::os::Time::now();
  } else {
    // updated in another method
  }
  std::string features;

  if (m_useJointValues)
    for (size_t i = 0; i < m_actuatedDOFs; i++) {
      features += std::to_string(m_jointValues(i)) + " ";
    }

  if (m_useJointVelocities)
    for (size_t i = 0; i < m_actuatedDOFs; i++) {
      features += std::to_string(m_jointVelocities(i)) + " ";
    }
  /*
  if (m_useLeftFootWrench)
    for (size_t i = 0; i < 6; i++) {
      features += std::to_string(m_leftShoes(i)) + " ";
    }

  if (m_useRightFootWrench)
    for (size_t i = 0; i < 6; i++) {
      features += std::to_string(m_rightShoes(i)) + " ";
    }
  */

  if (m_useFeetWrench) {
    for (size_t i = 0; i < 6; i++) {
      features += std::to_string(m_leftShoes(i)) + " ";
    }

    for (size_t i = 0; i < 6; i++) {
      features += std::to_string(m_rightShoes(i)) + " ";
    }
  }

  if (m_useBaseValues)
    for (size_t i = 0; i < m_baseValues.size(); i++) {
      features += std::to_string(m_baseValues(i)) + " ";
    }

  if (m_useComValues)
    for (size_t i = 0; i < m_CoMValues.size(); i++) {
      features += std::to_string(m_CoMValues(i)) + " ";
    }

  if (m_useForAnnotation) {
    std::lock_guard<std::mutex> lock(m_mutex);
    features += m_latestAnnotation;
  }

  if (features.back() == ' ')
    features.pop_back();

  features = std::to_string(m_time) + " " + features + "\n";

  // if this condition is true, it means another thread emptied m_logBuffer, so
  // we should avoid adding to the buffer and we should wait for the update
  // method to be called
  if (m_readDataFromFile)
    if (m_time != (m_readDataMapVector[m_DataLineIndex])["time"]) {
      yInfo() << "[logData] avoid adding to the buffer, wait to call for "
                 "update.";
      yInfo() << "[logData] logBuffer size should be zero"
              << m_logBuffer.size();
      return true;
    }

  std::lock_guard<std::mutex> lock(m_mutex);
  m_logBuffer.push_back(features);
  m_annotationBuffer.push_back(m_latestAnnotation);

  // m_logBuffer can have maximum size of m_fastBackwardsSteps
  if (m_logBuffer.size() == m_fastBackwardSteps) {
    m_logger << m_logBuffer[0];

    m_logBuffer.erase(m_logBuffer.begin());
    m_annotationBuffer.erase(m_annotationBuffer.begin());
  }
  return true;
}

bool HumanDataAcquisitionModule::close() {
  m_isClosed = true;
  if (m_logData) {
    if (m_logBuffer.size() > 0) {
      yInfo() << "[close] log buffer size is" << m_logBuffer.size();
      yInfo() << "[close] it is going to be emptied before closing the "
                 "application ...";

      m_logger << m_logBuffer[0];
      m_logBuffer.erase(m_logBuffer.begin());
      return this->close(); // add the recursion here in order to ensure that
                            // all the data are saved.
    }
    yInfo() << "[close] log buffer is empty.";
    m_logger.close();

    yInfo() << "Data saved with the following order.";
    yInfo() << "<yarp time (s)>,<data>";
  }
  yInfo() << " [close] closing ...";

  return true;
}
bool HumanDataAcquisitionModule::impl::mapJointsHDE2Controller(
    std::vector<std::string> robotJointsListNames,
    std::vector<std::string> humanJointsListName,
    std::vector<unsigned> &humanToRobotMap) {
  if (!humanToRobotMap.empty()) {
    humanToRobotMap.clear();
  }

  bool foundMatch = false;
  for (unsigned i = 0; i < robotJointsListNames.size(); i++) {
    for (unsigned j = 0; j < humanJointsListName.size(); j++) {

      if (robotJointsListNames[i] == humanJointsListName[j]) {
        foundMatch = true;
        humanToRobotMap.push_back(j);
        break;
      }
    }
    if (!foundMatch) {
      yError() << "[HumanDataAcquisition::impl::mapJointsHDE2CONTROLLER] not "
                  "found match for: "
               << robotJointsListNames[i] << " , " << i;
      return false;
    }
    foundMatch = false;
  }

  yInfo() << "*** mapped joint names: ****";
  for (size_t i = 0; i < robotJointsListNames.size(); i++) {
    yInfo() << "(" << i << ", " << humanToRobotMap[i]
            << "): " << robotJointsListNames[i] << " , "
            << humanJointsListName[(humanToRobotMap[i])];
  }

  return true;
}
