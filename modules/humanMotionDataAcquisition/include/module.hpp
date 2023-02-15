#ifndef XSENSRETARGETING_H
#define XSENSRETARGETING_H

// std
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// YARP
#include <hde/msgs/HumanState.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Network.h>
#include <yarp/os/RFModule.h>
#include <yarp/sig/Vector.h>
#include <yarp/os/Searchable.h>

// wearable
#include <Wearable/IWear/IWear.h>

#include <chrono>
#include <yarp/os/Clock.h>

class mapJoints {
public:
  std::string name;
  int index;
};

class HumanDataAcquisitionModule : public yarp::os::RFModule {
private:
  /** An implementation class for spepcific functionalities required in this
   * module. */
  class impl;
  std::unique_ptr<impl> pImpl;
  /** target (robot) joint values (raw amd smoothed values) */
  yarp::sig::Vector m_jointValues, m_smoothedJointValues;
  /** target (robot) joint velocities (raw amd smoothed values) */
  yarp::sig::Vector m_jointVelocities;

  /** CoM joint values coming from human-state-provider */
  yarp::sig::Vector m_CoMValues;

  /** base values coming from human-state-provider */
  yarp::sig::Vector m_baseValues;

  /** wrenches value coming from wearables */
  yarp::sig::Vector m_leftShoes, m_rightShoes;
  yarp::sig::Vector m_wrenchValues;

  /** vector containting joint values, joint velocities, left and right foot
   * wrenches*/
  yarp::sig::Vector m_kinDynValues;

  /** the order of joints list arrived from human state provider is
   different from the one we want to send to the controller */
  std::vector<std::string> m_humanJointsListName;

  /** Port used to retrieve the human whole body joint pose. */
  yarp::os::BufferedPort<hde::msgs::HumanState> m_wholeBodyHumanJointsPort;
  
  /** Port used to retrieve the left shoes wrenches. */
  //yarp::os::BufferedPort<yarp::os::Bottle> m_leftShoesPort;

  /** Port used to retrieve the right shoes wrenches. */
  //yarp::os::BufferedPort<yarp::os::Bottle> m_rightShoesPort;
  

  /** Port used to retrieve both shoes wrenches. */
  //yarp::os::BufferedPort<yarp::os::Bottle> m_bothShoesPort;

  yarp::dev::PolyDriver m_wearableDevice;

  wearable::IWear *m_iWear{nullptr}; /**Sense FT Shoes wearable interface. */
  
  wearable::VectorOfSensorPtr<const wearable::sensor::IForceTorque6DSensor> m_leftShoeSensor;

  wearable::VectorOfSensorPtr<const wearable::sensor::IForceTorque6DSensor> m_rightShoeSensor;
  /** Port used to provide the smoothed joint pose to yarp port. */
  yarp::os::BufferedPort<yarp::sig::Vector> m_wholeBodyJointsPort;
  /** Port used to provide the human CoM position to the yarp network.  */
  yarp::os::BufferedPort<yarp::sig::Vector> m_HumanCoMPort;
  /** Port used to provide the human base pose to yarp network.  */
  yarp::os::BufferedPort<yarp::sig::Vector> m_basePort;
  /** Port used to provide the human wrench port to yarp port.  */
  yarp::os::BufferedPort<yarp::sig::Vector> m_wrenchPort;
  /** Port used to provide the human joint values, velocities, and wrenches to a
   * yarp port.  */
  yarp::os::BufferedPort<yarp::sig::Vector> m_KinDynPort;

  double m_dT;             /**< Module period. */
  bool m_useXsens;         /**< True if the Xsens is used in the retargeting */
  bool m_logData;          /**< True to log human data*/
  bool m_showAnnoData;     /**< True to show previously annotated data set */
  bool m_streamData;       /**< True to stream human data */
  bool m_isClosed;         /**< True if the module is not closed*/
  bool m_readDataFromFile; /**< True if reading data from file and used for
                              annotating data*/
  size_t
      m_DataLineIndex; /**< This is used to go forward and backward for
                      annotating data when checking with the visualized data*/
  double m_time;       /**< the time used for saving in the file*/

  std::mutex m_mutex;

  std::ofstream m_logger;

  std::vector<std::string> m_logBuffer;

  std::vector<std::string> m_annotationBuffer; /**< the latest buffered logged
                                      data annotation to use when logging the
                                      data (used when doing flash back in time)
                                      = annotation(time- flashbacktime)*/

  int m_fastBackwardSteps;

  std::vector<std::string>
      m_robotJointsListNames; /**< Vector containing the name of the controlled
                                 joints.*/
  size_t m_actuatedDOFs;      /**< Number of the actuated DoF */

  std::vector<unsigned> m_humanToRobotMap;

  bool m_firstIteration;
  double m_jointDiffThreshold;

  bool m_useJointValues;
  bool m_useJointVelocities;
  //bool m_useLeftFootWrench;
  //bool m_useRightFootWrench;
  bool m_useFeetWrench;
  bool m_useBaseValues;
  bool m_useComValues;

  bool m_useForAnnotation;

  std::string m_latestAnnotation; /**< the latest or updated annotation to use
                                     when logging the data*/

  std::string m_sensorNameLeftShoe;

  std::string m_sensorNameRightShoe;
  
  std::vector<std::string> m_annotationList; /**< Vector containing the name of
                                                the list of the annotations.*/

  std::vector<std::unordered_map<std::string, double>> m_readDataMapVector;

  std::vector<std::string> m_jointValuesFeatuersName;

  std::vector<std::string> m_jointVelocitiesFeatuersName;

  std::vector<std::string> m_baseFeatuersName;

  std::vector<std::string> m_comFeatuersName;

  std::vector<std::string> m_leftWrenchFeatuersName;

  std::vector<std::string> m_rightWrenchFeatuersName;

  bool m_isPaused; /// to pause for specific time when annotating the code

public:
  HumanDataAcquisitionModule();
  ~HumanDataAcquisitionModule();
  /*
   * Configure the whole body retargeting retargeting.
   * @param config reference to a resource finder object.
   * @return true in case of success and false otherwise
   */
  bool getVectorizeHumanStates();

  //bool getLeftShoesWrenches();

  //bool getRightShoesWrenches();

  bool getShoesWrenches();

  bool logData();

  //bool showAnnoData();

  bool dataHandler();

  void keyboardHandler();

  bool readDataFile(std::string fileName);

  bool readDataLineAtMoment();

  /**
   * Get the period of the RFModule.
   * @return the period of the module.
   */
  double getPeriod() final;

  /**
   * Main function of the RFModule.
   * @return true in case of success and false otherwise.
   */
  bool updateModule() final;

  /**
   * Configure the RFModule.
   * @param rf is the reference to a resource finder object
   * @return true in case of success and false otherwise.
   */
  bool configure(yarp::os::ResourceFinder &rf) final;

  /**
   * Close the RFModule.
   * @return true in case of success and false otherwise.
   */
  bool close() final;
};

inline bool yarpListToStringVector(yarp::os::Value *&input,
                                   std::vector<std::string> &output) {
  // clear the std::vector
  output.clear();

  // check if the yarp value is a list
  if (!input->isList()) {
    yError() << "[yarpListToStringVector] The input is not a list.";
    return false;
  }

  yarp::os::Bottle *bottle = input->asList();
  for (int i = 0; i < bottle->size(); i++) {
    // check if the elements of the bottle are strings
    if (!bottle->get(i).isString()) {
      yError()
          << "[yarpListToStringVector] There is a field that is not a string.";
      return false;
    }
    output.push_back(bottle->get(i).asString());
  }
  return true;
}

template <typename T> auto seconds_to_duration(T seconds) {
  return std::chrono::duration<T, std::ratio<1>>(seconds);
}

#endif // WHOLEBODYRETARGETING_H
