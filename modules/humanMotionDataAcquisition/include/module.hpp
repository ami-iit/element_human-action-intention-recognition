#ifndef XSENSRETARGETING_H
#define XSENSRETARGETING_H

// std
#include <cmath>
#include <memory>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <iterator>
#include <sstream>

// YARP
#include <yarp/os/Bottle.h>
#include <yarp/os/Network.h>
#include <HumanDynamicsEstimation/HumanState.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/RFModule.h>
#include <yarp/sig/Vector.h>
#include <yarp/os/LogStream.h>

#include <chrono>
#include <yarp/os/Clock.h>

class mapJoints
{
public:
    std::string name;
    int index;
};

class HumanDataAcquisitionModule : public yarp::os::RFModule
{
private:
    /** An implementation class for spepcific functionalities required in this module. */
    class impl;
    std::unique_ptr<impl> pImpl;
    /** target (robot) joint values (raw amd smoothed values) */
    yarp::sig::Vector m_jointValues, m_smoothedJointValues;
    /** target (robot) joint velocities (raw amd smoothed values) */
    yarp::sig::Vector m_jointVelocities;

    yarp::sig::Vector m_leftShoes, m_rightShoes;
    /** CoM joint values coming from human-state-provider */
    yarp::sig::Vector m_CoMValues;
    std::vector<std::string>
    m_humanJointsListName; // the order of joints list arrived from human state provider is
    // different from the one we want to send to the controller

    /** Port used to retrieve the human whole body joint pose. */
    yarp::os::BufferedPort<human::HumanState> m_wholeBodyHumanJointsPort;

    /** Port used to retrieve the left shoes wrenches. */
    yarp::os::BufferedPort<yarp::os::Bottle> m_leftShoesPort;

    /** Port used to retrieve the right shoes wrenches. */
    yarp::os::BufferedPort<yarp::os::Bottle> m_rightShoesPort;

    /** Port used to provide the smoothed joint pose to the controller. */
    yarp::os::BufferedPort<yarp::sig::Vector> m_wholeBodyHumanSmoothedJointsPort;
    /** Port used to provide the human CoM position to the controller.  */
    yarp::os::BufferedPort<yarp::sig::Vector> m_HumanCoMPort;

    double m_dT; /**< Module period. */
    bool m_useXsens; /**< True if the Xsens is used in the retargeting */

    std::ofstream m_logger;


    std::vector<std::string>
    m_robotJointsListNames; /**< Vector containing the name of the controlled joints.*/
    size_t m_actuatedDOFs; /**< Number of the actuated DoF */

    std::vector<unsigned> m_humanToRobotMap;

    bool m_firstIteration;
    double m_jointDiffThreshold;

public:
    HumanDataAcquisitionModule();
    ~HumanDataAcquisitionModule();
    /*
     * Configure the whole body retargeting retargeting.
     * @param config reference to a resource finder object.
     * @return true in case of success and false otherwise
     */
    bool getJointValues();


    bool getLeftShoesWrenches();
    bool getRightShoesWrenches();

    bool getSmoothedJointValues(yarp::sig::Vector& smoothedJointValues);


    bool logData();

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
    bool configure(yarp::os::ResourceFinder& rf) final;

    /**
     * Close the RFModule.
     * @return true in case of success and false otherwise.
     */
    bool close() final;

};

inline bool yarpListToStringVector(yarp::os::Value*& input, std::vector<std::string>& output)
{
    // clear the std::vector
    output.clear();

    // check if the yarp value is a list
    if (!input->isList())
    {
        yError() << "[yarpListToStringVector] The input is not a list.";
        return false;
    }

    yarp::os::Bottle* bottle = input->asList();
    for (int i = 0; i < bottle->size(); i++)
    {
        // check if the elements of the bottle are strings
        if (!bottle->get(i).isString())
        {
            yError() << "[yarpListToStringVector] There is a field that is not a string.";
            return false;
        }
        output.push_back(bottle->get(i).asString());
    }
    return true;
}

#endif // WHOLEBODYRETARGETING_H