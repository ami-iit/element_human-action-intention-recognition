//#include "yarp/ HumanState.h"
#include <module.hpp>

class HumanDataAcquisitionModule::impl
{
public:
    /*
     * map the joint values (order) coming from HDE to the controller order
     * @param robotJointsListNames list of the joint names used in controller
     * @param humanJointsListName list of the joint names received from the HDE
     * (human-dynamics-estimation repository)
     * @param humanToRobotMap the container for mapping of the human joints to the robot ones
     * @return true in case of success and false otherwise
     */
    bool mapJointsHDE2Controller(std::vector<std::string> robotJointsListNames,
                                 std::vector<std::string> humanJointsListName,
                                 std::vector<unsigned>& humanToRobotMap);
};

HumanDataAcquisitionModule::HumanDataAcquisitionModule()
    : pImpl{new impl()} {};

HumanDataAcquisitionModule::~HumanDataAcquisitionModule(){};

bool HumanDataAcquisitionModule::configure(yarp::os::ResourceFinder& rf)
{
    // check if the configuration file is empty
    if (rf.isNull())
    {
        yError() << "[XsensRetargeting::configure] Empty configuration for the OculusModule "
                    "application.";
        return false;
    }

    // get the period
    m_dT = rf.check("samplingTime", yarp::os::Value(0.1)).asDouble();

    // set the module name
    std::string name;
    name = rf.check( "name", yarp::os::Value("HumanDataAcquisition")).asString();

    setName(name.c_str());

    // initialize minimum jerk trajectory for the whole body

    yarp::os::Value* axesListYarp;

    if (!rf.check("joints_list", axesListYarp))
    {
        yError() << "[XsensRetargeting::configure] Unable to find joints_list"
                    "into config file.";
        return false;
    }

    yarpListToStringVector(axesListYarp,m_robotJointsListNames);


    m_actuatedDOFs = m_robotJointsListNames.size();

    m_jointValues.resize(m_actuatedDOFs, 0.0);
    m_jointVelocities.resize(m_actuatedDOFs,0.0);

    yInfo() << "XsensRetargeting::configure:  NoOfJoints: " << m_actuatedDOFs;

    std::string portName;
    portName = rf.check( "HDEJointsPort", yarp::os::Value("HDEJointsPort")).asString();

    if (!m_wholeBodyHumanJointsPort.open("/" + getName() + portName))
    {
        yError() << "[XsensRetargeting::configure] " << portName << " port already open.";
        return false;
    }
    yarp::os::Network::connect("/iCub/RobotStateWrapper/state:o", "/" + getName() + portName);

    //
    portName = rf.check( "WearablesLeftShoesPort", yarp::os::Value("/FTShoeLeft/WearableData/data:i")).asString();
    if (!m_leftShoesPort.open("/" + getName() + portName))
    {
        yError() << "[XsensRetargeting::configure] " << portName << " port already open.";
        return false;
    }
    yarp::os::Network::connect("/FTShoeLeft/WearableData/data:o", "/" + getName() + portName);


    //
    portName = rf.check( "WearablesRightShoesPort", yarp::os::Value("/FTShoeRight/WearableData/data:i")).asString();
    if (!m_rightShoesPort.open("/" + getName() + portName))
    {
        yError() << "[XsensRetargeting::configure] " << portName << " port already open.";
        return false;
    }
    yarp::os::Network::connect("/FTShoeRight/WearableData/data:o", "/" + getName() + portName);


    portName = rf.check( "controllerJointsPort",  yarp::os::Value("controllerJointsPort")).asString();

    if (!m_wholeBodyHumanSmoothedJointsPort.open("/" + getName() + portName))
    {
        yError() << "[XsensRetargeting::configure] Unable to open the port " << portName;
        return false;
    }

    portName = rf.check( "controllerCoMPort",  yarp::os::Value("controllerCoMPort")).asString();

    if (!m_HumanCoMPort.open("/" + getName() + portName))
    {
        yError() << "[XsensRetargeting::configure] Unable to open the port " << portName;
        return false;
    }

    m_firstIteration = true;

    double jointThreshold;
    jointThreshold = rf.check("jointDifferenceThreshold", yarp::os::Value(0.01)).asDouble();
    m_jointDiffThreshold = jointThreshold;
    m_CoMValues.resize(3, 0.0);

    m_leftShoes.resize(6,0.0);
    m_rightShoes.resize(6,0.0);


    //***********************
    // Chose an unique filename
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    std::stringstream fileName;
    fileName << "Dataset_" << std::put_time(&tm, "%Y_%m_%d_%H_%M_%S") << ".txt";

    // open the log file
    m_logger.open(fileName.str().c_str());
    yInfo() << "data will save in: " << fileName.str();

    std::string features= "time ";
    for (size_t i = 0; i < m_robotJointsListNames.size(); i++)
    {
            features += (m_robotJointsListNames[i]) + "_val ";
    }
    for (size_t i = 0; i < m_robotJointsListNames.size(); i++)
    {
            features += (m_robotJointsListNames[i]) + "_vel ";
    }

    features+="l_shoe_fx l_shoe_fy l_shoe_fz l_shoe_tx l_shoe_ty l_shoe_tz ";
    features+="r_shoe_fx r_shoe_fy r_shoe_fz r_shoe_tx r_shoe_ty r_shoe_tz";
    yInfo()<<"features: "<<features;

    m_logger << features + "\n";


    //***********************



    yInfo() << "[XsensRetargeting::configure]"
            << " Sampling time  : " << m_dT;
    yInfo() << "[XsensRetargeting::configure]"
            << " Joint threshold: " << m_jointDiffThreshold;

    yInfo() << " [XsensRetargeting::configure] done!";
    return true;
}
bool HumanDataAcquisitionModule::getJointValues()
{

    human::HumanState* desiredHumanStates = m_wholeBodyHumanJointsPort.read(false);

    if (desiredHumanStates == NULL)
    {
        return true;
    }

    // get the new joint values
    std::vector<double> newHumanjointsValues = desiredHumanStates->positions;
    std::vector<double> newHumanjointsVelocities= desiredHumanStates->velocities;


    // get the new CoM positions
    human::Vector3 CoMValues = desiredHumanStates->CoMPositionWRTGlobal;

    m_CoMValues(0) = CoMValues.x;
    m_CoMValues(1) = CoMValues.y;
    m_CoMValues(2) = CoMValues.z;

    if (!m_firstIteration)
    {
        yInfo() << "[XsensRetargeting::getJointValues] Xsens Retargeting Module is Running ...";

        for (unsigned j = 0; j < m_actuatedDOFs; j++)
        {
            // check for the spikes in joint values
            if (std::abs(newHumanjointsValues[m_humanToRobotMap[j]] - m_jointValues(j))
                < m_jointDiffThreshold)
            {
                m_jointValues(j) = newHumanjointsValues[m_humanToRobotMap[j]];
                m_jointVelocities(j)=newHumanjointsVelocities[m_humanToRobotMap[j]];

            } else
            {
                yWarning() << "spike in data: joint : " << j << " , " << m_robotJointsListNames[j]
                           << " ; old data: " << m_jointValues(j)
                           << " ; new data:" << newHumanjointsValues[m_humanToRobotMap[j]];
            }
        }
    } else
    {
        m_firstIteration = false;

        /* We should do a maping between two vectors here: human and robot joint vectors, since
         their order are not the same! */

        // check the human joints name list
        m_humanJointsListName = desiredHumanStates->jointNames;

        /* print human and robot joint name list */
        yInfo() << "Human joints name list: [human joints list] [robot joints list]"
                << m_humanJointsListName.size() << " , " << m_robotJointsListNames.size();

        for (size_t i = 0; i < m_humanJointsListName.size(); i++)
        {
            if (i < m_robotJointsListNames.size())
                yInfo() << "(" << i << "): " << m_humanJointsListName[i] << " , "
                        << m_robotJointsListNames[i];
            else
            {
                yInfo() << "(" << i << "): " << m_humanJointsListName[i] << " , --";
            }
        }
        /* find the map between the human and robot joint list orders*/
        if (!pImpl->mapJointsHDE2Controller(
                m_robotJointsListNames, m_humanJointsListName, m_humanToRobotMap))
        {
            yError() << "[XsensRetargeting::getJointValues()] mapping is not possible";
            return false;
        }
        if (m_humanToRobotMap.size() == 0)
        {
            yError() << "[XsensRetargeting::getJointValues()] m_humanToRobotMap.size is zero";
        }

        /* fill the robot joint list values*/
        for (unsigned j = 0; j < m_actuatedDOFs; j++)
        {
            m_jointValues(j) = newHumanjointsValues[m_humanToRobotMap[j]];
            m_jointVelocities(j)=newHumanjointsVelocities[m_humanToRobotMap[j]];
            yInfo() << " robot initial joint value: (" << j << "): " << m_jointValues[j];
        }
    }

    yInfo() << "joint [0]: " << m_robotJointsListNames[0] << " : "
            << newHumanjointsValues[m_humanToRobotMap[0]] << " , " << m_jointValues(0) <<" , "<<m_jointVelocities(0);

    return true;
}


bool HumanDataAcquisitionModule::getLeftShoesWrenches(){

    yarp::os::Bottle* leftShoeWrench = m_leftShoesPort.read(false);

    if (leftShoeWrench == NULL)
    {
        return true;
    }
        yInfo()<<"left shoes: leftShoeWrench size: "<<leftShoeWrench->size();

    // data are located in 5th element:
    yarp::os::Value list4 = leftShoeWrench->get(4);

    yarp::os::Bottle* tmp1 = list4.asList();
    yarp::os::Bottle* tmp2 = tmp1->get(0).asList();
    yarp::os::Bottle* tmp3 = tmp2->get(1).asList();

    for (size_t i=0; i<6; i++)
        m_leftShoes(i)=tmp3->get(i+2).asDouble();

    yInfo()<<"m_leftShoes: "<<m_leftShoes.toString();


    return true;
}


bool HumanDataAcquisitionModule::getRightShoesWrenches(){


    yarp::os::Bottle* rightShoeWrench = m_rightShoesPort.read(false);

    if (rightShoeWrench == NULL)
    {
        yInfo()<<"[HumanDataAcquisitionModule::getRightShoesWrenches()] right shoes port is empty";
        return true;
    }
        yInfo()<<"right shoes: rightShoeWrench size: "<<rightShoeWrench->size();

    // data are located in 5th element:
    yarp::os::Value list4 = rightShoeWrench->get(4);

    yarp::os::Bottle* tmp1 = list4.asList();
    yarp::os::Bottle* tmp2 = tmp1->get(0).asList();
    yarp::os::Bottle* tmp3 = tmp2->get(1).asList();

    for (size_t i=0; i<6; i++)
        m_rightShoes(i)=tmp3->get(i+2).asDouble();

    yInfo()<<"m_rightShoes: "<<m_rightShoes.toString();


    return true;
}

double HumanDataAcquisitionModule::getPeriod()
{
    return m_dT;
}

bool HumanDataAcquisitionModule::updateModule()
{

    getJointValues();
    getLeftShoesWrenches();
    getRightShoesWrenches();
    logData();

    if (m_wholeBodyHumanJointsPort.isClosed())
    {
        yError() << "[XsensRetargeting::updateModule] m_wholeBodyHumanJointsPort port is closed";
        return false;
    }

    if (m_HumanCoMPort.isClosed())
    {
        yError() << "[XsensRetargeting::updateModule] m_HumanCoMPort port is closed";
        return false;
    }

    if (m_leftShoesPort.isClosed())
    {
        yError() << "[XsensRetargeting::updateModule] m_HumanCoMPort port is closed";
        return false;
    }
    if (m_rightShoesPort.isClosed())
    {
        yError() << "[XsensRetargeting::updateModule] m_HumanCoMPort port is closed";
        return false;
    }



    if (!m_firstIteration)
    {
        yarp::sig::Vector& CoMrefValues = m_HumanCoMPort.prepare();
        CoMrefValues = m_CoMValues;
        m_HumanCoMPort.write();

        yarp::sig::Vector& refValues = m_wholeBodyHumanSmoothedJointsPort.prepare();

            refValues = m_jointValues;
        m_wholeBodyHumanSmoothedJointsPort.write();
    }

    return true;
}

bool HumanDataAcquisitionModule::logData()
{
    double time = yarp::os::Time::now();
    std::string features;

    for (size_t i = 0; i < m_actuatedDOFs; i++)
    {
        features += std::to_string(m_jointValues(i)) + " ";
    }


    for (size_t i = 0; i < m_actuatedDOFs; i++)
    {
        features += std::to_string(m_jointVelocities(i)) + " ";
    }

    for (size_t i = 0; i < 6; i++)
    {
        features += std::to_string(m_leftShoes(i)) + " ";
    }


    for (size_t i = 0; i < 6; i++)
    {

        features += std::to_string(m_rightShoes(i));
        if(i<5)
            features +=" ";
    }

    m_logger << std::to_string(time) + " " + features + "\n";
    return true;
}

bool HumanDataAcquisitionModule::close()
{
    m_logger.close();

    yInfo() << "Data saved with the following order.";
    yInfo() << "<yarp time (s)>,<data>";
    yInfo() << "closing ...";


    return true;
}
bool HumanDataAcquisitionModule::impl::mapJointsHDE2Controller(std::vector<std::string> robotJointsListNames,
                                                     std::vector<std::string> humanJointsListName,
                                                     std::vector<unsigned>& humanToRobotMap)
{
    if (!humanToRobotMap.empty())
    {
        humanToRobotMap.clear();
    }

    bool foundMatch = false;
    for (unsigned i = 0; i < robotJointsListNames.size(); i++)
    {
        for (unsigned j = 0; j < humanJointsListName.size(); j++)
        {

            if (robotJointsListNames[i] == humanJointsListName[j])
            {
                foundMatch = true;
                humanToRobotMap.push_back(j);
                break;
            }
        }
        if (!foundMatch)
        {
            yError() << "[XsensRetargeting::impl::mapJointsHDE2CONTROLLER] not found match for: "
                     << robotJointsListNames[i] << " , " << i;
            return false;
        }
        foundMatch = false;
    }

    yInfo() << "*** mapped joint names: ****";
    for (size_t i = 0; i < robotJointsListNames.size(); i++)
    {
        yInfo() << "(" << i << ", " << humanToRobotMap[i] << "): " << robotJointsListNames[i]
                << " , " << humanJointsListName[(humanToRobotMap[i])];
    }

    return true;
}
