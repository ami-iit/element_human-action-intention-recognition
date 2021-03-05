/**
 * @file SequencePrediction.cpp
 * @authors Kourosh Darvish <Kourosh.Darvish@iit.it>
 * @copyright 2021 iCub Facility - Istituto Italiano di Tecnologia
 *            Released under the terms of the LGPLv2.1 or later, see LGPL.TXT
 * @date 2021
 */

#include<SequencePrediction.hpp>


SequencePrediction::SequencePrediction(){}

SequencePrediction::~SequencePrediction(){}

bool SequencePrediction::configure(yarp::os::ResourceFinder& rf)
{

    return true;
}

double SequencePrediction::getPeriod()
{
    return m_dT;
}

bool SequencePrediction::updateModule()
{
    return true;
}


bool SequencePrediction::close()
{
    return true;
}
