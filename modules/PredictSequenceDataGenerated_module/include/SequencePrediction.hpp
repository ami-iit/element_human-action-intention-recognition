/**
 * @file SequencePrediction.hpp
 * @authors Kourosh Darvish <Kourosh.Darvish@iit.it>
 * @copyright 2021 iCub Facility - Istituto Italiano di Tecnologia
 *            Released under the terms of the LGPLv2.1 or later, see LGPL.TXT
 * @date 2021
 */
#ifndef SEQUENCEPREDICTION_H
#define SEQUENCEPREDICTION_H

#include <iostream>
#include <chrono>

// yarp
#include <yarp/dev/PolyDriver.h>
#include <yarp/dev/IFrameTransform.h>
#include <yarp/dev/IJoypadController.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/RFModule.h>
#include <yarp/os/RpcClient.h>
#include <yarp/sig/Vector.h>
#include <yarp/os/Clock.h>

// tensorflow
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/util/abstract_stack_trace.h>

namespace tf = tensorflow;


class SequencePrediction : public yarp::os::RFModule
{
private:
    double m_dT;
    tf::Tensor m_input;
    tf::Status m_status;
    tf::Session* m_session;
    tf::SessionOptions sess_opts;

    tf::GraphDef _graph_def;   // graph from frozen protobuf
    tf::SessionOptions _opts;  // gpu options
    tf::Session *_session = 0; // session to run the graph in tf back end
    tf::Status _status;        // status check for each tf action trial
    std::string _dev;          // device for the graph "\cpu:0" or "\cpu:0"
    std::string _model_path = "/home/hallab/Github/project/joint_stem_detection_and_crop-weed_classification/testing_model/output_graph.pb";
    std::string _img_path = "/home/hallab/Github/project/joint_stem_detection_and_crop-weed_classification/dataset/img/rgb_000008.png";


    tf::keras



public:

    SequencePrediction();

    ~SequencePrediction();

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


#endif // SEQUENCEPREDICTION_H
