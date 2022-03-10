# element_human-action-intention-recognition

## Responsible
|                Kourosh Darvish                              |
:------------------------------------------------------------:|
<img src="https://github.com/kouroshD.png" width="180"> |

## Background
The problem of human action & intention recognition (HAIR) rises several challenges and opportunity to the robotic community. We define the human action and intention recognition at _symbolic level_ (classification problem) and _motion level_ (regression problem) as "the process of classification of the human motion among the existing ones (i.e. modeled ones) in a library of actions, the starting point and ending point of an action, predicting the human motion in future reasoning according to incomplete temporal data, and the belief degree of the action recognition and prediction".
The effectiveness of the human action recognition (high precision, recall, accuracy, and f1 scores) for a given objective and application depends on the methods and sensory information we use. For our use case in this element, we learn or model the actions offline, whereas human action recognition is performed online.
HAIR can be performed by using the following sensory data coming from the human:

-  human limb and joint kinematics and dynamic or a mixture of them (technologies: data coming from the mocap, RGB-d data, raw imu wearable sensors, shoes,  etc)
- RGB-d data of the human environment
- Eye-tracking (technologies: Scleral search coil method, Infrared occulography (IROG), Electro-occulography (EOG), Video Occulography (VOG))
- human physiological measures such as heart rate, EEG, ECG, EMG, blood pressure, or skin conductance.
- Speech Recognition and graphical user interfaces.

Different methods has been applied in the literature in order to recognize and predict the human action and intention, such as Neural Network, Expectation-Maximization (E-M) method, Hidden Markov Model (HMM), Gaussian Mixture Model and Regression (GMM and GMR), Dynamic Time warping (DTW), Bayesian Networks (BN), Inverse Optimal Control or Inverse Reinforcement Learning (IOC or IRL). 

There are several Challenges to recognize the human actions and intentions in an unstructured environment, where the human performs the actions naturally; namely, variability of the time series data, changes in the speed of the action execution, and performing an action with different time series (e.g., grasping an object with different grasping poses and position).


## Objectives

In this element, at the first step, we will try to recognize the human action and intention based on the whole body kinematics and dynamics measures. The goal is to recognize human action and intention while acting and predict the evolution of features in future.

## Applications

- Retargeting of human motion to robot motion
- Human-Robot Collaboration 
- Joint-action scenarios
- Human Ergonomy prediction (in the context of human-robot collaboration)


## Outcomes
-  A repository for human action and intention recognition (Matlab, Python, or C++)

## Dependencies

- **IMPORTANT: To install properly `tensorflow 2.1` you should have python version `3.7`. Check [here](https://www.tensorflow.org/install/pip), [here](https://stackoverflow.com/questions/48720833/could-not-find-a-version-that-satisfies-the-requirement-tensorflow).** you can use the following command in macOS to install the required version of python: `brew install python@3.7`

- I'll suggest you to install the repo inside a virtual environment to avoid clashing. To do so, you need to:
  - if you do not have virtual environment, first install it `pip3 install virtualenv`
  - Identify the directory of virtual environment: `virtualenv <Directory of virtual environment>`
  - Activate the virtual environment: `source <Directory of virtual environment>/bin/activate`
  - After finishing you work, you can `deactivate` the virtual environment

- To install the dependencies, please run the following command:

  `$pip3 install --user -r requirements.txt`

  *Notes:* 

  - If you want to install the code, you can use virtual environment, as mentioned before.

  - if you want to install in system directoy, remove `--user` from the command line.

  *Notes about compiling the code:*
  - commenting the requirements of TensorflowCC in CMakelists.txt file in `modules.`
  - commenting `numpy>=1.20.0` in requirements.txt file
  - modifying `opt-einsum==3.1.0` to `opt-einsum>=3.1.0`

  
