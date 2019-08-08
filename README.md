# element_human-action-intention-recognition

## Responsible
|                Kourosh Darvish                              |
:------------------------------------------------------------:|
<img src="https://github.com/kouroshD.png" width="180"> |

## Background
The problem of human action & intention recgonition (HAIR) rises several challenges and opportunity to the robotic community. We define the human action and intention recognition at _symbolic level_ (classification problem) and _motion level_ (regression problem) as "the process of classification of the human motion among the existing ones (i.e. modeled ones) in a library of actions, the starting point and ending point of an action, predicting the human motion in future reasoning according to incomplete temporal data, and the belief degree of the action recognition and prediction".
The effectiveness of the human action recognition (high precision, recall, accuracy, and f1 scores) for a given objective and application depends on the methods and sensory information we use. For our use case in this element, we learn or model the actions offline, whereas the human action recognition is performed online.
HAIR can be performed by using following sensory data comming from the human:

-  human limb and joint kinematic and dynamic or a mixture of them (technologies: data coming from the mocap, RGBd data, raw imu werables sensors, shoes,  etc)
- RGBd data of the human environment
- Eye tracking (technologies: Scleral search coil method, Infrared occulography (IROG), Electro-occulography (EOG), Video Occulography (VOG))
- human physiological measures such as heart rate, EEG, ECG, EMG, blood pressure, or skin conductance.
- Speech Recognition and garphical user interfaces.

Different methods has been applied in the literature in order to recognize and predict the human action and intention, such as Neural Network, Expectation-Maximization (E-M) method, Hidden MArkov Model (HMM), Gaussian Mixture Model and Regression (GMM and GMR), Dynamic Time warping (DTW), Bayesian Networks (BN), Inverse Optimal Control or Inverse Reinforcement Learning (IOC or IRL). 

There are several Challenges to recognize the human actions and intentions in an unstructured environment, where the human performs the actions naturally; namely, variability of the time series data, changes in the speed of the action execution, and performing an action with differet time series (e.g., grasping an object with different grasping poses and position).


## Objectives

In this element, at the first step we will try to recognize the human action and intention based on the whole body kinematics and dyanmics measures. The goal is to recgnize the human action and intention, while performing an action, and predict the future trajectory of the features.

## Applications

- Retargeting of human motion to robot motion
- Human-Robot Collaboration 
- Joint-action scenarios
- Human Ergonomy predicion (in the context of human-robot collaboration)


## Outcomes
-  A repository for human action and intention recognition (Matlab, Python, or C++)

