### risk assessment 


This folder contains the codes for the risk assessment based on RNN.

The `Example` folder contains some example codes while developing the codes for the risk assessment. Those files may not work, and they are work in progress.

- `A_Generate_NN_Structure.m` is the script to learn to solve the risk assessment problem as a regression problem.
- `B_Test_Online.m` is used for the onlie test of the risk assessment. 
	- The input to this code is the human states data which are continuesly arriving, using yarp port.
	- Through another yarp port, the moment at which the _lifting_ task starting and ending should be passed as well.
	- An example for the online experiment is shown in https://github.com/dic-iit/element_risk-assessment/issues/48#issuecomment-825073793
	
- How to run online Demo:
- run the yarp data player where lifting task is done. The wearable data is located in https://istitutoitalianotecnologia-my.sharepoint.com/:f:/g/personal/lorenzo_rapetti_iit_it/EvNdKzSFSx9BvlFIStGUKe4Bj_ZULy5ZH0XNsbbZyvnuJg?e=sEEsYA

- Run the HDE human state provider with 48 DOF in order to stream the human joint angles. (ref: https://github.com/dic-iit/element_human-action-intention-recognition/issues/36#issuecomment-822335951)

- Run the following code https://github.com/dic-iit/element_human-action-intention-recognition/tree/devel/modules/humanMotionDataAcquisition with the following configuration file https://github.com/dic-iit/element_human-action-intention-recognition/blob/devel/modules/humanMotionDataAcquisition/app/robots/humanDataForRiskAssessment.ini . This allows to stream standard vector files with the user desired order.

- Run the humanVisualizer in https://github.com/dic-iit/element_ergonomy-control/tree/master/modules/MultiRobotVisualizer . This is used to understand when the human start the lifting task and when it is stoped.

- Open two yarp ports called : `/liftingStarted:o` and `/liftingEnded:o` and press `1` when the lifting task started and ended.

- Run the code on `B_Test_Online.m` this will connect the ports related to human data streaming, and the two port mentioned before.

- Play the werable data, wait for starting and ending of the lifting task and press 1 on the ports mentioned before in order to collect the data in the matlan script. At the end you will see the resutls of the risk assessment.

- Some examples of expected runs and results are shown in https://github.com/dic-iit/element_risk-assessment/issues/48#issuecomment-825073793 



