import numpy as np

model_path = '/home/kourosh/icub_ws/robotology-superbuild/build/install/share/human-gazebo/'
file_name = 'humanSubject01_66dof.urdf'
jointOrder = np.array(['jL5S1_rotx_val', 'jRightHip_rotx_val', 'jLeftHip_rotx_val',
                       'jLeftHip_roty_val', 'jLeftHip_rotz_val', 'jLeftKnee_rotx_val',
                       'jLeftKnee_roty_val', 'jLeftKnee_rotz_val', 'jLeftAnkle_rotx_val',
                       'jLeftAnkle_roty_val', 'jLeftAnkle_rotz_val',
                       'jLeftBallFoot_rotx_val', 'jLeftBallFoot_roty_val',
                       'jLeftBallFoot_rotz_val', 'jRightHip_roty_val',
                       'jRightHip_rotz_val', 'jRightKnee_rotx_val', 'jRightKnee_roty_val',
                       'jRightKnee_rotz_val', 'jRightAnkle_rotx_val',
                       'jRightAnkle_roty_val', 'jRightAnkle_rotz_val',
                       'jRightBallFoot_rotx_val', 'jRightBallFoot_roty_val',
                       'jRightBallFoot_rotz_val', 'jL5S1_roty_val', 'jL5S1_rotz_val',
                       'jL4L3_rotx_val', 'jL4L3_roty_val', 'jL4L3_rotz_val',
                       'jL1T12_rotx_val', 'jL1T12_roty_val', 'jL1T12_rotz_val',
                       'jT9T8_rotx_val', 'jT9T8_roty_val', 'jT9T8_rotz_val',
                       'jLeftC7Shoulder_rotx_val', 'jT1C7_rotx_val',
                       'jRightC7Shoulder_rotx_val', 'jRightC7Shoulder_roty_val',
                       'jRightC7Shoulder_rotz_val', 'jRightShoulder_rotx_val',
                       'jRightShoulder_roty_val', 'jRightShoulder_rotz_val',
                       'jRightElbow_rotx_val', 'jRightElbow_roty_val',
                       'jRightElbow_rotz_val', 'jRightWrist_rotx_val',
                       'jRightWrist_roty_val', 'jRightWrist_rotz_val', 'jT1C7_roty_val',
                       'jT1C7_rotz_val', 'jC1Head_rotx_val', 'jC1Head_roty_val',
                       'jC1Head_rotz_val', 'jLeftC7Shoulder_roty_val',
                       'jLeftC7Shoulder_rotz_val', 'jLeftShoulder_rotx_val',
                       'jLeftShoulder_roty_val', 'jLeftShoulder_rotz_val',
                       'jLeftElbow_rotx_val', 'jLeftElbow_roty_val',
                       'jLeftElbow_rotz_val', 'jLeftWrist_rotx_val',
                       'jLeftWrist_roty_val', 'jLeftWrist_rotz_val', 'jL5S1_rotx_vel',
                       'jRightHip_rotx_vel', 'jLeftHip_rotx_vel', 'jLeftHip_roty_vel',
                       'jLeftHip_rotz_vel', 'jLeftKnee_rotx_vel', 'jLeftKnee_roty_vel',
                       'jLeftKnee_rotz_vel', 'jLeftAnkle_rotx_vel', 'jLeftAnkle_roty_vel',
                       'jLeftAnkle_rotz_vel', 'jLeftBallFoot_rotx_vel',
                       'jLeftBallFoot_roty_vel', 'jLeftBallFoot_rotz_vel',
                       'jRightHip_roty_vel', 'jRightHip_rotz_vel', 'jRightKnee_rotx_vel',
                       'jRightKnee_roty_vel', 'jRightKnee_rotz_vel',
                       'jRightAnkle_rotx_vel', 'jRightAnkle_roty_vel',
                       'jRightAnkle_rotz_vel', 'jRightBallFoot_rotx_vel',
                       'jRightBallFoot_roty_vel', 'jRightBallFoot_rotz_vel',
                       'jL5S1_roty_vel', 'jL5S1_rotz_vel', 'jL4L3_rotx_vel',
                       'jL4L3_roty_vel', 'jL4L3_rotz_vel', 'jL1T12_rotx_vel',
                       'jL1T12_roty_vel', 'jL1T12_rotz_vel', 'jT9T8_rotx_vel',
                       'jT9T8_roty_vel', 'jT9T8_rotz_vel', 'jLeftC7Shoulder_rotx_vel',
                       'jT1C7_rotx_vel', 'jRightC7Shoulder_rotx_vel',
                       'jRightC7Shoulder_roty_vel', 'jRightC7Shoulder_rotz_vel',
                       'jRightShoulder_rotx_vel', 'jRightShoulder_roty_vel',
                       'jRightShoulder_rotz_vel', 'jRightElbow_rotx_vel',
                       'jRightElbow_roty_vel', 'jRightElbow_rotz_vel',
                       'jRightWrist_rotx_vel', 'jRightWrist_roty_vel',
                       'jRightWrist_rotz_vel', 'jT1C7_roty_vel', 'jT1C7_rotz_vel',
                       'jC1Head_rotx_vel', 'jC1Head_roty_vel', 'jC1Head_rotz_vel',
                       'jLeftC7Shoulder_roty_vel', 'jLeftC7Shoulder_rotz_vel',
                       'jLeftShoulder_rotx_vel', 'jLeftShoulder_roty_vel',
                       'jLeftShoulder_rotz_vel', 'jLeftElbow_rotx_vel',
                       'jLeftElbow_roty_vel', 'jLeftElbow_rotz_vel',
                       'jLeftWrist_rotx_vel', 'jLeftWrist_roty_vel',
                       'jLeftWrist_rotz_vel', 'l_shoe_fx', 'l_shoe_fy', 'l_shoe_fz'])
