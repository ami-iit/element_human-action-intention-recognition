#!/bin/sh
python main_online_animation_action_prediction.py &
python main_online_animation_joint_angle_prediction.py &
python main_online_animation_wrench_prediction.py
