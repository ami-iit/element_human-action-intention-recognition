function [var, VariableNames, VariableTypes ] = importfile(filename, dataLines)
%IMPORTFILE Import data from a text file
%  [TIME, JL5S1_ROTX_VAL, JRIGHTHIP_ROTX_VAL, JLEFTHIP_ROTX_VAL,
%  JLEFTHIP_ROTY_VAL, JLEFTHIP_ROTZ_VAL, JLEFTKNEE_ROTX_VAL,
%  JLEFTKNEE_ROTY_VAL, JLEFTKNEE_ROTZ_VAL, JLEFTANKLE_ROTX_VAL,
%  JLEFTANKLE_ROTY_VAL, JLEFTANKLE_ROTZ_VAL, JLEFTBALLFOOT_ROTX_VAL,
%  JLEFTBALLFOOT_ROTY_VAL, JLEFTBALLFOOT_ROTZ_VAL, JRIGHTHIP_ROTY_VAL,
%  JRIGHTHIP_ROTZ_VAL, JRIGHTKNEE_ROTX_VAL, JRIGHTKNEE_ROTY_VAL,
%  JRIGHTKNEE_ROTZ_VAL, JRIGHTANKLE_ROTX_VAL, JRIGHTANKLE_ROTY_VAL,
%  JRIGHTANKLE_ROTZ_VAL, JRIGHTBALLFOOT_ROTX_VAL,
%  JRIGHTBALLFOOT_ROTY_VAL, JRIGHTBALLFOOT_ROTZ_VAL, JL5S1_ROTY_VAL,
%  JL5S1_ROTZ_VAL, JL4L3_ROTX_VAL, JL4L3_ROTY_VAL, JL4L3_ROTZ_VAL,
%  JL1T12_ROTX_VAL, JL1T12_ROTY_VAL, JL1T12_ROTZ_VAL, JT9T8_ROTX_VAL,
%  JT9T8_ROTY_VAL, JT9T8_ROTZ_VAL, JLEFTC7SHOULDER_ROTX_VAL,
%  JT1C7_ROTX_VAL, JRIGHTC7SHOULDER_ROTX_VAL, JRIGHTC7SHOULDER_ROTY_VAL,
%  JRIGHTC7SHOULDER_ROTZ_VAL, JRIGHTSHOULDER_ROTX_VAL,
%  JRIGHTSHOULDER_ROTY_VAL, JRIGHTSHOULDER_ROTZ_VAL,
%  JRIGHTELBOW_ROTX_VAL, JRIGHTELBOW_ROTY_VAL, JRIGHTELBOW_ROTZ_VAL,
%  JRIGHTWRIST_ROTX_VAL, JRIGHTWRIST_ROTY_VAL, JRIGHTWRIST_ROTZ_VAL,
%  JT1C7_ROTY_VAL, JT1C7_ROTZ_VAL, JC1HEAD_ROTX_VAL, JC1HEAD_ROTY_VAL,
%  JC1HEAD_ROTZ_VAL, JLEFTC7SHOULDER_ROTY_VAL, JLEFTC7SHOULDER_ROTZ_VAL,
%  JLEFTSHOULDER_ROTX_VAL, JLEFTSHOULDER_ROTY_VAL,
%  JLEFTSHOULDER_ROTZ_VAL, JLEFTELBOW_ROTX_VAL, JLEFTELBOW_ROTY_VAL,
%  JLEFTELBOW_ROTZ_VAL, JLEFTWRIST_ROTX_VAL, JLEFTWRIST_ROTY_VAL,
%  JLEFTWRIST_ROTZ_VAL, JL5S1_ROTX_VEL, JRIGHTHIP_ROTX_VEL,
%  JLEFTHIP_ROTX_VEL, JLEFTHIP_ROTY_VEL, JLEFTHIP_ROTZ_VEL,
%  JLEFTKNEE_ROTX_VEL, JLEFTKNEE_ROTY_VEL, JLEFTKNEE_ROTZ_VEL,
%  JLEFTANKLE_ROTX_VEL, JLEFTANKLE_ROTY_VEL, JLEFTANKLE_ROTZ_VEL,
%  JLEFTBALLFOOT_ROTX_VEL, JLEFTBALLFOOT_ROTY_VEL,
%  JLEFTBALLFOOT_ROTZ_VEL, JRIGHTHIP_ROTY_VEL, JRIGHTHIP_ROTZ_VEL,
%  JRIGHTKNEE_ROTX_VEL, JRIGHTKNEE_ROTY_VEL, JRIGHTKNEE_ROTZ_VEL,
%  JRIGHTANKLE_ROTX_VEL, JRIGHTANKLE_ROTY_VEL, JRIGHTANKLE_ROTZ_VEL,
%  JRIGHTBALLFOOT_ROTX_VEL, JRIGHTBALLFOOT_ROTY_VEL,
%  JRIGHTBALLFOOT_ROTZ_VEL, JL5S1_ROTY_VEL, JL5S1_ROTZ_VEL,
%  JL4L3_ROTX_VEL, JL4L3_ROTY_VEL, JL4L3_ROTZ_VEL, JL1T12_ROTX_VEL,
%  JL1T12_ROTY_VEL, JL1T12_ROTZ_VEL, JT9T8_ROTX_VEL, JT9T8_ROTY_VEL,
%  JT9T8_ROTZ_VEL, JLEFTC7SHOULDER_ROTX_VEL, JT1C7_ROTX_VEL,
%  JRIGHTC7SHOULDER_ROTX_VEL, JRIGHTC7SHOULDER_ROTY_VEL,
%  JRIGHTC7SHOULDER_ROTZ_VEL, JRIGHTSHOULDER_ROTX_VEL,
%  JRIGHTSHOULDER_ROTY_VEL, JRIGHTSHOULDER_ROTZ_VEL,
%  JRIGHTELBOW_ROTX_VEL, JRIGHTELBOW_ROTY_VEL, JRIGHTELBOW_ROTZ_VEL,
%  JRIGHTWRIST_ROTX_VEL, JRIGHTWRIST_ROTY_VEL, JRIGHTWRIST_ROTZ_VEL,
%  JT1C7_ROTY_VEL, JT1C7_ROTZ_VEL, JC1HEAD_ROTX_VEL, JC1HEAD_ROTY_VEL,
%  JC1HEAD_ROTZ_VEL, JLEFTC7SHOULDER_ROTY_VEL, JLEFTC7SHOULDER_ROTZ_VEL,
%  JLEFTSHOULDER_ROTX_VEL, JLEFTSHOULDER_ROTY_VEL,
%  JLEFTSHOULDER_ROTZ_VEL, JLEFTELBOW_ROTX_VEL, JLEFTELBOW_ROTY_VEL,
%  JLEFTELBOW_ROTZ_VEL, JLEFTWRIST_ROTX_VEL, JLEFTWRIST_ROTY_VEL,
%  JLEFTWRIST_ROTZ_VEL, L_SHOE_FX, L_SHOE_FY, L_SHOE_FZ, L_SHOE_TX,
%  L_SHOE_TY, L_SHOE_TZ, R_SHOE_FX, R_SHOE_FY, R_SHOE_FZ, R_SHOE_TX,
%  R_SHOE_TY, R_SHOE_TZ] = IMPORTFILE(FILENAME) reads data from text
%  file FILENAME for the default selection.  Returns the data as column
%  vectors.
%
%  [TIME, JL5S1_ROTX_VAL, JRIGHTHIP_ROTX_VAL, JLEFTHIP_ROTX_VAL,
%  JLEFTHIP_ROTY_VAL, JLEFTHIP_ROTZ_VAL, JLEFTKNEE_ROTX_VAL,
%  JLEFTKNEE_ROTY_VAL, JLEFTKNEE_ROTZ_VAL, JLEFTANKLE_ROTX_VAL,
%  JLEFTANKLE_ROTY_VAL, JLEFTANKLE_ROTZ_VAL, JLEFTBALLFOOT_ROTX_VAL,
%  JLEFTBALLFOOT_ROTY_VAL, JLEFTBALLFOOT_ROTZ_VAL, JRIGHTHIP_ROTY_VAL,
%  JRIGHTHIP_ROTZ_VAL, JRIGHTKNEE_ROTX_VAL, JRIGHTKNEE_ROTY_VAL,
%  JRIGHTKNEE_ROTZ_VAL, JRIGHTANKLE_ROTX_VAL, JRIGHTANKLE_ROTY_VAL,
%  JRIGHTANKLE_ROTZ_VAL, JRIGHTBALLFOOT_ROTX_VAL,
%  JRIGHTBALLFOOT_ROTY_VAL, JRIGHTBALLFOOT_ROTZ_VAL, JL5S1_ROTY_VAL,
%  JL5S1_ROTZ_VAL, JL4L3_ROTX_VAL, JL4L3_ROTY_VAL, JL4L3_ROTZ_VAL,
%  JL1T12_ROTX_VAL, JL1T12_ROTY_VAL, JL1T12_ROTZ_VAL, JT9T8_ROTX_VAL,
%  JT9T8_ROTY_VAL, JT9T8_ROTZ_VAL, JLEFTC7SHOULDER_ROTX_VAL,
%  JT1C7_ROTX_VAL, JRIGHTC7SHOULDER_ROTX_VAL, JRIGHTC7SHOULDER_ROTY_VAL,
%  JRIGHTC7SHOULDER_ROTZ_VAL, JRIGHTSHOULDER_ROTX_VAL,
%  JRIGHTSHOULDER_ROTY_VAL, JRIGHTSHOULDER_ROTZ_VAL,
%  JRIGHTELBOW_ROTX_VAL, JRIGHTELBOW_ROTY_VAL, JRIGHTELBOW_ROTZ_VAL,
%  JRIGHTWRIST_ROTX_VAL, JRIGHTWRIST_ROTY_VAL, JRIGHTWRIST_ROTZ_VAL,
%  JT1C7_ROTY_VAL, JT1C7_ROTZ_VAL, JC1HEAD_ROTX_VAL, JC1HEAD_ROTY_VAL,
%  JC1HEAD_ROTZ_VAL, JLEFTC7SHOULDER_ROTY_VAL, JLEFTC7SHOULDER_ROTZ_VAL,
%  JLEFTSHOULDER_ROTX_VAL, JLEFTSHOULDER_ROTY_VAL,
%  JLEFTSHOULDER_ROTZ_VAL, JLEFTELBOW_ROTX_VAL, JLEFTELBOW_ROTY_VAL,
%  JLEFTELBOW_ROTZ_VAL, JLEFTWRIST_ROTX_VAL, JLEFTWRIST_ROTY_VAL,
%  JLEFTWRIST_ROTZ_VAL, JL5S1_ROTX_VEL, JRIGHTHIP_ROTX_VEL,
%  JLEFTHIP_ROTX_VEL, JLEFTHIP_ROTY_VEL, JLEFTHIP_ROTZ_VEL,
%  JLEFTKNEE_ROTX_VEL, JLEFTKNEE_ROTY_VEL, JLEFTKNEE_ROTZ_VEL,
%  JLEFTANKLE_ROTX_VEL, JLEFTANKLE_ROTY_VEL, JLEFTANKLE_ROTZ_VEL,
%  JLEFTBALLFOOT_ROTX_VEL, JLEFTBALLFOOT_ROTY_VEL,
%  JLEFTBALLFOOT_ROTZ_VEL, JRIGHTHIP_ROTY_VEL, JRIGHTHIP_ROTZ_VEL,
%  JRIGHTKNEE_ROTX_VEL, JRIGHTKNEE_ROTY_VEL, JRIGHTKNEE_ROTZ_VEL,
%  JRIGHTANKLE_ROTX_VEL, JRIGHTANKLE_ROTY_VEL, JRIGHTANKLE_ROTZ_VEL,
%  JRIGHTBALLFOOT_ROTX_VEL, JRIGHTBALLFOOT_ROTY_VEL,
%  JRIGHTBALLFOOT_ROTZ_VEL, JL5S1_ROTY_VEL, JL5S1_ROTZ_VEL,
%  JL4L3_ROTX_VEL, JL4L3_ROTY_VEL, JL4L3_ROTZ_VEL, JL1T12_ROTX_VEL,
%  JL1T12_ROTY_VEL, JL1T12_ROTZ_VEL, JT9T8_ROTX_VEL, JT9T8_ROTY_VEL,
%  JT9T8_ROTZ_VEL, JLEFTC7SHOULDER_ROTX_VEL, JT1C7_ROTX_VEL,
%  JRIGHTC7SHOULDER_ROTX_VEL, JRIGHTC7SHOULDER_ROTY_VEL,
%  JRIGHTC7SHOULDER_ROTZ_VEL, JRIGHTSHOULDER_ROTX_VEL,
%  JRIGHTSHOULDER_ROTY_VEL, JRIGHTSHOULDER_ROTZ_VEL,
%  JRIGHTELBOW_ROTX_VEL, JRIGHTELBOW_ROTY_VEL, JRIGHTELBOW_ROTZ_VEL,
%  JRIGHTWRIST_ROTX_VEL, JRIGHTWRIST_ROTY_VEL, JRIGHTWRIST_ROTZ_VEL,
%  JT1C7_ROTY_VEL, JT1C7_ROTZ_VEL, JC1HEAD_ROTX_VEL, JC1HEAD_ROTY_VEL,
%  JC1HEAD_ROTZ_VEL, JLEFTC7SHOULDER_ROTY_VEL, JLEFTC7SHOULDER_ROTZ_VEL,
%  JLEFTSHOULDER_ROTX_VEL, JLEFTSHOULDER_ROTY_VEL,
%  JLEFTSHOULDER_ROTZ_VEL, JLEFTELBOW_ROTX_VEL, JLEFTELBOW_ROTY_VEL,
%  JLEFTELBOW_ROTZ_VEL, JLEFTWRIST_ROTX_VEL, JLEFTWRIST_ROTY_VEL,
%  JLEFTWRIST_ROTZ_VEL, L_SHOE_FX, L_SHOE_FY, L_SHOE_FZ, L_SHOE_TX,
%  L_SHOE_TY, L_SHOE_TZ, R_SHOE_FX, R_SHOE_FY, R_SHOE_FZ, R_SHOE_TX,
%  R_SHOE_TY, R_SHOE_TZ] = IMPORTFILE(FILE, DATALINES) reads data for
%  the specified row interval(s) of text file FILENAME. Specify
%  DATALINES as a positive scalar integer or a N-by-2 array of positive
%  scalar integers for dis-contiguous row intervals.
%
%  Example:
%  [time, jL5S1_rotx_val, jRightHip_rotx_val, jLeftHip_rotx_val, jLeftHip_roty_val, jLeftHip_rotz_val, jLeftKnee_rotx_val, jLeftKnee_roty_val, jLeftKnee_rotz_val, jLeftAnkle_rotx_val, jLeftAnkle_roty_val, jLeftAnkle_rotz_val, jLeftBallFoot_rotx_val, jLeftBallFoot_roty_val, jLeftBallFoot_rotz_val, jRightHip_roty_val, jRightHip_rotz_val, jRightKnee_rotx_val, jRightKnee_roty_val, jRightKnee_rotz_val, jRightAnkle_rotx_val, jRightAnkle_roty_val, jRightAnkle_rotz_val, jRightBallFoot_rotx_val, jRightBallFoot_roty_val, jRightBallFoot_rotz_val, jL5S1_roty_val, jL5S1_rotz_val, jL4L3_rotx_val, jL4L3_roty_val, jL4L3_rotz_val, jL1T12_rotx_val, jL1T12_roty_val, jL1T12_rotz_val, jT9T8_rotx_val, jT9T8_roty_val, jT9T8_rotz_val, jLeftC7Shoulder_rotx_val, jT1C7_rotx_val, jRightC7Shoulder_rotx_val, jRightC7Shoulder_roty_val, jRightC7Shoulder_rotz_val, jRightShoulder_rotx_val, jRightShoulder_roty_val, jRightShoulder_rotz_val, jRightElbow_rotx_val, jRightElbow_roty_val, jRightElbow_rotz_val, jRightWrist_rotx_val, jRightWrist_roty_val, jRightWrist_rotz_val, jT1C7_roty_val, jT1C7_rotz_val, jC1Head_rotx_val, jC1Head_roty_val, jC1Head_rotz_val, jLeftC7Shoulder_roty_val, jLeftC7Shoulder_rotz_val, jLeftShoulder_rotx_val, jLeftShoulder_roty_val, jLeftShoulder_rotz_val, jLeftElbow_rotx_val, jLeftElbow_roty_val, jLeftElbow_rotz_val, jLeftWrist_rotx_val, jLeftWrist_roty_val, jLeftWrist_rotz_val, jL5S1_rotx_vel, jRightHip_rotx_vel, jLeftHip_rotx_vel, jLeftHip_roty_vel, jLeftHip_rotz_vel, jLeftKnee_rotx_vel, jLeftKnee_roty_vel, jLeftKnee_rotz_vel, jLeftAnkle_rotx_vel, jLeftAnkle_roty_vel, jLeftAnkle_rotz_vel, jLeftBallFoot_rotx_vel, jLeftBallFoot_roty_vel, jLeftBallFoot_rotz_vel, jRightHip_roty_vel, jRightHip_rotz_vel, jRightKnee_rotx_vel, jRightKnee_roty_vel, jRightKnee_rotz_vel, jRightAnkle_rotx_vel, jRightAnkle_roty_vel, jRightAnkle_rotz_vel, jRightBallFoot_rotx_vel, jRightBallFoot_roty_vel, jRightBallFoot_rotz_vel, jL5S1_roty_vel, jL5S1_rotz_vel, jL4L3_rotx_vel, jL4L3_roty_vel, jL4L3_rotz_vel, jL1T12_rotx_vel, jL1T12_roty_vel, jL1T12_rotz_vel, jT9T8_rotx_vel, jT9T8_roty_vel, jT9T8_rotz_vel, jLeftC7Shoulder_rotx_vel, jT1C7_rotx_vel, jRightC7Shoulder_rotx_vel, jRightC7Shoulder_roty_vel, jRightC7Shoulder_rotz_vel, jRightShoulder_rotx_vel, jRightShoulder_roty_vel, jRightShoulder_rotz_vel, jRightElbow_rotx_vel, jRightElbow_roty_vel, jRightElbow_rotz_vel, jRightWrist_rotx_vel, jRightWrist_roty_vel, jRightWrist_rotz_vel, jT1C7_roty_vel, jT1C7_rotz_vel, jC1Head_rotx_vel, jC1Head_roty_vel, jC1Head_rotz_vel, jLeftC7Shoulder_roty_vel, jLeftC7Shoulder_rotz_vel, jLeftShoulder_rotx_vel, jLeftShoulder_roty_vel, jLeftShoulder_rotz_vel, jLeftElbow_rotx_vel, jLeftElbow_roty_vel, jLeftElbow_rotz_vel, jLeftWrist_rotx_vel, jLeftWrist_roty_vel, jLeftWrist_rotz_vel, l_shoe_fx, l_shoe_fy, l_shoe_fz, l_shoe_tx, l_shoe_ty, l_shoe_tz, r_shoe_fx, r_shoe_fy, r_shoe_fz, r_shoe_tx, r_shoe_ty, r_shoe_tz] = importfile("/home/kourosh/icub_ws/external/element_human-action-intention-recognition/matlab/HumanActionIntentionPrediction/RawData/Dataset01/Dataset_2021_03_23_13_45_06.txt", [2, Inf]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 23-Mar-2021 15:10:16

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [2, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 145);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = " ";

% Specify column names and types
opts.VariableNames = ["time", "jL5S1_rotx_val", "jRightHip_rotx_val", "jLeftHip_rotx_val", "jLeftHip_roty_val", "jLeftHip_rotz_val", "jLeftKnee_rotx_val", "jLeftKnee_roty_val", "jLeftKnee_rotz_val", "jLeftAnkle_rotx_val", "jLeftAnkle_roty_val", "jLeftAnkle_rotz_val", "jLeftBallFoot_rotx_val", "jLeftBallFoot_roty_val", "jLeftBallFoot_rotz_val", "jRightHip_roty_val", "jRightHip_rotz_val", "jRightKnee_rotx_val", "jRightKnee_roty_val", "jRightKnee_rotz_val", "jRightAnkle_rotx_val", "jRightAnkle_roty_val", "jRightAnkle_rotz_val", "jRightBallFoot_rotx_val", "jRightBallFoot_roty_val", "jRightBallFoot_rotz_val", "jL5S1_roty_val", "jL5S1_rotz_val", "jL4L3_rotx_val", "jL4L3_roty_val", "jL4L3_rotz_val", "jL1T12_rotx_val", "jL1T12_roty_val", "jL1T12_rotz_val", "jT9T8_rotx_val", "jT9T8_roty_val", "jT9T8_rotz_val", "jLeftC7Shoulder_rotx_val", "jT1C7_rotx_val", "jRightC7Shoulder_rotx_val", "jRightC7Shoulder_roty_val", "jRightC7Shoulder_rotz_val", "jRightShoulder_rotx_val", "jRightShoulder_roty_val", "jRightShoulder_rotz_val", "jRightElbow_rotx_val", "jRightElbow_roty_val", "jRightElbow_rotz_val", "jRightWrist_rotx_val", "jRightWrist_roty_val", "jRightWrist_rotz_val", "jT1C7_roty_val", "jT1C7_rotz_val", "jC1Head_rotx_val", "jC1Head_roty_val", "jC1Head_rotz_val", "jLeftC7Shoulder_roty_val", "jLeftC7Shoulder_rotz_val", "jLeftShoulder_rotx_val", "jLeftShoulder_roty_val", "jLeftShoulder_rotz_val", "jLeftElbow_rotx_val", "jLeftElbow_roty_val", "jLeftElbow_rotz_val", "jLeftWrist_rotx_val", "jLeftWrist_roty_val", "jLeftWrist_rotz_val", "jL5S1_rotx_vel", "jRightHip_rotx_vel", "jLeftHip_rotx_vel", "jLeftHip_roty_vel", "jLeftHip_rotz_vel", "jLeftKnee_rotx_vel", "jLeftKnee_roty_vel", "jLeftKnee_rotz_vel", "jLeftAnkle_rotx_vel", "jLeftAnkle_roty_vel", "jLeftAnkle_rotz_vel", "jLeftBallFoot_rotx_vel", "jLeftBallFoot_roty_vel", "jLeftBallFoot_rotz_vel", "jRightHip_roty_vel", "jRightHip_rotz_vel", "jRightKnee_rotx_vel", "jRightKnee_roty_vel", "jRightKnee_rotz_vel", "jRightAnkle_rotx_vel", "jRightAnkle_roty_vel", "jRightAnkle_rotz_vel", "jRightBallFoot_rotx_vel", "jRightBallFoot_roty_vel", "jRightBallFoot_rotz_vel", "jL5S1_roty_vel", "jL5S1_rotz_vel", "jL4L3_rotx_vel", "jL4L3_roty_vel", "jL4L3_rotz_vel", "jL1T12_rotx_vel", "jL1T12_roty_vel", "jL1T12_rotz_vel", "jT9T8_rotx_vel", "jT9T8_roty_vel", "jT9T8_rotz_vel", "jLeftC7Shoulder_rotx_vel", "jT1C7_rotx_vel", "jRightC7Shoulder_rotx_vel", "jRightC7Shoulder_roty_vel", "jRightC7Shoulder_rotz_vel", "jRightShoulder_rotx_vel", "jRightShoulder_roty_vel", "jRightShoulder_rotz_vel", "jRightElbow_rotx_vel", "jRightElbow_roty_vel", "jRightElbow_rotz_vel", "jRightWrist_rotx_vel", "jRightWrist_roty_vel", "jRightWrist_rotz_vel", "jT1C7_roty_vel", "jT1C7_rotz_vel", "jC1Head_rotx_vel", "jC1Head_roty_vel", "jC1Head_rotz_vel", "jLeftC7Shoulder_roty_vel", "jLeftC7Shoulder_rotz_vel", "jLeftShoulder_rotx_vel", "jLeftShoulder_roty_vel", "jLeftShoulder_rotz_vel", "jLeftElbow_rotx_vel", "jLeftElbow_roty_vel", "jLeftElbow_rotz_vel", "jLeftWrist_rotx_vel", "jLeftWrist_roty_vel", "jLeftWrist_rotz_vel", "l_shoe_fx", "l_shoe_fy", "l_shoe_fz", "l_shoe_tx", "l_shoe_ty", "l_shoe_tz", "r_shoe_fx", "r_shoe_fy", "r_shoe_fz", "r_shoe_tx", "r_shoe_ty", "r_shoe_tz"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ImportErrorRule = "omitrow";
opts.MissingRule = "omitrow";
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts.ConsecutiveDelimitersRule = "join";
opts.LeadingDelimitersRule = "ignore";

% Import the data
tbl = readtable(filename, opts);

%% Convert to output type
var(:,1) = tbl.time;
var(:,2) = tbl.jL5S1_rotx_val;
var(:,3) = tbl.jRightHip_rotx_val;
var(:,4) = tbl.jLeftHip_rotx_val;
var(:,5) = tbl.jLeftHip_roty_val;
var(:,6) = tbl.jLeftHip_rotz_val;
var(:,7) = tbl.jLeftKnee_rotx_val;
var(:,8) = tbl.jLeftKnee_roty_val;
var(:,9) = tbl.jLeftKnee_rotz_val;
var(:,10) = tbl.jLeftAnkle_rotx_val;
var(:,11) = tbl.jLeftAnkle_roty_val;
var(:,12) = tbl.jLeftAnkle_rotz_val;
var(:,13) = tbl.jLeftBallFoot_rotx_val;
var(:,14) = tbl.jLeftBallFoot_roty_val;
var(:,15) = tbl.jLeftBallFoot_rotz_val;
var(:,16) = tbl.jRightHip_roty_val;
var(:,17) = tbl.jRightHip_rotz_val;
var(:,18) = tbl.jRightKnee_rotx_val;
var(:,19) = tbl.jRightKnee_roty_val;
var(:,20) = tbl.jRightKnee_rotz_val;
var(:,21) = tbl.jRightAnkle_rotx_val;
var(:,22) = tbl.jRightAnkle_roty_val;
var(:,23) = tbl.jRightAnkle_rotz_val;
var(:,24) = tbl.jRightBallFoot_rotx_val;
var(:,25) = tbl.jRightBallFoot_roty_val;
var(:,26) = tbl.jRightBallFoot_rotz_val;
var(:,27) = tbl.jL5S1_roty_val;
var(:,28) = tbl.jL5S1_rotz_val;
var(:,29) = tbl.jL4L3_rotx_val;
var(:,30) = tbl.jL4L3_roty_val;
var(:,31) = tbl.jL4L3_rotz_val;
var(:,32) = tbl.jL1T12_rotx_val;
var(:,33) = tbl.jL1T12_roty_val;
var(:,34) = tbl.jL1T12_rotz_val;
var(:,35) = tbl.jT9T8_rotx_val;
var(:,36) = tbl.jT9T8_roty_val;
var(:,37) = tbl.jT9T8_rotz_val;
var(:,38) = tbl.jLeftC7Shoulder_rotx_val;
var(:,39) = tbl.jT1C7_rotx_val;
var(:,40) = tbl.jRightC7Shoulder_rotx_val;
var(:,41) = tbl.jRightC7Shoulder_roty_val;
var(:,42) = tbl.jRightC7Shoulder_rotz_val;
var(:,43) = tbl.jRightShoulder_rotx_val;
var(:,44) = tbl.jRightShoulder_roty_val;
var(:,45) = tbl.jRightShoulder_rotz_val;
var(:,46) = tbl.jRightElbow_rotx_val;
var(:,47) = tbl.jRightElbow_roty_val;
var(:,48) = tbl.jRightElbow_rotz_val;
var(:,49) = tbl.jRightWrist_rotx_val;
var(:,50) = tbl.jRightWrist_roty_val;
var(:,51) = tbl.jRightWrist_rotz_val;
var(:,52) = tbl.jT1C7_roty_val;
var(:,53) = tbl.jT1C7_rotz_val;
var(:,54) = tbl.jC1Head_rotx_val;
var(:,55) = tbl.jC1Head_roty_val;
var(:,56) = tbl.jC1Head_rotz_val;
var(:,57) = tbl.jLeftC7Shoulder_roty_val;
var(:,58)  = tbl.jLeftC7Shoulder_rotz_val;
var(:,59)  = tbl.jLeftShoulder_rotx_val;
var(:,60)  = tbl.jLeftShoulder_roty_val;
var(:,61)  = tbl.jLeftShoulder_rotz_val;
var(:,62)  = tbl.jLeftElbow_rotx_val;
var(:,63)  = tbl.jLeftElbow_roty_val;
var(:,64)  = tbl.jLeftElbow_rotz_val;
var(:,65)  = tbl.jLeftWrist_rotx_val;
var(:,66)  = tbl.jLeftWrist_roty_val;
var(:,67)  = tbl.jLeftWrist_rotz_val;
var(:,68)  = tbl.jL5S1_rotx_vel;
var(:,69)  = tbl.jRightHip_rotx_vel;
var(:,70)  = tbl.jLeftHip_rotx_vel;
var(:,71)  = tbl.jLeftHip_roty_vel;
var(:,72)  = tbl.jLeftHip_rotz_vel;
var(:,73)  = tbl.jLeftKnee_rotx_vel;
var(:,74)  = tbl.jLeftKnee_roty_vel;
var(:,75)  = tbl.jLeftKnee_rotz_vel;
var(:,76)  = tbl.jLeftAnkle_rotx_vel;
var(:,77)  = tbl.jLeftAnkle_roty_vel;
var(:,78)  = tbl.jLeftAnkle_rotz_vel;
var(:,79)  = tbl.jLeftBallFoot_rotx_vel;
var(:,80)  = tbl.jLeftBallFoot_roty_vel;
var(:,81)  = tbl.jLeftBallFoot_rotz_vel;
var(:,82)  = tbl.jRightHip_roty_vel;
var(:,83)  = tbl.jRightHip_rotz_vel;
var(:,84)  = tbl.jRightKnee_rotx_vel;
var(:,85)  = tbl.jRightKnee_roty_vel;
var(:,86)  = tbl.jRightKnee_rotz_vel;
var(:,87)  = tbl.jRightAnkle_rotx_vel;
var(:,88)  = tbl.jRightAnkle_roty_vel;
var(:,89)  = tbl.jRightAnkle_rotz_vel;
var(:,90)  = tbl.jRightBallFoot_rotx_vel;
var(:,91)  = tbl.jRightBallFoot_roty_vel;
var(:,92)  = tbl.jRightBallFoot_rotz_vel;
var(:,93)  = tbl.jL5S1_roty_vel;
var(:,94)  = tbl.jL5S1_rotz_vel;
var(:,95)  = tbl.jL4L3_rotx_vel;
var(:,96)  = tbl.jL4L3_roty_vel;
var(:,97)  = tbl.jL4L3_rotz_vel;
var(:,98)  = tbl.jL1T12_rotx_vel;
var(:,99)  = tbl.jL1T12_roty_vel;
var(:,100)  = tbl.jL1T12_rotz_vel;
var(:,101)  = tbl.jT9T8_rotx_vel;
var(:,102)  = tbl.jT9T8_roty_vel;
var(:,103)  = tbl.jT9T8_rotz_vel;
var(:,104)  = tbl.jLeftC7Shoulder_rotx_vel;
var(:,105)  = tbl.jT1C7_rotx_vel;
var(:,106)  = tbl.jRightC7Shoulder_rotx_vel;
var(:,107)  = tbl.jRightC7Shoulder_roty_vel;
var(:,108)  = tbl.jRightC7Shoulder_rotz_vel;
var(:,109)  = tbl.jRightShoulder_rotx_vel;
var(:,110)  = tbl.jRightShoulder_roty_vel;
var(:,111)  = tbl.jRightShoulder_rotz_vel;
var(:,112)  = tbl.jRightElbow_rotx_vel;
var(:,113)  = tbl.jRightElbow_roty_vel;
var(:,114)  = tbl.jRightElbow_rotz_vel;
var(:,115)  = tbl.jRightWrist_rotx_vel;
var(:,116)  = tbl.jRightWrist_roty_vel;
var(:,117)  = tbl.jRightWrist_rotz_vel;
var(:,118)  = tbl.jT1C7_roty_vel;
var(:,119)  = tbl.jT1C7_rotz_vel;
var(:,120)  = tbl.jC1Head_rotx_vel;
var(:,121)  = tbl.jC1Head_roty_vel;
var(:,122)  = tbl.jC1Head_rotz_vel;
var(:,123)  = tbl.jLeftC7Shoulder_roty_vel;
var(:,124)  = tbl.jLeftC7Shoulder_rotz_vel;
var(:,125)  = tbl.jLeftShoulder_rotx_vel;
var(:,126)  = tbl.jLeftShoulder_roty_vel;
var(:,127)  = tbl.jLeftShoulder_rotz_vel;
var(:,128)  = tbl.jLeftElbow_rotx_vel;
var(:,129)  = tbl.jLeftElbow_roty_vel;
var(:,130)  = tbl.jLeftElbow_rotz_vel;
var(:,131)  = tbl.jLeftWrist_rotx_vel;
var(:,132)  = tbl.jLeftWrist_roty_vel;
var(:,133)  = tbl.jLeftWrist_rotz_vel;
var(:,134)  = tbl.l_shoe_fx;
var(:,135)  = tbl.l_shoe_fy;
var(:,136)  = tbl.l_shoe_fz;
var(:,137)  = tbl.l_shoe_tx;
var(:,138)  = tbl.l_shoe_ty;
var(:,139) = tbl.l_shoe_tz;
var(:,140)  = tbl.r_shoe_fx;
var(:,141)  = tbl.r_shoe_fy;
var(:,142)  = tbl.r_shoe_fz;
var(:,143)  = tbl.r_shoe_tx;
var(:,144)  = tbl.r_shoe_ty;
var(:,145)  = tbl.r_shoe_tz;

VariableNames = opts.VariableNames;
VariableTypes = opts.VariableTypes;

end