# =====================
# === Configuration ===
# =====================
import guidedMoE as GMoE

# high level flags for training
learn_moe_model = True
relearn_moe_model = False
learn_cnn_model = False
learn_lstm_model = False
do_performance_analysis = True
normalize_input = False
output_categorical = True
save_model = True
verbose = False
reduce_joints = False

# save model
model_name = 'model'
models_path = 'NN_models/' + GMoE.get_time_now()
    
# read data path
data_path01 = '~/element_human-action-intention-recognition/dataset/lifting_test/2023_02_09_lifitng_data_labeled/01_cheng_labeled.txt'
data_path02 = '~/element_human-action-intention-recognition/dataset/lifting_test/2023_02_09_lifitng_data_labeled/02_cheng_labeled.txt'
data_path03 = '~/element_human-action-intention-recognition/dataset/lifting_test/2023_02_09_lifitng_data_labeled/03_cheng_labeled.txt'
data_path04 = '~/element_human-action-intention-recognition/dataset/lifting_test/2023_02_09_lifitng_data_labeled/01_lorenzo_labeled.txt'
data_path05 = '~/element_human-action-intention-recognition/dataset/lifting_test/2023_02_09_lifitng_data_labeled/02_lorenzo_labeled.txt'
data_path06 = '~/element_human-action-intention-recognition/dataset/lifting_test/2023_02_09_lifitng_data_labeled/03_lorenzo_labeled.txt'

# load_model_path = 'NN_models/2023-02-17 11:08:18'
# load_model_name = 'model_MoE_Best'
# split data for training/validation/test
train_percentage = 0.7
val_percentage = 0.2
test_percentage = 1.0 - (train_percentage + val_percentage)

# L1 and L2 regularization
regularization_l2_gate = 1.0e-2
regularization_l1_gate = 1.0e-2
regularization_l2_experts = 1.0e-2
regularization_l1_experts = 1.0e-2

# dropout rate
dropout_rate = 0.4

# denormalization parameters
user_mass = 75.0
gravity = 9.81

# NN setup
output_steps = 50 # prediction horizon 1s (1/0.04=25, 1/0.1=10, 1/0.01=100)
shift = output_steps
input_width = 10 # previous data used during prediction 10*0.01=0.1s
max_subplots = 5
max_epochs = 40
patience = 10 # used for eary stopping
number_experts_outputs = 43 # 31 joint valuse + 12 feet wrenches
exp_output_idx = 31

#input_feature_list = ['jT9T8_rotx_val', 'jT9T8_roty_val', 'jT9T8_rotz_val',
#                      'jRightShoulder_rotx_val', 'jRightShoulder_roty_val', 'jRightShoulder_rotz_val',
#                      'jRightElbow_roty_val', 'jRightElbow_rotz_val',
#                      'jLeftShoulder_rotx_val', 'jLeftShoulder_roty_val', 'jLeftShoulder_rotz_val',
#                      'jLeftElbow_roty_val', 'jLeftElbow_rotz_val',
#                      'jLeftHip_rotx_val', 'jLeftHip_roty_val', 'jLeftHip_rotz_val',
#                      'jLeftKnee_roty_val', 'jLeftKnee_rotz_val',
#                      'jLeftAnkle_rotx_val', 'jLeftAnkle_roty_val', 'jLeftAnkle_rotz_val',
#                      'jLeftBallFoot_roty_val', 'jRightBallFoot_roty_val',
#                      'jRightHip_rotx_val', 'jRightHip_roty_val', 'jRightHip_rotz_val',
#                      'jRightKnee_roty_val', 'jRightKnee_rotz_val',
#                      'jRightAnkle_rotx_val', 'jRightAnkle_roty_val', 'jRightAnkle_rotz_val',
#                      'jT9T8_rotx_vel', 'jT9T8_roty_vel', 'jT9T8_roty_vel',
#                      'jRightShoulder_rotx_vel', 'jRightShoulder_roty_vel', 'jRightShoulder_rotz_vel',
#                      'jRightElbow_roty_vel', 'jRightElbow_rotz_vel',
#                      'jLeftShoulder_rotx_vel', 'jLeftShoulder_roty_vel', 'jLeftShoulder_rotz_vel',
#                      'jLeftElbow_roty_vel', 'jLeftElbow_rotz_vel',
#                      'jLeftHip_rotx_vel', 'jLeftHip_roty_vel', 'jLeftHip_rotz_vel',
#                      'jLeftKnee_roty_vel', 'jLeftKnee_rotz_vel',
#                      'jLeftAnkle_rotx_vel', 'jLeftAnkle_roty_vel', 'jLeftAnkle_rotz_vel',
#                      'jLeftBallFoot_roty_vel', 'jRightBallFoot_roty_vel',
#                      'jRightHip_rotx_vel', 'jRightHip_roty_vel', 'jRightHip_rotz_vel',
#                      'jRightKnee_roty_vel', 'jRightKnee_rotz_vel',
#                      'jRightAnkle_rotx_vel', 'jRightAnkle_roty_vel', 'jRightAnkle_rotz_vel',
#                      'l_shoe_fx', 'l_shoe_fy', 'l_shoe_fz', 'l_shoe_tx', 'l_shoe_ty', 'l_shoe_tz',
#                      'r_shoe_fx', 'r_shoe_fy', 'r_shoe_fz', 'r_shoe_tx', 'r_shoe_ty', 'r_shoe_tz']

output_feature_list = ['label']
pop_list = ['time', 'label']

#full_labels = ["none", "standing", "stooping", "bending", "straightening", "rising", "placing", 
#              "fetching", "stoop-lowering", "bend-lowering", "stoop-back", "bend-back"]
full_labels = ["rising", "squatting", "standing"]

reduced_input_indices = [34, 35, 36, 
                         42, 43, 44, 
                         46, 47, 
                         58, 59, 60, 
                         62, 63, 
                         3, 4, 5, 
                         7, 8,
                         9, 10, 11,
                         13, 24, 
                         15, 16, 
                         17, 18, 19, 
                         20, 21, 22, 
                         100, 101, 102, 
                         108, 109, 110, 
                         112, 113, 
                         124, 125, 126, 
                         128, 129, 
                         69, 70, 71, 
                         73, 74, 
                         75, 76, 77, 
                         79, 90, 
                         68, 81, 82, 
                         84, 85, 
                         86, 87, 88, 
                         133, 134, 135, 136, 137, 138, 
                         139, 140, 141, 142, 143, 144]

