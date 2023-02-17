#####################
##### Libraries #####
#####################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import guidedMoE as GMoE
import dataProcessing as dp
import dataConfig as cfg
import keras
from Utilities import load_model_from_file

#########################
##### main training #####
#########################
if __name__ == "__main__":
    # ========================
    # === Data praperation ===
    # ========================
    if cfg.reduce_joints:
        # 13 joints, 74 features (31 positions, 31 velocities, 12 wrenches)
        input_feature_list = cfg.input_feature_list
    else:
        input_feature_list = []
   
    output_feature_list = cfg.output_feature_list
    pop_list = cfg.pop_list

    # data_path = cfg.data_path01
    # df_raw = pd.read_csv(data_path, sep=' ')
    df_raw = dp.makeLargeDataSet()

    if input_feature_list:
        df_input_action01 = df_raw[input_feature_list].copy()
    else:
        df_input_action01 = df_raw.copy()
    
    if output_feature_list:  # ! define the output feature list
        df_output01 = df_raw[output_feature_list].copy()
    else:
        df_output01 = df_raw.copy()
    
    if cfg.output_categorical:  # ! get the output label in case of categorical ouputs
        df_output01 = pd.get_dummies(df_output01)
        output_labels = df_output01.keys()

        # number of categories should be equal to the size of annotation list
        number_categories = len(output_labels)  # ! the number of categories
        print('total output labels: {}'.format(output_labels))

    # ! start the time from the zero, depends on the application
    if 'time' in df_raw:
        df_time = df_raw['time'] - df_raw['time'][0]

    # test if the object has the right type of data
    df_input_action01.head()

    if pop_list is not None:
        for pop_name in pop_list:
            if pop_name in df_input_action01:
                df_input_action01.pop(pop_name)
                print("[INFO] show now input data: ", df_input_action01)

    # ! normalize the force/torque values with the user weight
    wrench_keys = [key for key in df_input_action01.keys() if 'shoe' in key.lower()]
    df_input_weight_normalized01 = df_input_action01
    
    for key in wrench_keys:
        df_input_weight_normalized01[key] = df_input_action01[key] / (cfg.user_mass * cfg.gravity)
   
    n01 = len(df_input_weight_normalized01)

    train_input_df01 = df_input_weight_normalized01[0:int(n01 * cfg.train_percentage)].copy()
    print("[INFO] show train input: ", train_input_df01)
    val_input_df01 = df_input_weight_normalized01[int(n01 * cfg.train_percentage):int(n01 * (cfg.train_percentage + cfg.val_percentage))].copy()
    print("[INFO] show validation input: ", val_input_df01)
    test_input_df01 = df_input_weight_normalized01[int(n01 * (cfg.train_percentage + cfg.val_percentage)):].copy()
    print("[INFO] show test input: ", test_input_df01)

    train_target_df01 = df_output01[0:int(n01 * cfg.train_percentage)]
    val_target_df01 = df_output01[int(n01 * cfg.train_percentage):int(n01 * (cfg.train_percentage + cfg.val_percentage))]
    test_target_df01 = df_output01[int(n01 * (cfg.train_percentage + cfg.val_percentage)):]

    # ! normalize all the data based on the training data
    if cfg.normalize_input:
        train_input_mean01 = train_input_df01.mean()
        train_input_std01 = train_input_df01.std()
    
        #savetxt('mean_data.csv', train_input_mean, delimiter=' ')
        #savetxt('std_data.csv', train_input_std, delimiter=' ')
        # np.any(np.all(train_input_df01))
        print('NaN value in mean - :', train_input_df01.isnull().values.any())
        print('NaN value in mean - :', train_input_mean01.isnull().values.any())
        print('NaN value in std - :', train_input_std01.isnull().values.any())

        train_input_df01 = (train_input_df01 - train_input_mean01) / train_input_std01
        val_input_df01 = (val_input_df01 - train_input_mean01) / train_input_std01
        test_input_df01 = (test_input_df01 - train_input_mean01) / train_input_std01
        # train_input_df01 = train_input_df01.apply(lambda x: (x-x.mean())/ x.std(), axis=0)
        # val_input_df01 = val_input_df01.apply(lambda x: (x-x.mean())/ x.std(), axis=0)
        # test_input_df01 = test_input_df01.apply(lambda x: (x-x.mean())/ x.std(), axis=0)
        print('NaN value in train - :', train_input_df01['jLeftBallFoot_roty_val'].isnull().values.any())
        print(train_input_df01['jLeftBallFoot_roty_val'].isnull().sum(axis = 0))
        print('NaN value in val - :', val_input_df01['jLeftBallFoot_roty_val'].isnull().values.any())
        print(val_input_df01['jLeftBallFoot_roty_val'].isnull().sum(axis = 0))
        print('NaN value in test - :', test_input_df01['jLeftBallFoot_roty_val'].isnull().values.any())
        print(test_input_df01['jLeftBallFoot_roty_val'].isnull().sum(axis = 0))
    
    # merge the inputs and targets
    train_input_df = train_input_df01
    val_input_df = val_input_df01
    test_input_df = test_input_df01
  
    train_target_df = train_target_df01
    val_target_df = val_target_df01
    test_target_df = test_target_df01

    # ! concatenate the two datasets to have one df
    gate_train_df = pd.concat([train_input_df, train_target_df].copy(), axis=1)
    gate_val_df = pd.concat([val_input_df, val_target_df].copy(), axis=1)
    gate_test_df = pd.concat([test_input_df, test_target_df].copy(), axis=1)
    
    expert_train_df = train_input_df.copy()
    expert_val_df = val_input_df.copy()
    expert_test_df = test_input_df.copy()

    multi_window = dp.WindowGenerator(input_width=cfg.input_width,
                                   label_width=cfg.output_steps,
                                   shift=cfg.shift,
                                   train_df=gate_train_df,
                                   val_df=gate_val_df,
                                   test_df=gate_test_df,
                                   output_labels=output_labels)

    multi_window.train

    input_data_example, __ = multi_window.example
    input_shape = (input_data_example.shape[1], input_data_example.shape[2])
    multi_val_performance = {}
    multi_test_performance = {}

    # ========================
    # === Learn GMoE model ===
    # ========================
    if cfg.learn_moe_model:
        # create an initial moe model
        model_moe = GMoE.get_moe_nn_model(number_categories=number_categories,
                                     number_experts_outputs=cfg.number_experts_outputs,
                                     output_steps=cfg.output_steps,
                                     input_shape=input_shape,
                                     reg_l1_gate=cfg.regularization_l1_gate,
                                     reg_l2_gate=cfg.regularization_l2_gate,
                                     reg_l1_experts=cfg.regularization_l1_experts,
                                     reg_l2_experts=cfg.regularization_l2_experts,
                                     dp_rate=cfg.dropout_rate)
        #model_moe.layers[-1]._name = 'moe_output'
        model_moe.summary()

        model_moe = GMoE.compile_model(model_moe)

        history_moe = GMoE.fit_model(model=model_moe,
                                window=multi_window,
                                patience=cfg.patience,
                                max_epochs=cfg.max_epochs,
                                model_path=cfg.models_path,
                                model_name=cfg.model_name + '_MoE_Best') 
        
        multi_val_performance['MoE'] = model_moe.evaluate(multi_window.val)
        multi_test_performance['MoE'] = model_moe.evaluate(multi_window.test, verbose=0)   
    elif cfg.relearn_moe_model:
        # load pre-trained model
        load_model_path = cfg.load_model_path
        load_model_name = cfg.load_model_name
        #file_name = 'model_MoE_Best'
        #model_moe = keras.models.load_model(file_path)
        model_moe = load_model_from_file(file_path=load_model_path, file_name=load_model_name)
        #model_moe.layers[-1]._name = 'moe_output'
        model_moe.summary()
        model_moe = GMoE.compile_model(model_moe)
        history_moe = GMoE.fit_model(model=model_moe,
                                window=multi_window,
                                patience=cfg.patience,
                                max_epochs=cfg.max_epochs,
                                model_path=cfg.models_path,
                                model_name=cfg.model_name + '_MoE_Best_v1') 
        multi_val_performance['MoE'] = model_moe.evaluate(multi_window.val)
        multi_test_performance['MoE'] = model_moe.evaluate(multi_window.test, verbose=0)
    # ============================
    # === Performance analysis ===
    # ============================
    if cfg.do_performance_analysis:
        x = np.arange(len(multi_test_performance))
        width = 0.3

        metrics_list = ['gate_output_accuracy', 'moe_output_mae']
        metrics_names = []
        for metrics_name in metrics_list:
            if cfg.learn_moe_model:
                metric_index = model_moe.metrics_names.index(metrics_name)
                metrics_names = model_moe.metrics_names

            val_mae = [v[metric_index] for v in multi_val_performance.values()]
            test_mae = [v[metric_index] for v in multi_test_performance.values()]
            plt.figure(figsize=(12, 8))
            plt.bar(x - 0.17, val_mae, width, label='Validation')
            plt.bar(x + 0.17, test_mae, width, label='Test')
            plt.xticks(ticks=x, labels=multi_test_performance.keys(),
                       rotation=45)
            plt.ylabel('{}'.format(metrics_name))
            _ = plt.legend()

        print('==============================')
        print('======== Test Metrics ========')
        print('==============================')
        for name, value in multi_test_performance.items():
            print('=========== {} ==========='.format(name))
            for i in range(len(metrics_names)):
                print(f'{metrics_names[i]:20s}: {value[i]:0.4f}')
        print('==============================')

    # ==================
    # === Save model ===
    # ==================
    if cfg.save_model:
        if cfg.learn_moe_model:
            GMoE.save_nn_model(model_moe, file_path=cfg.models_path, file_name=cfg.model_name + '_MoE')
            GMoE.visualize_model(model_moe, file_path=cfg.models_path, file_name=cfg.model_name + '_MoE')
    
