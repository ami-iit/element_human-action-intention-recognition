#####################
##### Libraries #####
#####################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import guidedMoE as GMoE
import dataProcessing as dp
import dataConfig as cfg

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

    df_raw = pd.read_csv(cfg.data_path01, sep=' ')
    #print("[INFO] show raw input data: ", df_raw)
    #df_raw_action02 = pd.read_csv(data_path02, sep=' ')

    if input_feature_list:
        df_input_action01 = df_raw[input_feature_list].copy()
        #print("[INFO] show input data: ", df_input_action01)
        #df_input_action02 = df_raw_action02[input_feature_list].copy()
    else:
        df_input_action01 = df_raw.copy()
        #df_input_action02 = df_raw_action02.copy()
    
    if output_feature_list:  # ! define the output feature list
        df_output01 = df_raw[output_feature_list].copy()
        #df_output02 = df_raw_action02[output_feature_list].copy()
    else:
        df_output01 = df_raw.copy()
        #df_output02 = df_raw_action02.copy()
    
    if cfg.output_categorical:  # ! get the output label in case of categorical ouputs
        df_output01 = pd.get_dummies(df_output01)
        output_labels01 = df_output01.keys()

        #df_output02 = pd.get_dummies(df_output02)
        #output_labels02 = df_output02.keys()

        #output_labels = output_labels01.union(output_labels02)
        output_labels = output_labels01
        # number of categories should be equal to the size of annotation list
        number_categories = len(output_labels)  # ! the number of categories
        #print('output labels set 1: {}'.format(output_labels01))
        #print('output labels set 2: {}'.format(output_labels02))
        print('total output labels: {}'.format(output_labels))

    # ! start the time from the zero, depends on the application
    if 'time' in df_raw:
        #print("[INFO] show time: ", df_raw['time'])
        df_time = df_raw['time'] - df_raw['time'][0]
        #print("[INFO] show reduced time: ", df_time)
    # test if the object has the right type of data
    df_input_action01.head()

    if pop_list is not None:
        for pop_name in pop_list:
            if pop_name in df_input_action01:
                df_input_action01.pop(pop_name)
                print("[INFO] show now input data: ", df_input_action01)
            #if pop_name in df_input_action02:
            #    df_input_action02.pop(pop_name)

    # ! normalize the force/torque values with the user weight
    wrench_keys = [key for key in df_input_action01.keys() if 'shoe' in key.lower()]
    df_input_weight_normalized01 = df_input_action01
    #df_input_weight_normalized02 = df_input_action02
    for key in wrench_keys:
        df_input_weight_normalized01[key] = df_input_action01[key] / (cfg.user_mass * cfg.gravity)
        #df_input_weight_normalized02[key] = df_input_action02[key] / (user_mass * gravity)
    #print("[INFO] normalized data: ", df_input_weight_normalized01)
    n01 = len(df_input_weight_normalized01)
    #n02 = len(df_input_weight_normalized02)

    train_input_df01 = df_input_weight_normalized01[0:int(n01 * cfg.train_percentage)].copy()
    print("[INFO] show train input: ", train_input_df01)
    val_input_df01 = df_input_weight_normalized01[int(n01 * cfg.train_percentage):int(n01 * (cfg.train_percentage + cfg.val_percentage))].copy()
    print("[INFO] show validation input: ", val_input_df01)
    test_input_df01 = df_input_weight_normalized01[int(n01 * (cfg.train_percentage + cfg.val_percentage)):].copy()
    print("[INFO] show test input: ", test_input_df01)

    #train_input_df02 = df_input_weight_normalized02[0:int(n02 * train_percentage)].copy()
    #val_input_df02 = df_input_weight_normalized02[int(n02 * train_percentage):int(n02 * (train_percentage + val_percentage))].copy()
    #test_input_df02 = df_input_weight_normalized02[int(n02 * (train_percentage + val_percentage)):].copy()

    train_target_df01 = df_output01[0:int(n01 * cfg.train_percentage)]
    val_target_df01 = df_output01[int(n01 * cfg.train_percentage):int(n01 * (cfg.train_percentage + cfg.val_percentage))]
    test_target_df01 = df_output01[int(n01 * (cfg.train_percentage + cfg.val_percentage)):]

    #train_target_df02 = df_output02[0:int(n02 * train_percentage)]
    #val_target_df02 = df_output02[int(n02 * train_percentage):int(n02 * (train_percentage + val_percentage))]
    #test_target_df02 = df_output02[int(n02 * (train_percentage + val_percentage)):]

    # ! normalize all the data based on the training data
    if cfg.normalize_input:
        #scaler = RobustScaler() 
        #train_input_df = preprocessing.normalize(train_input_df)
        #val_input_df = preprocessing.normalize(val_input_df)
        #test_input_df = preprocessing.normalize(test_input_df)

        train_input_mean01 = train_input_df01.mean()
        train_input_std01 = train_input_df01.std()

        #train_input_mean01 = train_input_df01.mean()
        #train_input_std01 = train_input_df01.std()
        #savetxt('mean_data.csv', train_input_mean, delimiter=' ')
        #savetxt('std_data.csv', train_input_std, delimiter=' ')

        print('NaN value in mean - :', np.any(np.all(train_input_mean01)))
        print('NaN value in std - :', np.any(np.all(train_input_std01)))

        train_input_df01 = (train_input_df01 - train_input_mean01) / train_input_std01
        val_input_df01 = (val_input_df01 - train_input_mean01) / train_input_std01
        test_input_df01 = (test_input_df01 - train_input_mean01) / train_input_std01

        #train_input_df01 = (train_input_df01 - train_input_mean01) / train_input_std01
        #val_input_df01 = (val_input_df01 - train_input_mean01) / train_input_std01
        #test_input_df01 = (test_input_df01 - train_input_mean01) / train_input_std01
    
    # merge the inputs and targets
    #train_input_df = train_input_df00.append(train_input_df01, ignore_index=True)
    train_input_df = train_input_df01

    #val_input_df = val_input_df00.append(val_input_df01, ignore_index=True)
    val_input_df = val_input_df01

    #test_input_df = test_input_df00.append(test_input_df01, ignore_index=True)
    test_input_df = test_input_df01
  
    #train_target_df = train_target_df00.append(train_target_df01, ignore_index=True)
    #train_target_df = train_target_df.replace(np.nan, 0)
    train_target_df = train_target_df01

    #val_target_df = val_target_df00.append(val_target_df01, ignore_index=True)
    #val_target_df = val_target_df.replace(np.nan, 0)
    val_target_df = val_target_df01

    #test_target_df = test_target_df00.append(test_target_df01, ignore_index=True)
    #test_target_df = test_target_df.replace(np.nan, 0)
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
        # create a moe model
        model_moe = GMoE.get_moe_nn_model(number_categories=number_categories,
                                     number_experts_outputs=cfg.number_experts_outputs,
                                     output_steps=cfg.output_steps,
                                     input_shape=input_shape,
                                     reg_l1_gate=cfg.regularization_l1_gate,
                                     reg_l2_gate=cfg.regularization_l2_gate,
                                     reg_l1_experts=cfg.regularization_l1_experts,
                                     reg_l2_experts=cfg.regularization_l2_experts,
                                     dp_rate=cfg.dropout_rate)
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
    
