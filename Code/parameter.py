def parameters(mode):

    import pandas as pd

    # Feature Extraction
    subframe_shift_list = [128, 256, 512]  # 2
    subframe_length_list = [256, 512, 1024]  # 2
    number_features_list = [16, 32, 40]  # 3
    feature_mode_list = ['MFCC', 'MEL']  # 2

    # CNN
    dropout_rate_list = [0, 0.3, 0.7, 0.5]  # 3
    ks1_list = [3, 5]  # 2
    ks2_list = [2, 3]  # 2
    num_fil_1_layer_list = [32, 64]  # 2
    num_fil_2_layer_list = [64, 128]  # 2
    dense_layer_list = [64, 1024, 2048]  # 3
    batch_size_list = [1, 50, 150, 300, 500]  # 3
    epochs_list = [1, 10, 15, 20, 30, 50, 100]  # 5

    neural_network_list = ['gender', 'speaker']  # 2

    #####################################################
    ############    PLACE     ###########################
    #####################################################
    place = 'uni'

    # Choose Gender or Speaker // 0 = Gender // 1 = Speaker

    neural_network = neural_network_list[0]
    number_speakers = 20


    feature_mode = feature_mode_list[0]
    subframe_shift = subframe_shift_list[1]
    subframe_length = subframe_length_list[1]
    number_features = number_features_list[1]



    if neural_network == 'gender':
        dropout_rate = dropout_rate_list[2]
        ks1 = ks1_list[0]
        ks2 = ks2_list[1]
        num_fil_1_layer = num_fil_1_layer_list[1]
        num_fil_2_layer = num_fil_2_layer_list[1]
        dense_layer = dense_layer_list[1]
        batch_size = batch_size_list[3]
        epochs = epochs_list[3]
        number_bibs = 0

    else:
        dropout_rate = dropout_rate_list[3]
        ks1 = ks1_list[1]
        ks2 = ks1_list[1]
        num_fil_1_layer = num_fil_1_layer_list[1]
        num_fil_2_layer = num_fil_2_layer_list[1]
        dense_layer = dense_layer_list[1]
        batch_size = batch_size_list[2]
        epochs = epochs_list[6]

        number_bibs = 0

    if place == 'home':
        filename_audiofile = "C:\\Users\\srech\\PycharmProjects\\cnn_mnist.py\\Databases\\audio_files\\audio_file{0}.json"
        filename_meta = "C:\\Users\\srech\\PycharmProjects\\cnn_mnist.py\\databases\\meta.json"
        path_config = "C:\\Users\\srech\\PycharmProjects\\cnn_mnist.py\\Databases\\configuration.json"
        filename_observer = "C:\\Users\\srech\\desktop\\log\\Databases\\observer.json"
        path_tensorboard = 'C:\\Users\\srech\\desktop\\log\\Tensorboard\\Graph{0}'
        filename_testing = "C:\\Users\\srech\\desktop\\log\\Databases\\testingMFCC_64_512_256_7.json"
        filename_eval = "C:\\Users\\srech\\desktop\\log\\Databases\\evaluationMFCC_64_512_256_7.json"
        filename_training = "C:\\Users\\srech\\desktop\\log\\Databases\\trainingMFCC_64_512_256_7.json"
        filename_save = "C:\\Users\\srech\\Desktop\\log\\Databases\\models\\model.h5"
        filename_retesting = "C:\\Users\\srech\\desktop\\log\\Databases\\retesting_32_512_256_7.json"
        filename_retraining = "C:\\Users\\srech\\Desktop\\log\\Databases\\retraining_32_512_256_7.json"
        path_model = 'C:\\Users\\srech\\desktop\\log\\Model\\model_siamese.h5'
    else:
        filename_audiofile = "C:\\Users\\jonny\\Desktop\\log\\Databases\\audio_files\\audio_file{0}.json"
        filename_meta = "C:\\Users\\jonny\\Desktop\\log\\databases\\meta.json"
        path_tensorboard = 'C:\\Users\\jonny\\desktop\\log\\Tensorboard\\Graph{0}'
        path_config = "C:\\Users\\jonny\\PycharmProjects\\feature_extraction\\Databases\\configuration.json"
        filename_observer = "C:\\Users\\Jonny\\Desktop\\log\\observer.json"
        filename_training = "C:\\Users\\Jonny\\Desktop\\log\\Databases\\database_training_MFCC_32_64_512_256.json"
        filename_eval = "C:\\Users\\Jonny\\Desktop\\log\\Databases\\eval_training_MFCC_32_64_512_256.json"
        filename_testing = "C:\\Users\\Jonny\\Desktop\\log\\Databases\\testing_MFCC_32_64_512_256.json"
        filename_retesting = "C:\\Users\\Jonny\\desktop\\log\\Databases\\retesting_MFCC_32_64_512_256_7.json"
        filename_retraining = "C:\\Users\\Jonny\\Desktop\\log\\Databases\\retraining_MFCC_32_64_512_256_7.json"
        filename_save = "C:\\Users\\Jonny\\Desktop\\log\\models\\model.h5"
        path_model = 'C:\\Users\\jonny\\desktop\\log\\Models\\model_siamese.h5'

    if mode == 'database':
        df_return = pd.DataFrame({'neural_network': [neural_network],
                              'audiofile': [filename_audiofile], 'training': [filename_training],
                              'eval': [filename_eval], 'meta': [filename_meta], 'tensor': [path_tensorboard],
                              'config': [path_config], 'observer': [filename_observer], 'features': [number_features],
                              'testing': [filename_testing], 'retraining': [filename_retraining], 'retesting':[filename_retesting],
                              'number_speakers': [number_speakers]})
    elif mode == 'cnn':
        df_return = pd.DataFrame({'index': [number_bibs], 'features': [number_features], 'dropout': [dropout_rate],
                            'kernel1': [ks1], 'kernel2': [ks2], 'filter1': [num_fil_1_layer],
                            'filter2': [num_fil_2_layer], 'dense': [dense_layer], 'batch_size': [batch_size], 'epochs': [epochs],
                            'save_path': [filename_save], 'classifier': [neural_network], 'tensorboard': [path_tensorboard],
                            'model': [path_model], 'number_speaker': [number_speakers]})
    elif mode == 'mfcc':
        df_return = pd.DataFrame({'shift': [subframe_shift], 'length': [subframe_length], 'mode': [feature_mode], 'features': [number_features]})
    else:
        raise ValueError('wrong mode')

    return df_return
