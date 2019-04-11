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


    # Path to normalized audiofiles from GetWavToPython (EDIT) Destination
    filename_audiofile = "C:\\Users\\silas\\Desktop\\Neuer Ordner\\audio_files\\audio_file{0}.json"

    # Path to Extracted Features (EDIT) Destination 1 (MFCC/LMBE Features)
    # Paths For Database Provider
    path_feature_database = "C:\\Users\\silas\\Desktop\\Neuer Ordner\\FeaturesForDatabase\\{0}.json"

    # Paths For Neural Network (EDIT) Destination 2 (Final Features)
    path_features = 'C:\\Users\\silas\\Desktop\\Neuer Ordner\\InputData\\{0}.json'

    # Choose Gender or Speaker // 0 = Gender // 1 = Speaker

    neural_network = neural_network_list[0]
    number_speakers = 20

    feature_mode = feature_mode_list[0]
    subframe_shift = subframe_shift_list[1]
    subframe_length = subframe_length_list[1]
    number_features = number_features_list[1]

    dropout_rate = dropout_rate_list[2]
    ks1 = ks1_list[0]
    ks2 = ks2_list[1]
    num_fil_1_layer = num_fil_1_layer_list[1]
    num_fil_2_layer = num_fil_2_layer_list[1]
    dense_layer = dense_layer_list[1]
    batch_size = batch_size_list[3]
    epochs = epochs_list[3]
    number_bibs = 0

    if mode == 'database':
        df_return = pd.DataFrame({'neural_network': [neural_network],
                              'audiofile': [filename_audiofile], 'features': [number_features],
                              'number_speakers': [number_speakers], 'path_features': [path_feature_database]})
    elif mode == 'cnn':
        df_return = pd.DataFrame({'index': [number_bibs], 'features': [number_features], 'dropout': [dropout_rate],
                            'kernel1': [ks1], 'kernel2': [ks2], 'filter1': [num_fil_1_layer],
                            'filter2': [num_fil_2_layer], 'dense': [dense_layer], 'batch_size': [batch_size], 'epochs': [epochs],
                            'classifier': [neural_network], 'tensorboard': [path_tensorboard],
                            'number_speaker': [number_speakers], 'extracted_features': [path_features]})
    elif mode == 'mfcc':
        df_return = pd.DataFrame({'shift': [subframe_shift], 'length': [subframe_length], 'mode': [feature_mode], 'features': [number_features]})


    else:
        raise ValueError('wrong mode')

    return df_return
