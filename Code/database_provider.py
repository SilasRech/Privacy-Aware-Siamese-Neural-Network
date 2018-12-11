"""
Create Database for Neural Network using Pandas
"""
import numpy as np
import pandas as pd
from parameter import parameters
from feature_extraction import feature_extraction
db_df = parameters('database')
features = db_df.iloc[0]['features']

number_speakers = db_df.iloc[0]['number_speakers']

def database(mode, saved):
    """
    Returns data and label for a given mode, from the timit database
    :param mode: define test or training mode
    :param saved: 0 for no saved file, 1 for an already existing file
    :return: for mode = training returns data matrix and label vector for training and evaluation,
             for mode = test return, data matrix and label vector for testing
    """
    test_f = []
    test_m = []

    # Paths for home and university and number features

    female_speaker_list = pd.DataFrame()
    male_speaker_list = pd.DataFrame()
    test_data = pd.DataFrame()
    training1 = pd.DataFrame()
    eval_con = pd.DataFrame()
    retest_con = pd.DataFrame()
    retrain_con = pd.DataFrame()
    test = pd.DataFrame()

    meta = pd.read_json(db_df.iloc[0]['meta'], orient='split')

    female = meta.loc[meta['gender'] == 'f']
    male = meta.loc[meta['gender'] == 'm']
    male_indeces = male.index.values
    female_indeces = female.index.values

    # find 10m/10f longest samples for each speaker
    for i in range(int(len(female.index) / 10)):
        indeces_f = female_indeces[i * 10:i * 10 + 10]
        length_speaker_f = sum(meta[indeces_f[0]:indeces_f[-1]]['sample_length'])
        female_speaker = pd.DataFrame({'sample_length': [length_speaker_f], 'index_list': [female_indeces[10*i]]})
        female_speaker_list = pd.concat([female_speaker_list, female_speaker], sort=False)

    for k in range(int(len(male.index) / 10)):
        indeces_m = male_indeces[k * 10:k * 10 + 10]
        length_speaker_m = sum(meta[indeces_m[0]:indeces_m[-1]]['sample_length'])
        male_speaker = pd.DataFrame({'sample_length': [length_speaker_m], 'index_list': [male_indeces[10*k]]})
        male_speaker_list = pd.concat([male_speaker_list, male_speaker], sort=False)

    female_list = list(female_speaker_list.nlargest(20, 'sample_length').index_list)
    male_list = list(male_speaker_list.nlargest(20, 'sample_length').index_list)

    for i in range(10):
        test_f.append([meta.iloc[female_list[i]]])
        test_m.append([meta.iloc[male_list[i]]])

    # Compute remaining speaker for test list
    test_list = list(meta.index[0:4200:10])    # Just get the indeces and divide by the number of samples(10)
    error_list = []

    for i in range(10):
        error_list.append(int(female_list[i] / 10))
        error_list.append(int(male_list[i] / 10))

    for i in sorted(error_list, reverse=True):
        del test_list[i]

    ############################################################
    ############   TRAINING MODE ###############################
    ############################################################

    if mode == 'train':
        print('setting up training/evaluation data')
        if saved == 0:
            for m in range(10):
                # Extract features from each of the 10 samples for one speaker, male and female
                filename_sig = db_df.iloc[0]['audiofile']
                # Split into training and evaluation data
                male_data_con = extract_feature(m, meta, male_list[m], filename_sig)
                female_data_con = extract_feature(int(round(number_speakers/2))+m, meta, female_list[m], filename_sig)

                evalf, training_f = np.split(female_data_con, [int(.3 * len(female_data_con))])
                evalm, training_m = np.split(male_data_con, [int(.3 * len(male_data_con))])

                eval1 = pd.concat([evalf, evalm], sort=False)
                eval_con = pd.concat([eval_con, eval1])
                training = pd.concat([training_f, training_m])
                training1 = pd.concat([training1, training], sort=False)

                print(m)

            eval_con.reset_index()
            training1.reset_index()

            with open(db_df.iloc[0]['training'], 'w') as fs:
                df_json = training1.to_json(orient='split')
                fs.write(df_json)
            with open(db_df.iloc[0]['eval'], 'w') as f:
                df_json = eval_con.to_json(orient='split')
                f.write(df_json)
        else:
            training1 = pd.read_json(db_df.iloc[0]['training'], orient='split')
            eval_con = pd.read_json(db_df.iloc[0]['eval'], orient='split')

        if db_df.neural_network.item() == 'gender':

            # create labels for gender mode
            train_label = training1.loc[:, ['labeled', 'gender']]
            train_label.index = range(len(train_label.index))

            train_labels = gender_labels(train_label, features)

            eval_label = eval_con.loc[:, ['labeled', 'gender']]
            eval_label.index = range(len(eval_label.index))

            eval_labels = gender_labels(eval_label, features)

        elif db_df.neural_network.item() == 'speaker':

            # Create labels for speaker mode
            train_label = training1.loc[:, 'labeled']
            train_label.index = range(len(train_label.index))
            train_labels = speaker_labels(train_label, features)

            eval_label = eval_con.loc[:, 'labeled']
            eval_label.index = range(len(eval_label.index))
            eval_labels = speaker_labels(eval_label, features)

        else:
            raise ValueError('Neural Network Label not specified')

        # reshape the data to a 4-dimensional matrix for the network
        train_data = reshape_data(training1, features, train_label)
        eval_data = reshape_data(eval_con, features, eval_label)


        print('training data/evaluation data finished')
        return eval_data, eval_labels, train_data, train_labels

    ##########################################################
    #####   TESTING MODE    ##################################
    ##########################################################

    elif mode == 'test':

        if saved == 0:
            filename_sig = db_df.iloc[0]['audiofile']
            if db_df.neural_network.item() == 'gender':
                for m in range(400):
                    testing = extract_feature(m, meta, test_list[m], filename_sig)
                    print(m)
                    test = pd.concat([test, testing], sort=False)

                with open(db_df.iloc[0]['testing'], 'w') as fs:
                    df_json = test.to_json(orient='split')
                    fs.write(df_json)

            elif db_df.neural_network.item() == 'speaker':
                test_list = list(meta.index[0:4200:10])
                for m in range(420):
                    print(m)
                    testing = extract_feature(m, meta, test_list[m], filename_sig)
                    # Split into re-training and re-testing data
                    retest, retrain = np.split(testing, [int(.3*len(testing))])

                    retrain_con = pd.concat([retrain_con, retrain])
                    retest_con = pd.concat([retest_con, retest])

                with open(db_df.iloc[0]['retesting'], 'w') as fs:
                    df_json = retest_con.to_json(orient='split')
                    fs.write(df_json)

                with open(db_df.iloc[0]['retraining'], 'w') as fs:
                    df_json =retrain_con.to_json(orient='split')
                    fs.write(df_json)
            else:
                raise ValueError('Neural Network Label not specified')

        if db_df.neural_network.item() == 'gender':
            testing = pd.read_json(db_df.iloc[0]['testing'], orient='split')

        elif db_df.neural_network.item() == 'speaker':
            retesting = pd.read_json(db_df.iloc[0]['retesting'], orient='split')
            retraining = pd.read_json(db_df.iloc[0]['retraining'], orient='split')

        else:
            print("Network Label not specified")

        print('setting up testing data')

        if db_df.neural_network.item() == 'gender':

            test_label = testing.loc[:, ['labeled', 'gender']]

            test_label.index = range(len(test_label.index))

            test_labels = gender_labels(test_label, features)

            test_data = reshape_data(testing, features, test_label)

        elif db_df.neural_network.item() == 'speaker':

            test_data_list = []
            train_data_list = []
            test_labels_list = []
            train_labels_list = []
            names_list_test = []
            old_index = 0

            test_label = retesting.loc[:, 'labeled']
            train_label = retraining.loc[:, 'labeled']

            test_label.index = range(len(test_label.index))
            train_label.index = range(len(train_label.index))

            for i in range(len(test_label) - len(test_label) % features):
                if i == 0:
                    num_speaker_test = 1
                elif test_label[i] != test_label[i-1]:
                    num_speaker_test += 1
                names_list_test.append(num_speaker_test)

            names_list_test = np.asarray(names_list_test)

            for k in range(1, int(round(420/number_speakers))+1):
                index_ = [i for i, x in enumerate(names_list_test) if x == k*number_speakers]
                last_index = index_[-1]
                data_one = retesting.iloc[old_index:last_index][:]
                label_one = test_label.iloc[old_index:last_index][:]

                test_data = reshape_data(data_one, features, label_one)
                test_labels = speaker_labels(label_one, features)

                test_labels_list.append(test_labels)
                test_data_list.append(test_data)
                old_index = index_[-1]+1

            names_list_train = []
            for i in range(len(train_label) - len(train_label) % features):
                if i == 0:
                    num_speaker_train = 1
                elif train_label[i] != train_label[i-1]:
                    num_speaker_train += 1
                names_list_train.append(num_speaker_train)

            names_list_train = np.asarray(names_list_train)
            old_index = 0
            for k in range(1, int(round(420/number_speakers))+1):
                index_ = [i for i, x in enumerate(names_list_train) if x == k*number_speakers]
                last_index = index_[-1]
                data_one = retraining.iloc[old_index:last_index][:]
                label_one = train_label.iloc[old_index:last_index][:]

                train_data = reshape_data(data_one, features, label_one)
                train_labels = speaker_labels(label_one, features)

                train_labels_list.append(train_labels)
                train_data_list.append(train_data)
                old_index = index_[-1]+1

        print('testing data finished')

        if db_df.neural_network.item() == 'speaker':
            return test_data_list, test_labels_list, train_data_list, train_labels_list
        else:
            return test_data, test_labels


def speaker_labels(label, features):
    """
    Transforms names list in to a number based 1-D vector
    :param label: list of names
    :param features: num of features to extract
    :return: reshaped and value based 1-D vector
    """

    label_list = []
    label.index = range(len(label.index))
    label = label % number_speakers

    for m in range(max(label)+1):
        index_ = [i for i, x in enumerate(label) if x == m]
        last_index = index_[-1] - index_[-1] % features
        first_index = index_[0]
        num_rows = last_index - first_index
        num_rows = num_rows - num_rows % features
        number = label[first_index+5]

        one_label = np.repeat(int(number), int(num_rows / features))
        label_list = np.concatenate((label_list, one_label))

    return label_list


def reshape_data(data, features, label):
    """
    reshapes the feature matric into a 3-D or 4-D vector
    :param data:
    :param label: label_list to divide into set of 20 speakers
    :param features:
    :return: reshaped 4-D matrix
    """

    label.index = range(len(label.index))
    data.index = range(len(data.index))
    if db_df.neural_network.item() == 'gender':
        names_list = label.loc[:, 'labeled']
    else:
        names_list = label
        names_list = names_list % number_speakers

    for m in range(max(names_list)+1):
        index_ = [i for i, x in enumerate(names_list) if x == m]
        last_index = index_[-1] - index_[-1] % features
        first_index = index_[0]
        num_rows = last_index - first_index
        num_rows = num_rows - num_rows % features

        train_data = np.asarray(data.iloc[first_index:first_index+num_rows, 1: features+1])

        data_3d = np.empty((int(num_rows / features), features, features))

        for i in range(int(num_rows / features)):
            data_3d[i] = train_data[i * features: i * features + features, :]

        if m == 0:
            data_con = data_3d
        else:
            data_con = np.concatenate((data_con, data_3d))

    data_imaged = np.reshape(data_con, (-1, features, features, 1))
    return data_imaged


def gender_labels(label, features):
    """
    Labels genders in 1-D array value based
    :param labels: gender/id labeled 2-D array
    :param features:
    :return:
    """
    label_list = []

    label.index = range(len(label.index))
    label_names = label.loc[:, 'labeled']
    label_gender = label.loc[:, 'gender']


    for m in range(max(label_names)+1):
        index_ = [i for i, x in enumerate(label_names) if x == m]
        last_index = index_[-1] - index_[-1] % features
        first_index = index_[0]
        num_rows = last_index - first_index
        num_rows = num_rows - num_rows % features

        if label_gender[first_index+5] == 'f':
            number = 1
        else:
            number = 0

        one_label = np.repeat(int(number), int(num_rows / features))
        label_list = np.concatenate((label_list, one_label))

    return np.asarray(label_list)


def extract_feature(m, names_list, speaker_list, pathfile):
    data = pd.DataFrame()
    meta = pd.DataFrame()
    for n in range(10):
        file_num = speaker_list+n
        filename_sig = pathfile.format(file_num)
        audiofile = pd.read_json(filename_sig, orient='split')
        audiofile = audiofile[audiofile.columns[3]].values

        test_feature, num_rows = feature_extraction(audiofile=audiofile)
        data = pd.concat([data, test_feature])
        meta_one = pd.DataFrame({'name': names_list.iloc[file_num]['name'], 'gender': names_list.iloc[file_num]['gender'], 'sample': names_list.iloc[file_num]['sample'], 'labeled': m, 'utterance': m*n}, index=[file_num])
        meta_one = meta_one.loc[meta_one.index.repeat(num_rows)]
        meta = pd.concat([meta, meta_one])

    return pd.concat([data.reset_index(), meta.reset_index()], axis=1, sort=False)
