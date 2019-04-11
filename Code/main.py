from database_provider import database
from parameter import parameters
import numpy as np
from NeuralNetworkSetup import neural_network

#
# Important Run GetWAVToPythonFile before main one time for initialisation
#


# If already run code once, database is loaded(1)/saved in file(edit in parameter.py), the program will just skip creating the database and load the files
database_loaded = 0


cnn_list = parameters('cnn')
path_features = cnn_list.iloc[0]['extracted_features']

accuracy = 0

if database_loaded == 0:
    eval_data, eval_labels, train_data, train_labels = database('train', database_loaded, 'gender')

    np.save(path_features.format('eval_data.npy'), eval_data)
    np.save(path_features.format('test_label.npy'), eval_labels)

    np.save(path_features.format('train_data.npy'), train_data)
    np.save(path_features.format('train_label.npy'), train_labels)

    test_data, test_labels = database('test', database_loaded, 'gender')
    np.save(path_features.format('test_data.npy'), test_data)
    np.save(path_features.format('test_label.npy'), test_labels)

    retest_data, retest_labels, retrain_data, retrain_labels = database('test', database_loaded, 'speaker')

    np.save(path_features.format('retest_data.npy'), retest_data)
    np.save(path_features.format('retest_labels.npy'), retest_labels)
    np.save(path_features.format('retrain_data.npy'), retrain_data)
    np.save(path_features.format('retrain_labels.npy'), retrain_labels)
   
else:

    eval_data = np.load(path_features.format('eval_data.npy'))
    eval_labels = np.load(path_features.format('eval_labels.npy'))

    train_data = np.load(path_features.format('train_data.npy'))
    train_labels = np.load(path_features.format('train_labels.npy'))

    retest_data = np.load(path_features.format('retest_data.npy'))
    retest_labels = np.load(path_features.format('retest_labels.npy'))

    retrain_data = np.load(path_features.format('retrain_data.npy'))
    retrain_labels = np.load(path_features.format('retrain_labels.npy'))

# Loss either ='Sequential' or 'Triple'
scores, accuracy = neural_network(x_eval=eval_data, y_eval=eval_labels, x_train=train_data, y_train=train_labels, loss='Sequential',
                        x_test=test_data, y_test=test_labels, y_retest=retest_labels, y_retrain=retrain_labels ,x_retrain=retrain_data, x_retest=retest_data, utterance=False)

