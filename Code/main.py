from cnn import neural_network
from database_provider import database
from parameter import parameters

import numpy as np
cnn_list = parameters('cnn')
accuracy = 0
done = 1
if cnn_list.iloc[0]['classifier'] == 'gender':
    eval_data, eval_labels, train_data, train_labels = database('train', 1, 'gender')


if cnn_list.iloc[0]['classifier'] == 'gender':
    #test_data, test_labels = database('test', 1)
    #np.save('C:\\Users\\jonny\\Desktop\\log\\InputData\\test_data_utterance.json.npy', test_data)
    #np.save('C:\\Users\\jonny\\Desktop\\log\\InputData\\test_label_utterance.json.npy', test_labels)
    retest_data, retest_labels, retrain_data, retrain_labels = database('test', 1, 'speaker')
    #retest_data = 0
    #retest_labels = 0
    #retrain_data = 0
    #retrain_labels = 0
    test_data = np.load('C:\\Users\\jonny\\Desktop\\log\\InputData\\test_data_utterance.json.npy')
    test_labels = np.load('C:\\Users\\jonny\\Desktop\\log\\InputData\\test_label_utterance.json.npy')
    #retest_data = np.load('C:\\Users\\Jonny\\Desktop\\log\\InputData\\retest_data_utterance.json.npy')
    #retest_labels = np.load('C:\\Users\\Jonny\\Desktop\\log\\InputData\\retest_labels_utterance.json.npy')
    #retrain_data = np.load('C:\\Users\\Jonny\\Desktop\\log\\InputData\\retrain_data_utterance.json.npy')
    #retrain_labels = np.load('C:\\Users\\Jonny\\Desktop\\log\\InputData\\retrain_labels_utterance.json.npy')
   
else:

    retest_data, retest_labels, retrain_data, retrain_labels = database('test', 1, 'speaker')
    np.save('C:\\Users\\Jonny\\Desktop\\log\\InputData\\retest_data_utterance.json.npy', arr=retest_data)
    np.save('C:\\Users\\Jonny\\Desktop\\log\\InputData\\retest_labels_utterance.json.npy', arr=retest_labels)
    np.save('C:\\Users\\Jonny\\Desktop\\log\\InputData\\retrain_data_utterance.json.npy', arr=retrain_data)
    np.save('C:\\Users\\Jonny\\Desktop\\log\\InputData\\retrain_labels_utterance.json.npy', arr=retrain_labels)
    test_data = 0
    test_labels = 0


scores, accuracy = neural_network(x_eval=eval_data, y_eval=eval_labels, x_train=train_data, y_train=train_labels, loss=1,
                        x_test=test_data, y_test=test_labels, y_retest=retest_labels, y_retrain=retrain_labels ,x_retrain=retrain_data, x_retest=retest_data, utterance=False)

