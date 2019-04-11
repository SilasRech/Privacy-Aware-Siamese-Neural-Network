import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda, Input
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.models import Model
from keras.callbacks import TensorBoard
from parameter import parameters
import tensorflow as tf
import numpy as np
import random

cnn_df = parameters('cnn')
kernel1 = (cnn_df.iloc[0]['kernel1'], cnn_df.iloc[0]['kernel1'])
kernel2 = (cnn_df.iloc[0]['kernel2'], cnn_df.iloc[0]['kernel2'])
number_speaker = cnn_df.iloc[0]['number_speaker']
accuracy_speaker = []
epochs = 10

# Callbacks
# Creates EarlyStoppingFunction for Training
stopper = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.03, patience=3, verbose=0, mode='auto',
                                        baseline=None)


def neural_network(x_eval, y_eval, x_train, y_train, loss, x_test, y_test, x_retest, y_retest, x_retrain, y_retrain, utterance=False):

    y_test_utt = y_test[:, 1]
    y_eval = y_eval[:, 0]
    y_test = y_test[:, 0]
    y_train = y_train[:, 0]

    average_siamese = []
    average_gender = []
    average_speaker = []

    # Clear Model
    tf.keras.backend.clear_session()
    history = AccuracyHistory()

    # Basics
    input_shape = (32, 32, 1)
    learning_rate = 0.0002
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=True)

    if loss == 'Sequential':

        # Transform into digits
        digits_indeces_train = [np.where(y_train == i)[0] for i in range(2)]
        digits_indeces_eval = [np.where(y_eval == i)[0] for i in range(2)]
        digits_indeces_test = [np.where(y_test == i)[0] for i in range(2)]

        # Create Pairs
        tr_pairs, tr_y, tr_y1, tr_y2 = create_pairs_ratio(x_train, digits_indeces_train, 0.5, 1)
        eval_pairs, eval_y, eval_y1, eval_y2 = create_pairs_ratio(x_eval, digits_indeces_eval, 0.5, 1)
        te_pairs, te_y, te_y1, te_y2 = create_pairs_ratio(x_test, digits_indeces_test, 0.5, 1)

        # Build Basenetwork (Convolutional Part)
        base_network = create_base_network(input_shape)

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        # Define two outputs for the network with the same weigths
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        # Build the last layer of the convolutional part
        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape)([processed_a, processed_b])

        model = Model([input_a, input_b], distance)
        model.summary()
        base_network.summary()

        # Training phase
        model.compile(loss=contrastive_loss, optimizer=adam, metrics=[accuracy_siamese])
        model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                  batch_size=150,
                  epochs=75,
                  validation_data=([eval_pairs[:, 0], eval_pairs[:, 1]], eval_y), shuffle=True)

        test_predictions = model.predict([te_pairs[:, 0], te_pairs[:, 1]], verbose=1)
        accuracy_siamese1 = compute_accuracy(te_y, test_predictions)
        average_siamese.append(accuracy_siamese1)

        print('Test accuracy siamese network {0}'.format(accuracy_siamese1))

        # Save model and define model output as input for DNN part
        model_new = Model(inputs=base_network.get_input_at(0), outputs=base_network.get_layer('max_pooling2d_5').output)

        # Predict the new inputs
        x_pred_train = model_new.predict(x_train)
        x_pred_eval = model_new.predict(x_eval)
        x_pred_test = model_new.predict(x_test)


        #display_data(x_pred_test, y_test)
        # Hot encode labels
        classes = 2
        y_train = keras.utils.to_categorical(y_train, classes)
        y_eval = keras.utils.to_categorical(y_eval, classes)
        y_test = keras.utils.to_categorical(y_test, classes)

        # Build and train the DNN for gender discrimination
        input_new = (1, 1,   512)
        model_dense = Sequential()
        model_dense.add(Flatten(input_shape=input_new))
        model_dense.add(Dense(1024, activation='relu'))
        model_dense.add(Dropout(0.5))
        model_dense.add(Dense(2, activation='softmax'))

        model_dense.summary()
        model_dense.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        model_dense.fit(x_pred_train, y_train,
                        batch_size=200,
                        epochs=30,
                        validation_data=(x_pred_eval, y_eval))

        acc = model_dense.evaluate(x=x_pred_test, y=y_test)

        x_pred_test_utt = model_dense.predict(x_pred_test)

        if utterance:
            acc_utterance_gender = utterance_accuracy(x_pred_test_utt, y_test, y_test_utt)
            acc_gender = acc_utterance_gender
        else:
            acc_gender = acc[1]
        print('test accuracy gender discrimination {0} %'.format(acc_gender))
        average_gender.append(acc_gender)
        # Build DNN for speaker identification
        classes = number_speaker

        input_speaker = (1, 1, 512)
        model_dense_speaker = Sequential()
        model_dense_speaker.add(Flatten(input_shape=input_speaker))
        model_dense_speaker.add(Dense(1024, activation='relu', input_shape=input_speaker))
        model_dense_speaker.add(Dropout(0.5))
        model_dense_speaker.add(Dense(number_speaker, activation='softmax'))

        adam = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=True)

        model_dense_speaker.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        # Train DNN for speaker identification
        accuracy_speaker = []
        for k in range(int(420/number_speaker)):
            print('------------Speaker Batch Iteration {0}-------------'.format(k+1))

            retest_label_one = y_retest[k]
            retrain_labels_one = y_retrain[k]

            if utterance:
                y_retest_utt = retest_label_one[:, 1]
                retest_label_one_true = retest_label_one[:, 0]
                retrain_label_one_true = retrain_labels_one[:, 0]
            else:
                y_retest_utt = retest_label_one[:, 1]
                retest_label_one_true = retest_label_one[:, 0]
                retrain_label_one_true = retrain_labels_one[:, 0]

            y_retest_one = keras.utils.to_categorical(retest_label_one_true % classes, classes)
            y_retrain_one = keras.utils.to_categorical(retrain_label_one_true % classes, classes)

            x_input_train = model_new.predict(x_retrain[k])
            x_input_eval = model_new.predict(x_retest[k])

            reset_weights(model_dense_speaker)
            model_dense_speaker.fit(x=x_input_train, y=y_retrain_one, shuffle=True, epochs=150,
                                    batch_size=90,
                                    validation_data=(x_input_eval, y_retest_one), callbacks=history)

            if utterance:
                pred_speaker = model_dense_speaker.predict(x_input_eval)
                utterance_acc = utterance_accuracy(pred_speaker, y_retest_one, y_retest_utt)
                accuracy_speaker.append(utterance_acc)

            else:
                accuracy_1 = list(history.acc)
                accuracy_speaker.append(accuracy_1[-1])

        acc_speaker = np.mean(accuracy_speaker)

        average_speaker.append(acc_speaker)

    else:
        # So row 0 are the males, and row 1 are the females
        digits_indeces_train = [np.where(y_train == i)[0] for i in range(2)]
        digits_indeces_eval = [np.where(y_eval == i)[0] for i in range(2)]
        digits_indeces_test = [np.where(y_test == i)[0] for i in range(2)]

        tr_pairs, tr_y, tr_y1, tr_y2 = create_pairs_ratio(x_train, digits_indeces_train, 1, 1)
        eval_pairs, eval_y, eval_y1, eval_y2 = create_pairs_ratio(x_eval, digits_indeces_eval, 1, 1)
        te_pairs, te_y, te_y1, te_y2 = create_pairs_ratio(x_test, digits_indeces_test, 1, 1)

        # Transform into hot-encoded vectors
        classes = 2
        tr_y1 = keras.utils.to_categorical(tr_y1, classes)
        eval_y1 = keras.utils.to_categorical(eval_y1, classes)
        te_y1 = keras.utils.to_categorical(te_y1, classes)

        tr_y2 = keras.utils.to_categorical(tr_y2, classes)
        eval_y2 = keras.utils.to_categorical(eval_y2, classes)
        te_y2 = keras.utils.to_categorical(te_y2, classes)

        y_train = keras.utils.to_categorical(y_train[:, 0], classes)
        y_eval = keras.utils.to_categorical(y_eval[:, 0], classes)
        y_test = keras.utils.to_categorical(y_test[:, 0], classes)

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        # network definition
        network = create_whole_network(input_shape)

        processed_a = network(input_a)
        processed_b = network(input_b)

        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape, name='distance')([processed_a, processed_b])

        network.summary()
        dense = create_dense_network((1, 1, 512))
        dense_a = dense(processed_a)
        dense_b = dense(processed_b)
        dense_a = Dense(2, activation='softmax')(dense_a)
        output_1 = Flatten(name='output_1')(dense_a)
        dense_b = Dense(2, activation='softmax')(dense_b)
        output_2 = Flatten(name='output_2')(dense_b)

        model_whole = Model(inputs=[input_a, input_b], outputs=[output_1, output_2, distance], name='CompositeModel')

        # train

        alpha = 0.1
        model_whole.compile(loss={'output_1': 'categorical_crossentropy', 'output_2': 'categorical_crossentropy', 'distance': contrastive_loss},
                            loss_weights={'output_1': alpha, 'output_2': alpha, 'distance': 700},
                            optimizer=adam,
                            metrics={'output_1': 'accuracy', 'output_2': 'accuracy', 'distance': accuracy_siamese()})
        model_whole.fit([tr_pairs[:, 0], tr_pairs[:, 1]], [tr_y1, tr_y2, tr_y], shuffle=True,
                  batch_size=150,
                  epochs=50,
                  validation_data=([eval_pairs[:, 0], eval_pairs[:, 1]], [eval_y1,  eval_y2,  eval_y]))

        acc_siamese_training = model_whole.evaluate([te_pairs[:, 0], te_pairs[:, 1]], [te_y1,  te_y2,  te_y], verbose=1)

        pred_test = model_whole.predict([te_pairs[:, 0], te_pairs[:, 1]])
        utterance_accuracy(pred_test[0][:], te_y1, y_test[:, 1])
        utterance_accuracy(pred_test[1][:], te_y2, y_test[:, 1])

        print('--- accuracies for composite loss functions {0} %---'.format(acc_siamese_training))

        model_new = Model(inputs=network.get_input_at(0), outputs=network.get_layer('max_pooling2d_5').output)

        x_pred_train = model_new.predict(x_train)
        x_pred_eval = model_new.predict(x_eval)
        x_pred_test = model_new.predict(x_test)

        # Build and train the DNN for gender discrimination
        input_new = (1, 1, 512)
        model_dense = Sequential()
        model_dense.add(Flatten(input_shape=input_new))
        model_dense.add(Dense(1024, activation='relu'))
        model_dense.add(Dropout(0.5))
        model_dense.add(Dense(2, activation='softmax'))

        model_dense.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        model_dense.fit(x_pred_train, y_train,
                        batch_size=200,
                        epochs=30,
                        validation_data=(x_pred_eval, y_eval))

        acc = model_dense.evaluate(x=x_pred_test, y=y_test)
        acc_gender = acc[1]
        print('test accuracy gender discrimination {0} %'.format(acc_gender))

        classes = number_speaker

        accuracy_speaker = []
        input_speaker = (1, 1, 512)
        model_dense_speaker = Sequential()
        model_dense_speaker.add(Flatten(input_shape=input_speaker))
        model_dense_speaker.add(Dense(1024, activation='relu', input_shape=input_speaker))
        model_dense_speaker.add(Dropout(0.5))
        model_dense_speaker.add(Dense(number_speaker, activation='softmax'))

        learning_rate = 0.0002
        adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=True)

        model_dense_speaker.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        for k in range(int(round(420/number_speaker))):
            print('---------Speaker Batch Iteration {0} of {1}------------'.format(k+1), int(round(420/number_speaker)))
            retest_label_one = y_retest[k]
            retrain_labels_one = y_retrain[k]

            y_retest_utt = retest_label_one[:, 1]
            retest_label_one_true = retest_label_one[:, 0]
            retrain_label_one_true = retrain_labels_one[:, 0]
            y_retest_one = keras.utils.to_categorical(retest_label_one_true % classes, classes)
            y_retrain_one = keras.utils.to_categorical(retrain_label_one_true % classes, classes)

            x_input_train = model_new.predict(x_retrain[k])
            x_input_eval = model_new.predict(x_retest[k])

            reset_weights(model_dense_speaker)
            model_dense_speaker.fit(x=x_input_train, y=y_retrain_one, shuffle=True, epochs=150,
                                    batch_size=90, validation_data=(x_input_eval, y_retest_one), callbacks=history)

            if not utterance:
                accuracy_1 = list(history.acc)
                accuracy_speaker.append(accuracy_1[-1])

            else:
                pred_speaker = model_dense_speaker.predict(x_input_eval)
                utterance_acc = utterance_accuracy(pred_speaker, y_retest_one, y_retest_utt)
                accuracy_speaker.append(utterance_acc)

            acc_speaker = np.mean(accuracy_speaker)
        return acc_speaker, acc_gender, accuracy_siamese


def utterance_accuracy(y_pred, y_true, y_utt):

    """
    Computes the utterance based accuracy of a predicted vector
    :param secs:    duration of utterance in seconds
    :param y_pred: predicted labels
    :param y_true: true labels
    :return: utterance based accuracy
    """
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    pred_list = []
    true_list = []

    for i in range(int(max(y_utt)+1)):
        index_ = [m for m, x in enumerate(y_utt) if x == i]
        if index_ == []:
            i += 1
            index_ = [m for m, x in enumerate(y_utt) if x == i]
        last_index = index_[-1]
        first_index = index_[0]

        # Class for predicted utterance
        print(i)
        utterance = y_pred[first_index:last_index+1]
        counted_pred = np.unique(utterance, return_counts=True)
        classes = counted_pred[0]
        counted = np.argmax(counted_pred[1])
        class_pred = classes[counted]
        pred_list.append(class_pred)
        pred_class = np.asarray(pred_list)

        # Class for true utterance
        utterance = y_true[first_index:last_index+1]
        counted_true = np.unique(utterance, return_counts=True)
        classes = counted_true[0]
        counted = np.argmax(counted_true[1])
        class_true = classes[counted]
        true_list.append(class_true)
        true_class = np.asarray(true_list)

    accuracy = np.mean(pred_class == true_class)

    return accuracy


def reshape_speaker(data):
    reshaped = []
    for i in range(len(data)):
        data_one = np.reshape(data[i][:], (1, 131072))
        reshaped.append(data_one)

    reshaped = np.asarray(reshaped)

    return reshaped


def reshape_gender(data):
    reshaped = []
    for i in range(len(data)):
        data_one = np.reshape(data[i][:], (1, 131072))
        reshaped.append(data_one)

    reshaped = np.asarray(reshaped)

    return reshaped


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def euclidean_distance_divided(x,y):
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    euclidean = K.sqrt(K.maximum(sum_square, K.epsilon()))
    return euclidean


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    output_shape = (shape1[0], 1)
    return output_shape


def create_pairs_ratio(x, digit_indices, ratio, times):
    """
    :param x: input data
    :param digits_indices: labeled data
    :param ratio: ratio for similar pairs in percentage
    :param times: total number of times the data is repeatedly chosen
    :return: pairs, new labels
    """

    if ratio > 1:
        raise ValueError('Value must be between 0 and 1')

    pairs_sim = []
    pairs_dif = []

    len_male = len(digit_indices[0])
    len_female = len(digit_indices[1])

    n = min([len(digit_indices[d]) for d in range(2)]) - 1

    for m in range(times):
        for i in range(len_male):
            # Similiar
            random_sim = random.randint(0, n)
            if random_sim == i:
                random_sim = random.randint(0, n)
            z1 = digit_indices[0][i]
            z2 = digit_indices[0][random_sim]
            pairs_sim += [[x[z1], x[z2], 1, 0, 0]]

            # Unsimiliar
            random_unsim = random.randint(0, n)
            if random_unsim == i:
                random_unsim = random.randint(0, n)
            z1 = digit_indices[0][i]
            z2 = digit_indices[1][random_unsim]
            pairs_dif += [[x[z1], x[z2], 0, 0, 1 ]]

        for i in range(len_female):
            # Similiar
            random_sim = random.randint(0, n)
            if random_sim == i:
                random_sim = random.randint(0, n)
            z1 = digit_indices[1][i]
            z2 = digit_indices[1][random_sim]
            pairs_sim += [[x[z1], x[z2], 1, 1, 1]]

            # Unsimiliar
            random_unsim = random.randint(0, n)
            if random_unsim == i:
                random_unsim = random.randint(0, n)
            z1 = digit_indices[1][i]
            z2 = digit_indices[0][random_unsim]
            pairs_dif += [[x[z1], x[z2], 0, 1, 0]]

    n = len(pairs_sim)
    pairs_sim = random.sample(pairs_sim, int(round(n*ratio)))
    pairs_dif = random.sample(pairs_dif, int(round(n*(1-ratio))))
    pairs = pairs_sim + pairs_dif

    new_labels_1 = []
    new_labels_2 = []
    labels = []
    pairs_return = []
    for i in range(len(pairs)):
        labels.append(pairs[i][2])
        pairs_return.append(pairs[i][0:2])
        new_labels_1.append(pairs[i][3])
        new_labels_2.append(pairs[i][4])

    return np.array(pairs_return), np.array(labels), np.array(new_labels_1), np.array(new_labels_2)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)

    x = BatchNormalization()(input)
    x = Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(64, kernel_size=(5, 5), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(64, kernel_size=(5, 5), activation='relu', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = BatchNormalization()(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    # x = Dropout(0.1)(x)

    x = BatchNormalization()(x)
    x = Conv2D(256, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(256, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(256, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    #x = Dropout(0.1)(x)

    x = BatchNormalization()(x)
    x = Conv2D(512, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(512, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(512, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    #x = Dropout(0.1)(x)

    x = BatchNormalization()(x)
    x = Conv2D(512, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(512, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(512, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)


    x = Flatten()(x)

    return Model(input, x)


def create_dense_network(input_shape):
    '''
    Base network to be shared (eq. to feature extraction).
    '''
    input_ = Input(shape=input_shape)

    x = Dense(1024, activation='relu')(input_)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)

    return Model(input_, x, name='Dense')


def create_whole_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)

    # Conv part
    x = BatchNormalization()(input)
    x = Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(64, kernel_size=(5, 5), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(64, kernel_size=(5, 5), activation='relu', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = BatchNormalization()(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = BatchNormalization()(x)
    x = Conv2D(256, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(256, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(256, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(0.2)(x)

    x = BatchNormalization()(x)
    x = Conv2D(512, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(512, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(512, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(0.2)(x)

    x = BatchNormalization()(x)
    x = Conv2D(512, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(512, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(512, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)

    return Model(input, x)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy_siamese(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def accuracy_triple(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def contrastive_loss(y_true, y_pred):
    margin = float(1)
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(float(0), margin - y_pred)))


def triple_loss(y_true, y_label):
    a, b, c = y_true
    alpha = 1
    loss_a = keras.losses.categorical_crossentropy(y_label, a)
    loss_b = keras.losses.categorical_crossentropy(y_label, b)
    contrastive = contrastive_loss(c, y_label)

    return K.square(contrastive) + alpha*(loss_a + loss_b)


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('val_acc'))


def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

