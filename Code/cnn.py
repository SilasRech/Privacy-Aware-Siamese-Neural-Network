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
path_model = cnn_df.iloc[0]['model']
path_tensorboard = cnn_df.iloc[0]['tensorboard']
kernel1 = (cnn_df.iloc[0]['kernel1'], cnn_df.iloc[0]['kernel1'])
kernel2 = (cnn_df.iloc[0]['kernel2'], cnn_df.iloc[0]['kernel2'])
accuracy_speaker = []
epochs = 10


def neural_network(x_eval, y_eval, x_train, y_train, loss, alpha, x_test=0, y_test=0, x_retest=0, y_retest=0, x_retrain=0, y_retrain=0):

    # Clear Model
    tf.keras.backend.clear_session()

    if loss == 1:

        # Basics
        input_shape = (32, 32, 1)
        learning_rate = 0.0002
        adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=True)

        # Transform into digits
        digits_indeces_train = [np.where(y_train == i)[0] for i in range(2)]
        digits_indeces_eval = [np.where(y_eval == i)[0] for i in range(2)]
        digits_indeces_test = [np.where(y_test == i)[0] for i in range(2)]


        # Preate Pairs
        tr_pairs, tr_y = create_pairs(x_train, digits_indeces_train)
        eval_pairs, eval_y = create_pairs(x_eval, digits_indeces_eval)
        te_pairs, te_y = create_pairs(x_test, digits_indeces_test)

        # Build Basenetwork (Convolutional Part)
        base_network = create_base_network(input_shape)

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        #Define two outputs for the network with the same weigths
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        # Build the last layer of the convolutional part
        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape)([processed_a, processed_b])

        model = Model([input_a, input_b], distance)
        model.summary()
        base_network.summary()

        # Training phase
        model.compile(loss=contrastive_loss, optimizer=adam, metrics=[accuracy])
        model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                  batch_size=200,
                  epochs=50,
                  validation_data=([eval_pairs[:, 0], eval_pairs[:, 1]], eval_y), callbacks=[tensor_board])
        test_predictions = model.predict([te_pairs[:, 0], te_pairs[:, 1]], verbose=1)
        accuracy_siamese = compute_accuracy(te_y, test_predictions)

        print('Test accuracy siamese network {0}'.format(accuracy_siamese))

        # Save model and define model output as input for DNN part
        model.save('C:\\Users\\Jonny\\Desktop\\log\\Models\\model_siamese.h5')
        model_new = Model(inputs=base_network.get_input_at(0), outputs=base_network.get_layer('max_pooling2d_5').output)

        # Predict the new inputs
        x_pred_train = model_new.predict(x_train)
        x_pred_eval = model_new.predict(x_eval)
        x_pred_test = model_new.predict(x_test)

        # Hot encode labels
        classes = 2
        y_train = keras.utils.to_categorical(y_train, classes)
        y_eval = keras.utils.to_categorical(y_eval, classes)
        y_test = keras.utils.to_categorical(y_test, classes)

        # Build and train the DNN for gender discrimination
        input_new = (1, 1, 256)
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
        acc_gender = acc[1]
        print('test accuracy gender discrimination {0} %'.format(acc_gender))

        # Build DNN for speaker identification
        classes = 20
        y_train = keras.utils.to_categorical(y_train, classes)
        y_eval = keras.utils.to_categorical(y_eval, classes)
        for k in range(20):
            y_retest[k] = keras.utils.to_categorical(y_retest[k] % classes, classes)
            y_retrain[k] = keras.utils.to_categorical(y_retrain[k] % classes, classes)

        input_speaker = (1, 1, 256)
        model_dense_speaker = Sequential()
        model_dense_speaker.add(Flatten(input_shape=input_speaker))
        model_dense_speaker.add(Dense(1024, activation='relu', input_shape=input_speaker))
        model_dense_speaker.add(Dropout(0.5))
        model_dense_speaker.add(Dense(20, activation='softmax'))

        x_input_train = model_new.predict(x_train)
        x_input_eval = model_new.predict(x_eval)

        adam = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=True)

        model_dense_speaker.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        # Train DNN for speaker identification
        model_dense_speaker.fit(x=x_input_train, y=y_train, shuffle=True, epochs=cnn_df.iloc[0]['epochs'],
                                batch_size=cnn_df.iloc[0]['batch_size'],
                                validation_data=(x_input_eval, y_eval), callbacks=[tensor_board, history, stopper])

        accuracy_1 = list(history.acc)
        accuracy_speaker.append(accuracy_1[-1])

        for k in range(20):
            x_input_train = model_new.predict(x_retrain[k])
            x_input_eval = model_new.predict(x_retest[k])

            reset_weights(model_dense_speaker)
            model_dense_speaker.fit(x=x_input_train, y=y_retrain[k], shuffle=True, epochs=100,
                                    batch_size=150,
                                    validation_data=(x_input_eval, y_retest[k]), callbacks=[tensor_board, history])

            accuracy_1 = list(history.acc)
            accuracy_speaker.append(accuracy_1[-1])

        acc_speaker = sum(accuracy_speaker) / 20

    else:
        input_shape = (32, 32, 1)
        learning_rate = 0.0002
        adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=True)

        digits_indeces_train = [np.where(y_train == i)[0] for i in range(2)]
        digits_indeces_eval = [np.where(y_eval == i)[0] for i in range(2)]
        digits_indeces_test = [np.where(y_test == i)[0] for i in range(2)]

        tr_pairs, tr_y = create_pairs_sim(x_train, digits_indeces_train)
        eval_pairs, eval_y = create_pairs_sim(x_eval, digits_indeces_eval)
        te_pairs, te_y = create_pairs_sim(x_test, digits_indeces_test)

        classes = 2
        y_train = keras.utils.to_categorical(y_train, classes)
        y_eval = keras.utils.to_categorical(y_eval, classes)
        y_test = keras.utils.to_categorical(y_test, classes)

        # network definition
        network = create_whole_network(input_shape)

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        processed_a = network(input_a)
        processed_b = network(input_b)

        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape)([processed_a, processed_b])
        network.summary()

        model_whole = Model(input=[input_a, input_b], output=[processed_a, processed_b, distance])
        # input_dense = keras.layers.concatenate([dense_processed_a, dense_processed_b])

        model_whole.summary()
        # train

        model_whole.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        model_whole.fit([tr_pairs[:, 0], tr_pairs[:, 1]], y_train,
                  batch_size=150,
                  epochs=50,
                  validation_data=([eval_pairs[:, 0], eval_pairs[:, 1]], y_eval))

        acc = model_whole.evaluate([te_pairs[:, 0], te_pairs[:, 1]], verbose=1)

        acc_gender = acc[1]
        print('test accuracy gender discrimination {0} %'.format(acc_gender))

        classes = 20
        # y_train = keras.utils.to_categorical(y_train, classes)
        # y_eval = keras.utils.to_categorical(y_eval, classes)
        for k in range(20):
            y_retest[k] = keras.utils.to_categorical(y_retest[k] % 20, classes)
            y_retrain[k] = keras.utils.to_categorical(y_retrain[k] % 20, classes)

        input_speaker = (1, 1, 256)
        model_dense_speaker = Sequential()
        model_dense_speaker.add(Flatten(input_shape=input_speaker))
        model_dense_speaker.add(Dense(1024, activation='relu', input_shape=input_speaker))
        model_dense_speaker.add(Dropout(0.5))
        model_dense_speaker.add(Dense(20, activation='softmax'))

        # x_input_train = model_new.predict(x_train)
        # x_input_train = reshape_re(x_input_train)
        # x_input_eval = model_new.predict(x_eval)
        # x_input_eval = reshape_re(x_input_eval)

        learning_rate = 0.0002
        adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=True)

        model_dense_speaker.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        # model_dense_speaker.fit(x=x_input_train, y=y_train, shuffle=True, epochs=cnn_df.iloc[0]['epochs'],
        #                        batch_size=cnn_df.iloc[0]['batch_size'],
        #                        validation_data=(x_input_eval, y_eval), callbacks=[tensor_board, history, stopper])

        # accuracy_1 = list(history.acc)
        # accuracy_speaker.append(accuracy_1[-1])

        for k in range(20):
            x_input_train = model_new.predict(x_retrain[k])
            # x_input_train = reshape_re(x_input_train)
            x_input_eval = model_new.predict(x_retest[k])
            # x_input_eval = reshape_re(x_input_eval)

            reset_weights(model_dense_speaker)
            model_dense_speaker.fit(x=x_input_train, y=y_retrain[k], shuffle=True, epochs=100,
                                    batch_size=150,
                                    validation_data=(x_input_eval, y_retest[k]))

            #accuracy_1 = list(history.acc)
            #accuracy_speaker.append(accuracy_1[-1])

        acc_speaker = sum(accuracy_speaker) / 20

    return acc_speaker, acc_gender


def reshape_speaker(data):
    reshaped = []
    for i in range(len(data)):
        data_one = np.reshape(data[i][:], (1, 8192))
        reshaped.append(data_one)

    reshaped = np.asarray(reshaped)

    return reshaped


def reshape_gender(data):
    reshaped = []
    for i in range(len(data)):
        data_one = np.reshape(data[i][:], (1, 8192))
        reshaped.append(data_one)

    reshaped = np.asarray(reshaped)

    return reshaped


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    euclidean = K.sqrt(K.maximum(sum_square, K.epsilon()))
    return euclidean

def euclidean_distance_divided(x,y):
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    euclidean = K.sqrt(K.maximum(sum_square, K.epsilon()))
    return euclidean


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    output_shape = (shape1[0], 1)
    return output_shape


def output_shape_triple_loss(shapes):
    #shape1, shape2, shape3, shape4 = shapes
    #output_shape = (shape1[0], 2)
    return shapes


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(2)]) - 1
    for d in range(2):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 2)
            dn = (d + inc) % 2
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

def create_pairs_sim(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(2)]) - 1
    for d in range(2):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            labels += [1]
    return np.array(pairs), np.array(labels)

def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)

    x = BatchNormalization()(input)
    x = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(32, kernel_size=(5, 5), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(32, kernel_size=(5, 5), activation='relu', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = BatchNormalization()(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(0.25)(x)

    x = BatchNormalization()(x)
    x = Conv2D(256, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(256, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(256, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(0.25)(x)

    x = BatchNormalization()(x)
    x = Conv2D(256, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(256, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(256, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)

    return Model(input, x)

def create_whole_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)

    # Conv part
    x = BatchNormalization()(input)
    x = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(32, kernel_size=(5, 5), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(32, kernel_size=(5, 5), activation='relu', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = BatchNormalization()(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(0.25)(x)

    x = BatchNormalization()(x)
    x = Conv2D(256, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(256, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(256, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(0.25)(x)

    x = BatchNormalization()(x)
    x = Conv2D(256, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(256, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = Conv2D(256, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)

    #Dense part
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)

    return Model(input, x)


def create_dense_network(input_shape):
    '''
    Base network to be shared (eq. to feature extraction).
    '''

    input = Input(shape=input_shape)

    x = Dense(1024, activation='relu')(input)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)

    return Model(input, x)

def compute_accuracy(y_true, y_pred):
    '''
    Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''
    Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def contrastive_loss(y_pred, y_true):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(float(0), margin - y_pred)))


def triple_loss_through(vects):
    #a, b, c, d = vects
    a, b = vects

    return [a, b]

def triple_loss(y_true, y_label):
    #a, b, c, d = y_true
    #alpha = 1
    #loss_a = keras.losses.categorical_crossentropy(y_label, a)
    #loss_b = keras.losses.categorical_crossentropy(y_label, b)
    #distance = euclidean_distance_divided(c, d)
    #contrastive = contrastive_loss(distance, y_label)

    #return K.square(contrastive) + alpha*(loss_a + loss_b)

    a, b = y_true
    alpha = 1
    loss_a = keras.losses.categorical_crossentropy(y_label, a)
    loss_b = keras.losses.categorical_crossentropy(y_label, b)

    return alpha*(loss_a + loss_b)


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

