from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow import keras
from src.tf_models.eeg_net import EEGNetTF
from src.tf_models.tcn_net import EEGTCNet
from src.tf_models.final_net import FinalNet
from src.tf_models.eegsym import EEGSym
from sklearn.metrics import classification_report

import numpy as np


def cnn_method_eegnet_tf(x_train, x_test, y_train, y_test, lr=1e-3, epochs=250, weight_decay=1e-2, batch_size=64):
    x_train, x_test, y_train, y_test = x_train.copy(), x_test.copy(), y_train.copy(), y_test.copy()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    model = EEGNetTF(nb_classes = 4, Chans = x_train.shape[1], Samples = x_train.shape[2],
                     dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16,
                     dropoutType = 'Dropout')

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=lr, weight_decay=weight_decay), #
                  metrics = ['accuracy'])
    callbacks = [
        ReduceLROnPlateau(monitor="loss", factor=0.85, patience=35, verbose=1, min_lr=0.00005),
    ]
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),callbacks=callbacks)

    # model.save('saved_model/')

    probs = model.predict(x_test)
    preds = probs.argmax(axis = -1)
    acc = np.mean(preds == y_test)

    print(classification_report(y_test, preds))

    return acc


def cnn_method_eegtcn_tf(x_train, x_test, y_train, y_test, lr=1e-3, epochs=250, weight_decay=1e-2, batch_size=64):
    x_train, x_test, y_train, y_test = x_train.copy(), x_test.copy(), y_train.copy(), y_test.copy()
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])

    model = EEGTCNet(nb_classes = 4, Chans = x_train.shape[2], Samples = x_train.shape[3],
                     dropout = 0.25, kernLength = 32, F1 = 8, D = 2)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=lr, weight_decay=weight_decay), #
                  metrics = ['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

    # model.save('saved_model/')

    probs = model.predict(x_test)
    preds = probs.argmax(axis = -1)
    acc = np.mean(preds == y_test)

    print(classification_report(y_test, preds))

    return acc


def cnn_method_final_tf(x_train, x_test, y_train, y_test, lr=1e-3, epochs=250, weight_decay=1e-2, batch_size=64):
    x_train, x_test, y_train, y_test = x_train.copy(), x_test.copy(), y_train.copy(), y_test.copy()
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])

    model = FinalNet(n_classes = 4, in_chans=x_train.shape[2], in_samples = x_train.shape[3])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=lr, weight_decay=weight_decay), #
                  metrics = ['accuracy'])
    callbacks = [
        ReduceLROnPlateau(monitor="loss", factor=0.85, patience=35, verbose=1, min_lr=0.00005),
        ModelCheckpoint(filepath='best_model.h5', monitor='val_accuracy', save_best_only=True)
        # EarlyStopping(monitor='val_accuracy', verbose=1, mode='max', patience=patience)
    ]
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), callbacks=callbacks)

    # Оценка модели на тестовых данных
    train_loss, train_accuracy = model.evaluate(x_train, y_train)
    print('Точность на тренировочных данных:', train_accuracy)
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print('Точность на тестовых данных:', test_accuracy)

    # model.save('saved_model/')

    probs = model.predict(x_test)
    preds = probs.argmax(axis = -1)
    acc = np.mean(preds == y_test)

    print(classification_report(y_test, preds))

    return acc


def cnn_method_eegsym_tf(x_train, x_test, y_train, y_test, lr=1e-3, epochs=250, weight_decay=1e-2, batch_size=64):
    x_train, x_test, y_train, y_test = x_train.copy(), x_test.copy(), y_train.copy(), y_test.copy()
    x_train = x_train.transpose(0, 2, 1).reshape(x_train.shape[0], x_train.shape[2], x_train.shape[1], 1)
    x_test = x_test.transpose(0, 2, 1).reshape(x_test.shape[0], x_test.shape[2], x_test.shape[1], 1)

    model = EEGSym()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=lr, weight_decay=weight_decay), #
                  metrics = ['accuracy'])
    callbacks = [
        ReduceLROnPlateau(monitor="loss", factor=0.85, patience=35, verbose=1, min_lr=0.00005),
        ModelCheckpoint(filepath='best_model.h5', monitor='val_accuracy', save_best_only=True)
    ]
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), callbacks=callbacks)

    # model.save('saved_model/')

    probs = model.predict(x_test)
    preds = probs.argmax(axis = -1)
    acc = np.mean(preds == y_test)

    print(classification_report(y_test, preds))

    return acc
