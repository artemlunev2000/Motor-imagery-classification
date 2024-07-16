import tensorflow as tf
from mne_realtime import LSLClient
import joblib
import mne
import cv2
import numpy as np
import logging
import time

logging.basicConfig(level=logging.WARNING)
mne.set_log_level('WARNING')

if __name__ == '__main__':
    info = mne.create_info(
        ch_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'T3', 'T4', 'Fz', 'Cz', 'Pz', 'Aux'],
        sfreq=250,
        ch_types='eeg'
    )
    full_movement_cycle_hz = round(9.5 * 250)  # sec * sfreq
    model = tf.keras.models.load_model(f'models/pretrained.h5')
    wait_img = cv2.imread('../images/wait.png')
    up_img = cv2.imread('../images/imagined_up.png')
    down_img = cv2.imread('../images/imagined_down.png')
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)

    while True:
        user_input = input("Enter s to make move and e to exit: ")
        if user_input == 's':
            for _ in range(8):
                with LSLClient(info=info, host='NBEEG16_1075_Data', wait_max=100,
                               buffer_size=full_movement_cycle_hz) as client:
                    cv2.imshow('Display', wait_img)
                    cv2.waitKey(8500)
                    client.get_data_as_epoch(n_samples=1)

                    cv2.imshow('Display', wait_img)
                    cv2.waitKey(1500)

                    cv2.imshow('Display', up_img)
                    cv2.waitKey(2500)

                    cv2.imshow('Display', wait_img)
                    cv2.waitKey(1500)

                    cv2.imshow('Display', down_img)
                    cv2.waitKey(2500)

                    cv2.imshow('Display', wait_img)
                    cv2.waitKey(1500)

                    epoch = client.get_data_as_epoch(n_samples=full_movement_cycle_hz)
                    data = epoch.get_data()[0]
                    data_to_save = data.copy()
                    raw_new = mne.io.RawArray(data, info)
                    raw_new.filter(2, None)
                    # raw_new.plot(block=True)
                    data = raw_new.get_data()

                    # data = data[:16] - data[15]  # if with aux ref
                    data = data.reshape(1, data.shape[0], data.shape[1])
                    for j in range(data.shape[1]):
                        scaler = joblib.load(f'models/scaler{j}.pkl')
                        data[:, j, :] = scaler.transform(data[:, j, :])

                    data = data.reshape(data.shape[0], 1, data.shape[1], data.shape[2])

                    probs = model.predict(data)
                    print(f'probs - {probs}')
                    preds = probs.argmax(axis=-1)
                    print(f'preds - {preds}')
                    if preds[0] == 0:
                        result = 'Right horizontal'
                    elif preds[0] == 1:
                        result = 'Right Vertical'
                    elif preds[0] == 2:
                        result = 'Left horizontal'
                    else:
                        result = 'Left vertical'
                    print(f'Your movement type: {result} ({preds[0] + 1})')
                    expected = 1
                    np.save(f'np_savings/{preds[0]}-{int(expected) - 1}-{time.time()}.npy', data_to_save)

                    cv2.destroyWindow('Display')
                    time.sleep(2)

        elif user_input == 'e':
            break
