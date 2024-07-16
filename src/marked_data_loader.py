import mne
import numpy as np
import joblib
from typing import Optional, List, Union
from dataclasses import dataclass
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from math import floor, ceil

import logging
logging.basicConfig(level=logging.WARNING)
mne.set_log_level('WARNING')


@dataclass
class MarkedRaw:
    raw_filename: str
    mark_filename: str
    need_augmentation: bool


class MarkedDataLoader:
    def __init__(self, marked_raw_list: List[MarkedRaw],
                 with_reference_process: bool = False,
                 with_filter: bool = True, lower_freq: float = 0, upper_freq: float = 100,
                 invert_fp_signal: bool = False, to_remove_channels: Optional[list] = None,
                 reference: Optional[str] = None,
                 use_ica: bool = True, ica_components: int = 8, ica_exclude: Optional[list] = None
                 ):

        self.raw_list = []
        for marked_raw in marked_raw_list:
            raw = mne.io.read_raw_edf(marked_raw.raw_filename, preload=True)

            data = raw.get_data()
            if invert_fp_signal:
                data[:2] = -data[:2]
            if with_reference_process:
                data_len = 15 - (0 if not to_remove_channels else len(to_remove_channels))
                data[:data_len] = data[:data_len] - data[data_len]

            info = mne.create_info(ch_names=raw.ch_names, sfreq=raw.info['sfreq'], ch_types='eeg')
            raw_new = mne.io.RawArray(data, info)
            if with_filter:
                raw_new.filter(lower_freq, upper_freq)
            if reference:
                raw_new = raw_new.set_eeg_reference(ref_channels=reference)
            if to_remove_channels:
                raw_new.drop_channels(to_remove_channels)

            if use_ica:
                ica = mne.preprocessing.ICA(n_components=ica_components, random_state=0)
                ica.fit(raw_new)
                ica.exclude = ica_exclude
                ica.apply(raw_new)

            self.raw_list.append(raw_new)

        self.X = []
        self.y = []
        self.from_to_files = []
        current_position = 0
        for i, raw in enumerate(self.raw_list):
            samples_count = \
                self.process_raw(raw, marked_raw_list[i].mark_filename, marked_raw_list[i].need_augmentation)
            self.from_to_files.append((current_position, current_position + samples_count))
            current_position = current_position + samples_count

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def get_normalized_train_test(self, cross_val=False, first_num=False, with_normalize=False, use_ready_scalers=False):
        reshape_val = 5

        if cross_val:
            x_shape = self.X.shape
            X_train, X_test, y_train, y_test = \
                np.empty((0, x_shape[1], x_shape[2])), np.empty((0, x_shape[1], x_shape[2])), np.array([]), np.array([])
            for start, end in self.from_to_files:
                x_part, y_part = self.X[start:end], self.y[start:end]
                initial_cnt = dict(Counter(y_part))
                max_val, min_val = max(initial_cnt.values()), min(initial_cnt.values())
                coef = ceil((max_val // reshape_val) * cross_val) / floor((min_val // reshape_val) * cross_val) * 1.1
                while True:
                    X_train_part, X_test_part, y_train_part, y_test_part = \
                        train_test_split(x_part.reshape(-1, reshape_val, x_shape[1], x_shape[2]), y_part.reshape(-1, reshape_val), test_size=cross_val)
                    X_train_part, X_test_part, y_train_part, y_test_part = \
                        X_train_part.reshape(-1, x_shape[1], x_shape[2]), X_test_part.reshape(-1, x_shape[1], x_shape[2]), \
                        y_train_part.reshape(-1), y_test_part.reshape(-1)

                    cnt = Counter(y_test_part)
                    if max(dict(cnt).values()) / min(dict(cnt).values()) < coef and len(dict(cnt).keys()) == 4:
                        X_train, X_test, y_train, y_test = \
                            np.concatenate((X_train, X_train_part)), np.concatenate((X_test, X_test_part)), \
                            np.concatenate((y_train, y_train_part)), np.concatenate((y_test, y_test_part))
                        break
                    else:
                        print('skipped')

        elif first_num is not None:
            X_train, X_test, y_train, y_test = self.X[:first_num], self.X[first_num:], self.y[:first_num], self.y[first_num:]

        if with_normalize:
          if use_ready_scalers:
            for j in range(X_train.shape[1]):
                scaler = joblib.load(f'scaler{j}.pkl')
                X_train[:, j, :] = scaler.transform(X_train[:, j, :])
                X_test[:, j, :] = scaler.transform(X_test[:, j, :])
          else:
            for j in range(X_train.shape[1]):
                scaler = StandardScaler()
                scaler.fit(X_train[:, j, :])
                X_train[:, j, :] = scaler.transform(X_train[:, j, :])
                X_test[:, j, :] = scaler.transform(X_test[:, j, :])
                joblib.dump(scaler, f'scaler{j}.pkl')

        return X_train, X_test, y_train, y_test

    def get_normalized_train_val_test(self, cross_val_per=0.15, test_per=0.08, with_normalize=False, use_ready_scalers=False):
        reshape_val = 5

        x_shape = self.X.shape
        x_train, x_test, y_train, y_test = train_test_split(
            self.X.reshape(-1, reshape_val, x_shape[1], x_shape[2]),
            self.y.reshape(-1, reshape_val), test_size=cross_val_per + test_per, stratify=self.y.reshape(-1, reshape_val)
        )
        x_train, y_train = x_train.reshape(-1, x_shape[1], x_shape[2]), y_train.reshape(-1)
        x_test, y_test = x_test[:, 0, :, :], y_test[:, 0]
        if test_per == 0:
            x_cross_val, x_test, y_cross_val, y_test = x_test, np.array([]), y_test, np.array([])
        else:
            x_cross_val, x_test, y_cross_val, y_test = train_test_split(
                x_test, y_test, test_size=test_per/(cross_val_per + test_per), stratify=y_test
            )

        if with_normalize:
            if use_ready_scalers:
                for j in range(x_train.shape[1]):
                    scaler = joblib.load(f'scaler{j}.pkl')
                    x_train[:, j, :] = scaler.transform(x_train[:, j, :])
                    x_cross_val[:, j, :] = scaler.transform(x_cross_val[:, j, :])
                    if test_per != 0:
                        x_test[:, j, :] = scaler.transform(x_test[:, j, :])
            else:
                for j in range(x_train.shape[1]):
                    scaler = StandardScaler()
                    scaler.fit(x_train[:, j, :])
                    x_train[:, j, :] = scaler.transform(x_train[:, j, :])
                    x_cross_val[:, j, :] = scaler.transform(x_cross_val[:, j, :])
                    if test_per != 0:
                        x_test[:, j, :] = scaler.transform(x_test[:, j, :])
                    joblib.dump(scaler, f'scaler{j}.pkl')

        np.save('x_train.npy', x_train)
        np.save('x_cross_val.npy', x_cross_val)
        np.save('x_test.npy', x_test)
        np.save('y_train.npy', y_train)
        np.save('y_cross_val.npy', y_cross_val)
        np.save('y_test.npy', y_test)

        return x_train, x_cross_val, x_test, y_train, y_cross_val, y_test

    def process_raw(self, raw, mark_filename: str, with_augmentation: bool):
        current_samples_count = 0
        info = raw.info
        if self.ica:
            self.ica.apply(raw)

        data = raw.get_data(picks=info.ch_names)
        hz = 250
        video_start = 15

        with open(mark_filename, 'r') as f:
            for movement_info in f.readlines():
                current_samples_count += 5
                movement_info = movement_info.split(', ')
                movement_start_sec, movement_end_sec, label = \
                    float(movement_info[0]) + video_start, float(movement_info[1]) + video_start, int(movement_info[2])

                self.X.append(data[:, round(movement_start_sec * hz):round(movement_end_sec * hz)])
                self.y.append(label)

                if with_augmentation:

                    self.X.append(data[:, round((movement_start_sec - 0.5) * hz):round((movement_end_sec - 0.5) * hz)])
                    self.y.append(label)

                    self.X.append(data[:, round((movement_start_sec + 0.5) * hz):round((movement_end_sec + 0.5) * hz)])
                    self.y.append(label)

                    self.X.append(data[:, round((movement_start_sec - 1) * hz):round((movement_end_sec - 1) * hz)])
                    self.y.append(label)

                    self.X.append(data[:, round((movement_start_sec + 1) * hz):round((movement_end_sec + 1) * hz)])
                    self.y.append(label)

        return current_samples_count
