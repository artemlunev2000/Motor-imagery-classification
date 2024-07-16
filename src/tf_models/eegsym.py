from tensorflow.keras.layers import Activation, Input, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import Conv3D, Add, AveragePooling3D
from tensorflow.keras.layers import Dense
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np


def EEGSym(input_time=9500, fs=250, ncha=16, filters_per_branch=8,
           scales_time=(1000, 500, 250), dropout_rate=0.25, activation='elu',
           n_classes=4, spatial_resnet_repetitions=1, residual=True, symmetric=True):
    def general_module(input, scales_samples, filters_per_branch, ncha,
                         activation, dropout_rate, average,
                         spatial_resnet_repetitions=1, residual=True,
                         init=False):
        block_units = list()
        unit_conv_t = list()
        unit_batchconv_t = list()

        for i in range(len(scales_samples)):
            unit_conv_t.append(Conv3D(filters=filters_per_branch,
                                      kernel_size=(1, scales_samples[i], 1),
                                      kernel_initializer='he_normal',
                                      padding='same'))
            unit_batchconv_t.append(BatchNormalization())

        if ncha != 1:
            unit_dconv = list()
            unit_batchdconv = list()
            unit_conv_s = list()
            unit_batchconv_s = list()
            for i in range(spatial_resnet_repetitions):
                # 3D Implementation of DepthwiseConv
                unit_dconv.append(Conv3D(kernel_size=(1, 1, ncha),
                                         filters=filters_per_branch * len(
                                             scales_samples),
                                         groups=filters_per_branch * len(
                                             scales_samples),
                                         use_bias=False,
                                         padding='valid'))
                unit_batchdconv.append(BatchNormalization())

                unit_conv_s.append(Conv3D(kernel_size=(1, 1, ncha),
                                          filters=filters_per_branch,
                                          # groups=filters_per_branch,
                                          use_bias=False,
                                          strides=(1, 1, 1),
                                          kernel_initializer='he_normal',
                                          padding='valid'))
                unit_batchconv_s.append(BatchNormalization())

            unit_conv_1 = Conv3D(kernel_size=(1, 1, 1),
                                 filters=filters_per_branch,
                                 use_bias=False,
                                 kernel_initializer='he_normal',
                                 padding='valid')
            unit_batchconv_1 = BatchNormalization()

        for j in range(len(input)):
            block_side_units = list()
            for i in range(len(scales_samples)):
                unit = input[j]
                unit = unit_conv_t[i](unit)

                unit = unit_batchconv_t[i](unit)
                unit = Activation(activation)(unit)
                unit = Dropout(dropout_rate)(unit)

                block_side_units.append(unit)
            block_units.append(block_side_units)
        # Concatenation
        block_out = list()
        for j in range(len(input)):
            if len(block_units[j]) != 1:
                block_out.append(
                    keras.layers.concatenate(block_units[j], axis=-1))
            else:
                block_out.append(block_units[j][0])

            if residual:
                if len(block_units[j]) != 1:
                    block_out_temp = input[j]
                else:
                    block_out_temp = input[j]
                    block_out_temp = unit_conv_1(block_out_temp)

                    block_out_temp = unit_batchconv_1(block_out_temp)
                    block_out_temp = Activation(activation)(block_out_temp)
                    block_out_temp = Dropout(dropout_rate)(block_out_temp)

                block_out[j] = Add()([block_out[j], block_out_temp])

            if average != 1:
                block_out[j] = AveragePooling3D((1, average, 1))(block_out[j])

        if ncha != 1:
            for i in range(spatial_resnet_repetitions):
                block_out_temp = list()
                for j in range(len(input)):
                    if len(scales_samples) != 1:
                        if residual:
                            block_out_temp.append(block_out[j])

                            block_out_temp[j] = unit_dconv[i](block_out_temp[j])

                            block_out_temp[j] = unit_batchdconv[i](
                                block_out_temp[j])
                            block_out_temp[j] = Activation(activation)(
                                block_out_temp[j])
                            block_out_temp[j] = Dropout(dropout_rate)(
                                block_out_temp[j])

                            block_out[j] = Add()(
                                [block_out[j], block_out_temp[j]])

                        elif init:
                            block_out[j] = unit_dconv[i](block_out[j])
                            block_out[j] = unit_batchdconv[i](block_out[j])
                            block_out[j] = Activation(activation)(block_out[j])
                            block_out[j] = Dropout(dropout_rate)(block_out[j])
                    else:
                        if residual:
                            block_out_temp.append(block_out[j])

                            block_out_temp[j] = unit_conv_s[i](
                                block_out_temp[j])
                            block_out_temp[j] = unit_batchconv_s[i](
                                block_out_temp[j])
                            block_out_temp[j] = Activation(activation)(
                                block_out_temp[j])
                            block_out_temp[j] = Dropout(dropout_rate)(
                                block_out_temp[j])

                            block_out[j] = Add()(
                                [block_out[j], block_out_temp[j]])
        return block_out
    input_samples = int(input_time * fs / 1000)
    scales_samples = [int(s * fs / 1000) for s in scales_time]

    input_layer = Input((input_samples, ncha, 1))
    input = tf.expand_dims(input_layer, axis=1)

    if symmetric:
        left_idx = [0, 2, 4, 6, 8, 10]
        right_idx = [1, 3, 5, 7, 9, 11]
        central_idx = [12, 13, 14, 15]
        ncha = ncha - len(left_idx)

        ch_left = tf.gather(input, indices=left_idx, axis=-2)
        ch_right = tf.gather(input, indices=right_idx, axis=-2)
        ch_central = tf.gather(input, indices=central_idx, axis=-2)

        left_init = keras.layers.concatenate((ch_left, ch_central), axis=-2)
        right_init = keras.layers.concatenate((ch_right, ch_central), axis=-2)

        input = keras.layers.concatenate((left_init, right_init), axis=1)
        division = 2
    else:
        division = 1

    b1_out = general_module([input],
                              scales_samples=scales_samples,
                              filters_per_branch=filters_per_branch,
                              ncha=ncha,
                              activation=activation,
                              dropout_rate=dropout_rate, average=2,
                              spatial_resnet_repetitions=spatial_resnet_repetitions,
                              residual=residual, init=True)

    b2_out = general_module(b1_out, scales_samples=[int(x / 4) for x in
                                                      scales_samples],
                              filters_per_branch=filters_per_branch,
                              ncha=ncha,
                              activation=activation,
                              dropout_rate=dropout_rate, average=2,
                              spatial_resnet_repetitions=spatial_resnet_repetitions,
                              residual=residual)

    b3_u1 = general_module(b2_out, scales_samples=[16],
                             filters_per_branch=int(
                                 filters_per_branch * len(
                                     scales_samples) / 2),
                             ncha=ncha,
                             activation=activation,
                             dropout_rate=dropout_rate, average=2,
                             spatial_resnet_repetitions=spatial_resnet_repetitions,
                             residual=residual)

    b3_u1 = general_module(b3_u1,
                             scales_samples=[8],
                             filters_per_branch=int(
                                 filters_per_branch * len(
                                     scales_samples) / 2),

                             ncha=ncha,
                             activation=activation,
                             dropout_rate=dropout_rate, average=2,
                             spatial_resnet_repetitions=spatial_resnet_repetitions,
                             residual=residual)
    b3_u2 = general_module(b3_u1, scales_samples=[4],
                             filters_per_branch=int(
                                 filters_per_branch * len(
                                     scales_samples) / 4),
                             ncha=ncha,
                             activation=activation,
                             dropout_rate=dropout_rate, average=2,
                             spatial_resnet_repetitions=spatial_resnet_repetitions,
                             residual=residual)

    t_red = b3_u2[0]
    for _ in range(1):
        t_red_temp = t_red
        t_red_temp = Conv3D(kernel_size=(1, 4, 1),
                            filters=int(filters_per_branch * len(
                                scales_samples) / 4),
                            use_bias=False,
                            strides=(1, 1, 1),
                            kernel_initializer='he_normal',
                            padding='same')(t_red_temp)

        t_red_temp = BatchNormalization()(t_red_temp)
        t_red_temp = Activation(activation)(t_red_temp)
        t_red_temp = Dropout(dropout_rate)(t_red_temp)

        if residual:
            t_red = Add()([t_red, t_red_temp])
        else:
            t_red = t_red_temp

    t_red = AveragePooling3D((1, 2, 1))(t_red)

    ch_merg = t_red
    if residual:
        for _ in range(2):
            ch_merg_temp = ch_merg
            ch_merg_temp = Conv3D(kernel_size=(division, 1, ncha),
                                  filters=int(filters_per_branch * len(
                                      scales_samples) / 4),
                                  use_bias=False,
                                  strides=(1, 1, 1),
                                  kernel_initializer='he_normal',
                                  padding='valid')(ch_merg_temp)
            ch_merg_temp = BatchNormalization()(ch_merg_temp)
            ch_merg_temp = Activation(activation)(ch_merg_temp)
            ch_merg_temp = Dropout(dropout_rate)(ch_merg_temp)
            ch_merg = Add()([ch_merg, ch_merg_temp])

        ch_merg = Conv3D(kernel_size=(division, 1, ncha),
                         filters=int(
                             filters_per_branch * len(scales_samples) / 4),
                         groups=int(
                             filters_per_branch * len(scales_samples) / 8),
                         use_bias=False,
                         padding='valid')(ch_merg)
        ch_merg = BatchNormalization()(ch_merg)
        ch_merg = Activation(activation)(ch_merg)
        ch_merg = Dropout(dropout_rate)(ch_merg)
    else:
        if symmetric:
            ch_merg = Conv3D(kernel_size=(division, 1, 1),
                             filters=int(
                                 filters_per_branch * len(
                                     scales_samples) / 4),
                             groups=int(
                                 filters_per_branch * len(
                                     scales_samples) / 8),
                             use_bias=False,
                             padding='valid')(ch_merg)
            ch_merg = BatchNormalization()(ch_merg)
            ch_merg = Activation(activation)(ch_merg)
            ch_merg = Dropout(dropout_rate)(ch_merg)

    t_merg = ch_merg
    for _ in range(1):
        if residual:
            t_merg_temp = t_merg
            t_merg_temp = Conv3D(kernel_size=(1, input_samples // 64, 1),
                                 filters=int(filters_per_branch * len(
                                     scales_samples) / 4),
                                 use_bias=False,
                                 strides=(1, 1, 1),
                                 kernel_initializer='he_normal',
                                 padding='valid')(t_merg_temp)
            t_merg_temp = BatchNormalization()(t_merg_temp)
            t_merg_temp = Activation(activation)(t_merg_temp)
            t_merg_temp = Dropout(dropout_rate)(t_merg_temp)

            t_merg = Add()([t_merg, t_merg_temp])
        else:
            t_merg_temp = t_merg
            t_merg_temp = Conv3D(kernel_size=(1, input_samples // 64, 1),
                                 filters=int(filters_per_branch * len(
                                     scales_samples) / 4),
                                 use_bias=False,
                                 strides=(1, 1, 1),
                                 kernel_initializer='he_normal',
                                 padding='same')(t_merg_temp)
            t_merg_temp = BatchNormalization()(t_merg_temp)
            t_merg_temp = Activation(activation)(t_merg_temp)
            t_merg_temp = Dropout(dropout_rate)(t_merg_temp)
            t_merg = t_merg_temp

    t_merg = Conv3D(kernel_size=(1, input_samples // 64, 1),
                    filters=int(
                        filters_per_branch * len(scales_samples) / 4) * 2,
                    groups=int(
                        filters_per_branch * len(scales_samples) / 4),
                    use_bias=False,
                    padding='valid')(t_merg)
    t_merg = BatchNormalization()(t_merg)
    t_merg = Activation(activation)(t_merg)
    t_merg = Dropout(dropout_rate)(t_merg)
    output = t_merg
    for _ in range(4):
        output_temp = output
        output_temp = Conv3D(kernel_size=(1, 1, 1),
                             filters=int(
                                 filters_per_branch * len(
                                     scales_samples) / 2),
                             use_bias=False,
                             strides=(1, 1, 1),
                             kernel_initializer='he_normal',
                             padding='valid')(output_temp)
        output_temp = BatchNormalization()(output_temp)
        output_temp = Activation(activation)(output_temp)
        output_temp = Dropout(dropout_rate)(output_temp)
        if residual:
            output = Add()([output, output_temp])
        else:
            output = output_temp

    output = Flatten()(output)
    output_layer = Dense(n_classes, activation='softmax')(output)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model
