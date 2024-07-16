import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, Lambda, Add
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPooling2D, AveragePooling2D, MultiHeadAttention
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.layers import Reshape, Concatenate
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import L2
from tensorflow import keras


def FinalNet(n_classes, in_chans=16, in_samples=2375, n_windows=1, attention='mha',
             eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=18, eegn_dropout=0.3,
             tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
             tcn_activation='elu', fuse='average'):
    input_1 = Input(shape=(1, in_chans, in_samples))
    input_2 = Permute((3, 2, 1))(input_1)

    left_idx = [0, 2, 4, 6, 8, 10]
    right_idx = [1, 3, 5, 7, 9, 11]
    central_idx = [12, 13, 14, 15]
    in_chans = in_chans - len(left_idx)

    ch_left = tf.gather(input_2, indices=left_idx, axis=-2)
    ch_right = tf.gather(input_2, indices=right_idx, axis=-2)
    ch_central = tf.gather(input_2, indices=central_idx, axis=-2)

    left_init = keras.layers.concatenate((ch_left, ch_central), axis=-2)
    right_init = keras.layers.concatenate((ch_right, ch_central), axis=-2)

    # input = keras.layers.concatenate((left_init, right_init), axis=1)
    sw_concat = []  # to store concatenated or averaged sliding window outputs
    conv_blocks = []
    for tensor in [left_init, right_init]:
        dense_weightDecay = 0.09
        conv_weightDecay = 0.009
        conv_maxNorm = 0.6
        from_logits = False

        numFilters = eegn_F1
        F2 = numFilters * eegn_D
        block1 = eegnet(input_layer=tensor, F1=eegn_F1, D=eegn_D,
                        kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                        weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                        in_chans=in_chans, dropout=eegn_dropout)
        block1 = Lambda(lambda x: x[:, :, -1, :])(block1)
        conv_blocks.append(block1)
    # Sliding window
    for i in range(n_windows):
        st = i
        attention_blocks = []
        for block_index in [0, 1]:
            end = conv_blocks[block_index].shape[1] - n_windows + i + 1
            block2 = conv_blocks[block_index][:, st:end, :]

            # Attention_model
            if attention is not None:
                if (attention == 'se' or attention == 'cbam'):
                    block2 = Permute((2, 1))(block2)  # shape=(None, 32, 16)
                    block2 = attention(block2, attention)
                    block2 = Permute((2, 1))(block2)  # shape=(None, 16, 32)
                else:
                    block2 = attention(block2, attention)

            attention_blocks.append(block2)

        # Temporal convolutional network (TCN)
        block2 = tf.stack([attention_blocks[0], attention_blocks[1]], axis=1)
        block2 = Conv2D(32, (2, 1),
                    data_format='channels_last',
                    kernel_regularizer=L2(conv_weightDecay),
                    kernel_constraint=max_norm(conv_maxNorm, axis=[0, 1, 2]),
                    use_bias=False)(block2)
        block2 = BatchNormalization(axis=-1)(block2)
        block2 = Activation('elu')(block2)
        block2 = Dropout(0.25)(block2)
        block2 = Lambda(lambda x: x[:, -1, :, :])(block2)
        block3 = tcn(input_layer=block2, input_dimension=F2, depth=tcn_depth,
                     kernel_size=tcn_kernelSize, filters=tcn_filters,
                     weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                     dropout=tcn_dropout, activation=tcn_activation)
        # Get feature maps of the last sequence
        block3 = Lambda(lambda x: x[:, -1, :])(block3)

        # Outputs of sliding window: Average_after_dense or concatenate_then_dense
        if (fuse == 'average'):
            sw_concat.append(Dense(n_classes, kernel_regularizer=L2(dense_weightDecay))(block3))
        elif (fuse == 'concat'):
            if i == 0:
                sw_concat = block3
            else:
                sw_concat = Concatenate()([sw_concat, block3])

    if (fuse == 'average'):
        if len(sw_concat) > 1:  # more than one window
            sw_concat = tf.keras.layers.Average()(sw_concat[:])
        else:  # one window (# windows = 1)
            sw_concat = sw_concat[0]
    elif (fuse == 'concat'):
        sw_concat = Dense(n_classes, kernel_regularizer=L2(dense_weightDecay))(sw_concat)

    if from_logits:  # No activation here because we are using from_logits=True
        out = Activation('linear')(sw_concat)
    else:  # Using softmax activation
        out = Activation('softmax')(sw_concat)

 

    return Model(inputs=input_1, outputs=out)


def eegnet(input_layer, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22,
           weightDecay=0.009, maxNorm=0.6, dropout=0.25):

    F2 = F1 * D
    block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last',
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]),
                    use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)  # bn_axis = -1 if data_format() == 'channels_last' else 1

    block2 = DepthwiseConv2D((1, in_chans),
                             depth_multiplier=D,
                             data_format='channels_last',
                             depthwise_regularizer=L2(weightDecay),
                             depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]),
                             use_bias=False)(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)

    block3 = Conv2D(F2, (16, 1),
                    data_format='channels_last',
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]),
                    use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)

    block3 = AveragePooling2D((poolSize, 1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)

    return block3


def tcn(input_layer, input_dimension, depth, kernel_size, filters, dropout, maxNorm=0.6, activation='relu'):

    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   kernel_regularizer=L2(0.01),
                   kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
                   padding='causal', kernel_initializer='he_uniform')(input_layer)

    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   kernel_regularizer=L2(0.01),
                   kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
                   padding='causal', kernel_initializer='he_uniform')(block)

    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if (input_dimension != filters):
        conv = Conv1D(filters, kernel_size=1,
                      kernel_regularizer=L2(0.01),
                      kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
                      padding='same')(input_layer)

        added = Add()([block, conv])
    else:
        added = Add()([block, input_layer])
    out = Activation(activation)(added)

    for i in range(depth - 1):
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
                       kernel_regularizer=L2(0.01),
                       kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
                       padding='causal', kernel_initializer='he_uniform')(out)

        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
                       kernel_regularizer=L2(0.01),
                       kernel_constraint=max_norm(maxNorm, axis=[0, 1]),
                       padding='causal', kernel_initializer='he_uniform')(block)

        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)
    return out


def attention(in_layer):
    in_sh = in_layer.shape
    in_len = len(in_sh)
    expanded_axis = 2

    if (in_len > 3):
        in_layer = Reshape((in_sh[1], -1))(in_layer)
    out_layer = mha_block(in_layer)

    if (in_len == 3 and len(out_layer.shape) == 4):
        out_layer = tf.squeeze(out_layer, expanded_axis)
    elif (in_len == 4 and len(out_layer.shape) == 3):
        out_layer = Reshape((in_sh[1], in_sh[2], in_sh[3]))(out_layer)
    return out_layer


def mha_block(input_feature, dropout=0.3):
    x = LayerNormalization()(input_feature)
    x = MultiHeadAttention(key_dim=8, num_heads=2, dropout=dropout)(x, x)

    x = Dropout(0.3)(x)
    mha_feature = Add()([input_feature, x])

    return mha_feature
