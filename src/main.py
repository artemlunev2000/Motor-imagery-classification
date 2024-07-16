from marked_data_loader import MarkedDataLoader, MarkedRaw
from collections import Counter
from tf_models.methods import cnn_method_final_tf, cnn_method_ae_atc_tf, cnn_method_aelstm_atc_tf, cnn_method_mlstm_tf, \
    cnn_method_lstm_atc_tf, cnn_method_atc3_tf, cnn_method_eegsym_tf

if __name__ == '__main__':
    marked_raw_list = [
        MarkedRaw(raw_filename='eeg_data/3 subject/1/15-3-5a9a464e-890a-4608-8ab4-b1c02415e563.edf',
                  mark_filename='markups/mark 15-3-5a9a464e-890a-4608-8ab4-b1c02415e563.txt',
                  need_augmentation=True),
        MarkedRaw(raw_filename='eeg_data/3 subject/1/15-3-5d2366b3-f490-4e72-b266-f0262d0287a6.edf',
                  mark_filename='markups/mark 15-3-5d2366b3-f490-4e72-b266-f0262d0287a6.txt',
                  need_augmentation=True)
    ]
    marked_data_loader = MarkedDataLoader(
        marked_raw_list=marked_raw_list,
        with_reference_process=False,
        with_filter=False,
        lower_freq=2,
        upper_freq=50,
        invert_fp_signal=False,
        to_remove_channels=None,  # ['Aux', 'T3', 'T4']
        reference=None,  # average | ['Aux']
        use_ica=False,
        ica_components=10,
        ica_exclude=[0]
    )

    X_train, X_test, y_train, y_test = \
        marked_data_loader.get_normalized_train_test(cross_val=0.2, with_normalize=True)
    cnt_train = Counter(y_train)
    print(dict(cnt_train))
    cnt_test = Counter(y_test)
    print(dict(cnt_test))

    final_acc = cnn_method_final_tf(
        X_train, X_test, y_train, y_test, lr=9e-4, epochs=550, weight_decay=1e-2, batch_size=64
    )
    print(f'accuracy: {final_acc}')
