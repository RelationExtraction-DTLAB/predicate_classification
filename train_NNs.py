import numpy as np
import sys
import os


def main(model_type:str, label_type:str, process:str):
    '''

    Train models based on CNN, LSTM algorithms to predict predicate/framenet labels.

    :param model_type: biobert, scibert, bert_base
    :param label_type: predicate, framenet
    :param process: train, pred

    '''
    random_state = 42
    np.random.seed(random_state)

    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir != '' else '.'

    output_dir_path = current_dir + '/models'
    data_file_path = current_dir + '/data/predicate_train_201211_v7.csv'

    from utility.simple_data_loader import load_text_label_pairs, load_csv_dataset
    from utility.text_fit import fit_text

    text_data_model = fit_text(data_file_path, label_type=label_type)
    text_label_pairs = load_csv_dataset(data_file_path, label_type=label_type)

    if model_type == 'CNN':
        from cnn import WordVecCnn
        model = WordVecCnn()
    elif model_type == 'MC_CNN':
        from cnn import WordVecMultiChannelCnn
        model = WordVecMultiChannelCnn()
    elif model_type == 'LSTM':
        from lstm import WordVecLstmSoftmax
        model = WordVecLstmSoftmax()
    elif model_type == 'BiLSTM':
        from lstm import WordVecBidirectionalLstmSoftmax
        model = WordVecBidirectionalLstmSoftmax()
    elif model_type == 'CNN_LSTM':
        from cnn_lstm import WordVecCnnLstm
        model = WordVecCnnLstm()
    else:
        raise Exception('Please enter correct NN types!')

    batch_size = 64
    epochs = 20

    history = model.fit(text_data_model=text_data_model,
                             model_dir_path=output_dir_path,
                             text_label_pairs=text_label_pairs,
                             batch_size=batch_size,
                             epochs=epochs,
                             test_size=0.2,
                             random_state=random_state)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])

