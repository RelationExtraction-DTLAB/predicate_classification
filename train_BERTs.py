## Library
# base
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"

import sys
import random
import warnings
warnings.filterwarnings('ignore')

# data manipulation
import pandas as pd
import numpy as np

# tools
import torch

# user-made
from berts import BERT_for_classification
from utility.simple_data_loader import load_bert_data

## Main Function
def main(model_type: str, label_type: str, process: str):

    '''
    model_type: biobert, scibert, bert_base
    label_type: predicate, framenet
    process: train, pred
    '''

    # Set seed value 
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Set directory
    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir != '' else '.'

    # Output directory
    # output_dir_path = current_dir + './models'

    '''
    please enter file path for dataset, and also model path if you want to predict
    '''

    data_file_path = current_dir + '/data/train.csv'
    model_file_name = 'BioBERT_0107_09_20_20_0.8462.pth'

    # Load data
    X, y, num_labels = load_bert_data(data_file_path, label_type=label_type)
    # Generate model
    model = BERT_for_classification(model_type, num_labels)
    model.set_base(X, y, test_size=0.2, random_state=seed_val)

    if process == 'train':
        model.fit(random_state=seed_val)
        
        # Draw and save a precision-recall curve for framenet labels
        if label_type == 'framenet':
            model.plot()
    
    elif process == 'pred':
        predictions = model.pred(model_file_name, model_type)

  
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
