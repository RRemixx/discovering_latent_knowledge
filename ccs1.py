from tqdm import tqdm
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression

from ccs_model import CCS
from ccs_utils import *


def get_model_and_data(model_name="deberta", cache_dir=None):
    # Let's just try IMDB for simplicity
    data = load_dataset("amazon_polarity")["test"]

    if model_name == "deberta":
        model_type = "encoder"
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xxlarge", cache_dir=cache_dir)
        model = AutoModelForMaskedLM.from_pretrained("microsoft/deberta-v2-xxlarge", cache_dir=cache_dir)
        model.cuda()
    elif model_name == "gpt-j":
        model_type = "decoder"
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=cache_dir)
        model.cuda()
    elif model_name == "t5":
        model_type = "encoder_decoder"
        tokenizer = AutoTokenizer.from_pretrained("t5-11b", cache_dir=cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-11b", cache_dir=cache_dir)
        model.parallelize()  # T5 is big enough that we may need to run it on multiple GPUs
    else:
        raise ValueError(f"Model {model_name} not implemented!")

    neg_hs, pos_hs, y = get_hidden_states_many_examples(model, tokenizer, data, model_type)
    
    # Save the processed data to a single file
    np.savez('processed_data.npz', neg_hs=neg_hs, pos_hs=pos_hs, y=y)
    
    return neg_hs, pos_hs, y


def validate(neg_hs, pos_hs, y):
    # let's create a simple 50/50 train split (the data is already randomized)
    n = len(y)
    neg_hs_train, neg_hs_test = neg_hs[:n//2], neg_hs[n//2:]
    pos_hs_train, pos_hs_test = pos_hs[:n//2], pos_hs[n//2:]
    y_train, y_test = y[:n//2], y[n//2:]

    # for simplicity we can just take the difference between positive and negative hidden states
    # (concatenating also works fine)
    x_train = neg_hs_train - pos_hs_train
    x_test = neg_hs_test - pos_hs_test

    lr = LogisticRegression(class_weight="balanced")
    lr.fit(x_train, y_train)
    print("Logistic regression accuracy: {}".format(lr.score(x_test, y_test)))
    
    return neg_hs_train, neg_hs_test, pos_hs_train, pos_hs_test, y_train, y_test


def eval_css(neg_hs_train, neg_hs_test, pos_hs_train, pos_hs_test, y_train, y_test):
    # Train CCS without any labels
    ccs = CCS(neg_hs_train, pos_hs_train)
    ccs.repeated_train()

    # Evaluate
    ccs_acc = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
    print("CCS accuracy: {}".format(ccs_acc))

    # Train CCS with labels
    ccs_supervised = CCS(neg_hs_train, pos_hs_train)
    ccs_supervised.supervised_train(y_train, batch_size=-1, verbose=False, nepochs=1000)
    # ccs_supervised.repeated_train()

    # Evaluate
    ccs_supervised_acc = ccs_supervised.get_acc(neg_hs_test, pos_hs_test, y_test)
    print("CCS supervised accuracy: {}".format(ccs_supervised_acc))


def main():
    data_file = 'processed_data.npz'
    # Load data otherwise download and process data
    try:
        data = np.load(data_file)
        neg_hs = data['neg_hs']
        pos_hs = data['pos_hs']
        y = data['y']
    except FileNotFoundError:
        # If the file doesn't exist, process the data
        neg_hs, pos_hs, y = get_model_and_data(model_name="deberta", cache_dir=None)
    
    # Validate the data
    neg_hs_train, neg_hs_test, pos_hs_train, pos_hs_test, y_train, y_test = validate(neg_hs, pos_hs, y)
    
    # Evaluate CCS
    eval_css(neg_hs_train, neg_hs_test, pos_hs_train, pos_hs_test, y_train, y_test)


if __name__ == "__main__":
    main()
