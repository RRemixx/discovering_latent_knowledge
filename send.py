from tqdm import tqdm
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression


def get_encoder_hidden_states(model, tokenizer, input_text, layer=-1):
    """
    Given an encoder model and some text, gets the encoder hidden states (in a given layer, by default the last) 
    on that input text (where the full text is given to the encoder).

    Returns a numpy array of shape (hidden_dim,)
    """
    # tokenize
    encoder_text_ids = tokenizer(input_text, truncation=True, return_tensors="pt").input_ids.to(model.device)

    # forward pass
    with torch.no_grad():
        output = model(encoder_text_ids, output_hidden_states=True)

    # get the appropriate hidden states
    hs_tuple = output["hidden_states"]
    
    hs = hs_tuple[layer][0, -1].detach().cpu().numpy()

    return hs

def get_encoder_decoder_hidden_states(model, tokenizer, input_text, layer=-1):
    """
    Given an encoder-decoder model and some text, gets the encoder hidden states (in a given layer, by default the last) 
    on that input text (where the full text is given to the encoder).

    Returns a numpy array of shape (hidden_dim,)
    """
    # tokenize
    encoder_text_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    decoder_text_ids = tokenizer("", return_tensors="pt").input_ids.to(model.device)

    # forward pass
    with torch.no_grad():
        output = model(encoder_text_ids, decoder_input_ids=decoder_text_ids, output_hidden_states=True)

    # get the appropriate hidden states
    hs_tuple = output["encoder_hidden_states"]
    hs = hs_tuple[layer][0, -1].detach().cpu().numpy()

    return hs

def get_decoder_hidden_states(model, tokenizer, input_text, layer=-1):
    """
    Given a decoder model and some text, gets the hidden states (in a given layer, by default the last) on that input text

    Returns a numpy array of shape (hidden_dim,)
    """
    # tokenize (adding the EOS token this time)
    input_ids = tokenizer(input_text + tokenizer.eos_token, return_tensors="pt").input_ids.to(model.device)

    # forward pass
    with torch.no_grad():
        output = model(input_ids, output_hidden_states=True)

    # get the last layer, last token hidden states
    hs_tuple = output["hidden_states"]
    hs = hs_tuple[layer][0, -1].detach().cpu().numpy()

    return hs

def get_hidden_states(model, tokenizer, input_text, layer=-1, model_type="encoder"):
    fn = {"encoder": get_encoder_hidden_states, "encoder_decoder": get_encoder_decoder_hidden_states,
          "decoder": get_decoder_hidden_states}[model_type]

    return fn(model, tokenizer, input_text, layer=layer)


def format_imdb(text, label):
    """
    Given an imdb example ("text") and corresponding label (0 for negative, or 1 for positive), 
    returns a zero-shot prompt for that example (which includes that label as the answer).
    
    (This is just one example of a simple, manually created prompt.)
    """
    return "The following movie review expresses a " + ["negative", "positive"][label] + " sentiment:\n" + text


def get_hidden_states_many_examples(model, tokenizer, data, model_type, n=100):
    """
    Given an encoder-decoder model, a list of data, computes the contrast hidden states on n random examples.
    Returns numpy arrays of shape (n, hidden_dim) for each candidate label, along with a boolean numpy array of shape (n,)
    with the ground truth labels
    
    This is deliberately simple so that it's easy to understand, rather than being optimized for efficiency
    """
    # setup
    model.eval()
    all_neg_hs, all_pos_hs, all_gt_labels = [], [], []

    # loop
    for _ in tqdm(range(n)):
        # for simplicity, sample a random example until we find one that's a reasonable length
        # (most examples should be a reasonable length, so this is just to make sure)
        while True:
            idx = np.random.randint(len(data))
            text, true_label = data[idx]["content"], data[idx]["label"]
            # the actual formatted input will be longer, so include a bit of a marign
            if len(tokenizer(text)) < 400:  
                break
                
        # get hidden states
        neg_hs = get_hidden_states(model, tokenizer, format_imdb(text, 0), model_type=model_type)
        pos_hs = get_hidden_states(model, tokenizer, format_imdb(text, 1), model_type=model_type)

        # collect
        all_neg_hs.append(neg_hs)
        all_pos_hs.append(pos_hs)
        all_gt_labels.append(true_label)

    all_neg_hs = np.stack(all_neg_hs)
    all_pos_hs = np.stack(all_pos_hs)
    all_gt_labels = np.stack(all_gt_labels)

    return all_neg_hs, all_pos_hs, all_gt_labels


def get_model_and_data(model_name="deberta", cache_dir=None, save_path=None, n=2000):
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

    neg_hs, pos_hs, y = get_hidden_states_many_examples(model, tokenizer, data, model_type, n=n)
    
    # Save the processed data to a single file
    np.savez(save_path, neg_hs=neg_hs, pos_hs=pos_hs, y=y)
    
    return neg_hs, pos_hs, y


if __name__ == "__main__":
    # Example usage
    # Example usage
    models = ["deberta"]
    cache_dir = None
    n = 1000

    for model_name in models:
        save_path = f"./{model_name}.npz"
        print(f"Processing {model_name}...")
        neg_hs, pos_hs, y = get_model_and_data(model_name=model_name, cache_dir=cache_dir, save_path=save_path, n=n)
        
    # An example to load the saved data
    if False:
        model_name = 'gpt-j'
        data_file = f'data/{model_name}.npz'
        # Load data otherwise download and process data
        try:
            data = np.load(data_file)
            neg_hs = data['neg_hs']
            pos_hs = data['pos_hs']
            y = data['y']
        except FileNotFoundError:
            # If the file doesn't exist, process the data
            neg_hs, pos_hs, y = get_model_and_data(model_name=f"{model_name}", cache_dir=None, n=2)

        # Validate the data
        neg_hs_train, neg_hs_test, pos_hs_train, pos_hs_test, y_train, y_test = validate(neg_hs, pos_hs, y)