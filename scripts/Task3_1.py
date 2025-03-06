# dev: Hong-Viet Tran
import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import sys
sys.path.append("../notebooks")
from train import supervised_training

from dataloader import SMILESDataset, ExternalDataset
from model import MoLFormerWithRegressionHead

def random_selection(ext_data, n):
    """
    Randomly select n data points from the external dataset.
    """
    random.seed(42)
    selected_indices = random.choices(range(len(ext_data)), k=n)    
    ext_set = [{'SMILES': ext_data['SMILES'][i], 'label': ext_data['Label'][i]} for i in selected_indices]
    return ext_set

def loss_based_selection(ext_data, model, tokenizer, n, device):
    """
    Select n data points from the external dataset based on the prediction loss of the model.
    """
    ext_set = ExternalDataset(ext_data, tokenizer)
    ext_loader = DataLoader(ext_set, batch_size=16, shuffle=False)
    
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in ext_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = (outputs.squeeze() - label)**2
            losses.extend([l.item() for l in loss])
    
    selected_indices = np.argsort(losses)[-n:] # select n data points with the largest loss
    ext_set = [{'SMILES': ext_data['SMILES'][i], 'label': ext_data['Label'][i]} for i in selected_indices]
    return ext_set

def influence_based_selection(ext_data, influences, n):
    """
    Select n data points from the external dataset based on the influence scores.
    They are selected corresponding to the n smallest influences scores.
    This is because dL/de = I_up_loss. 
    In the paper, in 4.1, they used I_up,loss to represent the influence of removing 
    a training point on the loss at test point. We have de approximates -1/n, 
    then dL = (-1/n)I_up_loss. Meanwhile, we always want dL to decrease after removing
    a training point, so we should remove training points with large I_up_loss.

    Reversely, in this case, we use I_up_loss as the influence of adding (not removing 
    as in the paper) an external data point on the loss at a test data point. 
    Now de approximates 1/n > 0 (adding data), then dL = 1/n * I_up_loss. 
    So for smaller dL, we want I_up_loss to be as small as possible. 
    """
    selected_indices = np.argsort(influences)[:n]
    ext_set = [{'SMILES': ext_data['SMILES'][i], 'label': ext_data['Label'][i]} for i in selected_indices]
    return ext_set

def generate_method(choice, ext_data, model, tokenizer, influences=None, fraction=10, device=torch.device("cpu")):
    allowed_methods = ["random", "loss_based", "influence_based"]
    n = np.floor(fraction * len(ext_data))

    if choice not in allowed_methods:
        print("Not a valid method, try again!")
        exit(-1)
    elif choice == allowed_methods[0]:
        ext_set = random_selection(ext_data, n)
        return ext_set
    elif choice == allowed_methods[1]:
        ext_set = loss_based_selection(ext_data, model, tokenizer, n, device)
        return ext_set
    elif choice == allowed_methods[2]:
        ext_set = influence_based_selection(ext_data, influences, n)
        return ext_set

if __name__ == "__main__":
    DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
    MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"
    CHOICE = "loss_based"
    FRACTION = 50

    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)
    
    # original data preparation
    dataset = load_dataset(DATASET_PATH)
    train_indices, test_indices = train_test_split(range(len(dataset['train'])), test_size=0.2, random_state=42)
    train_set = Subset(dataset['train'], train_indices)
    test_set = Subset(dataset['train'], test_indices)
    
    train_dataset    = SMILESDataset(train_set, tokenizer)
    test_dataset     = SMILESDataset(test_set, tokenizer)
    print(f"Train dataLoader with   {len(train_dataset)} data points created.") # for Hessian
    print(f"Test dataLoader with    {len(test_dataset)} data points created.")  # for z_test

    # model preparation for data selection
    model = AutoModel.from_pretrained("../notebooks/postMLM(3)-model", deterministic_eval=True, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_for_data_selection = MoLFormerWithRegressionHead(model).to(device)
    model_for_data_selection.regression_head.load_state_dict(torch.load("../notebooks/postMLM(3)-model/postMLM(3)_head.pth", weights_only=True))

    # load external data and select data
    ext_data = pd.read_csv("../tasks/External-Dataset_for_Task2.csv")
    ext_set = generate_method(CHOICE, ext_data, model_for_data_selection, tokenizer, fraction=FRACTION, device=device)
    external_dataset = SMILESDataset(ext_set, tokenizer)
    print(f"External dataLoader with {len(external_dataset)} data points created.") # for \nabla L(z,\theta)

    # merge external data with training data
    merged_dataset = ConcatDataset([train_dataset, external_dataset])
    print(f"Merged dataLoader with   {len(merged_dataset)} data points created.") # for \nabla L(z,\theta)

    BATCH_SIZE = 16
    merged_loader   = DataLoader(merged_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader     = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # prepare model for training
    # model preparation
    model = AutoModel.from_pretrained("../notebooks/finetuned-mlm(3)-model", deterministic_eval=True, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regression_model = MoLFormerWithRegressionHead(model).to(device)
    # regression_model.regression_head.load_state_dict(torch.load("../notebooks/postMLM(3)-model/postMLM(3)_head.pth", weights_only=True))

    # start training
    num_epochs = 100
    save_name = CHOICE + "_" + str(FRACTION) + "_finetunedMLM"
    supervised_training(regression_model,
                        merged_loader,
                        test_loader, 5e-5,
                        num_epochs, "NNTI-Task1",
                        save_name, device)