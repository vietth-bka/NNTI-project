# dev: Hong-Viet Tran
import torch
from torch.utils.data import DataLoader, Dataset, Subset, SequentialSampler
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

from dataloader import SMILESDataset, ExternalDataset
from model import MoLFormerWithRegressionHead

DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

ext_data = pd.read_csv("../tasks/External-Dataset_for_Task2.csv")

def random_selection(ext_data, n):
    """
    Randomly select n data points from the external dataset.
    """
    selected_indices = random.choices(range(len(ext_data)), k=n)
    ext_set = Subset(ext_data, selected_indices)
    return ext_set

def loss_based_selection(ext_data, model, tokenizer, n, device):
    """
    Select n data points from the external dataset based on the prediction loss of the model.
    """
    ext_set = ExternalDataset(ext_data, tokenizer)
    ext_loader = DataLoader(ext_set, batch_size=16, shuffle=False)
    
    model.eval()
    losses = []
    for batch in ext_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = (outputs.squeeze() - label)**2
            loss = [l.item() for l in loss]
            losses.extend(loss)
    
    selected_indices = np.argsort(losses)[-n:] # select n data points with the largest loss
    ext_set = Subset(ext_data, selected_indices)
    return ext_set

if __name__ == "__main__":
    # TODO: your code goes here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)
    
    dataset = load_dataset(DATASET_PATH)
    train_indices, test_indices = train_test_split(range(len(dataset['train'])), test_size=0.2, random_state=42)
    train_set = Subset(dataset['train'], train_indices)
    test_set = Subset(dataset['train'], test_indices)
    
    train_dataset    = SMILESDataset(train_set, tokenizer)
    test_dataset     = SMILESDataset(test_set, tokenizer)
    external_dataset = ExternalDataset(ext_data, tokenizer)

    print(f"Train    DataLoader with {len(train_dataset)} data points created.") # for Hessian
    print(f"Test     DataLoader with {len(test_dataset)} data points created.")  # for z_test
    print(f"External DataLoader with {len(external_dataset)} data points created.") # for \nabla L(z,\theta)