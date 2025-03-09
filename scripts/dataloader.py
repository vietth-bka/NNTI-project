import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import pandas as pd
import random

class SMILESDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        smiles = self.dataset[idx]['SMILES']        

        encoded = self.tokenizer(smiles, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
        encoded = {k: v.squeeze() for k, v in encoded.items()}

        return {**encoded, 'labels': torch.tensor(self.dataset[idx]['label'], dtype=torch.float32)}
        

class ExternalDataset(Dataset):
    def __init__(self, ext_data, tokenizer):
        self.dataset = ext_data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        smiles = self.dataset['SMILES'][idx]

        encoded = self.tokenizer(smiles, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
        encoded = {k: v.squeeze() for k, v in encoded.items()}

        return {**encoded, 'labels': torch.tensor(self.dataset['Label'][idx], dtype=torch.float32)}
