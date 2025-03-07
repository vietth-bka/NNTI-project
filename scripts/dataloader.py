import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import pandas as pd
import random
from rdkit import Chem

def augment_smiles(smiles, augment_prob=0.5):
    """
    Augment a SMILES string by randomly generating an alternative
    SMILES representation using RDKit, with probability augment_prob.
    If augmentation fails, return the original SMILES.
    """
    if random.random() < augment_prob:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            # Generate a random SMILES string (different atom ordering)
            return Chem.MolToSmiles(mol, doRandom=True)
        except Exception as e:
            return smiles
    else:
        return smiles

class SMILESDataset(Dataset):
    def __init__(self, dataset, tokenizer, augment=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        smiles = self.dataset[idx]['SMILES']

        if self.augment:
            smiles = augment_smiles(smiles)            

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
