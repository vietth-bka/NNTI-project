# import dependencies
import torch
import sklearn
import datasets
import numpy as np
import transformers
import pandas as pd
from tqdm import tqdm

DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

ext_data = pd.read_csv("../tasks/External-Dataset_for_Task2.csv")

########################################################
# Entry point
########################################################

if __name__ == "__main__":
    # TODO: your code goes here
    pass


