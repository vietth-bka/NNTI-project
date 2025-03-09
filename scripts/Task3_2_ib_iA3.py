# import dependencies
import wandb
import torch 
import datasets
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType, IA3Config
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from model import MoLFormerWithRegressionHead, HiddenModel
from Task3_1 import generate_method
from dataloader import SMILESDataset

import pandas as pd
import sys
sys.path.append("../notebooks")
from train import supervised_training, supervised_training_lr_scheduler, supervised_training_lrs_val

MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

########################################################
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def bitfit(model):
    # Unfreeze only bias parameters
    for name, param in model.named_parameters():
        param.requires_grad = True if "bias" in name else False
    print_trainable_parameters(model)
    model = MoLFormerWithRegressionHead(model)
    return model

def LoRA(model):
    peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,            # low-rank dimension
    lora_alpha=16,  # scaling factor
    lora_dropout=0.1,
    bias="none",
    # target_modules=["query", "value", "key", "dense"],
    target_modules=["query", "value"]
    )
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    model = MoLFormerWithRegressionHead(model)
    return model

def iA3(model):
    peft_config = IA3Config(
    task_type="TOKEN_CLS",  # Adjust based on the task (e.g., MASKED_LM)
    target_modules=["key", "value", "dense"], 
    feedforward_modules=["dense"],  # Modules to apply IA3 in FFN
    inference_mode=False,
    )
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    model = MoLFormerWithRegressionHead(model)
    return model
########################################################

if __name__ == "__main__":
    # TODO: your code goes here
    DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
    MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"
    CHOICE = "influence_based"
    FRACTION = 75
    PEFT = "iA3"

    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)
    
    # original data preparation
    dataset = load_dataset(DATASET_PATH)
    train_indices, test_indices = train_test_split(range(len(dataset['train'])), test_size=0.2, random_state=42)
    train_set = [dataset['train'][i] for i in train_indices]
    test_set = [dataset['train'][i] for i in test_indices]
    
    ext_data = pd.read_csv("../tasks/External-Dataset_for_Task2.csv")

    if FRACTION != 0:
        if CHOICE == "loss_based":
            # model preparation for data selection
            model = AutoModel.from_pretrained("../notebooks/postMLM(3)-model", deterministic_eval=True, trust_remote_code=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_for_data_selection = MoLFormerWithRegressionHead(model).to(device)
            model_for_data_selection.regression_head.load_state_dict(torch.load("../notebooks/postMLM(3)-model/postMLM(3)_head.pth", weights_only=True))

            # load external data and select data
            ext_set = generate_method("loss_based", ext_data, model_for_data_selection, tokenizer, fraction=FRACTION, device=device)
        
        elif CHOICE == "random":
            ext_set = generate_method("random", ext_data, fraction=FRACTION)

        elif CHOICE == "influence_based":
            influences = np.load("./influences(3).npy")
            ext_set = generate_method("influence_based", ext_data, influences=influences, fraction=FRACTION)

    else:
        ext_set = []

    subtrain_indices, val_indices = train_test_split(range(len(train_set)), test_size=0.15, random_state=42)
    # actual_train_set = [train_set[i] for i in subtrain_indices] + ext_set
    actual_train_set = ext_set
    val_set = [train_set[i] for i in val_indices]

    train_dataset = SMILESDataset(actual_train_set, tokenizer)
    val_dataset   = SMILESDataset(val_set, tokenizer)
    test_dataset  = SMILESDataset(test_set, tokenizer)

    print("Data added:", len(ext_set), " , total data points:", len(train_set + ext_set))
    print(f"Train dataLoader with    {len(train_dataset)} data points created.")
    print(f"Validate dataLoader with {len(val_dataset)} data points created.")
    print(f"Test dataLoader with     {len(test_dataset)} data points created.") 

    BATCH_SIZE = 16
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # prepare model for training
    # model preparation
    model = AutoModel.from_pretrained("../baseline_0_val_lrs_ftMLM-lrs-model", deterministic_eval=True, trust_remote_code=True)
    model = HiddenModel(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_trainable_parameters(model)

    if PEFT == "bitfit":
        print("\nFine-tuning with bitfit!")
        regression_model = bitfit(model).to(device)
    elif PEFT == "LoRA":
        print("\nFine-tuning with LoRA!")
        regression_model = LoRA(model).to(device)
    elif PEFT == "iA3":
        print("\nFine-tuning with iA3!")
        regression_model = iA3(model).to(device)
    else:
        print("INVALID peft!")
        exit(-1)
    regression_model.regression_head.load_state_dict(torch.load("../baseline_0_val_lrs_ftMLM-lrs-model/baseline_0_val_lrs_ftMLM_head.pth", weights_only=True))
    print(regression_model.regression_head.weight.requires_grad)
    print(regression_model.regression_head.bias.requires_grad)
    # start training
    num_epochs = 100
    save_name = f"{CHOICE}_{PEFT}_{str(FRACTION)}_val_lrs_ftMLM"
    supervised_training_lrs_val(regression_model,
                        train_loader,
                        val_loader,
                        test_loader, 5e-5,
                        num_epochs, "NNTI-Task1",
                        save_name, device, 101)