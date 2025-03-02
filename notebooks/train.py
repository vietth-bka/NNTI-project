import torch
from datasets import load_dataset
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import random

# specify dataset name and model name
DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"  #MoLFormer model

# load the dataset from HuggingFace
dataset = load_dataset(DATASET_PATH)

# Explore the dataset
# For example, print the column names and display a few sample rows
# TODO: your code goes here
print(dataset.column_names)
print(len(dataset['train']['SMILES']))
print(len(dataset['train']['label']))
print(dataset['train'][0])

# define a PyTorch Dataset class for handling SMILES strings and targets

# TODO: your code goes here
class SMILESDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        smiles = self.dataset[idx]['SMILES']

        encoded = self.tokenizer(smiles, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
        encoded = {k: v.squeeze() for k, v in encoded.items()}

        return {**encoded, 'labels': torch.tensor(self.dataset[idx]['label'], dtype=torch.float32)}
    

# tokenize the data
# load a pre-trained tokenizer from HuggingFace
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)

encode = tokenizer(dataset['train'][0]['SMILES'], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
encode['attention_mask'].shape

len(dataset['train'])

# split the data into training and test datasets
# TODO: your code goes here
train_indices, test_indices = train_test_split(range(len(dataset['train'])), test_size=0.2, random_state=42)
train_set = Subset(dataset['train'], train_indices)
test_set = Subset(dataset['train'], test_indices)

train_dataset = SMILESDataset(train_set, tokenizer)
test_dataset = SMILESDataset(test_set, tokenizer)

print(f"Train DataLoader with {len(train_dataset)} data points created.")
print(f"Test DataLoader with {len(test_dataset)} data points created.")

# construct Pytorch data loaders for both train and test datasets
BATCH_SIZE = 16 # adjust based on memory constraints

# TODO: your code goes here
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# load pre-trained model from HuggingFace
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)

# We need to add a regression head on the language model as we are doing a regression task.

# specify model with a regression head

class MoLFormerWithRegressionHead(nn.Module):
    # TODO: your code goes here
    def __init__(self, model):
        super(MoLFormerWithRegressionHead, self).__init__()
        self.model = model
        self.hidden_size = model.config.hidden_size
        self.regression_head = nn.Linear(self.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        # cls_token = outputs.last_hidden_state[:, 0, :]
        sequence_output = outputs[0]
        cls_token = sequence_output[:, 0, :]
        outputs_head = self.regression_head(cls_token)
        return outputs_head
# instantiate the model
# model = MoLFormerWithRegressionHead(model)

# initialize the regression model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
regression_model = MoLFormerWithRegressionHead(model).to(device)

############# TODO: your code goes here: supervised training #############
num_epochs = 100
optimizer = torch.optim.AdamW(regression_model.parameters(), lr=5e-5)
criterion = nn.MSELoss()
best_loss, count = float('inf'), 0

for epoch in range(num_epochs):
    regression_model.train()
    total_loss = 0

    for data in tqdm(train_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        label = data['labels'].to(device)
        optimizer.zero_grad()
        outputs = regression_model(input_ids, attention_mask)
        loss = criterion(outputs.squeeze(), label)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * label.shape[0]
    
    epoch_loss = total_loss / len(train_dataset)
    
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        count = 0    
        regression_model.model.save_pretrained("./baseline-model")
        torch.save(regression_model.regression_head.state_dict(), "./baseline-model/baseline_head.pth")
    else:
        count += 1

    print(f"SUP: Epoch {epoch+1}, Loss: {total_loss / len(train_dataset)}, Count: {count}")

    ###TODO: Evaluation
    regression_model.eval()
    total_loss = 0

    with torch.no_grad():
        for data in tqdm(test_loader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            label = data['labels'].to(device)
            outputs = regression_model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), label)
            total_loss += loss.item() * label.shape[0]

    print(f"SUP: Test Loss: {total_loss / len(test_dataset)}")

    if count == 10: # early stop
        print("Early stop !")
        break

######### TODO: your code goes here: unsupervised training #########
# from transformers import get_scheduler

# # unlabel_dataset = SMILESDataset(train_set, tokenizer, False)
# # print(f"Train DataLoader with {len(unlabel_dataset)} unsupervised data points created.")

# unsup_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME,
#                                                    deterministic_eval=True,
#                                                    trust_remote_code=True).to(device)

# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
#                                                 mlm=True,
#                                                 mlm_probability=0.15)

# train_dataloader = DataLoader(train_dataset,
#                               batch_size=16,
#                               shuffle=True,
#                               collate_fn=data_collator)

# optimizer = torch.optim.AdamW(unsup_model.parameters(), lr=5e-5)
# num_epochs = 100
# num_training_steps = num_epochs * len(train_dataloader)
# scheduler = get_scheduler(
#     name="linear",
#     optimizer=optimizer,
#     num_warmup_steps=int(0.1 * num_training_steps),
#     num_training_steps=num_training_steps
# )

# unsup_model.train()
# best_loss = float('inf')
# count = 0

# for epoch in range(num_epochs):
#     total_loss = 0

#     for data in tqdm(train_dataloader):
#         data = {k: v.to(device) for k, v in data.items()}
#         outputs = unsup_model(**data)
#         loss = outputs.loss

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#         total_loss += loss.item() * data['input_ids'].shape[0]

#     epoch_loss = total_loss / len(train_dataset)

#     if epoch_loss < best_loss:
#         best_loss = epoch_loss
#         count = 0
#         #save model
#         unsup_model.save_pretrained("./finetuned-mlm-model")
#         tokenizer.save_pretrained("./finetuned-mlm-token")
#         print("Model saved ...")
#     else:
#         count += 1

#     print(f"MLM: Epoch {epoch+1}, Loss: {epoch_loss}, Count: {count}")

#     if count == 10: # early stop
#         print("Early stop !")
#         break

######## TODO: your code goes here for fine-tuning the model MLM ########

finetuned_mlm_model = AutoModel.from_pretrained(
    "./finetuned-mlm-model",
    deterministic_eval=True,
    trust_remote_code=True,
)

finetune_model = MoLFormerWithRegressionHead(finetuned_mlm_model).to(device)

num_epochs = 100
optimizer = torch.optim.AdamW(finetune_model.parameters(), lr=5e-5)
criterion = nn.MSELoss()
best_loss = float('inf')
count = 0

for epoch in range(num_epochs):
    finetune_model.train()
    train_loss = 0

    for data in tqdm(train_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        label = data['labels'].to(device)
        outputs = finetune_model(input_ids, attention_mask)
        loss = criterion(outputs.squeeze(), label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * label.shape[0]

    epoch_loss = train_loss / len(train_dataset)

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        count = 0
        finetune_model.model.save_pretrained("./postMLM-model")
        torch.save(finetune_model.regression_head.state_dict(), "./postMLM-model/postMLM_head.pth")
    else:
        count += 1
    
    print(f"FT: Epoch {epoch+1}, Loss: {train_loss / len(train_dataset)}, Count: {count}")

    finetune_model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            label = data['labels'].to(device)
            outputs = finetune_model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), label)
            test_loss += loss.item() * label.shape[0]

        print(f"FT: Test Loss: {test_loss / len(test_dataset)}")

    if count >= 10: # early stop
        print("Early stop !")
        break