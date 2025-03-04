import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from sklearn.model_selection import train_test_split

from train import supervised_training, unsupervised_learning

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
model = AutoModel.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)

# We need to add a regression head on the language model as we are doing a regression task.

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


# initialize the regression model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
regression_model = MoLFormerWithRegressionHead(model).to(device)
num_epochs = 100

############# TODO: your code goes here: supervised training   #############
# save_name = "baseline"
# supervised_training(regression_model, 
#                     train_loader, 
#                     test_loader, 
#                     num_epochs, 
#                     save_name, device)
############# TODO: your code goes here: unsupervised training #############
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

# unsupervised_learning(unsup_model, train_dataloader, 100, "finetuned-mlm", device)

############# TODO: your code goes here for fine-tuning the model MLM #############
save_name = "postMLM"
finetuned_mlm_model = AutoModel.from_pretrained(
    "./finetuned-mlm",
    deterministic_eval=True,
    trust_remote_code=True,
)

finetune_model = MoLFormerWithRegressionHead(finetuned_mlm_model).to(device)

supervised_training(finetune_model, 
                    train_loader, 
                    test_loader, 
                    num_epochs, 
                    save_name, device)