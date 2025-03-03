# import dependencies
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataloader import SMILESDataset, ExternalDataset
from model import MoLFormerWithRegressionHead

DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

ext_data = pd.read_csv("../tasks/External-Dataset_for_Task2.csv")

########################################################
# Entry point

def get_flat_grad(loss, model):
    """
    Computes the flattened gradient of a training loss w.r.t. model parameters.
    
    Args:
        loss: A scalar training loss value.
        model: Any neural network model.
    
    Returns:
        A flattened vector of the gradient of the training loss.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if loss.dim() == 0:
        grad = torch.autograd.grad(loss, params, create_graph=True)
        out_grad = torch.cat([e.contiguous().view(-1) for g in grad for e in g])
        return out_grad
    else:
        grads = [torch.autograd.grad(l, params, create_graph=True) for l in loss]
        out_grads = []        
        for grad in grads:
            out_grad = torch.cat([e.contiguous().view(-1) for g in grad for e in g])
            out_grads.append(out_grad)
        # print("Test:", torch.vstack(out_grads).shape)
        return torch.vstack(out_grads)


def hessian_vector_product(loss, model, v):
    """
    Computes the Hessian-vector product H * v.
    
    Args:
        loss: A scalar training loss value.
        model: Any neural network model.
        v: A flattened vector (e.g., gradient of a training loss) to multiply with H.
    
    Returns:
        The Hessian-vector product H * v.
    """
    grad = get_flat_grad(loss, model)
    # print(grad.shape, v.shape)
    # grad_dot_v = torch.dot(grads, v)
    grad_dot_v = torch.matmul(v, grad)
    # print("grad_dot_v:", grad_dot_v.shape)
    output = get_flat_grad(grad_dot_v, model)
    return output.clone().detach()


def LiSSA_iHVP(train_loss, model, v, alpha=0.04, damp=0.01, recursion_depth=100, tol=1e-5):
    """
    Approximates the inverse Hessian-vector product H^{-1}v using LiSSA.
    
    Args:
        train_loss: A list of scalar training losses.
        model: Any neural network model.
        v: A flattened vector (e.g., gradient of a training loss) to multiply with H^{-1}.
        Here it will be the gradient of the loss at a test point.
        alpha: A small step size.
        recursion_depth: Maximum number of iterations.
        tol: Tolerance for convergence.
    
    Returns:
        An approximation of H^{-1}v denoted by u.
    """
    # Initialize the approximation: u = v
    u = v.clone().detach()
    v = v.clone().detach()

    for loss in train_loss:
        # LiSSA update: u_next = v + (I - H(z))u
        u_next = v + (1-damp) * u - alpha * hessian_vector_product(loss, model, u)
        if torch.norm(u_next - u) < tol:
            u = u_next
            break
        u = u_next
        
    return u.clone().detach()

def s_test_single(train_loader, model, v):
    """
    Calculates s_test for a single test image taking into account the whole
    training dataset. s_test = invHessian * nabla(Loss(test_img, model params))

    Arguments:
        train_loader: pytorch dataloader, which can load the train data
        len(train_loader): number of iterations of which to take the avg.
        recursion_depth: batch-size
        as number of iterations * recursion_depth should equal the training dataset size
    Returns:
        s_test: torch tensor, contains s_test for a single test image
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    s_test = torch.zeros(train_loader.batch_size, num_params).to(device) # ultilizing vectorization, computing multiple s_test at once

    model.train()
    print("Computing s_test single")
    for data in tqdm(train_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        label = data['labels'].to(device)
        outputs = model(input_ids, attention_mask)
        batch_losses = (outputs.squeeze() - label)**2

        s_test += LiSSA_iHVP(batch_losses, model, v)
    
    s_test /= len(train_loader) # averaging out all s_test
    return s_test


def compute_influence_per_test(train_loader, model, s_test):
    """
    Computes the influence function for all trained images given a test point.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    print("Computing influence per test img ..")
    influence_row = []
    for data in train_loader: # loops in the loops to avoid memory issues
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        label = data['labels'].to(device)
        outputs = model(input_ids, attention_mask)
        batch_losses = (outputs.squeeze() - label)**2
        
        grads = get_flat_grad(batch_losses, model) # gradient of the loss at a training point
        influence_mat = -torch.matmul(s_test, grads.T) # negative sign as in the formula of I_up,loss.
        influence_row.extend(influence_mat.mean(0))
    
    return torch.tensor(influence_row).to(device)


def compute_influences(test_loader, train_loader, model):
    """
    Calculates s_test for all test images 
    then for each test image, compute influences of all training points
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    print("Computing influence ..")
    influences = torch.zeros(len(train_loader.dataset)).to(device)
    for data in tqdm(test_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        label = data['labels'].to(device)
        outputs = model(input_ids, attention_mask)
        batch_losses = (outputs.squeeze() - label)**2
        v = get_flat_grad(batch_losses, model) # gradients of the losses at a test point
        s_test = s_test_single(train_loader, model, v)

        # loops in the loops to avoid memory issues
        influence_row = compute_influence_per_test(train_loader, model, s_test)
        # influence_row = influence_row / len(test_loader.dataset) * (-1 / len(train_loader.dataset)) 
        influence_row = influence_row / len(test_loader.dataset) * (-1 / 3360) # 3360 is the size of the training dataset of the pretrained model
        influences += influence_row # for a given test img, each row rep
        print("influences:", influences)
    
    return influences.cpu().detach().numpy()

########################################################

if __name__ == "__main__":
    # TODO: your code goes here
    dataset = load_dataset(DATASET_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)
    
    train_dataset = ExternalDataset(ext_data, tokenizer)
    _, test_indices = train_test_split(range(len(dataset['train'])), test_size=0.2, random_state=42)
    test_set = Subset(dataset['train'], test_indices)
    test_dataset = SMILESDataset(test_set, tokenizer)

    print(f"Train DataLoader with {len(train_dataset)} data points created.")
    print(f"Test  DataLoader with {len(test_dataset)} data points created.")
    
    BATCH_SIZE = 2 # adjust based on memory constraints
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False) # shuffle=False to keep the order
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = AutoModel.from_pretrained("../notebooks/postMLM-model", trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regression_model = MoLFormerWithRegressionHead(model).to(device)
    regression_model.regression_head.load_state_dict(torch.load("../notebooks/postMLM-model/postMLM_head.pth", weights_only=True))

    all_influences = compute_influences(test_loader, train_loader, regression_model)
    print(all_influences)