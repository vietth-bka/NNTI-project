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

########################################################
# Entry point

def get_flat_grad(loss, model):
    """
    Computes the flattened gradient of a training loss w.r.t. model parameters.
    
    Args:
        loss: A scalar training loss value or a loss vector.
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
    v = v.data
    grad = get_flat_grad(loss, model)
    # grad_dot_v = torch.dot(grad, v)
    # print(v.shape, grad.shape, loss.dim())
    grad_dot_v = torch.matmul(v, grad)
    output = get_flat_grad(grad_dot_v, model)
    return output.data


def LiSSA_iHVP(train_loss, model, v, alpha=0.04, damp=0.01, tol=1e-5):
    """
    Approximates the inverse Hessian-vector product H^{-1}v using LiSSA.
    
    Args:
        train_loss: A list of scalar training losses.
        model: Any neural network model.
        v: A flattened vector (e.g., gradient of a training loss) to multiply with H^{-1}.
        Here it will be the gradient of the loss at a test point.
        alpha: A small step size.
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
            u = u_next.data
            break
        u = u_next.data
        
    return u.data

def s_test_single(train_loader, model, v, r, recursion_depth=16):
    """
    Calculates s_test for a single test image taking into account the whole
    training dataset. s_test = invHessian * nabla(Loss(test_img, model params))

    Arguments:
        train_loader: load the train data to compute unbias estimator of Hessian
        r: number of iterations of which to take the avg
        recursion_depth: recursion depth of LiSSA, the higher the better approximation
        as number of iterations * recursion_depth should equal the training dataset size
    Returns:
        s_test: torch tensor, contains s_test for a single test image
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    s_test = torch.zeros_like(v).to(device) # ultilizing vectorization, computing multiple s_test at once

    model.train()
    print("Computing s_test single, batchsize: ", train_loader.batch_size)
    for _ in tqdm(range(r)):
        # data = next(iter(train_loader))
        # input_ids = data['input_ids'].to(device)
        # attention_mask = data['attention_mask'].to(device)
        # label = data['labels'].to(device)
        # outputs = model(input_ids, attention_mask)
        # batch_losses = (outputs.squeeze() - label)**2
        batch_losses = [] # batch computation doesn't work for large recursion_depth
        for _ in range(recursion_depth):
            data = next(iter(train_loader))
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            label = data['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            batch_losses.append(((outputs.squeeze() - label)**2).squeeze())
            assert label.shape[0] == 1, "Batch size of train_loader should be 1 for memory efficiency."
        s_test += LiSSA_iHVP(batch_losses, model, v)
    
    s_test /= r # averaging out all s_test
    return s_test


def compute_influence_per_test(ext_loader, model, s_test):
    """
    Computes the influence function for all trained images given a test point.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    print("Computing influence per test img ..")
    influence_row = []
    for data in ext_loader: # ext_loader is not shuffled
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        label = data['labels'].to(device)
        outputs = model(input_ids, attention_mask)
        batch_losses = (outputs.squeeze() - label)**2
        
        grads = get_flat_grad(batch_losses, model) # gradient of the loss at a training point
        influence_mat = -torch.matmul(s_test, grads.T) # negative sign as in the formula of I_up,loss.
        influence_mat = influence_mat.cpu().detach().numpy()
        influence_row.extend(influence_mat.sum(0))
    assert len(influence_row) == len(ext_loader.dataset), f"dimensional mismatch {len(influence_row)} vs {len(ext_loader.dataset)}"
    
    return torch.tensor(influence_row).to(device)


def compute_influences(test_loader, train_loader, ext_loader, model, r=10, recursion_depth=16):
    """
    Calculates s_test for all test images 
    then for each test image, compute influences of all training points
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    print("Computing influence ..")
    influences = torch.zeros(len(ext_loader.dataset)).to(device)
    for data in test_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        label = data['labels'].to(device)
        outputs = model(input_ids, attention_mask)
        batch_losses = (outputs.squeeze() - label)**2
        v = get_flat_grad(batch_losses, model) # gradients of the losses at a test point, 
        # larger batch size -> faster vectorization but larger size of v => memory issues
        s_test = s_test_single(train_loader, model, v, r, recursion_depth)

        # loops in the loops to avoid memory issues
        influence_row = compute_influence_per_test(ext_loader, model, s_test) # given s_test of a test img, compute influences of all external imgs
        influence_row = influence_row / len(test_loader.dataset) * (1 / 3360) 
        # divide by len(test_loader.dataset) to get the average influence of a specific external img on test imgs
        # 3360 is the size of the training dataset of the pretrained model
        influences += influence_row
        print("influences:", influences)
    
    return influences.cpu().detach().numpy()

########################################################

if __name__ == "__main__":
    # TODO: your code goes here
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)
    
    dataset = load_dataset(DATASET_PATH)
    train_indices, test_indices = train_test_split(range(len(dataset['train'])), test_size=0.005, random_state=42)
    train_set = Subset(dataset['train'], train_indices)
    test_set = Subset(dataset['train'], test_indices)
    train_dataset    = SMILESDataset(train_set, tokenizer)
    test_dataset     = SMILESDataset(test_set, tokenizer)
    external_dataset = ExternalDataset(ext_data, tokenizer)

    print(f"Train DataLoader with {len(train_dataset)} data points created.")
    print(f"Test  DataLoader with {len(test_dataset)} data points created.")
    print(f"External DataLoader with {len(external_dataset)} data points created.")
    
    BATCH_SIZE = 1 # adjust based on memory constraints
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    ext_loader   = DataLoader(external_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = AutoModel.from_pretrained("../postMLM-model", trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regression_model = MoLFormerWithRegressionHead(model).to(device)
    regression_model.regression_head.load_state_dict(torch.load("../postMLM-model/postMLM_head.pth", weights_only=True))

    all_influences = compute_influences(test_loader, train_loader, ext_loader, regression_model, r=40)
    print(all_influences)