import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import get_scheduler
import wandb

def supervised_training(regression_model, train_loader, test_loader, lr, num_epochs, project, save_name, device):
    """
    Train the regression model with supervised learning using early stop.
    """
    print(f"Training {save_name} model ...")
    run = wandb.init(
        # Set the wandb name.
        name=save_name,
        # Set the wandb project where this run will be logged.
        project=project,
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": lr,
            "architecture": "MoLFormer",
            "dataset": "SMILES",
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
        },
    )
    optimizer = torch.optim.AdamW(regression_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss, count = float('inf'), 0

    for epoch in range(num_epochs):
        regression_model.train()
        train_loss = 0

        for data in tqdm(train_loader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            label = data['labels'].to(device)
            outputs = regression_model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * label.shape[0]
        
        epoch_loss = train_loss / len(train_loader.dataset)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            count = 0    
            regression_model.model.save_pretrained(f"./{save_name}-model")
            torch.save(regression_model.regression_head.state_dict(), f"./{save_name}-model/{save_name}_head.pth")
        else:
            count += 1

        print(f"Sup: Epoch {epoch+1}, Loss: {epoch_loss}, Count: {count}")

        ###TODO: Evaluation
        regression_model.eval()
        test_loss = 0

        with torch.no_grad():
            for data in tqdm(test_loader):
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                label = data['labels'].to(device)
                outputs = regression_model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), label)
                test_loss += loss.item() * label.shape[0]

        print(f"Sup: Test Loss: {test_loss / len(test_loader.dataset)}")
        run.log({"train_loss": epoch_loss, 
                 "test_loss": test_loss / len(test_loader.dataset),
                 "learning rate": optimizer.param_groups[0]['lr']})
        
        if count >= 10: # early stop
            print("Early stop !")
            run.finish()
            break
    
    run.finish()

def unsupervised_learning(unsup_model, train_loader, num_epochs, save_name, lr, device):
    """
    Train the masked language model with unsupervised learning using early stop and lr scheduler.
    """
    run = wandb.init(
        # Set the wandb name.
        name=save_name,
        # Set the wandb project where this run will be logged.
        project="NNTI-Task1-unsup",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": lr,
            "architecture": "MLM-MoLFormer",
            "dataset": "SMILES",
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
        },
    )

    optimizer = torch.optim.AdamW(unsup_model.parameters(), lr=lr)
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    unsup_model.train()
    best_loss = float('inf')
    count = 0

    for epoch in range(num_epochs):
        total_loss = 0

        for data in tqdm(train_loader):
            data = {k: v.to(device) for k, v in data.items()}
            outputs = unsup_model(**data)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * data['input_ids'].shape[0]

        epoch_loss = total_loss / len(train_loader.dataset)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            count = 0
            #save model
            unsup_model.save_pretrained(f"./{save_name}-model")
            print("Model saved ...")
        else:
            count += 1

        print(f"MLM: Epoch {epoch+1}, Loss: {epoch_loss}, Count: {count}")
        run.log({"train_loss": epoch_loss, "learning rate": optimizer.param_groups[0]['lr']})

        if count >= 10: # early stop
            print("Early stop !")
            run.finish()
            break
    
    run.finish()


def supervised_training_lr_scheduler(regression_model, train_loader, test_loader, lr, num_epochs, project, save_name, device):
    """
    Train the regression model with supervised learning using early stop.
    """
    print(f"Training {save_name} model ...")
    run = wandb.init(
        # Set the wandb name.
        name=save_name,
        # Set the wandb project where this run will be logged.
        project=project,
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": lr,
            "architecture": "MoLFormer",
            "dataset": "SMILES",
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
        },
    )
    optimizer = torch.optim.AdamW(regression_model.parameters(), lr=lr)
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    criterion = nn.MSELoss()
    best_loss, count = float('inf'), 0

    for epoch in range(num_epochs):
        regression_model.train()
        train_loss = 0

        for data in tqdm(train_loader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            label = data['labels'].to(device)
            outputs = regression_model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * label.shape[0]
        
        epoch_loss = train_loss / len(train_loader.dataset)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            count = 0    
            regression_model.model.save_pretrained(f"./{save_name}-lrs-model")
            torch.save(regression_model.regression_head.state_dict(), f"./{save_name}-lrs-model/{save_name}_head.pth")
        else:
            count += 1

        print(f"Sup: Epoch {epoch+1}, Loss: {epoch_loss}, Count: {count}")

        ###TODO: Evaluation
        regression_model.eval()
        test_loss = 0

        with torch.no_grad():
            for data in tqdm(test_loader):
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                label = data['labels'].to(device)
                outputs = regression_model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), label)
                test_loss += loss.item() * label.shape[0]

        print(f"Sup: Test Loss: {test_loss / len(test_loader.dataset)}")
        run.log({"train_loss": epoch_loss, 
                 "test_loss": test_loss / len(test_loader.dataset),
                 "learning rate": optimizer.param_groups[0]['lr']})
        
        if count >= 10: # early stop
            print("Early stop !")
            run.finish()
            break
    
    run.finish()


def supervised_training_lrs_val(regression_model, train_loader, val_loader, test_loader, lr, num_epochs, project, save_name, device):
    """
    Train the regression model with supervised learning using early stop.
    """
    print(f"Training {save_name} model ...")
    run = wandb.init(
        # Set the wandb name.
        name=save_name,
        # Set the wandb project where this run will be logged.
        project=project,
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": lr,
            "architecture": "MoLFormer",
            "dataset": "SMILES",
            "epochs": num_epochs,
            "num_data_train": len(train_loader.dataset),
            "batch_size": train_loader.batch_size,
        },
    )
    optimizer = torch.optim.AdamW(regression_model.parameters(), lr=lr)
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    criterion = nn.MSELoss()
    best_loss, count = float('inf'), 0

    for epoch in range(num_epochs):
        regression_model.train()
        train_losses = 0.0

        for data in tqdm(train_loader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            label = data['labels'].to(device)
            outputs = regression_model(input_ids, attention_mask)
            train_loss = criterion(outputs.squeeze(), label)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses += train_loss.item() * label.shape[0]
        
        # validating
        regression_model.eval()
        val_losses = 0
        with torch.no_grad():
            for data in tqdm(val_loader):
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                label = data['labels'].to(device)
                outputs = regression_model(input_ids, attention_mask)
                val_loss = criterion(outputs.squeeze(), label)
                val_losses += val_loss.item() * label.shape[0]

        epoch_val_loss = val_losses / len(val_loader.dataset)
        
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            count = 0
            regression_model.model.save_pretrained(f"./{save_name}-lrs-model")
            torch.save(regression_model.regression_head.state_dict(), f"./{save_name}-lrs-model/{save_name}_head.pth")
        else:
            count += 1

        print(f"Epoch {epoch+1}, Loss: {train_losses/len(train_loader.dataset)}, \
                                 Val loss: {epoch_val_loss},\
                                 Count: {count}")

        ### testing
        regression_model.eval()
        test_losses = 0
        with torch.no_grad():
            for data in tqdm(test_loader):
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                label = data['labels'].to(device)
                outputs = regression_model(input_ids, attention_mask)
                test_loss = criterion(outputs.squeeze(), label)
                test_losses += test_loss.item() * label.shape[0]

        print(f"Test Loss: {test_losses / len(test_loader.dataset)}")
        run.log({"train_loss": train_losses/len(train_loader.dataset), 
                 "val_loss": epoch_val_loss,
                 "test_loss": test_losses / len(test_loader.dataset),
                 "learning rate": optimizer.param_groups[0]['lr']})
        
        if count >= 10: # early stop
            print("Early stop !")
            run.finish()
            break
    
    run.finish()