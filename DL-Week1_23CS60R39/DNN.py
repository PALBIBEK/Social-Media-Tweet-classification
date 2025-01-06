import optuna
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import pickle

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        

device = "cuda" if torch.cuda.is_available() else "cpu"

class CustomDataset(Dataset):
    def __init__(self, df):
        self.x = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)  
        self.y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)     
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class Network(nn.Module):
    def __init__(self, input_dim=12572, output_dim=1, activation_1='ReLU', activation_2='ReLU', units_layer1=6, units_layer2=6, state_dict=None):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, units_layer1)
        self.activation_1 = getattr(nn, activation_1)()
        self.linear_2 = nn.Linear(units_layer1, units_layer2)
        self.activation_2 = getattr(nn, activation_2)()
        self.linear_3 = nn.Linear(units_layer2, output_dim)
        if state_dict is not None:
            self.load_state_dict(state_dict)

    def forward(self, x):
        out = self.linear_1(x)
        out = self.activation_1(out)
        out = self.linear_2(out)
        out = self.activation_2(out)
        out = self.linear_3(out)
        return out

if __name__=="__main__":
    with open('train-dnn.pkl', 'rb') as f:
        train_df = pickle.load(f)
    with open('val-dnn.pkl', 'rb') as f:
        val_df = pickle.load(f)

    train_dataset = CustomDataset(train_df)
    val_dataset = CustomDataset(val_df)


    def objective(trial):
        # Sample hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1,log=True)
        units_layer1 = trial.suggest_int('units_layer1', 4, 16)
        units_layer2 = trial.suggest_int('units_layer2', 4, 16)
        batch_size   = batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        activation_1 = trial.suggest_categorical('activation_1', ['ReLU', 'LeakyReLU'])
        activation_2 = trial.suggest_categorical('activation_2', ['ReLU', 'LeakyReLU'])

        # Define and set hyperparameters
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        epochs = trial.suggest_int('epochs', 5, 10)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training loop
        train_epoch_loss = []
        eval_epoch_loss = []
        for epoch in tqdm(range(epochs)):
            network.train()
            curr_loss = 0
            total = 0
            for train_x, train_y in train_dataloader:
                train_x = train_x.to(device)
                train_y = train_y.unsqueeze(1).to(device)
                optimizer.zero_grad()

                y_pred = network(train_x)
                loss = criterion(y_pred, train_y.float())

                loss.backward()
                optimizer.step()

                curr_loss += loss.item()
                total += len(train_y)
            train_epoch_loss.append(curr_loss / total)

            # Evaluation loop
            network.eval()
            curr_loss = 0
            total = 0
            for val_x, val_y in val_dataloader:
                val_x = val_x.to(device)
                val_y = val_y.unsqueeze(1).to(device)
                with torch.no_grad():
                    y_pred = network(val_x)

                loss = criterion(y_pred, val_y.float())

                curr_loss += loss.item()
                total += len(val_y)
            eval_epoch_loss.append(curr_loss / total)

        
        return eval_epoch_loss[-1]


    criterion = nn.BCEWithLogitsLoss()
    network = Network().to(device)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15) 

    best_params = study.best_params
    best_learning_rate = best_params['learning_rate']
    best_units_layer1 = best_params['units_layer1']
    best_units_layer2 = best_params['units_layer2']
    best_activation_1 = best_params['activation_1']
    best_activation_2 = best_params['activation_2']
    best_batch_size   = best_params['batch_size']
    best_epochs   = best_params['epochs']

    #best hyperparameters for final training
    network = Network(input_dim=len(train_dataset[0][0]), output_dim=1, activation_1=best_activation_1, activation_2=best_activation_2, units_layer1=best_units_layer1, units_layer2=best_units_layer2).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=best_learning_rate)
    batch_size = best_batch_size
    epochs = best_epochs
    train_dataloader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
    for epoch in tqdm(range(epochs)):
        network.train()
        curr_loss = 0
        total = 0
        for train_x, train_y in train_dataloader:
            train_x = train_x.to(device)
            train_y = train_y.unsqueeze(1).to(device)
            optimizer.zero_grad()
            y_pred = network(train_x)
            loss = criterion(y_pred, train_y.float())
            loss.backward()
            optimizer.step()
            
    model_state_dict = network.state_dict()

    model_info = {
        'input_dim': len(train_dataset[0][0]),
        'output_dim': 1,
        'activation_1': best_activation_1,
        'activation_2': best_activation_2,
        'units_layer1': best_units_layer1,
        'units_layer2': best_units_layer2,
        'state_dict': model_state_dict
    }
    directory_name = 'saved models'
    create_directory(directory_name)
    model_save_name = f'dnn.pth'
    path = f"{directory_name}/{model_save_name}"
    torch.save(model_info, path)

