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
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class Network(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1, dropout=0.2,state_dict=None):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        if state_dict is not None:
            self.load_state_dict(state_dict)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out) 
        out = self.fc(out[:, -1, :])
        return out



class CustomDataset(Dataset):
    def __init__(self, df):
        self.x = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


if __name__=="__main__":

    with open('train.pkl', 'rb') as f:
        train_df = pickle.load(f)
    with open('val.pkl', 'rb') as f:
        val_df = pickle.load(f)

    train_dataset = CustomDataset(train_df)
    val_dataset = CustomDataset(val_df)

    def objective(trial):
       
        hidden_size = trial.suggest_int('hidden_size', 64, 128)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1,log=True)
        epochs = trial.suggest_int('epochs', 5, 10)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        network = Network(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
        network.to(device)

        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        
        train_epoch_loss=[]
        eval_epoch_loss=[]
        for epoch in range(epochs):
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

    #input size of the data
    input_size = 100  
    output_size = 1  

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15)

    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    # hyperparameters to train the final Network
    best_hidden_size = best_params['hidden_size']
    best_num_layers = best_params['num_layers']
    best_learning_rate = best_params['learning_rate']
    batch_size = best_params['batch_size']
    epochs = best_params['epochs']

    network = Network(input_size=input_size, hidden_size=best_hidden_size, num_layers=best_num_layers, output_size=output_size).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=best_learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    for epoch in range(epochs):
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
        'input_size': input_size,
        'hidden_size': best_hidden_size,
        'num_layers': best_num_layers,
        'output_size': output_size,
        'state_dict': model_state_dict
    }
    directory_name = 'saved models'
    create_directory(directory_name)
    model_save_name = f'lstm.pth'
    path = f"{directory_name}/{model_save_name}"
    torch.save(model_info, path)



