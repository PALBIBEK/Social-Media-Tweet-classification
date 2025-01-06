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
device = "cuda" if torch.cuda.is_available() else "cpu"
import pickle
import optuna


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class CustomDataset(Dataset):
    def __init__(self, df):
        self.x = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class Network(nn.Module):
    def __init__(self, input_channel=1, output_dim=1, kernel_size=10, stride=10, padding=0, state_dict=None):
        super().__init__()
        self.conv_1 = nn.Conv1d(input_channel, 128, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation_1 = nn.ReLU()
        self.conv_2 = nn.Conv1d(128, 256, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation_2 = nn.ReLU()
        # Calculating output size for flattening
        output_size_conv1 = ((100 + 2 * padding - kernel_size) // stride) + 1
        output_size_conv2 = ((output_size_conv1 + 2 * padding - kernel_size) // stride) + 1
        self.flatten = nn.Flatten()
        self.linear_3 = nn.Linear(256 * output_size_conv2, output_dim)
        if state_dict is not None:
            self.load_state_dict(state_dict)


    def forward(self, x):
        out = self.conv_1(x)
        out = self.activation_1(out)
        out = self.conv_2(out)
        out = self.activation_2(out)
        out = self.flatten(out)
        out = self.linear_3(out)
        return out



if __name__=="__main__":
    
    with open('train.pkl', 'rb') as f:
        train_df = pickle.load(f)
    with open('val.pkl', 'rb') as f:
        val_df = pickle.load(f)

    train_dataset = CustomDataset(train_df)
    val_dataset = CustomDataset(val_df)

    def objective(trial):
        # Sample hyperparameters
        kernel_size = trial.suggest_int('kernel_size', 3, 10)
        stride = trial.suggest_int('stride', 3, 10)
        padding = trial.suggest_int('padding', 0, 5)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1,log=True)
        network = Network(kernel_size=kernel_size, stride=stride, padding=padding)
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        epochs = trial.suggest_int('epochs', 5, 10)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        eval_epoch_loss=[]
        train_epoch_loss=[]
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

    # Get the best hyperparameters
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)
    batch_size = best_params['batch_size']
    epochs = best_params['epochs']
    kernel_size=best_params['kernel_size']
    stride=best_params['stride']
    padding=best_params['padding']
    network = Network(kernel_size=best_params['kernel_size'], stride=best_params['stride'], padding=best_params['padding']).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=best_params['learning_rate'])
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)


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
    
    model_state_dict = network.state_dict()
    model_info = {
        'input_channel': 1,
        'output_dim': 1,
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'state_dict': model_state_dict
    }
    directory_name = 'saved models'
    create_directory(directory_name)
    model_save_name = f'cnn.pth'
    path = f"{directory_name}/{model_save_name}"
    torch.save(model_info, path)



