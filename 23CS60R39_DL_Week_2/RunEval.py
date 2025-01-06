import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=x, out_channels=128, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(6144, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)  
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class MyDNN(nn.Module):
    def __init__(self):
        super(MyDNN, self).__init__()
        self.fc1 = nn.Linear(train_data.shape[2], 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Aggregate token embeddings across the sequence dimension
        pooled_output = torch.mean(x, dim=1)  # Average pooling

        # Pass through the fully connected layers
        x = self.fc1(pooled_output)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x.view(-1, 1)  # Ensure output shape is [batch_size, 1]

def evaluate_model(model_path, test_data_path, model_type):
    # Load the saved model
    if model_type == "CNN":
        model = MyCNN()
    elif model_type == 'DNN':
        model = MyDNN()
    else:
        print("Invalid model type. Please provide either 'CNN'.")
        return
    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load test data
    test_data = np.load(test_data_path)
    X_test = torch.tensor(test_data, dtype=torch.float32)

    # Make predictions on the test data
    with torch.no_grad():
        outputs = model(X_test)
        predictions = (outputs > 0.5).float()
        predictions = predictions.cpu().numpy().flatten()

    # Load true labels for evaluation
    filename_parts = test_data_path.split('.')
    filename_without_extension = filename_parts[0]
    extension = '.' + filename_parts[-1]

    # Change 'data' to 'labels' in the filename
    test_label_path = filename_without_extension.replace('data', 'labels')

    # Combine the new filename with the extension
    test_label_path = test_label_path + extension
    true_labels = np.load(test_label_path)  # Provide the path to the true labels file

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print("Accuracy:", accuracy)

    # Generate and print classification report
    class_report = classification_report(true_labels, predictions)
    print("Classification Report:")
    print(class_report)
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Evaluate deep learning model.")
parser.add_argument("model_path", type=str, help="Path to the saved model file.")
parser.add_argument("test_data_path", type=str, help="Path to the test data file.")
parser.add_argument("model_type", type=str, help="Type of deep learning model (e.g., 'CNN').")
args = parser.parse_args()
original_filename = args.test_data_path
new_filename = original_filename.replace('test', 'train')
train_data = np.load(new_filename)  # Provide the path to the true labels file
x = train_data.shape[2]

# Evaluate the model
evaluate_model(args.model_path, args.test_data_path, args.model_type)

