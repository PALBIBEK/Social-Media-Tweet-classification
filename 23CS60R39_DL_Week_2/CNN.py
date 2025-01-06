import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=train_data.shape[2], out_channels=128, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(6144, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Reshaping for 1D convolution
        x = x.permute(0, 2, 1)  
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the vectorized data
model_type = input("Enter BERT model type: ")

# Load vectorized train data and labels
train_data = np.load(model_type + "-train_vectorized_data.npy")
train_labels = np.load(model_type + "-train_vectorized_labels.npy")

# Load vectorized test data and labels
test_data = np.load(model_type + "-test_vectorized_data.npy")
test_labels = np.load(model_type + "-test_vectorized_labels.npy")

# Load vectorized validation data and labels
val_data = np.load(model_type + "-val_vectorized_data.npy")
val_labels = np.load(model_type + "-val_vectorized_labels.npy")

# print(train_data.shape)
# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(train_data, dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension
X_val = torch.tensor(val_data, dtype=torch.float32)
y_val = torch.tensor(val_labels, dtype=torch.float32).unsqueeze(1)      # Add an extra dimension
X_test = torch.tensor(test_data, dtype=torch.float32)
y_test = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)    # Add an extra dimension

# Instantiate the CNN model
model = MyCNN().to(device)

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create DataLoader for batching and shuffling the train data
train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_dataloader)}")

# Validate the model
model.eval()
with torch.no_grad():
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    outputs = model(X_val)
    val_loss = criterion(outputs, y_val)

print(f"Validation Loss: {val_loss.item()}")

# Make predictions on the test data
model.eval()
with torch.no_grad():
    X_test = X_test.to(device)
    outputs = model(X_test)
    predictions = (outputs > 0.5).float()
    predictions = predictions.cpu().numpy().flatten()

# Convert true labels to numpy array
y_test_np = y_test.cpu().numpy().flatten()

# Calculate accuracy on the test data
accuracy = accuracy_score(y_test_np, predictions)
print(f"Test Accuracy: {accuracy}")

torch.save(model.state_dict(), model_type+"cnn_model.pth")

from sklearn.metrics import classification_report, confusion_matrix

# Load true labels for test data
true_labels = y_test_np

# Print classification report
print("Classification Report:")
print(classification_report(true_labels, predictions))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(true_labels, predictions))

