import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
model_type = input("Enter BERT model type: ")

# Load vectorized train data and labels
train_data = np.load(model_type + "-train_vectorized_data.npy")
train_labels = np.load(model_type + "-train_vectorized_labels.npy")
# print(train_data.shape)
# Load vectorized test data and labels
test_data = np.load(model_type + "-test_vectorized_data.npy")
test_labels = np.load(model_type + "-test_vectorized_labels.npy")
# print(test_data.shape)
# Load vectorized validation data and labels
val_data = np.load(model_type + "-val_vectorized_data.npy")
val_labels = np.load(model_type + "-val_vectorized_labels.npy")
# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(train_data, dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension
X_val = torch.tensor(val_data, dtype=torch.float32)
y_val = torch.tensor(val_labels, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension
X_test = torch.tensor(test_data, dtype=torch.float32)
y_test = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension

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


# Instantiate the model, loss function, and optimizer
model = MyDNN()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create DataLoader for batching and shuffling the train data
train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

batch_size = 64
# Instantiate the model, loss function, and optimizer
model = MyDNN().to(device)
criterion = nn.BCELoss().to(device)  # Move the criterion to the same device as the model
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i in range(0, len(train_data), batch_size):
        batch_data = train_data[i:i+batch_size]
        batch_labels = train_labels[i:i+batch_size]

        optimizer.zero_grad()
        batch_data = torch.tensor(batch_data, dtype=torch.float32).to(device)  # Move to device

        # Reshape the batch_labels to match the shape of the model output
        batch_labels = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(1).to(device)  # Move to device

        outputs = model(batch_data)

        # Calculate the loss
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_data)}")


# Validate the model
model.eval()
with torch.no_grad():
    X_val = X_val.to(device)  # Move input data to the same device as the model
    y_val = y_val.to(device)  # Move labels to the same device as the model
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

torch.save(model.state_dict(), model_type+"dnn_model.pth")

from sklearn.metrics import classification_report, confusion_matrix

# Load true labels for test data
true_labels = y_test_np

# Print classification report
print("Classification Report:")
print(classification_report(true_labels, predictions))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(true_labels, predictions))

