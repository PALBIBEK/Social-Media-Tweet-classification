import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Step 1: Load and preprocess the data
df = pd.read_csv('CL-II-MisinformationData - Sheet1.csv')
tweets = df['tweet'].tolist()
labels = df['label'].tolist()

# Step 2: Tokenization and Encoding
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encoded_inputs = tokenizer(tweets, padding=True, truncation=True, return_tensors='pt')

# Split data into training and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    encoded_inputs.input_ids, labels, test_size=0.2, random_state=42)

# Convert inputs back to tensors
train_inputs = {
    'input_ids': train_inputs,
    'attention_mask': encoded_inputs.attention_mask[:len(train_inputs)]
}
val_inputs = {
    'input_ids': val_inputs,
    'attention_mask': encoded_inputs.attention_mask[len(train_inputs):]
}

from torch.utils.data import Dataset, DataLoader, TensorDataset

## Step 4: Instantiate the model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Convert inputs to tensors
train_input_ids = torch.tensor(train_inputs['input_ids'])
train_attention_mask = torch.tensor(train_inputs['attention_mask'])
val_input_ids = torch.tensor(val_inputs['input_ids'])
val_attention_mask = torch.tensor(val_inputs['attention_mask'])

# Step 5: Create Dataset object
train_dataset = TensorDataset(train_input_ids, train_attention_mask, torch.tensor(train_labels))
val_dataset = TensorDataset(val_input_ids, val_attention_mask, torch.tensor(val_labels))

# Step 6: Training
training_args = TrainingArguments(
    output_dir='./results',  # Specify the output directory
    overwrite_output_dir=True,  # Overwrite the content of the output directory
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

# Step 6: Evaluation
predictions = trainer.predict(val_inputs)
predicted_classes = predictions.predictions.argmax(axis=1)
true_classes = val_labels

# Calculate accuracy
accuracy = accuracy_score(true_classes, predicted_classes)
print("Accuracy:", accuracy)

# Generate classification report
report = classification_report(true_classes, predicted_classes)
print("Classification Report:")
print(report)
