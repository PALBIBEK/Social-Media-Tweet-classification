import pandas as pd
import re
import argparse
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
import emoji
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
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
        self.fc1 = nn.Linear(x, 256)
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
parser.add_argument("test_data_path", type=str, help="Path to the test data file.")
parser.add_argument("model_type", type=str, help="Name of the BERT model type.")
parser.add_argument("model_path", type=str, help="Path to the saved model file.")
args = parser.parse_args()
model_type = args.model_type

def vectorize_data(model_name, input_data, output_file, device, batch_size=64):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    input_df = input_data
    
    # Tokenize and vectorize data
    max_length = 100  # Maximum sequence length for BERT models
    vectorized_data = []
    labels = []
    
    # Define label encoding dictionary
    label_encoding = {'real': 1, 'fake': 0}  # Map 'real' to 1 and 'fake' to 0
    
    for idx, row in input_df.iterrows():
        text = row['tweet']
        label = row['label']
        
        # Tokenize the text
        inputs = tokenizer(text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        inputs = inputs.to(device)

        # Pass the inputs through the model
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract the embeddings from the output and detach to move to CPU
        embeddings = outputs.last_hidden_state.detach().cpu().numpy()

        # Append the output to the list
        vectorized_data.append(embeddings)
        
        # Apply label encoding
        encoded_label = label_encoding[label]
        labels.append(encoded_label)

    # Stack all vectorized data into a single array
    vectorized_data = np.concatenate(vectorized_data, axis=0)
    # Save vectorized data and labels
    np.save(output_file.replace('.npy', '_data.npy'), vectorized_data)
    np.save(output_file.replace('.npy', '_labels.npy'), np.array(labels))
    print(f"Saved vectorized data and labels to {output_file}")
    return vectorized_data.shape[2]


# Reading data from .csv file
dataset = pd.read_csv(args.test_data_path)
# Function to replace emojis with their meanings
def replace_emojis(text):
    def emoji_meaning(match):
        emoji_char = match.group()
        return emoji.demojize(emoji_char, use_aliases=True)

    return re.sub(r'(:[a-zA-Z_]+:)', emoji_meaning, text)

twitter_data = pd.read_csv(args.test_data_path)

# Change the column name from 'text' to 'tweet'
twitter_data = twitter_data.rename(columns={'text': 'tweet'})

# Remove URLs
twitter_data['tweet'] = twitter_data['tweet'].apply(lambda text: re.sub(r'http\S+', '', text))

# Remove mentions (e.g., @username)
twitter_data['tweet'] = twitter_data['tweet'].apply(lambda text: re.sub(r'@[\w_]+', '', text))

# Replace emojis with their meanings
twitter_data['tweet'] = twitter_data['tweet'].apply(replace_emojis)

# Remove special characters, numbers, and punctuation
#twitter_data['tweet'] = twitter_data['tweet'].apply(lambda text: re.sub(r'[^a-zA-Z\s]', '', text))

# Convert to lowercase
twitter_data['tweet'] = twitter_data['tweet'].str.lower()

# Tokenization and removal of stopwords
stop_words = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)
twitter_data['tweet'] = twitter_data['tweet'].apply(lambda text: word_tokenize(text))
twitter_data['tweet'] = twitter_data['tweet'].apply(lambda tokens: [word for word in tokens if word not in stop_words])

# Stemming using Porter Stemmer
porter = PorterStemmer()
twitter_data['tweet'] = twitter_data['tweet'].apply(lambda tokens: [porter.stem(word) for word in tokens])

# Join the tokens back into text
twitter_data['tweet'] = twitter_data['tweet'].apply(lambda tokens: ' '.join(tokens))

test_data = twitter_data
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
output_file = model_type+"-test_vectorized.npy"
x = vectorize_data(model_type, test_data, output_file, device)
model_path = args.model_path
index = model_path.find("cnn")
training_type = ""
if index != -1:
    training_type = "CNN"
else:
    index = model_path.find("dnn")
    training_type = ""
    if index != -1:
        training_type = "DNN"
# Evaluate the model
path = model_type+"-test_vectorized_data.npy"
evaluate_model(args.model_path,path,training_type)