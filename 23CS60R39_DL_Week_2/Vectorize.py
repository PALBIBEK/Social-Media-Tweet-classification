import pandas as pd
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch

def vectorize_data(model_name, input_data, output_file, device, batch_size=64):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    # Load input data
    input_df = pd.read_csv(input_data)
    
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

if __name__ == "__main__":
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model_type = input("Enter BERT model type - ")

    # Vectorize validation data
    input_file = 'val_data.csv'
    output_file = model_type+"-val_vectorized.npy"
    vectorize_data(model_type, input_file, output_file, device)
    
    # Vectorize train data
    input_file = 'train_data.csv'
    output_file = model_type+"-train_vectorized.npy"
    vectorize_data(model_type, input_file, output_file, device)

    # Vectorize test data
    input_file = 'test_data.csv'
    output_file = model_type+"-test_vectorized.npy"
    vectorize_data(model_type, input_file, output_file, device)
