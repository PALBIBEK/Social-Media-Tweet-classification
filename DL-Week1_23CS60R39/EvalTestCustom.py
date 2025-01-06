import sys
import torch
from DNN import Network as DNNNetwork, CustomDataset as DNNDataset
from CNN import Network as CNNNetwork, CustomDataset as CNNDataset
from LSTM import Network as LSTMNetwork, CustomDataset as LSTMDataset
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import pandas as pd
import pickle
import numpy as np
import fasttext
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re
import emoji

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_model(model, test_dataset, batch_size=32):
    model.eval()
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            yp = model(x)
            yp = torch.sigmoid(yp)
            yp_classes = (yp > 0.5).float()
            yp_classes = yp_classes.squeeze(dim=1).long().cpu().numpy()  
            y_true.extend(y.cpu().numpy()) 
            y_pred.extend(yp_classes)

    target_names = ['fake', 'real']  
    print(classification_report(y_true, y_pred, target_names=target_names))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 EvalTestCustom.py <model_name> <test_csv_file>")
        sys.exit(1)

    model_name = sys.argv[1]
    test_csv_file = sys.argv[2]
    label_map = {'real': 1, 'fake': 0}

    df = pd.read_csv('CL-II-MisinformationData - Sheet1.csv')
    print("Preprocessing data")
    def convert_emojis(text):
        return emoji.demojize(text)

    #preprocess
    tknzr = TweetTokenizer()
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'http\S+', '', x))
    df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'#', '', x))
    df['tweet'] = df['tweet'].apply(convert_emojis)
    df['tweet'] = df['tweet'].apply(lambda x: tknzr.tokenize(x))
    df['tweet'] = df['tweet'].apply(lambda x: [word for word in x if word.lower() not in stop_words and word.isalnum()])
    df['tweet'] = df['tweet'].apply(lambda x: [stemmer.stem(word) for word in x])
    df['tweet'] = df['tweet'].apply(lambda x: ' '.join(x))

    if model_name == 'DNN':
        #preprocess
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        tfidf_test = tfidf_vectorizer.transform(df['tweet'])
        tfidf_test = tfidf_test.astype(np.float32)
        label_test = pd.DataFrame(df['label'].map(label_map).reset_index(drop=True))
        test_df = pd.DataFrame(tfidf_test.toarray())
        test_df = pd.concat([test_df, label_test], axis=1)
        test_dataset = DNNDataset(test_df)
        dnn_model_path = 'saved models/dnn.pth'
        dnn_model_dict = torch.load(dnn_model_path)
        dnn_model = DNNNetwork(**dnn_model_dict)
        print("Classification Report for DNN:")
        evaluate_model(dnn_model, test_dataset)

    elif model_name == 'CNN':

        model = fasttext.load_model('fasttext_model.bin')
        vectorized_data = []
        for text in df['tweet']:
            vector = model.get_sentence_vector(text)
            vectorized_data.append(vector)
        test_df = pd.DataFrame(vectorized_data)
        test_df['label'] = df['label'].map(label_map)



        test_dataset = CNNDataset(test_df)
        cnn_model_path = 'saved models/cnn.pth'
        cnn_model_dict = torch.load(cnn_model_path)
        cnn_model = CNNNetwork(**cnn_model_dict)
        print("Classification Report for CNN:")
        evaluate_model(cnn_model, test_dataset)
    elif model_name == 'LSTM':
        model = fasttext.load_model('fasttext_model.bin')
        vectorized_data = []
        for text in df['tweet']:
            vector = model.get_sentence_vector(text)
            vectorized_data.append(vector)
        test_df = pd.DataFrame(vectorized_data)
        test_df['label'] = df['label'].map(label_map)
        
        test_dataset = LSTMDataset(test_df)
        lstm_model_path = 'saved models/lstm.pth'
        lstm_model_dict = torch.load(lstm_model_path)
        lstm_model = LSTMNetwork(**lstm_model_dict)
        print("Classification Report for LSTM:")
        evaluate_model(lstm_model, test_dataset)
    else:
        print("Invalid model name. Choose from 'DNN', 'CNN', or 'LSTM'.")
