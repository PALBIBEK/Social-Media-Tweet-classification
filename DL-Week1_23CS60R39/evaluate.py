import torch
from DNN import Network as DNNNetwork, CustomDataset as DNNDataset
from CNN import Network as CNNNetwork, CustomDataset as CNNDataset
from LSTM import Network as LSTMNetwork, CustomDataset as LSTMDataset
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import pickle

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
    # Load test datasets and models
    with open('test-dnn.pkl', 'rb') as f:
        test_df_dnn = pickle.load(f)
    test_dataset_dnn = DNNDataset(test_df_dnn)
    dnn_model_path = 'saved models/dnn.pth'
    dnn_model_dict = torch.load(dnn_model_path)
    dnn_model = DNNNetwork(**dnn_model_dict)
    print("Classification Report for DNN:")
    evaluate_model(dnn_model, test_dataset_dnn)

    with open('test.pkl', 'rb') as f:
        test_df_cnn = pickle.load(f)
    test_dataset_cnn = CNNDataset(test_df_cnn)
    cnn_model_path = 'saved models/cnn.pth'
    cnn_model_dict = torch.load(cnn_model_path)
    cnn_model = CNNNetwork(**cnn_model_dict)
    print("Classification Report for CNN:")
    evaluate_model(cnn_model, test_dataset_cnn)

    with open('test.pkl', 'rb') as f:
        test_df_lstm = pickle.load(f)
    test_dataset_lstm = LSTMDataset(test_df_lstm)
    lstm_model_path = 'saved models/lstm.pth'
    lstm_model_dict = torch.load(lstm_model_path)
    lstm_model = LSTMNetwork(**lstm_model_dict)
    print("Classification Report for LSTM:")
    evaluate_model(lstm_model, test_dataset_lstm)
