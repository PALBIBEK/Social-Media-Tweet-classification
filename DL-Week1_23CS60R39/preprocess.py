import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import fasttext

df = pd.read_csv('CL-II-MisinformationData - Sheet1.csv')
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

#split into file
train, test_val = train_test_split(df, test_size=0.2, random_state=42)
val, test=train_test_split(test_val, test_size=0.5, random_state=42)
train.to_csv('train.csv', index= "False")
test.to_csv('test.csv', index= "False")
val.to_csv('val.csv', index= "False")

#vectorizer model
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(df['tweet'])
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

#vectorized df
label_map={'real':1,'fake':0}
tfidf_train = tfidf_vectorizer.transform(train['tweet'])
tfidf_train = tfidf_train.toarray()
tfidf_train = tfidf_train.astype(np.float32)
label_train = pd.DataFrame(train['label'].map(label_map).reset_index(drop=True))
train_df = pd.concat([pd.DataFrame(tfidf_train),label_train], axis=1)
train_df.to_pickle('train-dnn.pkl')

tfidf_val = tfidf_vectorizer.transform(val['tweet'])
tfidf_val = tfidf_val.toarray()
tfidf_val = tfidf_val.astype(np.float32)
label_val = pd.DataFrame(val['label'].map(label_map).reset_index(drop=True))
val_df = pd.concat([pd.DataFrame(tfidf_val),label_val], axis=1)
val_df.to_pickle('val-dnn.pkl')

tfidf_test = tfidf_vectorizer.transform(test['tweet'])
tfidf_test = tfidf_test.toarray()
tfidf_test = tfidf_test.astype(np.float32)
label_test = pd.DataFrame(test['label'].map(label_map).reset_index(drop=True))
test_df = pd.concat([pd.DataFrame(tfidf_test),label_test], axis=1)
test_df.to_pickle('test-dnn.pkl')



#For CNN and LSTM

df = pd.read_csv("train.csv")  
model = fasttext.train_unsupervised("train.csv", model='skipgram')
model.save_model("fasttext_model.bin")

label_map={'fake':0, 'real':1}
vectorized_data = []
for text in df['tweet']:
    vector = model.get_sentence_vector(text)
    vectorized_data.append(vector)
vectorized_df = pd.DataFrame(vectorized_data)
vectorized_df['label'] = df['label'].map(label_map)
vectorized_df.to_pickle('train.pkl')


vectorized_data = []
df = pd.read_csv("val.csv")
for text in df['tweet']:
    vector = model.get_sentence_vector(text)
    vectorized_data.append(vector)
val_df = pd.DataFrame(vectorized_data)
val_df['label'] = df['label'].map(label_map)
val_df.to_pickle('val.pkl')
print(val_df.head(5))
vectorized_data = []
df = pd.read_csv("test.csv")
for text in df['tweet']:
    vector = model.get_sentence_vector(text)
    vectorized_data.append(vector)
test_df = pd.DataFrame(vectorized_data)
test_df['label'] = df['label'].map(label_map)
test_df.to_pickle('test.pkl')
print(test_df.head(5))