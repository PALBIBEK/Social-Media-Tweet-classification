import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
import re
import emoji
import pickle

# Reading data from .csv file
dataset = pd.read_csv('CL-II-MisinformationData - Sheet1.csv')
# Function to replace emojis with their meanings
def replace_emojis(text):
    def emoji_meaning(match):
        emoji_char = match.group()
        return emoji.demojize(emoji_char, use_aliases=True)

    return re.sub(r'(:[a-zA-Z_]+:)', emoji_meaning, text)

# Assuming you have a DataFrame named 'twitter_data' with a column 'text' containing the tweets
# Replace 'twitter_data' with your actual DataFrame variable and column names

# Load your Twitter data (replace 'twitter_data.csv' with the actual file path)
twitter_data = pd.read_csv('CL-II-MisinformationData - Sheet1.csv')

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

# Split the dataset into train, validation, and test sets
train_data, temp_data = train_test_split(twitter_data, test_size=0.2, shuffle=True, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=True, random_state=42)

# Save the splits into CSV files
train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
