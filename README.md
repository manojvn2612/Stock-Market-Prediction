# Stock Price Prediction and Analysis

This project consists of two main parts:

1. **Stock Price Prediction Using LSTM**: A machine learning model that predicts stock prices of tech companies (Apple, Google, Microsoft, Amazon).
2. **Sentiment Analysis of Tweets Using Tweepy and a Simple Neural Network**: A model that analyzes sentiment (up or down) of tweets related to stock market movements using Tweepy to fetch tweets and a simple neural network for sentiment analysis.

---

## Part 1: Stock Price Prediction Using LSTM

### Overview

The first part of the project uses an LSTM (Long Short-Term Memory) model to predict the stock prices of four major tech companies. The model is trained on historical stock data using Yahoo Finance.

### Steps

1. **Data Collection**: Stock price data for `AAPL`, `GOOG`, `MSFT`, and `AMZN` is collected using the `yfinance` API.
2. **Preprocessing**: The data is scaled using `MinMaxScaler` to normalize the features between 0 and 1.
3. **Model Training**: An LSTM-based neural network is built and trained on the data.
4. **Model Evaluation**: Predictions are made on test data, and the Root Mean Squared Error (RMSE) is calculated to assess the performance of the model.
5. **Visualization**: The predicted and actual stock prices are plotted for comparison.

### Dependencies

- `numpy`
- `pandas`
- `yfinance(older version, use 0.2.40)`
- `sklearn`
- `keras`
- `matplotlib`

Install dependencies with the following command:

```bash
pip install numpy pandas yfinance=0.2.40 sklearn keras matplotlib
```

---

## Part 2: Sentiment Analysis of Tweets Using Tweepy and Simple NN

### Overview

This part of the project uses Tweepy to fetch tweets from Twitter, and a simple neural network (NN) to perform sentiment analysis (up or down). We use a basic NN model instead of LSTM for simplicity.

### Steps

1. **Fetch Tweets Using Tweepy**:
   - Tweets are fetched using the Tweepy API based on a search query or hashtag.
   
2. **Data Preprocessing**:
   - Text is cleaned by removing URLs, hashtags, and punctuation.
   - Tweets are tokenized and padded to ensure consistent input shape.

3. **Sentiment Labeling**:
   - Tweets are labeled as either `up` (bullish sentiment) or `down` (bearish sentiment).

4. **Model Building**:
   - A simple fully connected neural network is used for sentiment classification.
   
5. **Training**:
   - The model is trained on the labeled tweet data.

6. **Prediction**:
   - The trained model predicts sentiment for new tweets.

### Code Example

#### Fetching Tweets with Tweepy

Here is the code to fetch tweets using Tweepy:

```python
import tweepy
import pandas as pd

# Set up Twitter API credentials (replace with your own)
consumer_key = 'YOUR_CONSUMER_KEY'
consumer_secret = 'YOUR_CONSUMER_SECRET'
access_token = 'YOUR_ACCESS_TOKEN'
access_token_secret = 'YOUR_ACCESS_TOKEN_SECRET'

# Authenticate to Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Fetch tweets
query = 'stock market'  # Change to your desired search query
tweets = api.search(q=query, count=100, lang='en', tweet_mode='extended')

# Create a DataFrame with tweet data
tweet_data = pd.DataFrame([[tweet.created_at, tweet.full_text] for tweet in tweets], columns=['Date', 'Text'])

# Save the DataFrame to a CSV
tweet_data.to_csv('tweets.csv', index=False)

print("Tweets have been downloaded successfully.")
```

#### Data Preprocessing

```python
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Clean the tweet text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"#\w+", "", text)    # Remove hashtags
    text = re.sub(r"[^\w\s]", "", text) # Remove punctuation
    text = text.lower().strip()         # Convert to lowercase
    return text

tweet_data['clean_text'] = tweet_data['Text'].apply(clean_text)

# Labeling sentiments as 'up' or 'down' (dummy labels)
# In practice, these labels should come from manual annotation or sentiment analysis
tweet_data['label'] = tweet_data['Text'].apply(lambda x: 'up' if 'bullish' in x.lower() else 'down')

# Tokenization and padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweet_data['clean_text'])
sequences = tokenizer.texts_to_sequences(tweet_data['clean_text'])
max_length = 50
X = pad_sequences(sequences, maxlen=max_length)

# Encoding labels
encoder = LabelEncoder()
y = encoder.fit_transform(tweet_data['label'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### Building and Training the Simple NN Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.optimizers import Adam

# Simple NN model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_length))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
```

#### Evaluating and Predicting Sentiment

```python
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Example of predicting sentiment for new tweets
new_texts = [
    "The stock market is looking strong today. Bullish sentiments everywhere!",
    "Expecting a crash soon as things are overbought."
]

# Preprocess new texts
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=50)

# Predict sentiment
predictions = model.predict(new_padded)

# Interpret predictions
for text, pred in zip(new_texts, predictions):
    sentiment = "Up" if pred > 0.5 else "Down"
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {sentiment}")
    print("-" * 40)
```

### Dependencies

- `tweepy`
- `pandas`
- `numpy`
- `re`
- `scikit-learn`
- `tensorflow`
- `keras`

Install dependencies with the following command:

```bash
pip install tweepy pandas numpy scikit-learn tensorflow keras
```

### Usage

1. **Fetching Tweets**:
   - Ensure you have valid Twitter API keys and replace them in the Tweepy setup section.
   - Tweets are fetched using a search query and saved to a CSV.

2. **Training the Sentiment Analysis Model**:
   - The text is cleaned, tokenized, and padded.
   - A simple NN model is trained to predict sentiment (up or down).

3. **Prediction**:
   - Use the trained model to predict the sentiment of new tweets.


## License

Need Licensing for it

---
