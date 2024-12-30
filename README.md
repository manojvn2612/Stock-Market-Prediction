# Stock Price Prediction and Analysis

This project consists of two main parts:

1. **Stock Price Prediction Using LSTM**: A machine learning model that predicts stock prices of tech companies (Apple, Google, Microsoft, Amazon).
2. **Sentiment Analysis of Tweets**: A model that analyzes sentiment (up or down) of tweets related to stock market movements using data augmentation and an LSTM network.

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

## Part 2: Sentiment Analysis of Tweets Using LSTM and Data Augmentation

### Overview

This part of the project focuses on performing sentiment analysis on stock market-related tweets using an LSTM-based neural network. We perform data augmentation by replacing words with their synonyms using WordNet.

### Steps

1. **Data Augmentation**:
   - A custom function replaces random words in the text with synonyms using WordNet.
   - This helps generate more training data by augmenting the existing dataset.
2. **Text Preprocessing**:
   - Tweets are cleaned by removing URLs, hashtags, and punctuation.
   - Text is tokenized, and sequences are padded to a fixed length.
3. **Model Building**:
   - A Keras LSTM model is built to classify tweets as either "up" (bullish sentiment) or "down" (bearish sentiment).
4. **Training**:
   - The model is trained using the augmented data and validated on test data.
5. **Prediction**:
   - The trained model predicts sentiment for new tweets.
   
### Code Explanation

#### Data Augmentation

The following code augments the tweet data by replacing synonyms using WordNet:

```python
import random
from nltk.corpus import wordnet
import pandas as pd

# Load your dataset
file_path = "/content/tweets.csv"  # Update with your file path
data = pd.read_csv(file_path)

# Ensure the text column is named correctly
text_column = "text"  # Replace with the correct column name in your dataset
data = data.dropna(subset=[text_column])  # Drop rows with missing text

# Synonym replacement using WordNet
def synonym_replacement(sentence, n=2):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set(words))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_words)

# Helper function to get synonyms using WordNet
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").lower()
            if synonym != word:
                synonyms.add(synonym)
    return synonyms

# Augment data by applying transformations
def augment_data(data, text_column, n_augments=2):
    augmented_texts = []
    for _, row in data.iterrows():
        original_text = row[text_column]
        for _ in range(n_augments):
            augmented_text = synonym_replacement(original_text)
            augmented_texts.append(augmented_text)

    return augmented_texts

# Perform data augmentation
augmented_texts = augment_data(data, text_column)

# Create a new DataFrame for augmented data
augmented_df = pd.DataFrame({text_column: augmented_texts})

# Save augmented data to a CSV file
output_file = "/content/augmented_tweets.csv"
augmented_df.to_csv(output_file, index=False)

print(f"Augmented data saved to {output_file}")
```

#### Sentiment Analysis Model

The model is built using Keras with an LSTM layer for sequential data. Here's the code for training the sentiment analysis model:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load the dataset
file_path = '/content/tweets.csv'
tweets_data = pd.read_csv(file_path)

# Clean the text data
def clean_text(text):
    import re
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"#\w+", "", text)    # Remove hashtags
    text = re.sub(r"[^\w\s]", "", text) # Remove punctuation
    text = text.lower().strip()         # Convert to lowercase
    return text

tweets_data['clean_text'] = tweets_data['text'].apply(clean_text)

# Map labels to integers
tweets_data['label'] = tweets_data['prediction'].map({'up': 1, 'down': 0})

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets_data['clean_text'])
sequences = tokenizer.texts_to_sequences(tweets_data['clean_text'])
max_length = 50  # Define max length for padding
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, tweets_data['label'], test_size=0.2, random_state=42)

# Build the Keras model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_length),
    LSTM(128, dropout=0.2),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification (up or down)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the tokenizer and model
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

model.save('tweet_sentiment_model.h5')
```

### Usage

1. **Training the Sentiment Analysis Model**:
   - Ensure you have a CSV file with a `text` column and a `prediction` column for sentiment (either "up" or "down").
   - Clean the text, augment it with synonyms, and train the model as described.

2. **Prediction**:
   - Use the trained model to predict sentiment for new tweets:

```python
new_texts = [
    "The stock market is looking strong today. Bullish sentiments everywhere!",
    "Expecting a crash soon as things are overbought."
]

# Preprocess and predict sentiment
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=45, padding='post', truncating='post')
predictions = model.predict(new_padded)

for text, pred in zip(new_texts, predictions):
    sentiment = "Up" if pred > 0.5 else "Down"
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {sentiment}")
    print("-" * 40)
```

---
## License

Apache 2.0 License
2024 Manoj Nayak
