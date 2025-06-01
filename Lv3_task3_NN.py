import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Load and prepare data
df_1 = pd.read_csv('Datasets/SentimentDataset.csv')  # Replace with your dataset path
df = df_1[['Text', 'Sentiment']]
print(df.head())
print(df.info())

stop_words = set(stopwords.words('english'))

# Preprocessing function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub('<.*?>', '', text)  # Remove HTML tags
    text = re.sub('http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)  # Emoticons
    text = re.sub(r'[\U0001F300-\U0001F5FF]', '', text)  # Misc Symbols
    text = re.sub(r'[\U0001F680-\U0001F6FF]', '', text)  # Transport
    text = re.sub(r'[\U0001F1E0-\U0001F1FF]', '', text)  # Flags
    text = re.sub(r'[\U00002600-\U000027BF]', '', text)  # Misc symbols
    text = re.sub(r'[\U0001F900-\U0001F9FF]', '', text)  # Supplemental Symbols
    text = re.sub('[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

df['cleaned review'] = df['Text'].apply(clean_text)

# TF-IDF: converting the data into suitable format for neural network
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned review']).toarray()
y = df['Sentiment'].map({'positive': 1, 'negative': 0})

print(f"Actual number of features: {X.shape[1]}")
print(f"Shape of X: {X.shape}")

# Training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model building - Use actual feature count instead of hardcoded 5000
input_dim = X.shape[1]  # This will be the actual number of features
print(f"Using input dimension: {input_dim}")

model = Sequential([
    Dense(256, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.01)),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Model training
history = model.fit(X_train, y_train, epochs=15, batch_size=64, 
                   validation_data=(X_test, y_test), callbacks=[early_stopping])

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plots
# Sentiment Distribution (uncommented)
plt.figure(figsize=(12, 8))
sns.countplot(x='Sentiment', data=df)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()
# Plotting accuracy and loss over epochs
plt.figure(figsize=(14, 10))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 3)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label1='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()

# Additional: Show model summary
print("\nModel Summary:")
model.summary()


"""
Level 3

(Advanced)

Description: Build and train a simple feed-forward
neural network to classify images or structured data.

Task 3: Neural Networks with

TensorFlow/Keras

Load a dataset (e.g., MNIST digits or a structured
dataset) and preprocess it.
Design a neural network architecture using TensorFlow
or Keras.
Train the model using backpropagation and evaluate it
using accuracy and loss curves.
Tune hyperparameters (e.g., learning rate, batch size) to
improve performance.
Tools: Python, TensorFlow, Keras, pandas, matplotlib.

"""
