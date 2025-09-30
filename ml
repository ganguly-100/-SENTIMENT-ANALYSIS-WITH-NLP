# sentiment_analysis.py
# This script performs sentiment analysis on customer reviews using
# TF-IDF Vectorization and a Logistic Regression model.

import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load and Prepare the Dataset ---
# For this demonstration, we'll create a sample dataset of customer reviews.
# In a real-world scenario, you would load this from a CSV or database.
# 1 represents 'positive' sentiment, 0 represents 'negative' sentiment.
data = {
    'review': [
        "This product is absolutely fantastic! I love it.",
        "The quality is amazing, highly recommended.",
        "Completely satisfied with my purchase, will buy again.",
        "It's a wonderful experience, five stars!",
        "The best purchase I have ever made.",
        "I am so happy with this, it exceeded my expectations.",
        "This is a terrible product. I regret buying it.",
        "Awful experience, the item was broken on arrival.",
        "Do not buy this, it's a waste of money.",
        "Very disappointed with the quality and customer service.",
        "I would not recommend this to anyone.",
        "The product failed after just one week. Very poor.",
        "It works, but it's not great. Just okay.",
        "The product is decent for the price, not amazing but not bad either.",
        "An average product, does the job but nothing special."
    ],
    'sentiment': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0] # 1 for positive, 0 for negative
}
df = pd.DataFrame(data)

print("--- Original Data ---")
print(df.head())
print("\n")


# --- 2. Preprocess the Text Data ---
# Preprocessing is crucial for NLP tasks. We'll perform simple cleaning:
# - Convert text to lowercase
# - Remove punctuation
def preprocess_text(text):
    """
    Cleans and preprocesses a single text string.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = text.strip() # Remove leading/trailing whitespace
    return text

# Apply the preprocessing function to our reviews
df['cleaned_review'] = df['review'].apply(preprocess_text)

print("--- Data After Cleaning ---")
print(df[['review', 'cleaned_review']].head())
print("\n")


# --- 3. Feature Engineering: TF-IDF Vectorization ---
# We need to convert our text data into numerical vectors so the model can understand it.
# TF-IDF (Term Frequency-Inverse Document Frequency) is a great way to do this.
# It reflects how important a word is to a document in a collection.

# Define features (X) and target (y)
X = df['cleaned_review']
y = df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the TF-IDF Vectorizer
# `max_features` limits the number of features (words) to the top 5000.
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit the vectorizer on the training data and transform both training and test data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("--- TF-IDF Vectorization ---")
print(f"Shape of TF-IDF training matrix: {X_train_tfidf.shape}")
print(f"Shape of TF-IDF testing matrix: {X_test_tfidf.shape}")
print("\n")


# --- 4. Build and Train the Logistic Regression Model ---
# Logistic Regression is a simple yet powerful classification algorithm.
model = LogisticRegression(random_state=42)

print("--- Training the Logistic Regression Model ---")
model.fit(X_train_tfidf, y_train)
print("Model training complete.")
print("\n")


# --- 5. Evaluate the Model ---
# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Calculate and print evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(class_report)
print("Confusion Matrix:")
print(conf_matrix)
print("\n")

# Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Saved confusion matrix visualization to 'confusion_matrix.png'")
print("\n")


# --- 6. Test with New Reviews ---
# Let's see how our model performs on new, unseen data.
print("--- Testing with New Reviews ---")
new_reviews = [
    "The customer service was excellent and the product is top-notch.",
    "A complete waste of time and money, very frustrating.",
    "It's an okay product, not the best but it works."
]

# Preprocess and transform the new reviews
cleaned_new_reviews = [preprocess_text(review) for review in new_reviews]
new_reviews_tfidf = tfidf_vectorizer.transform(cleaned_new_reviews)

# Predict the sentiment
new_predictions = model.predict(new_reviews_tfidf)

# Display the results
for review, prediction in zip(new_reviews, new_predictions):
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    print(f"Review: '{review}'\nPredicted Sentiment: {sentiment}\n")

print("--- Script Finished ---")
