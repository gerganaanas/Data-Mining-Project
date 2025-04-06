# Model 2: Fake News Detector
# The model utilizes the Naïve Bayes (NB) algorithm
# The accuracy of the model is 0.94
# By using streamlit run in terminal the model becomes an application

import pandas as pd
import numpy as np  
import streamlit as st
import re  # Import regex for text preprocessing
import string  # Import string module for punctuation removal
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TF-IDF Vectorizer to convert text to numerical values
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import accuracy_score, classification_report  

# Load fake and real news datasets
df_fake = pd.read_csv("Fake.csv")  # Load fake news dataset
df_true = pd.read_csv("True.csv")  # Load real news dataset

# Assign class labels
df_fake["class"] = 0  # Label fake news as 0
df_true["class"] = 1  # Label real news as 1

# Combine both datasets
df = pd.concat([df_fake, df_true], axis=0)  # Merge datasets along rows (axis=0)

df = df.drop(["title", "subject", "date"], axis=1)  # Remove unnecessary columns (not useful for classification)
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle dataset and reset index

# Define text preprocessing function
def wordopt(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub('\[.*?\]', '', text)  # Remove text inside square brackets
    text = re.sub("\\W", " ", text)  # Remove non-word characters (special characters)
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)  # Remove URLs
    text = re.sub('<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(f'[{string.punctuation}]', '', text)  # Remove punctuation
    text = re.sub('\n', '', text)  # Remove newline characters
    text = re.sub('\w*\d\w*', '', text)  # Remove words containing numbers
    return text

# Apply text preprocessing
df["text"] = df["text"].apply(wordopt)  # Clean text data

# Define independent (x) and dependent (y) variables
x = df["text"]  # Independent variable (input text)
y = df["class"]  # Dependent variable (class label: 0 for fake, 1 for real)

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)  # 75% train, 25% test

# Convert text data into numerical features using TF-IDF
vectorization = TfidfVectorizer()  # Initialize TF-IDF Vectorizer
xv_train = vectorization.fit_transform(x_train)  # Fit and transform training data
xv_test = vectorization.transform(x_test)  # Transform test data (using same vocabulary as train)

# Train Naïve Bayes classifier
bayes = MultinomialNB(alpha=0.1)  # Set alpha parameter to 0.1 
bayes.fit(xv_train, y_train)  # Train model on training data

# Evaluate model
y_pred = bayes.predict(xv_test)  # Predict labels for test set
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy score

# Generate classification report as DataFrame
report_dict = classification_report(y_test, y_pred, target_names=["Fake", "Real"], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# Streamlit App Interface
st.title("Fake News Detector")  # App title
st.write(f"Naïve Bayes Model Accuracy: {accuracy:.2f}")  # Display model accuracy

# Display classification report as a table
st.subheader("Classification Report")
st.dataframe(report_df.style.format({
    "precision": "{:.2f}",
    "recall": "{:.2f}",
    "f1-score": "{:.2f}",
    "support": "{:.0f}"
}))

# User input for real-time classification
news = st.text_area("Enter a news article to check if it's fake or real:")  # Input field

# Function to classify user input
def classify_news(news):
    processed_news = wordopt(news)  # Preprocess input text
    vectorized_news = vectorization.transform([processed_news])  # Convert input text to TF-IDF vector
    prediction = bayes.predict(vectorized_news)[0]  # Predict class (0 or 1)
    return "Fake News" if prediction == 0 else "Real News"  # Return classification result

# Button to predict
if st.button("Check News"):  # Button to trigger prediction
    if news.strip():  # Ensure input is not empty
        prediction = classify_news(news)
        st.write("Prediction:", prediction)  # Display classification result
    else:
        st.write("Please enter some text to analyze.")
