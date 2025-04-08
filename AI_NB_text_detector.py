# AI vs Human Generated Text Detector
# The model utilizes the Naïve Bayes (NB) algorithm, 
# but it exhibits a bias towards Human-generated outcomes.
# The accuracy of the model is 0.97
# By using streamlit run in the terminal the model becomes an application

import streamlit as st
import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv('train_from_LLM-Detect_AI-GT_1MNB-3SGD.csv')

# Remove NaN values and ensure target column is integer
df.dropna(subset=['text', 'RDizzl3_seven'], inplace=True)
df['RDizzl3_seven'] = df['RDizzl3_seven'].astype(int)

# Function to shuffle the dataset randomly
def shuffle_data(df):
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

df = shuffle_data(df)  # Shuffle dataset for unbiased training

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace('\n', ' ').replace("'", "").replace('"', '')  # Remove both single and double quotes
    text = ''.join([x for x in text if x not in string.punctuation])  # Remove punctuation
    return text.lower()  # Convert to lowercase

df['text'] = df['text'].apply(clean_text)

# Define independent and dependent variables
X = df['text']  # Independent variable (input features: the text itself)
y = df['RDizzl3_seven']  # Dependent variable (target output: whether the text is AI-generated or human-written)

# Split dataset into training and test sets (70% for training, 30% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Convert text into TF-IDF vectors
vectorizer = CountVectorizer(stop_words='english', lowercase=True, strip_accents='unicode')
X_train_counts = vectorizer.fit_transform(X_train)  # Fit and transform training data
X_test_counts = vectorizer.transform(X_test)  # Transform test data

# TF-IDF Transformation
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)  # Fit and transform training data
X_test_tfidf = tfidf_transformer.transform(X_test_counts)  # Transform test data

# Train Naive Bayes model
model = MultinomialNB(alpha=0.1)  # Set alpha parameter to 0.1
model.fit(X_train_tfidf, y_train)

# Predict using the trained model
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy of the model on the test data
accuracy = accuracy_score(y_test, y_pred) * 100  # Accuracy as a percentage

# Generate classification report to evaluate model performance
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# Streamlit App
st.title("AI vs Human Text Classifier")
st.text(f"Model Accuracy: {accuracy:.2f}%")  # Display accuracy percentage
st.text("Model Performance on Test Data:")
st.dataframe(report_df.style.set_table_styles(
    [{'selector': 'table', 'props': [('border', '2px solid black')]}]
))

# Function to clean user input before prediction
def clean_user_input(text):
    return text.replace('"', '').replace("'", "")  # Remove quotes

# User input prediction
user_input = st.text_area("Enter a text to check AI vs Human-generated percentage:")
cleaned_input = clean_user_input(user_input)

if st.button("Predict"):
    # Transform user input into the same format as training data
    text_counts = vectorizer.transform([cleaned_input])  
    text_tfidf = tfidf_transformer.transform(text_counts)  # Apply TF-IDF transformation
    probability = model.predict_proba(text_tfidf)[0]  # Get probability scores from the model
    
    # Calculate AI and human percentages
    ai_percentage = round(probability[1] * 100, 2)  # AI-generated percentage
    human_percentage = round(probability[0] * 100, 2)  # Human-generated percentage
    
    # Display the percentages in the Streamlit app
    st.write(f"AI-generated text: {ai_percentage}%")
    st.write(f"Human-generated text: {human_percentage}%")
