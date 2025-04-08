# AI vs Human Generated Text Detector
# The model utilizes the Naïve Bayes (NB) algorithm, 
# but it exhibits a bias towards Human-generated outcomes.
# The accuracy of the model is 0.97
# By using streamlit run in the terminal the model becomes an application

import streamlit as st
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay  # [Added]

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
X = df['text'] # Independent
y = df['RDizzl3_seven'] #Dependent

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Convert text into TF-IDF vectors
vectorizer = CountVectorizer(stop_words='english', lowercase=True, strip_accents='unicode')
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# TF-IDF Transformation
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Train Naive Bayes model
model = MultinomialNB(alpha=0.1)
model.fit(X_train_tfidf, y_train)

# Predict using the trained model
y_pred = model.predict(X_test_tfidf)

# Accuracy and Classification Report
accuracy = accuracy_score(y_test, y_pred) * 100
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# Streamlit App
st.title("AI vs Human Text Classifier")
st.text(f"Model Accuracy: {accuracy:.2f}%")
st.text("Model Performance on Test Data:")
st.dataframe(report_df.style.set_table_styles(
    [{'selector': 'table', 'props': [('border', '2px solid black')]}]
))

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

# Plotting the Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap='Blues', ax=ax)  # Apply a blue color palette (Blues)
ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold', color='darkblue')  # Make title blue and bold
plt.grid(False)  # Disable gridlines for a cleaner look
st.pyplot(fig)

# Function to clean user input before prediction
def clean_user_input(text):
    return text.replace('"', '').replace("'", "")

# User input prediction
user_input = st.text_area("Enter a text to check AI vs Human-generated percentage:")
cleaned_input = clean_user_input(user_input)

if st.button("Predict"):
    text_counts = vectorizer.transform([cleaned_input])
    text_tfidf = tfidf_transformer.transform(text_counts)
    probability = model.predict_proba(text_tfidf)[0]
    
    ai_percentage = round(probability[1] * 100, 2)
    human_percentage = round(probability[0] * 100, 2)
    
    st.write(f"AI-generated text: {ai_percentage}%")
    st.write(f"Human-generated text: {human_percentage}%")

