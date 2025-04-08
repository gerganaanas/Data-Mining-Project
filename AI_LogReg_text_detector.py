# AI vs Human Generated Text Detector (Biased towards AI-generated)

# The model utilizes Logistic Regression, 
# The accuracy of the model is 0.99, 
# but it exhibits a bias toward AI-generated outcomes.
# By using streamlit run the model becomes an application

import streamlit as st
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('Training_Essay_Data.csv')

def shuffle_data(df):
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

df = shuffle_data(df)

# Cleaning function for both training and user input
def clean_text(text):
    text = text.replace("\n", "").replace("'", "").replace("\"", "")  # Remove quotes and newlines
    return ''.join([x for x in text if x not in string.punctuation])

df['text'] = df['text'].apply(clean_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['generated'], test_size=0.3, random_state=42)

# Convert text into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(C=100, max_iter=200, penalty='l2', solver='saga')
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# Streamlit App
st.title("AI vs Human Text Classifier")
st.text("Model accuracy: 99%")
st.text("Model Performance on Test Data:")
st.dataframe(report_df.style.set_table_styles(
    [{'selector': 'table', 'props': [('border', '2px solid black')]}]
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
labels = ['Human', 'AI']

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
st.pyplot(fig)

# User input prediction
def clean_user_input(text):
    text = text.replace("\"", "").replace("'", "")  # Remove quotes from user input
    return text

user_input = st.text_area("Enter a text to check AI vs Human-generated percentage:")
cleaned_input = clean_user_input(user_input)

if st.button("Predict"):
    text_tfidf = vectorizer.transform([cleaned_input])
    probability = model.predict_proba(text_tfidf)[0]
    ai_percentage = round(probability[1] * 100, 2)
    human_percentage = round(probability[0] * 100, 2)
    
    st.write(f"AI-generated: {ai_percentage}%")
    st.write(f"Human-generated: {human_percentage}%")


