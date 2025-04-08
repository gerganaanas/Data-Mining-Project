# Model 2: Fake News Detector
# The model utilizes the Naïve Bayes (NB) algorithm
# The accuracy of the model is 0.94
# By using streamlit run in terminal the model becomes an application

import pandas as pd
import numpy as np  
import streamlit as st
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # <-- Added confusion_matrix

# Load fake and real news datasets
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# Assign class labels
df_fake["class"] = 0
df_true["class"] = 1

# Combine both datasets
df = pd.concat([df_fake, df_true], axis=0)
df = df.drop(["title", "subject", "date"], axis=1)
df = df.sample(frac=1).reset_index(drop=True)

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub(f'[{string.punctuation}]', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Apply preprocessing
df["text"] = df["text"].apply(wordopt)

# Define features 
x = df["text"] # Independent variable
y = df["class"] # Dependent variable

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# TF-IDF vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Train model
bayes = MultinomialNB(alpha=0.1)
bayes.fit(xv_train, y_train)

# Predictions and metrics
y_pred = bayes.predict(xv_test)
accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, target_names=["Fake", "Real"], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
cm = confusion_matrix(y_test, y_pred)

# Streamlit App UI
st.title("Fake News Detector")
st.write(f"Naïve Bayes Model Accuracy: {accuracy:.2f}")

# Show classification report
st.subheader("Classification Report")
st.dataframe(report_df.style.format({
    "precision": "{:.2f}",
    "recall": "{:.2f}",
    "f1-score": "{:.2f}",
    "support": "{:.0f}"
}))

# Show confusion matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"], ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
st.pyplot(fig)

# Text input for prediction
news = st.text_area("Enter a news article to check if it's fake or real:")

def classify_news(news):
    processed_news = wordopt(news)
    vectorized_news = vectorization.transform([processed_news])
    prediction = bayes.predict(vectorized_news)[0]
    return "Fake News" if prediction == 0 else "Real News"

if st.button("Check News"):
    if news.strip():
        prediction = classify_news(news)
        st.write("Prediction:", prediction)
    else:
        st.write("Please enter some text to analyze.")
