
# Credit Card Default App
# Run the normal code and then the cross-validation in a Jupyter Notebook to see the best parameters

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# The dataset contains credit card customer information and whether they defaulted on payments.
df = pd.read_csv("UCI_Credit_Card (1).csv")

# Remove the 'ID' column as it is not a useful feature for prediction
df = df.drop(columns=['ID'])

# Define independent variables (features) used for prediction
selected_features = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'BILL_AMT1', 'BILL_AMT2', 'PAY_AMT1', 'PAY_AMT2']
X = df[selected_features]  # Independent variables

# Define dependent variable (target) which indicates if a customer defaults
y = df['default.payment.next.month']  # Dependent variable

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features to improve model performance
scaler = StandardScaler()
scaler.fit(X_train)  # Fit the scaler on training data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train k-Nearest Neighbors (kNN) classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_knn = knn.predict(X_test_scaled)

# Calculate and store accuracy score
accuracy = accuracy_score(y_test, y_pred_knn)

# Streamlit App UI
st.title("Credit Card Default Prediction")
st.write(f"### Model Accuracy: {accuracy:.2f}")

st.sidebar.header("User Input Features")

def user_input_features():
    """Creates input fields for user-provided data."""
    
    # Input for credit limit balance
    LIMIT_BAL = st.sidebar.number_input("LIMIT_BAL", min_value=0, max_value=1000000, value=50000)
    
    # Input for user's age
    AGE = st.sidebar.number_input("AGE", min_value=18, max_value=100, value=30)
    
    # Inputs for past repayment history (PAY_0, PAY_2, PAY_3) where -2 means no consumption, 0 means on-time payment, and positive values indicate delay
    PAY_0 = st.sidebar.slider("PAY_0 (Repayment Sep)", min_value=-2, max_value=8, value=0)
    PAY_2 = st.sidebar.slider("PAY_2 (Repayment Aug)", min_value=-2, max_value=8, value=0)
    PAY_3 = st.sidebar.slider("PAY_3 (Repayment Jul)", min_value=-2, max_value=8, value=0)
    
    # Inputs for bill amounts from previous months
    BILL_AMT1 = st.sidebar.number_input("BILL_AMT1", min_value=0, max_value=1000000, value=5000)
    BILL_AMT2 = st.sidebar.number_input("BILL_AMT2", min_value=0, max_value=1000000, value=4500)
    
    # Inputs for past payments made by the customer
    PAY_AMT1 = st.sidebar.number_input("PAY_AMT1", min_value=0, max_value=1000000, value=2000)
    PAY_AMT2 = st.sidebar.number_input("PAY_AMT2", min_value=0, max_value=1000000, value=1500)
    
    # Store user inputs in a dictionary
    data = {
        'LIMIT_BAL': [LIMIT_BAL],
        'AGE': [AGE],
        'PAY_0': [PAY_0],
        'PAY_2': [PAY_2],
        'PAY_3': [PAY_3],
        'BILL_AMT1': [BILL_AMT1],
        'BILL_AMT2': [BILL_AMT2],
        'PAY_AMT1': [PAY_AMT1],
        'PAY_AMT2': [PAY_AMT2]
    }
    
    return pd.DataFrame(data)

# Get user input
input_df = user_input_features()
st.write("### User Input Parameters", input_df)

# Scale user input using the previously fitted scaler
test_scaled = scaler.transform(input_df)

# Cross Validation to find the best parameters for KNN
from sklearn.model_selection import GridSearchCV

# Define parameter grid to search for best k and other hyperparameters
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
}

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
print(f"Best parameters found: {grid_search.best_params_}")

# Predict if the user is likely to default
prediction = knn.predict(test_scaled)

# Display prediction result
st.write("### Prediction Result")
st.write("Default" if prediction[0] == 1 else "No Default")

# Output: Best parameters found: {'metric': 'manhattan', 'n_neighbors': 11}

# Customer Churn Best Parameters

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from scipy.stats import randint

# Load dataset
data = pd.read_csv('Churn_Modelling.csv')

# Drop irrelevant columns
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Encode categorical variables
data['Geography'] = LabelEncoder().fit_transform(data['Geography'])
data['Gender'] = LabelEncoder().fit_transform(data['Gender'])

# Define features and target
X = data.drop('Exited', axis=1)
y = data['Exited']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameter space for random search
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'max_features': ['sqrt', 'log2', None],
}

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_scaled, y_train)

# Output best parameters
print(f"Best parameters found: {random_search.best_params_}")

# Use best model
best_model = random_search.best_estimator_

# Predict
y_pred = best_model.predict(X_test_scaled)

# Classification report
report = classification_report(y_test, y_pred, target_names=["Stayed", "Churned"], output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Display report 
print("Classification Report:")
print(report_df.style.format({
    "precision": "{:.2f}",
    "recall": "{:.2f}",
    "f1-score": "{:.2f}",
    "support": "{:.0f}"
}).set_table_styles([
    {'selector': 'th', 'props': [('background-color', '#4CAF50'), ('color', 'white')]},
    {'selector': 'td', 'props': [('border', '1px solid black'), ('padding', '10px')]},
    {'selector': 'tr:nth-child(odd)', 'props': [('background-color', '#f2f2f2')]},
    {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#ffffff')]},
]))

# Output: Best parameters found: {'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'min_samples_split': 6, 'n_estimators': 918}

# Fake News Detector

import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load data
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")
df_fake["class"] = 0
df_true["class"] = 1
df = pd.concat([df_fake, df_true], axis=0).drop(["title", "subject", "date"], axis=1)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Clean text function (same style)
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub(f'[{string.punctuation}]', '', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\w*\d\w*', '', text)
    return text.strip()

df["text"] = df["text"].apply(clean_text)

# Split into features and target
X = df["text"]
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Vectorize using CountVectorizer + TF-IDF Transformer
vectorizer = CountVectorizer(stop_words='english', lowercase=True, strip_accents='unicode')
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Define Naive Bayes model and grid search over alpha
model = MultinomialNB()
param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}
grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Train model
grid_search.fit(X_train_tfidf, y_train)

# Best model evaluation
print("Best Hyperparameters:", grid_search.best_params_)
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_pred) * 100

print(f"Test Accuracy: {test_accuracy:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Output: Best Hyperparameters: {'alpha': 0.1}

# AI vs Human Generated Text Detector NB
# Define best hyperparameters

import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Define Naive Bayes model
model = MultinomialNB()

# Set up hyperparameter grid for GridSearchCV
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],  # Smoothing parameter for Naive Bayes
}

# Set up GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the model using GridSearchCV
grid_search.fit(X_train_tfidf, y_train)

# Print the best parameters and best score
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best Cross-validation Accuracy: {grid_search.best_score_:.2f}")

# Get the best model
best_model = grid_search.best_estimator_

# Predict using the best model
y_pred = best_model.predict(X_test_tfidf)

# Calculate accuracy of the best model on the test data
accuracy = accuracy_score(y_test, y_pred) * 100  # Accuracy as a percentage

# Generate classification report to evaluate model performance
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# Print accuracy and classification report
print(f"Test Accuracy: {accuracy:.2f}%")
print("Model Performance on Test Data:")
print(report_df)

# Output: Best Hyperparameters: {'alpha': 0.1}

# AI Text Detector Logistic Regression

import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('Training_Essay_Data.csv')

# Shuffle data
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

# Define Logistic Regression model
model = LogisticRegression(max_iter=500)

# Define hyperparameters to tune
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization strength
    'solver': ['liblinear', 'saga'],  # Solvers to test
    'penalty': ['l2'],  # Regularization types
    'max_iter': [200, 500]  # Max iterations
}

# Perform GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_tfidf, y_train)

# Best parameters and model performance
print("Best Parameters from GridSearchCV:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

# Predict with the best model
y_pred = best_model.predict(X_test_tfidf)

# Generate classification report
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

# Output model performance
print("Model Performance on Test Data (Best Model):")
print(report_df)

# Example prediction for user input
def clean_user_input(text):
    text = text.replace("\"", "").replace("'", "")  # Remove quotes from user input
    return text

# Simulate user input (you can replace this part with actual input from your environment)
user_input = "Your sample text to check AI vs Human-generated content here."
cleaned_input = clean_user_input(user_input)

# Make prediction
text_tfidf = vectorizer.transform([cleaned_input])
probability = best_model.predict_proba(text_tfidf)[0]
ai_percentage = round(probability[1] * 100, 2)
human_percentage = round(probability[0] * 100, 2)

# Output prediction results
print(f"AI-generated: {ai_percentage}%")
print(f"Human-generated: {human_percentage}%")

# Output: Best Parameters from GridSearchCV:'C': 100, 'max_iter': 200, 'penalty': 'l2', 'solver': 'saga'

