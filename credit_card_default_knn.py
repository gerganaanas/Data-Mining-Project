# Credit Card Default App
# The model utilizes the k-nearest neighbors (KNN) algorithm
# The accuracy of the model is 0.81, biased towards predicting 
# that the customer will not default 
# By using streamlit run in terminal the model becomes an application

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# The dataset contains credit card customer information and whether they defaulted on payments.
df = pd.read_csv("UCI_Credit_Card (1).csv")

# Remove the 'ID' column as it is not a useful feature for prediction
df = df.drop(columns=['ID'])

# Define independent variables used for prediction
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

# Use the best hyperparameters found by GridSearchCV
knn = KNeighborsClassifier(n_neighbors=11, metric='manhattan')

# Train the model with the training data
knn.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_knn = knn.predict(X_test_scaled)

# Calculate and store accuracy score
accuracy = accuracy_score(y_test, y_pred_knn)

# Get the classification report
class_report = classification_report(y_test, y_pred_knn, output_dict=True)

# Convert classification report to DataFrame
class_report_df = pd.DataFrame(class_report).transpose()

# Create the confusion matrix and plot
cm = confusion_matrix(y_test, y_pred_knn)
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix")

# Streamlit App UI
st.title("Credit Card Default Prediction")
st.write(f"### Model Accuracy: {accuracy:.2f}")

# Display the classification report with table borders
st.write("### Classification Report")
st.dataframe(class_report_df.style.set_table_styles(
    [{'selector': 'thead th', 'props': [('border', '2px solid black')]},
     {'selector': 'tbody td', 'props': [('border', '1px solid black')]}]
))

# Display the confusion matrix
st.write("### Confusion Matrix")
st.pyplot(fig)

st.sidebar.header("User Input Features")

def user_input_features():
    """Creates input fields for user-provided data."""
    
    # Input for credit limit balance
    LIMIT_BAL = st.sidebar.number_input("LIMIT_BAL", min_value=0, max_value=1000000, value=50000)
    
    # Input for user's age
    AGE = st.sidebar.number_input("AGE", min_value=18, max_value=100, value=30)
    
    # Inputs for past repayment history (PAY_0, PAY_2, PAY_3)
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

# Predict if the user is likely to default
prediction = knn.predict(test_scaled)

# Display prediction result
st.write("### Prediction Result")
st.write("Default" if prediction[0] == 1 else "No Default")
