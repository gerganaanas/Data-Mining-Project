
# Data-Mining-Project
# Credit Card Default Prediction
## 1.Business Understanding
**Project Objectives:**
The **primary objective** of this project is to develop a machine-learning model that accurately predicts credit card default. The model utilizes the K-Nearest Neighbors (KNN) algorithm to analyze customer data and determine the likelihood of default.
**Background:**
Predicting credit card default is important for financial institutions to mitigate risk (Bhandary & Ghosh, 2025). This tool addresses the **business need** to identify customers likely to default, enabling proactive measures. **Stakeholders**, including banks and credit card companies, are concerned with minimizing losses and maintaining healthy portfolios. They require tools that provide actionable insights and enable timely interventions. **Considerations** include potential biases in the dataset, model interpretability, and the need for real-time predictions. Acknowledging the inherent limitations of predictive models, the project recognizes the possibility of misclassifications.
**Success Criteria:**
1. Achieve an accuracy exceeding 80% on the test dataset.
2. Provide reliable predictions of default probability.
3. Ensure a user-friendly interface for inputting customer data and receiving predictions.
## 2. Data Understanding
**Data Collection:** The dataset, "UCI_Credit_Card (1).csv," contains credit card customer information and default status. It serves as the foundation for the prediction model.
**Data Description:** The dataset includes features such as credit limit, age, payment history, and bill amounts serving as the independent variables. The dependent variable is "default.payment.next.month," indicating whether a customer defaulted.
**Initial Exploration:** The dataset was analyzed to understand feature distributions and class balance. There is an imbalance in the dataset, with more non-default cases. Potential biases from data sources were noted.
## 3. Data Preparation
To **prepare** the data for the K-Nearest Neighbors (KNN) model, several essential steps were taken. First, feature selection was performed by removing the "ID" column, which was deemed irrelevant for prediction, and identifying the most relevant features. Subsequently, the dataset was split into 80% training data and 20% testing data, allowing for strong model training and evaluation. Finally, the features were **standardized** using StandardScaler to ensure optimal model performance. These preprocessing steps collectively ensured that the data was appropriately **formatted and scaled**, making it suitable for effective training of the KNN model.
## 4. Modeling
The KNN algorithm was **selected** for its effectiveness in classification tasks. The model was trained using **optimized hyperparameters** (n_neighbors=11, metric='manhattan') to increase accuracy. The previous model was using default parameters which led to 79% accuracy. After the adjustment the model is at 81% accuracy. The data from the model was **split** into training (80%) and test data(20%). The training data was used to **fit the model**, and predictions were made on the test set.
## 5.Evaluation
The classification report indicates a model with a reasonable overall performance, achieving an **accuracy of 81.15%**. While not as high, this accuracy still signifies a degree of correctness in its predictions. However, there are notable differences in performance between the two classes. Specifically, the model exhibits a precision of 83.79% for class 0 (non-defaults), meaning that 83.79% of instances predicted as non-defaults were actually non-defaults. For class 1 (defaults), the precision is 62.33%, indicating that the model is less precise in identifying actual defaults. Therefore, the model is more biased toward non-defaults. 
Furthermore, the model demonstrates a high **recall** of 94.07% for class 0, successfully identifying a large proportion of all non-default cases. However, the recall for class 1 is significantly lower at 35.03%, suggesting that unfortunately the model misses a substantial number of actual defaults.
The **F1-scores**, which balance precision and recall, reflect this disparity. For class 0, the F1-score is 88.63%, indicating a good balance. For class 1, the F1-score is 44.86%, revealing a less effective balance. The macro average F1-score is 66.74%, and the weighted average F1-score is 79.05%, further confirming the model's varying performance across the classes. The **support values** indicate that class 0 has 4687 instances and class 1 has 1313 instances, demonstrating a class imbalance with a higher representation of non-default cases.
The model implements a credit card default prediction model using the K-Nearest Neighbors (KNN) algorithm. Its **strengths** lie in its structured code, effective feature scaling, and a user-friendly Streamlit interface. The data splitting ensures fair testing. Additionally, the classification report offers detailed performance insights to the user. However, as a **weakness** the model demonstrates a **bias** towards predicting non-defaults, as evidenced by the significantly lower recall and F1-score for defaults.
Additionally, the confusion matrix shows that this model demonstrates a strong ability to correctly identify customers who will not default (True Negative = 4409). However, its ability to correctly predict defaults is moderate (True Positive = 460). While the model has a relatively low rate of incorrectly pointing out non-defaulters as defaulters (False Positive = 278), it significantly struggles with identifying actual defaulters, as indicated by a high number of False Negatives (853). This is could be a major concern for a credit card company because failing to identify a substantial portion of customers who will default could lead to significant financial losses.
## 6. Deployment
The model was **deployed** as a Streamlit web application, allowing users to input customer data and receive default predictions. Continuous **monitoring** of model performance and periodic retraining with new data will ensure accuracy. **Future work** includes exploring other algorithms, addressing class imbalance, and further tuning hyperparameters. **Challenges** include maintaining accuracy over time and handling potential biases.
## Conclusion
A credit card default prediction model using KNN was successfully developed. The model achieves a reasonable accuracy, with room for improvement in predicting defaults. Future efforts will focus on enhancing performance and addressing limitations and bias. As a first model this one personally taught us what models need to operate and how we can increase the performance of the model through different functions. Furthermore, we also learned how to deploy our models as an application.


# Customer Churn Prediction
## 1. Business Understanding
**Project Objectives:**
The **primary goal** of this project is to develop a machine learning model that accurately predicts customer churn. The model utilizes a Random Forest classifier to analyze customer data and determine the likelihood of a customer leaving the bank.
**Background:**
Customer churn is a significant concern for banks, impacting revenue and customer lifetime value. This tool addresses the **primary business problem** of identifying customers at high risk of churn across various customer segments. The business need is to enable proactive retention strategies and minimize customer loss. **Stakeholders** include bank management, marketing, customer service, and data science teams, all concerned with customer retention (Singh et al., 2023). **Constraints** include data privacy regulations, the need for actionable insights, and the challenge of balancing model complexity with interpretability. A drawback of churn prediction models is the potential for false positives, leading to unnecessary retention efforts.
**Success Criteria:**
1. Achieve a high recall for the "Churned" class, minimizing missed churners.
2. Provide reliable predictions to inform targeted retention strategies.
3. Ensure an easy-to-use interface for bank personnel to input customer data and receive churn predictions.
## 2. Data Understanding
**Data Collection:**
The dataset consists of customer information, including demographic and transactional data, labeled as either "Stayed" (0) or "Churned" (1). It was sourced from "Churn_Modelling.csv" and serves as the foundation for churn prediction.
**Data Description:**
The dataset includes independent variables such as 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', and 'EstimatedSalary'. The dependent variable, 'Exited', indicates whether a customer churned (1) or stayed (0).
**Initial Exploration:**
Initial exploration focused on understanding the distribution of features and the class balance between "Stayed" and "Churned" customers. No outliers were identified.
## 3. Data Preparation
The data preparation process involved several steps to ensure high-quality input for analysis. First, **data cleaning** was performed by dropping irrelevant columns ('RowNumber', 'CustomerId', 'Surname') and encoding categorical features ('Geography', 'Gender') using LabelEncoder.
For **data integration**, if additional data sources were to be used, they would be merged based on customer identifiers while maintaining consistent labeling. **Feature engineering** was not extensively performed in this iteration but could be explored in future work. The dataset was split into 80% for training and 20% for testing.
Overall, the steps followed in data preparation included removing irrelevant data, encoding categorical variables, and splitting the data for training and testing.
## 4. Modeling
In the **modeling phase**, a Random Forest classifier was chosen for its strong performance in classification tasks and its ability to handle diverse data types. The dataset was divided into 80% for training and 20% for testing, allowing for proper evaluation of the model's predictive capabilities. The **optimal parameters** were found to be 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'min_samples_split': 6, 'n_estimators': 918 through a random grid serach improving the models performance from 86% to 87% . The Random Forest model was then created and trained. Evaluation metrics, including accuracy, precision, recall, and F1-score, were subsequently used to assess the model's performance on the test dataset and can be observed in the app.
## 5. Evaluation
After tuning the hyperparameters, the model now performs even better than the previous one. The current model achieves an overall accuracy of 87%, indicating an improvement in its predictive capabilities. Specifically, the model demonstrates strong precision for the "Stayed" class, with 88% of customers predicted to stay actually remaining. For the "Churned" class, the precision is slightly lower at 77%, suggesting that while the model identifies churners, it still produces some false positives.
Furthermore, the model exhibits a high recall for the "Stayed" class, successfully identifying 97% of customers who remained. However, the recall for the "Churned" class is 45%, indicating that the model still misses a significant portion of customers who actually churned.
The F1-scores, which balance precision and recall, are 92% for the "Stayed" class and 57% for the "Churned" class. The macro average F1-score is 75%, and the weighted average F1-score is 85%, indicating that the model performs well overall.
The support values remain the same, with 1607 instances for the "Stayed" class and 393 instances for the "Churned" class, highlighting the persistent class imbalance.
True Negatives (TN): 1565 — Correctly predicted stayed customers.
False Positives (FP): 42 — Customers predicted to churn, but they stayed.
False Negatives (FN): 217 — Customers predicted to stay, but they churned.
True Positives (TP): 176 — Correctly predicted churned customers.
This confusion matrix shows the model tends to favor predicting "Stayed", likely due to the imbalance in the dataset. While overall accuracy is high, the number of false negatives indicates a need for improved detection of customers likely to churn.

## 6. Deployment
For **deployment**, the model was implemented as a Streamlit web application, allowing bank personnel to input customer data and receive churn predictions. For **monitoring** the model, prediction accuracy and recall will be continuously tracked, and periodic retraining with updated data will ensure the model’s performance remains high.
**Future work** includes exploring more in depth hyperparameter tuning, incorporating additional models (e.g., XGBoost, Logistic Regression), and addressing the class imbalance to improve recall for the "Churned" class. **Challenges** moving forward include maintaining model accuracy over time and ensuring the model’s predictions are actionable for retention strategies. Regular maintenance will involve periodic retraining and updates to the dataset.
## Conclusion
This project demonstrates a machine learning approach to predicting customer churn. The Random Forest model achieves a reasonable accuracy but needs improvement in capturing all potential churners. Future efforts will focus on enhancing the model’s recall and exploring more advanced techniques to improve its reliability and validity for actionable retention strategies. This model taught us more about using Random Forest and its benefits.

# Fake News Detector
## 1. Business Understanding
**Project Objectives:**
The **primary goal** of this project is to develop a machine-learning model that classifies news articles as either fake or real. The model uses a Naïve Bayes (NB) classifier with TF-IDF vectorization to analyze input text and determine its authenticity.
**Background:**
With the spread of misinformation, distinguishing between fake and real news has become extremly important (Chen et al., 2023). This tool addresses the **primary business problem** of detecting fake news across various platforms. The business need is to ensure information integrity and deal with the spread of false narratives. **Stakeholders** include media organizations, social media platforms, and the general public, all concerned with the reliability of news (Chen et al., 2023). **Constraints** include the evolving nature of fake news, potential biases in the training dataset, and the need for real-time processing. A drawback of news detectors is the occurrence of both false positives and false negatives (Knilans, 2024).
**Success Criteria:**
1. Achieve an accuracy of at least 90% on the test dataset.
2. Provide reliable classification results for fake vs. real news.
3. Ensure an easy-to-use interface for users to input news articles and receive classification results.
## 2. Data Understanding
**Data Collection:**
The dataset consists of news articles labeled as either fake (0) or real (1). It has been sourced from "Kaggle" and serves as the foundation for classification. With an decent reliability.
**Data Description:**
The dataset is a **combination** of two files "True.csv" and "Fake.csv" they both include "title", "text", "subject" and "date" columns. The datasets were combined using pandas.concat(). In this model only two columns were used. The "text" column, which contains the textual content of the news articles, the independent variable. And a "class" column, which was developed after combining the data sets and serves as the dependent variable, indicating whether a news article is fake or real.
**Initial Exploration:**
To minimize bias, the dataset is **shuffled** before analysis. There were **no missing values** in the text column. The initial exploration focuses on understanding the class distribution to determine whether there is a balance between fake and real news samples. When it comes to the reliability of the data, potential biases in training samples might be present due to the data coming from different sources.
## 3. Data Preparation
The data preparation process involved several steps to ensure high-quality input for analysis. First, **data cleaning** was performed by removing punctuation, special characters, newline characters, unnecessary quotes, URLs, HTML tags, and words containing numbers to enhance consistency.
For **data integration**, if additional datasets are to be used, they would be merged based on text attributes while maintaining consistent labeling such as the two files that have already been combined. **Feature engineering** was also applied to improve classification performance. Specifically, text data was transformed into **TF-IDF vectors**, allowing for a more meaningful representation of textual features.
Overall, the steps followed in data preparation included normalizing text for consistency and applying TF-IDF transformation to extract useful features.
## 4. Modeling
In the **modeling phase**, Naïve Bayes (NB) was selected for its effectiveness in text classification tasks. The dataset was split into 75% for training and 25% for testing to evaluate the model's performance. The Naïve Bayes model was trained with an **optimized hyperparameter** (alpha=0.1) to improve classification accuracy. For evaluation, several metrics were used, including accuracy, precision, recall, and F1-score.
## 5. Evaluation
The classification report indicates a model with excellent overall performance, achieving an accuracy of 94%. This signifies a very high degree of correctness in its predictions. Specifically, the model exhibits excellent precision for fake news, with 94% of instances predicted as fake actually being fake, and a precision of 94% for real news. Furthermore, it demonstrates a high recall for fake news (95%) and a recall of 93% for real news.
The F1-scores, which balance precision and recall, are also very high for both classes, at 94% for fake news and 93% for real news. The macro and weighted average F1-scores are both 94%.
The support values show that fake news has 5895 instances and real news has 5330 instances, indicating a balanced dataset.
The model implements a text classification model using Naïve Bayes with TF-IDF vectorization, effectively distinguishing between fake and real news. Its **strengths** lie in its structured code, effective text preprocessing, and efficient use of TF-IDF, combined with a user-friendly Streamlit interface. The way it splits the data ensures fair testing. Additionally, the classification report offers detailed performance insights to the user. Through trial and error, the model was found to be **slightly biased** towards fake news.
True Negatives (TN): 5611 — Fake news correctly identified as fake.
False Positives (FP): 284 — Fake news incorrectly identified as real.
False Negatives (FN): 374 — Real news incorrectly identified as fake.
True Positives (TP): 4956 — Real news correctly identified as real.
This confusion matrix reflects a well-performing model. While there are some misclassifications, the model maintains a strong balance between false positives and false negatives and performs well across both categories.
## 6. Deployment
For **deployment**, the model was implemented as a Streamlit web app, allowing users to input news articles and receive results classifying them as fake or real. For **monitoring** the model, user inputs and classification results will be logged for continuous evaluation, and periodic retraining with new labeled data will ensure the model’s accuracy remains high.
**Future work** includes exploring deep learning models, such as Transformer-based classifiers, and expanding the dataset's diversity to enhance performance. **Challenges** moving forward include handling misleading text modifications and **maintaining** model accuracy over time. Regular maintenance will involve periodic retraining and updates to the dataset.
## Conclusion
This project successfully demonstrates a machine-learning approach to distinguishing fake news from real news. The model, using TF-IDF and Naïve Bayes classification, achieves a high accuracy of approximately 94%. Looking ahead, expanding dataset coverage and exploring more advanced models will help further enhance the model’s reliability and validity. This model taught us how to combine datasets, how the Naive Bayes algoithm works and how to learn more about a model through testing it with external data. 


# AI Text Detection NB Model
## 1. Business Understanding
**Project Objectives:**
The **primary goal** of this project is to develop a machine-learning model that classifies text as either AI-generated or human-written. The model uses a Naive Bayes classifier with TF-IDF vectorization to analyze input text and determine its origin with a probability score.
**Background:**
With the rise of AI-generated content, distinguishing between human-written and AI-generated text has become very important. This tool helps in solving the **primary business problem** of detecting AI-generated text in various domains, including academia, journalism, and content moderation. The business need is to ensure transparency and authenticity in textual data. Furthermore, **the stakeholders** are educational institutions, which are concerned about academic integrity, publishers, who worry about misinformation, and businesses needing to verify the authenticity of user-generated content (Marinkovic, 2024). Some **constraints that must be considered** include potential biases in the training dataset, model interpretability, and real-time processing requirements. A drawback of AI detectors is the occurrence of both false positives and false negatives (Knilans, 2024).
**Success Criteria:**
1.	Achieve an accuracy of at least 90% on the test dataset.
2.	Provide reliable probability scores for AI vs. human classification.
3.	Ensure an easy-to-use interface for users to input text and receive classification results.
## 2. Data Understanding  
**Data Collection:**
The dataset, “train_from_LLM-Detect_AI-GT_1MNB-3SGD.csv”, consists of text samples labeled as either AI-generated (1) or human-written (0). It has been sourced from “Kaggle” and serves as the foundation for classification.  
**Data Description:**
The dataset includes a “text” column, which contains the textual content to be classified. A “RDizzl3_seven” column, which serves as the target variable, indicating whether a text is AI-generated or human-written. It also includes a column “prompt name” used to generate AI text (if applicable) and “source” column signifying the origin of the text sample (e.g., AI model, human author, dataset source). 
**Initial Exploration:**
To minimize bias, the dataset is shuffled before analysis. There were no missing values in the text column, however just in case they were taken care of in the next step. The initial exploration focuses on understanding the class distribution to determine whether there is a balance between AI-generated and human-written samples, which was proven to be relatively balanced. When it comes to the reliability of the data potential biases in training samples might be present due to the data coming from different sources.
## 3. Data Preparation
The data preparation process involved several steps to ensure high-quality input for analysis. First, **data cleaning** was performed by **removing NaN values**, **standardizing text** by converting it to lowercase, and eliminating punctuation. Additionally, special characters, newline characters, and unnecessary quotes were removed to enhance consistency.
For **data integration**, if additional datasets are to be used, they would be merged based on text attributes while maintaining consistent labeling. **Feature engineering** was also applied to improve classification performance. Specifically, text data was transformed into **TF-IDF vectors**, allowing for a more meaningful representation of textual features.
Overall, the steps followed in data preparation included handling missing data, normalizing text for consistency, and applying TF-IDF transformation to extract useful features.
## 4. Modeling  
In the **modeling phase**, Naive Bayes was selected for its efficiency and strong performance in text classification tasks, but logistic regression was also considered, that is why it was used in the next model. Initially, default **hyperparameters** were used without defining alpha and the model had a 94.71% accuracy, but after a **cross-validation analysis** the best alpha was revealed to be 0.1, slightly improving the model to 97.6% accuracy. The dataset was split into 70% for training and 30% for testing to evaluate the model's performance. For evaluation, several metrics were used, including accuracy, precision, recall, and F1-score. The model outperformed random classification as well as the initial model before the hyperparameters were adjusted, demonstrating high accuracy.  
 ## 5. Evaluation  
The classification report indicates a model with good overall performance, achieving an **accuracy of 97.60%**. This signifies a very high degree of correctness in its predictions. Specifically, the model exhibits excellent **precision** for class 1, with 99.47% of instances predicted as class 1 actually belonging to that class. Furthermore, it demonstrates a **high recall** for class 0, successfully identifying 99.58% of all instances of this class. While the model performs strongly overall, there is a slight discrepancy in recall between the two classes. The recall for class 1 is 95.24%, which, while still high, is marginally lower than the recall for class 0. This suggests the model misses a small portion of actual class 1 instances. The **F1-scores**, which balance precision and recall, are also very high for both classes, at 97.83% for class 0 and 97.31% for class 1, indicating a strong balance between precision and recall. The macro average F1-score is 97.57%, and the weighted average F1-score is 97.60%, further confirming the model's good performance across both classes. The **support values** indicate that class 0 has 7326 instances and class 1 has 6135 instances, demonstrating a relatively balanced dataset.
The model implements a text classification model using Naive Bayes with TF-IDF vectorization, effectively distinguishing between AI and human-generated text. Its **strengths** lie in its structured code, effective text preprocessing, and efficient use of TF-IDF, combined with a user-friendly Streamlit interface. The way it splits the data ensures fair testing. Additionally, the classification report offers detailed performance insights to the user.
However, the approach has **limitations**. The reliance on NB, a relatively simple model, might restrict performance compared to more advanced methods. Limited feature engineering could lead to suboptimal results and potential overfitting. The black-box nature of user input predictions hinders interpretability, and the model's reliance on the dataset's quality shows the need for diverse and representative training data. Through trial and error, the model was detected to be **slightly biased** towards AI-generated content.
## 6. Deployment  
For **deployment**, the model was implemented as a Streamlit web app, allowing users to input text and receive results classifying it as AI-generated or human-written. For **monitoring** the model user inputs and classification results will be logged for continuous evaluation, and periodic retraining with new labeled data will ensure the model’s accuracy remains high.  
**Future work** includes exploring deep learning models, such as Transformer-based classifiers, and expanding the dataset's diversity to enhance performance. There could also be implemented a real-time API integration for broader application. **Challenges** moving forward include handling misleading text modifications and maintaining model accuracy over time. Regular **maintenance** will involve periodic retraining and updates to the dataset.  
## Conclusion  
This project successfully demonstrates a machine-learning approach to distinguishing AI-generated text from human-written content. The model, using TF-IDF and Naive Bayes classification, achieves high accuracy. Looking ahead, expanding dataset coverage and exploring more advanced models will help further enhance the model’s reliability and validity. This model taught us how difficult it is to truly make an accurate AI text detector. We also learned that the data the model has influences it to a large extent, despite the high accuracy.

# AI Text Detection Logistic Regression Model
## 1. Business Understanding
**Project Objectives:**
The **primary goal** of this project is to develop a machine-learning model that classifies text as either AI-generated or human-written. The model uses a Logistic Regression classifier with TF-IDF vectorization to analyze input text and determine its origin with a probability score.
**Background:**
With the rise of AI-generated content, distinguishing between human-written and AI-generated text has become very important. This tool helps in solving the **primary business problem** of detecting AI-generated text in various domains, including academia, journalism, and content moderation. The business need is to ensure transparency and authenticity in textual data. Furthermore, **the stakeholders** are educational institutions, which are concerned about academic integrity, publishers, who worry about misinformation, and businesses needing to verify the authenticity of user-generated content (Marinkovic, 2024). Some **constraints** that must be considered include potential biases in the training dataset, model interpretability, and real-time processing requirements. A drawback of AI detectors is the occurrence of both false positives and false negatives (Knilans, 2024).
**Success Criteria:**
1.	Achieve an accuracy of at least 90% on the test dataset.
2.	Provide reliable probability scores for AI vs. human classification.
3.	Ensure an easy-to-use interface for users to input text and receive classification results.
## 2. Data Understanding
**Data Collection:**
The dataset, “Training_Essay_Data.csv”, consists of text samples labeled as either AI-generated (1) or human-written (0). It has been sourced from “Kaggle” and serves as the foundation for classification. With an average reliability. 
**Data Description:**
The dataset includes a “text” column, which contains the textual content to be classified. A “generated” column, which serves as the dependent variable, indicating whether a text is AI-generated or human-written
**Initial Exploration:**
To minimize bias, the dataset is **shuffled** before analysis. There were no missing values in the text column. The initial exploration focuses on understanding the class distribution to determine whether there is a balance between AI-generated and human-written samples. When it comes to the reliability of the data potential biases in training samples might be present due to the data coming from different sources.
## 3. Data Preparation
The data preparation process involved several steps to ensure high-quality input for analysis. First, **data cleaning** was performed by removing punctuation, special characters, newline characters, and unnecessary quotes to enhance consistency.
For **data integration**, if additional datasets are to be used, they would be merged based on text attributes while maintaining consistent labeling. **Feature engineering** was also applied to improve classification performance. Specifically, text data was transformed into **TF-IDF vectors**, allowing for a more meaningful representation of textual features.
Overall, the steps followed in data preparation included normalizing text for consistency and applying TF-IDF transformation to extract useful features.
## 4. Modeling
In the **modeling phase**, Logistic Regression was selected for its effectiveness in binary classification tasks. The dataset was split into 70% for training and 30% for testing to evaluate the model's performance. The Logistic Regression model was trained with **optimized hyperparameters** (C=100, max_iter=500, penalty='l2', solver='saga') to improve the previous model’s accuracy of 98.7%. For evaluation, several metrics were used, including accuracy, precision, recall, and F1-score.
## 5. Evaluation
The **classification report** indicates a model with excellent overall performance, achieving an accuracy of 99.43%. This signifies a very high degree of correctness in its predictions. Specifically, the model exhibits **excellent precision** for class 0, with 99.48% of instances predicted as class 0 actually belonging to that class, and a precision of 99.35% for class 1. Furthermore, it demonstrates a high recall for class 0, successfully identifying 99.56% of all instances of this class, and a recall of 99.24% for class 1.
The **F1-scores**, which balance precision and recall, are also very high for both classes, at 99.52% for class 0 and 99.29% for class 1, indicating a strong balance between precision and recall. The macro average F1-score is 99.41%, and the weighted average F1-score is 99.43%, further confirming the model's good performance across both classes. The **support values** indicate that class 0 has 5208 instances and class 1 has 3536 instances, demonstrating a relatively balanced dataset.
The model implements a text classification model using Logistic Regression with TF-IDF vectorization, effectively distinguishing between AI and human-generated text. Its **strengths** lie in its structured code, effective text preprocessing, and efficient use of TF-IDF, combined with a user-friendly Streamlit interface. The way it splits the data ensures fair testing. Additionally, the classification report offers detailed performance insights to the user. Through trial and error the model was found to be **biased** towards human-generated content.
Additionally, the confusion matrix correctly identifies human-generated text (True Positive = 5185) and AI-generated text (True Negative = 3508) with high precision. The low False Positive (23) and False Negative (28) values indicate few incorrect classifications. Consequently, the F1-scores for both "Human" (99.51%) and "AI" (99.28%) classifications are also very high, signifying a well-balanced performance. While the model exhibits excellent accuracy, the implications of the few misclassifications (false positives potentially labeling legitimate human text and false negatives failing to detect AI text) should be carefully considered based on the model's specific application.
## 6. Deployment
For **deployment**, the model was implemented as a Streamlit web app, allowing users to input text and receive results classifying it as AI-generated or human-written. For **monitoring** the model, user inputs and classification results will be logged for continuous evaluation, and periodic retraining with new labeled data will ensure the model’s accuracy remains high.
**Future work** includes exploring deep learning models, such as Transformer-based classifiers, and expanding the dataset's diversity to enhance performance. There could also be implemented a real-time API integration for broader application. **Challenges** moving forward include handling misleading text modifications and **maintaining** model accuracy over time. Regular maintenance will involve periodic retraining and updates to the dataset.
## Conclusion
This project successfully demonstrates a machine-learning approach to distinguishing AI-generated text from human-written content. The model, using TF-IDF and Logistic Regression classification, achieves a high accuracy of approximately 99%. Looking ahead, expanding dataset coverage and exploring more advanced models will help further enhance the model’s reliability and validity. This model further confirmed the importance of the data the model was trained and tested on.

# References

Datasets were found on Kaggle, so was our inspiration for the code of the models. The datasets can be found in the repository. 

Bhandary, R., & Ghosh, B. K. (2025). Credit Card Default Prediction: An Empirical Analysis on Predictive Performance Using Statistical and Machine Learning Methods. Journal of Risk and Financial Management, 18(1), 23. https://doi.org/10.3390/jrfm18010023

Chen, S., Xiao, L., & Kumar, A. (2023). Spread of misinformation on social media: What contributes to it and how to combat it. Computers in Human Behavior, 141(107643), 107643. https://doi.org/10.1016/j.chb.2022.107643

Knilans, G. (2024, October 28). The Dark Side of AI Detectors: Why Accuracy Is Not Guaranteed - Trade Press Services. Trade Press Services. https://www.tradepressservices.com/ai-detectors/

Marinkovic, P. (2024, August 14). 4 Ways AI Content Detectors Work To Spot AI. Surfer; Surfer. https://surferseo.com/blog/how-do-ai-content-detectors-work/

Singh, P. P., Anik, F. I., Senapati, R., Sinha, A., Sakib, N., & Hossain, E. (2023). Investigating customer churn in banking: A machine learning approach and visualization app for data science and management. Data Science and Management, 7(1). https://doi.org/10.1016/j.dsm.2023.09.002



