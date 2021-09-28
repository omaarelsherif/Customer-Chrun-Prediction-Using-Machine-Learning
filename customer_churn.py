### Customer Churn ###
"""
Description:
            Customer churn is the percentage of customers that stopped using your company's product or service during a certain time frame,
            We will use machine learning algorithm called logistic regression to predict this value by train the model in dataset of customers
            information with their chrum value and then predict future chrum value for future customers.
"""

# Importing modules
import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score
warnings.simplefilter(action='ignore', category=FutureWarning)

## 1 | Data Preprocessing ##
"""Prepare, analyse and visualize the dataset before training"""

# 1.1 Load the dataset
dataset = pd.read_csv("Dataset/churn.csv")
print(f"Dataset shape : {dataset.shape}\n")
print(f"Dataset head :\n{dataset.head()}\n")

# 1.2 Check for missing values
print(f"Missing values :\n{dataset.isna().sum()}\n")

# 1.3 Data statistics
print(f"Dataset description :\n{dataset.describe()}\n")

# 1.4 Customer Churn count
print(f"Customer Churn count: \n{dataset['Churn'].value_counts()}\n")

# 1.5 Visualize the count of customer churn
sns.countplot(dataset['Churn'])
plt.title("The count of customer churn")
plt.show()

# 1.6 The percentage of customers that are leaving or staying
retained = dataset[dataset.Churn == 'No'].shape[0]
churned  = dataset[dataset.Churn == 'Yes'].shape[0]

# The percentage of customers that stayed
print(f"{round(retained/(retained + churned) * 100, 2)} % of customers stayed in the company")

# The percentage of customers that left
print(f"{round(churned/(retained + churned) * 100, 2)} % of customers left with the company\n")

# 1.7 Visualize the churn count for both males and females
sns.countplot(x ='gender', hue='Churn', data=dataset)
plt.title("The churn count for both males and females")
plt.show()

# 1.8 Visualize the churn count for the internet service
sns.countplot(x='InternetService', hue='Churn', data=dataset)
plt.title("The churn count for the internet service")
plt.show()

# 1.9 Visualize Numeric data
numericFeatures = ['tenure', 'MonthlyCharges']
fig, ax = plt.subplots(1,2, figsize=(28, 8))
dataset[dataset.Churn == "No"][numericFeatures].hist(bins=20, color='blue', alpha=0.5, ax=ax)
dataset[dataset.Churn == "Yes"][numericFeatures].hist(bins=20, color='orange', alpha=0.5, ax=ax)
plt.show()

# 1.10 Remove unnecessary columns
dataset = dataset.drop('customerID', axis=1)

# 1.11 Convert all the non-numeric columns to numeric
for column in dataset.columns:
  if dataset[column].dtype == np.int:
    continue
  dataset[column] = LabelEncoder().fit_transform(dataset[column])

# 1.12 Scale the data
x = dataset.drop('Churn', axis=1)
y = dataset['Churn']
x = StandardScaler().fit_transform(x)

# 1.13 Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

## 2 | Model Creation ##
"""Create model to fit it to the data"""

# 2.1 Create the model
model = LogisticRegression()

# 2.2 Train the model
model.fit(X_train, y_train)

# 2.3 Predictions on the test set
y_pred = model.predict(X_test)

## 3 | Model Evaluation ##
"""Evaluate model performance"""

# 3.1 Cross validation score
_ = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
results = (_.mean(), _.std())

# 3.2 Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {round(accuracy*100, 2)} %\n")

# 3.3 Classification report
print(f"\nClassification report : \n{classification_report(y_test, y_pred)}\n")

# 3.4 Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion matrix : \n")
sns.heatmap(cm, annot=True)
plt.title("Confusion matrix")
plt.show()
