<!-- PROJECT TITLE -->
**<h1 align="center">Customer Churn Prediction</h1>**

<!-- LOGO -->
<p align="center">
  <img src="Images/customer_chrun_cover.png"/>
</p>

<!-- PROJECT DESCRIPTION -->
## <br>**➲ Project description**
Customer churn is the percentage of customers that stopped using your company's product
or service during a certain time frame, We will use machine learning algorithm called logistic regression to predict this value by train the model in dataset of customers information with their churn value and then predict future churn value for future customers.

<!-- PREREQUISTIES -->
## <br>**➲ Prerequisites**
This is a list of required packages for the project to be installed :
* <a href="https://www.python.org/downloads/" target="_blank">Python 3.x</a>
* Pandas 
* Numpy
* Seaborn
* Matplotlib
* Scikit-learn

Install all required packages :
 ```sh
  pip install -r requirements.txt
  ```

<!-- THE DATASET -->
## <br>**➲ The Dataset**
The customer churn dataset contain 20 feature to describe customer state
and a target column **"Churn"** which decide if the customer stay or not.
<br>**Features and target :**
- gender
- SeniorCitizen
- Partner
- Dependents
- tenure
- PhoneService
- MultipleLines
- InternetServic 
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract 
- PaperlessBilling 
- PaymentMethod
- MonthlyCharges
- TotalCharges
- Churn (target)

<!-- CODING SECTIONS -->
## <br>**➲ Coding Sections**
In this part we will see the project code divided to sections as follows:
<br>

- Section 1 | The Data :<br>
In this section we aim to do some operations on the dataset before training the model on it,
<br>processes like:
  1. Data Loading : Load the dataset
  2. Data Visualization : Visualize the dataset features and frequinces
  3. Data Cleaning : Drop unwanted features and handling missing values 
  4. Data Encoding : Encode categorical features
  5. Data Scaling : Scale large numerical features
  6. Data Splitting : Split data into training and testing sets<br><br>

- Section 2 | The Model :
The dataset is ready for training, so we create a logistic regression model using scikit-learn and thin fit it to the data, and evaluate the model by getting accuracy, classification report and confusion matrix<br>

<!-- INSTALLATION -->
## <br>**➲ Installation**
1. Clone the repo
   ```sh
   git clone https://github.com/omaarelsherif/Customer-Chrun-Prediction-Using-Machine-Learning.git
   ```
2. Open 'main.ipynb' in Google Colab or VScode and enjoy

<!-- REFERENCES -->
## <br>**➲ References**
These links may help you to better understanding of the project idea and techniques used :
1. Customer chrum in machine learning : https://bit.ly/3B7zOte
2. Logistic regression : https://bit.ly/3kqIIeA
3. Model evaluation : https://bit.ly/3B12VOO

<!-- CONTACT -->
## <br>**➲ Contact**
- E-mail   : [omaarelsherif@gmail.com](mailto:omaarelsherif@gmail.com)
- LinkedIn : https://www.linkedin.com/in/omaarelsherif
- Facebook : https://www.facebook.com/omaarelshereif
