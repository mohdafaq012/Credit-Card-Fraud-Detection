Fraud Detection using Random Forest Classifier
Project Overview
This project focuses on building a machine learning model to detect fraudulent transactions in a financial dataset. Given the highly imbalanced nature of fraud data, the primary objective is to create a robust classification model that can effectively identify fraudulent activities while minimizing false negatives.

The solution utilizes a scikit-learn pipeline with a StandardScaler for feature preprocessing and a RandomForestClassifier for modeling. The model's performance is evaluated using metrics specifically suited for imbalanced datasets, such as the ROC AUC and Precision-Recall AUC.

Dataset
The analysis is based on a dataset named Fraud.csv. Key features of the dataset include:

step: Time step of the transaction.

type: The transaction type (e.g., PAYMENT, TRANSFER, CASH_OUT).

amount: The transaction amount.

oldbalanceOrg & newbalanceOrig: Balance of the originating account before and after the transaction.

oldbalanceDest & newbalanceDest: Balance of the destination account before and after the transaction.

isFraud: The target variable, indicating a fraudulent transaction (1) or a legitimate one (0).

Methodology
The notebook follows a clear machine learning workflow:

Data Loading & Exploration: The data is loaded, and initial checks for dimensions, missing values, and data types are performed. The severe class imbalance in the isFraud column is identified as a key challenge.

Feature Engineering: To enhance the model's predictive power, several new features were engineered from the raw data:

Balance Differences: orig_balance_diff and dest_balance_diff to capture the change in account balances.

Amount Ratios: amount_to_oldOrig and amount_to_oldDest to represent the transaction amount relative to the original balances.

Binary Flags: orig_zero_after and dest_zero_before to capture specific transactional patterns.

Frequency Features: orig_txn_count and dest_txn_count to measure the frequency of transactions for each account.

Recipient Type: dest_is_merchant to identify merchant accounts.

Model Training: A RandomForestClassifier was chosen for its ability to handle complex, non-linear relationships. To address the class imbalance, the class_weight='balanced' parameter was used, which adjusts the model to penalize misclassifications of the minority class more heavily. The model was trained using a scikit-learn Pipeline to ensure proper scaling of numerical features.

Model Evaluation: The model's performance was rigorously evaluated on a held-out test set using a stratified split to ensure proportional representation of the fraudulent class. Key results include:

ROC AUC Score: 0.9987

Precision-Recall AUC (Average Precision): 0.9584

Classification Report: Showed high precision, recall, and F1-score for both classes. The model achieved a high recall for the fraudulent class, indicating its effectiveness at identifying true fraud cases.

Dependencies
The following Python libraries are required to run this notebook:

numpy

pandas

scikit-learn

matplotlib

seaborn

joblib

You can install these dependencies using pip:

pip install numpy pandas scikit-learn matplotlib seaborn joblib

How to Run
Ensure you have the required dependencies installed.

Place the Fraud.csv dataset in the same directory as the notebook.

Execute the cells in the Untitled.ipynb notebook sequentially.

Future Work
Hyperparameter Tuning: Use GridSearchCV or RandomizedSearchCV to fine-tune the RandomForestClassifier for optimal performance.

Explore Other Models: Experiment with other tree-based models like XGBoost or LightGBM, which are also highly effective on tabular data and imbalanced problems.

Interactive Dashboard: Create a simple dashboard to visualize the model's predictions and key performance indicators.
