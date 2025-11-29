📊 Customer Churn Prediction – Machine Learning Project

This project focuses on predicting customer churn using machine learning techniques.
The dataset used is the Telco Customer Churn Dataset.

The workflow covers:

Exploratory Data Analysis (EDA)

Data preprocessing & feature engineering

Encoding categorical data

Model training (Logistic Regression, KNN, Random Forest, XGBoost, CatBoost)

Hyperparameter tuning

Model evaluation & comparison

Saving the best model for deployment

📁 Project Structure
├── data/
│   ├── train_final.csv
│   └── Telco-Customer-Churn-dataset.csv
│
├── models/
│   ├── best_model.pkl
│   ├── column_transformer.pkl
│   ├── label_encoder.pkl
│   └── scaler.pkl
│
├── notebooks/
│   ├── Exploratory_Data_Analysis.ipynb
│   ├── exploratory_data_analysis.py
│   ├── Model_selection_and_training.ipynb
│   ├── model_selection_and_training.py
│   ├── Final_Evaluation.ipynb
│   └── Final_Evaluation.pdf
│
└── README.md   ← (this file)

📘 1. Exploratory Data Analysis (EDA)

As shown in the EDA notebook, we:

Inspected missing values

Visualized churn distribution

Studied correlations

Analyzed categorical feature frequencies

Identified key drivers of churn (e.g., contract type, tenure, monthly charges)

📄 Reference:
EDA insights taken from uploaded files such as Final_Evaluation.pdf and Exploratory Data Analysis.

⚙️ 2. Data Preprocessing

From the preprocessing script:

✔ Label Encoding applied to binary categorical columns
✔ One-Hot Encoding applied to multi-class columns
✔ Numerical features scaled using StandardScaler
✔ All transformations stored using pickle for deployment

The final pipeline includes:

column_transformer.pkl – encodes categorical features

label_encoder.pkl – handles binary labels

scaler.pkl – scales numerical inputs

🤖 3. Models Trained

Based on Model Selection & Training notebook:

Models evaluated:

Logistic Regression

KNN

Random Forest

Random Forest (Tuned)

XGBoost

XGBoost (Tuned)

CatBoost

CatBoost (Tuned)

✔ Evaluation metrics used:

Accuracy

Precision

Recall

F1-Score

ROC-AUC

📄 Model reports and evaluation tables are visible in the training notebook and PDFs (see page sections in Model_selection_and_training.pdf).

🏆 4. Best Model Selection

The best-performing model was chosen using:

best_model_name = df_results.loc[df_results['Accuracy'].idxmax(), 'Model']
best_model = models[best_model_name]


(As shown on page 9 of Model_selection_and_training.pdf 

Model_selection_and_training

)

The best model was then serialized as:

models/best_model.pkl

🧪 5. Final Evaluation

The Final Evaluation notebook demonstrates:

Confusion Matrix

ROC Curve

Feature Importance

Comparison plots of all trained models

These results confirm the stability of the selected model for deployment.

🚀 6. Deployment-Ready Assets

The following files are essential for integrating the model into a Flask or FastAPI application:

File	Description
best_model.pkl	Final chosen model
column_transformer.pkl	Categorical encoder
label_encoder.pkl	Label encoder
scaler.pkl	Scaling object
🔧 7. Technologies Used

Python

Pandas, NumPy

Scikit-learn

XGBoost

CatBoost

Matplotlib / Seaborn

Jupyter Notebook

📈 8. Results Summary

According to the comparison chart (page 8 of training PDF):
CatBoost/XGBoost achieved the highest accuracy, and one of them was selected as the best model after tuning.

🙌 9. Author

Subhash Muda

If you like this project, ⭐ star the repo!
