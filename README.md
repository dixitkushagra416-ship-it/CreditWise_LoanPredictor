# 🏦 CreditWise — Loan Approval Prediction System

CreditWise is a Machine Learning project that predicts whether a loan application is likely to be approved or rejected based on an applicant's financial, personal, employment, and loan-related information.

The project uses a **Decision Tree Classifier** with preprocessing and hyperparameter tuning through GridSearchCV.

##  Project Objective

The objective of CreditWise is to build a machine learning model that can analyse applicant information and predict loan approval outcomes.

The project demonstrates a complete ML workflow, from data exploration and preprocessing to model training, evaluation, and deployment.

##  Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Dataset Overview

The dataset contains 1,000 records and 20 columns.

It includes information such as:

- Applicant income and co-applicant income
- Credit score
- Debt-to-income ratio
- Existing loans
- Savings and collateral value
- Loan amount and loan term
- Employment status
- Education level
- Marital status
- Loan purpose
- Property area

**Target Variable:** `Loan_Approved`

- `Yes` — Loan approved
- `No` — Loan not approved

Rows with missing target values are excluded during model training.

##  Machine Learning Workflow

1. **Exploratory Data Analysis (EDA)**
   - Analysed target distribution.
   - Explored applicant demographics.
   - Examined relationships between credit scores and loan approval.

2. **Data Preprocessing**
   - Removed `Applicant_ID` from model features.
   - Handled missing numerical values using median imputation.
   - Handled missing categorical values using most-frequent imputation.
   - Applied One-Hot Encoding to categorical features.

3. **Model Training**
   - Used a Decision Tree Classifier.
   - Applied class balancing using `class_weight="balanced"`.

4. **Hyperparameter Tuning**
   - Used GridSearchCV with 5-fold cross-validation.
   - Optimised using macro F1-score.
   - Applied Cost Complexity Pruning through `ccp_alpha`.

5. **Model Evaluation**
   - Evaluated the selected model on a separate test dataset.
   - Used accuracy, precision, recall, F1-score, and confusion matrix.

## Model Performance

| Metric | Score |
|---|---:|
| Test Accuracy | 96.84% |
| Macro F1-Score | 0.9641 |
| Best CV Macro F1 | 0.9582 |

**Best Hyperparameter:**

`ccp_alpha = 0.005`

These results are based on the current dataset and train-test split. They do not guarantee performance on unseen real-world loan applications.

##  Run Locally

Clone the repository:

```bash
git clone <https://github.com/dixitkushagra416-ship-it/CreditWise_LoanPredictor>
cd CreditWise
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Project Structure

```text
CreditWise/
│
├── app.py
├── train.py
├── model.joblib
├── loan_approval_data.csv
├── requirements.txt
├── metrics.json
├── README.md
└── .gitignore
```

##  Future Improvements

- Experiment with Random Forest and XGBoost.
- Improve model generalisation.
- Perform feature importance analysis.
- Add model explainability using SHAP.
- Improve the Streamlit interface.
- Evaluate fairness and probability calibration.



---

**Built as a Machine Learning project using Python and Scikit-learn.**

## 📁 Project Structure

```text
CreditWise_LoanPredictor/
│
├── README.md
├── loan_predictor_final.ipynb
├── loan_approval_data.csv
└── .gitignore
```

