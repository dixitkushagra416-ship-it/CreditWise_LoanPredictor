# CreditWise_LoanPredictor
ML-based Loan Approval Prediction using Decision Tree, with EDA, data preprocessing, GridSearchCV, and cost-complexity pruning. Achieved 96.84% test accuracy and 0.96 macro F1-score. Built using Python, Pandas, NumPy, and Scikit-learn.
# CreditWise – Loan Approval Prediction System

A Machine Learning project that predicts loan approval outcomes using a Decision Tree Classifier. The project focuses on exploratory data analysis, data preprocessing, model training, hyperparameter optimization, and performance evaluation.

## Project Overview

CreditWise aims to predict whether a loan application will be approved or rejected based on applicant and financial information.

## Key Features

- Exploratory Data Analysis (EDA)
- Missing value detection and imputation
- Numerical and categorical feature preprocessing
- One-Hot Encoding
- Decision Tree Classification
- Class balancing using `class_weight`
- Hyperparameter tuning with GridSearchCV
- Cost-complexity pruning using `ccp_alpha`
- 5-fold cross-validation
- Model evaluation using classification reports and confusion matrices

## Model Performance

| Metric | Score |
|---|---:|
| Test Accuracy | 96.84% |
| Macro Precision | 0.96 |
| Macro Recall | 0.97 |
| Macro F1-Score | 0.96 |
| Best CV Macro F1 | 0.9582 |

**Best Hyperparameter:** `ccp_alpha = 0.005`

## Visualizations

The project includes the following visualizations:

1. Loan Approval Distribution by Gender
2. Applicant Gender Distribution
3. Education Level by Gender
4. Credit Score vs Loan Approval
5. Confusion Matrix

All exported plots are available in the `assets/plots/` directory.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Project Structure

```text
CreditWise/
│
├── assets/
│   └── plots/
│       ├── loan_approval_by_gender.png
│       ├── applicant_gender_distribution.png
│       ├── education_level_by_gender.png
│       ├── credit_score_vs_loan_approval.png
│       └── confusion_matrix.png
│
├── loan_predictor_final.ipynb
├── loan_approval_data.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Usage

**1. Clone the repository**

```bash
git clone <YOUR_REPOSITORY_URL>
```

**2. Navigate to the project directory**

```bash
cd CreditWise
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Launch Jupyter Notebook**

```bash
jupyter notebook
```

Open `loan_predictor_final.ipynb` and run the cells.

**Note:** Place `loan_approval_data.csv` in the project root directory before executing the notebook.

## Future Improvements

- Compare Decision Tree with Random Forest and Logistic Regression.
- Implement ROC-AUC and Precision-Recall evaluation.
- Perform feature importance analysis.
- Investigate model fairness and potential data leakage.
- Explore deployment using Flask or Streamlit.

## Disclaimer

This project is developed for educational and portfolio purposes. Predictions should not be used as the sole basis for real-world lending decisions.

---

**Developed as part of my Machine Learning learning journey.**
