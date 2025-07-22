# Insurance Fraud Detection System

## Overview
Machine learning pipeline to identify fraudulent insurance claims using Random Forest classifier with SMOTE for handling class imbalance.

## Requirements
- Python 3.7+
- Libraries: pandas, scikit-learn, imbalanced-learn, joblib

## Installation
```bash
git clone https://github.com/yourusername/insurance-fraud-detection.git
cd insurance-fraud-detection
pip install -r requirements.txt

## Project Structure
insurance-claims-fraud-detection/
│
├── data/
│   └── insurance_fraud.csv              # Raw dataset
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb           # Jupyter Notebook for analysis and modeling
│
├── scripts/
│   ├── preprocess.py                    # Data cleaning and feature engineering
│   ├── model.py                         # Model training and evaluation
│   └── predict.py                       # Script to predict new cases
│
├── outputs/
│   ├── fraud_detection_model.pkl        # Trained model file
│   └── performance_report.txt           # Accuracy, classification report, etc.
│
├── requirements.txt                     # Dependencies
├── README.md                            # Project overview
└── .gitignore

