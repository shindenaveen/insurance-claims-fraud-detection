import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

def load_processed_data(file_path):
    """Load processed data from parquet file."""
    return pd.read_parquet(file_path)

def split_and_balance(df, target_col='fraud_reported_Y'):
    """Split data and apply SMOTE for class balancing."""
    X = df.drop(target_col, axis=1).copy()
    y = df[target_col].copy()
    
    # Ensure all columns are numeric
    X = X.apply(pd.to_numeric, errors='ignore')
    
    # Apply SMOTE
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X, y)
    
    return train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

def train_model(X_train, y_train):
    """Train and return Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, report_path):
    """Generate evaluation metrics and save report."""
    y_pred = model.predict(X_test)
    
    # Generate report content
    report = f"""Model Evaluation Report
{"="*40}
Confusion Matrix:
{confusion_matrix(y_test, y_pred)}

Classification Report:
{classification_report(y_test, y_pred)}

Dataset Shape: {X_test.shape}
"""
    # Save report
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Evaluation report saved to {report_path}")

def save_model(model, model_path):
    """Save trained model to disk."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    # Configuration
    data_path = "data/processed/claims_processed.parquet"
    model_path = "models/rf_model.pkl"
    report_path = "reports/performance_report.txt"
    
    # Run pipeline
    data = load_processed_data(data_path)
    X_train, X_test, y_train, y_test = split_and_balance(data)
    
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, report_path)
    save_model(model, model_path)
