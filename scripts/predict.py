import pandas as pd
import joblib
from pathlib import Path

def load_model(model_path):
    """Load trained model from file."""
    return joblib.load(model_path)

def preprocess_new_data(df, preprocessor_path=None):
    """
    Preprocess new data in the same way as training data.
    Optional: Load preprocessor if available
    """
    # Implement the same preprocessing steps as in preprocess.py
    df_processed = df.copy()
    
    # Handle datetimes
    datetime_cols = df_processed.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        df_processed.loc[:, col] = (df_processed[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1d')
    
    # One-hot encode (should match training data columns)
    df_processed = pd.get_dummies(df_processed)
    
    return df_processed

def predict(model, data):
    """Make predictions using the trained model."""
    return model.predict(data), model.predict_proba(data)

def save_predictions(predictions, output_path):
    """Save predictions to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == '__main__':
    # Configuration
    model_path = "models/rf_model.pkl"
    new_data_path = "data/new/new_claims.csv"
    output_path = "results/predictions.csv"
    
    # Run pipeline
    model = load_model(model_path)
    new_data = pd.read_csv(new_data_path)
    processed_data = preprocess_new_data(new_data)
    
    preds, pred_probs = predict(model, processed_data)
    results = pd.DataFrame({
        'prediction': preds,
        'probability_0': pred_probs[:, 0],
        'probability_1': pred_probs[:, 1]
    })
    
    save_predictions(results, output_path)
