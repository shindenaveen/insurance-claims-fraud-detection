import pandas as pd
from pathlib import Path

def load_data(file_path):
    """Load data from Excel file."""
    return pd.read_excel(file_path)

def preprocess_data(df):
    """
    Clean and preprocess the data:
    - Handle missing values
    - Convert datetime features
    - One-hot encode categorical variables
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_processed = df.copy()
    
    # Drop rows with missing values
    df_processed = df_processed.dropna()
    
    # Convert datetime columns to numerical (days since epoch)
    datetime_cols = df_processed.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        df_processed.loc[:, col] = (df_processed[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1d')
    
    # One-hot encode categorical variables
    df_processed = pd.get_dummies(df_processed, drop_first=True)
    
    return df_processed

def save_processed_data(df, output_path):
    """Save processed data to parquet format."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    print(f"Processed data saved to {output_path}")

if __name__ == '__main__':
    # Configuration
    input_path = "data/raw/insurance_claims.xlsx"
    output_path = "data/processed/claims_processed.parquet"
    
    # Run pipeline
    raw_data = load_data(input_path)
    processed_data = preprocess_data(raw_data)
    save_processed_data(processed_data, output_path)
