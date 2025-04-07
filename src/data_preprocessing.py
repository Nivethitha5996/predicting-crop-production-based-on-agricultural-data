import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
import os

def load_data():
    """Load raw dataset with robust error handling"""
    try:
        data_path = Path('data/FAOSTAT_data.csv')
        
        # Check if file exists
        if not data_path.exists():
            raise FileNotFoundError(
                f"Data file not found at: {data_path.absolute()}\n"
                "Please ensure:\n"
                "1. 'FAOSTAT_data.csv' exists in the 'data/' folder\n"
                "2. Your current working directory contains the project\n"
                f"Current directory: {os.getcwd()}"
            )
            
        # Check file size
        if os.path.getsize(data_path) == 0:
            raise ValueError("Data file is empty")
            
        return pd.read_csv(data_path)
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def clean_data(df):
    """Clean and reshape data with validation checks"""
    # Validate input DataFrame
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    required_columns = ['Area', 'Element', 'Item', 'Year', 'Value']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    try:
        # Drop unnecessary columns
        cols_to_drop = ['Domain Code', 'Domain', 'Element Code', 'Item Code (CPC)', 
                       'Year Code', 'Flag', 'Flag Description']
        df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
        
        # Rename columns
        df = df.rename(columns={
            'Area Code (M49)': 'area_code',
            'Area': 'country',
            'Element': 'parameter',
            'Item': 'crop',
            'Year': 'year',
            'Unit': 'unit',
            'Value': 'value'
        })
        
        # Pivot to wide format
        df_pivot = df.pivot_table(
            index=['country', 'crop', 'year', 'area_code'],
            columns='parameter',
            values='value'
        ).reset_index()
        
        # Clean column names
        df_pivot.columns = [str(col).lower().replace(' ', '_') for col in df_pivot.columns]
        
        # Convert yield from hg/ha to kg/ha if column exists
        if 'yield' in df_pivot.columns:
            df_pivot['yield'] = df_pivot['yield'] / 10
            
        return df_pivot
        
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        raise

def save_processed_data(df):
    """Save cleaned data with directory creation"""
    try:
        output_dir = Path('data/processed')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / 'cleaned_data.csv'
        df.to_csv(output_path, index=False)
        print(f"Data successfully saved to: {output_path.absolute()}")
    except Exception as e:
        print(f"Error saving processed data: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting data preprocessing...")
    try:
        raw_data = load_data()
        cleaned_data = clean_data(raw_data)
        save_processed_data(cleaned_data)
        print("Data preprocessing completed successfully!")
    except Exception as e:
        print(f"Data preprocessing failed: {str(e)}")
        exit(1)