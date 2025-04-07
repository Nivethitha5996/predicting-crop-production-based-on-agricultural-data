from pathlib import Path
import pandas as pd

def check_data_exists():
    """Check if required files exist"""
    required_files = [
        Path('data/raw/FAOSTAT_data.csv'),
        Path('data/processed/cleaned_data.csv'),
        Path('models/production_predictor.pkl')
    ]
    
    return all(f.exists() for f in required_files)