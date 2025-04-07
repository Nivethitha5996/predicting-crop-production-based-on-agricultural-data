import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path
import os

# New: Load constants
try:
    from utils.constants import DEFAULT_YIELD
except ImportError:
    DEFAULT_YIELD = 2500  # Fallback default

def load_data():
    """Enhanced data loading with validation"""
    data_path = Path('data/processed/cleaned_data.csv')
    
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Validate critical columns
    required_cols = ['country', 'crop', 'year', 'area_harvested', 'yield', 'production']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # New: Data quality checks
    if len(df) < 100:
        print("Warning: Small dataset may affect model performance")
    
    return df

def preprocess_data(df):
    """Enhanced preprocessing pipeline"""
    # 1. Handle missing values
    imputer = SimpleImputer(strategy='median')
    df[['area_harvested', 'yield']] = imputer.fit_transform(df[['area_harvested', 'yield']])
    
    # 2. Encode categoricals and save encoders
    country_encoder = LabelEncoder()
    crop_encoder = LabelEncoder()
    
    df['country_code'] = country_encoder.fit_transform(df['country'])
    df['crop_code'] = crop_encoder.fit_transform(df['crop'])
    
    # New: Save encoders to models/
    Path('models').mkdir(exist_ok=True)
    joblib.dump(country_encoder, 'models/country_encoder.pkl')
    joblib.dump(crop_encoder, 'models/crop_encoder.pkl')
    
    # 3. Feature selection
    features = ['country_code', 'crop_code', 'year', 'area_harvested', 'yield']
    X = df[features]
    y = df['production']
    
    return X, y

def train_model(X, y):
    """Enhanced model training with scaling"""
    # 1. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # 2. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # New: Save scaler to models/
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # 3. Initialize and train model
    model = RandomForestRegressor(
        n_estimators=200,  # Increased from 100
        max_depth=15,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # 4. Evaluation
    y_pred = model.predict(X_test_scaled)
    print("\nModel Performance:")
    print(f"- R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"- MAE: {mean_absolute_error(y_test, y_pred):.2f} tons")
    print(f"- Avg Error: {mean_absolute_error(y_test, y_pred)/y_test.mean()*100:.2f}% of average production")
    
    # 5. Save model
    joblib.dump(model, 'models/production_predictor.pkl')
    print("\nArtifacts saved to models/:")
    print("- production_predictor.pkl (model)")
    print("- country_encoder.pkl")
    print("- crop_encoder.pkl") 
    print("- scaler.pkl")

if __name__ == "__main__":
    print("Starting model training pipeline...")
    try:
        # 1. Load and validate data
        df = load_data()
        
        # 2. Preprocess
        print("\nPreprocessing data...")
        X, y = preprocess_data(df)
        print(f"Training set shape: {X.shape}")
        
        # 3. Train model
        print("\nTraining Random Forest model...")
        train_model(X, y)
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"\nError in training pipeline: {str(e)}")
        exit(1)