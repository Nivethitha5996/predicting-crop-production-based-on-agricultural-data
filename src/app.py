import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# New: Centralized configuration
MODELS_DIR = Path('models')
DATA_DIR = Path('data/processed')

class CropProductionPredictor:
    def __init__(self):
        """Initialize with loaded artifacts"""
        self.model = self._load_artifact('production_predictor.pkl')
        self.scaler = self._load_artifact('scaler.pkl')
        self.country_encoder = self._load_artifact('country_encoder.pkl')
        self.crop_encoder = self._load_artifact('crop_encoder.pkl')
        self.df = self._load_data()

    def _load_artifact(self, filename):
        """Helper to load model artifacts"""
        path = MODELS_DIR / filename
        if path.exists():
            return joblib.load(path)
        st.warning(f"Missing artifact: {filename}")
        return None

    def _load_data(self):
        """Load and cache processed data"""
        try:
            df = pd.read_csv(DATA_DIR / 'cleaned_data.csv')
            
            # Validate required columns
            required_cols = ['country', 'crop', 'year', 'area_harvested', 'yield']
            if not all(col in df.columns for col in required_cols):
                st.error("Processed data is missing required columns")
                st.stop()
                
            return df
            
        except FileNotFoundError:
            st.error("Processed data file not found. Please run preprocessing first.")
            st.stop()

    def predict(self, country, crop, year, area, yield_val=None):
        """Enhanced prediction with validation and fallback"""
        try:
            # 1. Validate inputs
            if not isinstance(area, (int, float)) or area <= 0:
                raise ValueError("Area must be a positive number")
            if not isinstance(year, int) or year < 1961 or year > 2030:
                raise ValueError("Year must be between 1961-2030")

            # 2. Encode categorical features
            country_code = self._encode_feature(country, self.country_encoder, 'country')
            crop_code = self._encode_feature(crop, self.crop_encoder, 'crop')

            # 3. Prepare input features
            features = np.array([[country_code, crop_code, year, area, yield_val or self._get_default_yield(crop)]])
            
            # 4. Scale features if scaler exists
            if self.scaler:
                features = self.scaler.transform(features)

            # 5. Make prediction
            prediction = self.model.predict(features)[0]
            
            # 6. Apply realistic constraints
            return self._apply_production_constraints(prediction, area, crop)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return self._fallback_prediction(area, crop, yield_val)

    def _encode_feature(self, value, encoder, feature_name):
        """Safe feature encoding with validation"""
        if encoder is None:
            raise ValueError(f"No encoder found for {feature_name}")
        
        try:
            return encoder.transform([value])[0]
        except ValueError:
            available = list(encoder.classes_)
            raise ValueError(
                f"Unknown {feature_name}: '{value}'. "
                f"Available options: {available[:10]}{'...' if len(available)>10 else ''}"
            )

    def _get_default_yield(self, crop):
        """Get crop-specific default yield (kg/ha)"""
        crop_yields = self.df.groupby('crop')['yield'].median()
        return crop_yields.get(crop, 2500)  # Default to 2500 kg/ha if crop unknown

    def _apply_production_constraints(self, prediction, area, crop):
        """Ensure predictions stay within realistic bounds"""
        # Get crop-specific yield range from historical data
        crop_data = self.df[self.df['crop'] == crop]
        min_yield = crop_data['yield'].quantile(0.05) / 1000  # kg/ha to tons/ha
        max_yield = crop_data['yield'].quantile(0.95) / 1000
        
        min_production = area * (min_yield if not np.isnan(min_yield) else 0.1)
        max_production = area * (max_yield if not np.isnan(max_yield) else 50)
        
        clipped = np.clip(prediction, min_production, max_production)
        
        if prediction != clipped:
            st.warning(
                f"Adjusted prediction from {prediction:,.1f} to {clipped:,.1f} tons "
                f"(expected range for {crop}: {min_production:,.1f}-{max_production:,.1f} tons)"
            )
            
        return clipped

    def _fallback_prediction(self, area, crop, yield_val=None):
        """Basic yield-based estimation when model fails"""
        default_yield = self._get_default_yield(crop)
        estimate = area * ((yield_val or default_yield) / 1000)
        st.warning(f"Using fallback estimation: {estimate:,.1f} tons")
        return estimate

def main():
    st.title("🌾 Crop Production Predictor")
    st.markdown("""
    Predict crop production based on historical agricultural data.
    """)
    
    # Initialize predictor
    predictor = CropProductionPredictor()
    
    # Get unique values for dropdowns
    countries = sorted(predictor.df['country'].unique())
    crops = sorted(predictor.df['crop'].unique())
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            country = st.selectbox("Country", countries, index=countries.index('India') if 'India' in countries else 0)
            crop = st.selectbox("Crop", crops, index=crops.index('Wheat') if 'Wheat' in crops else 0)
            
        with col2:
            year = st.number_input("Year", min_value=1961, max_value=2030, value=2023)
            area = st.number_input("Area Harvested (hectares)", min_value=0.1, value=100.0, step=1.0)
            yield_val = st.number_input(
                "Yield (kg/ha)", 
                min_value=1.0, 
                value=predictor._get_default_yield('Wheat'),  # Default based on selected crop
                help="Leave empty to use historical average for selected crop"
            )
        
        submitted = st.form_submit_button("Predict Production")
        
        if submitted:
            with st.spinner("Calculating..."):
                production = predictor.predict(country, crop, year, area, yield_val or None)
                
                st.success(f"**Predicted Production:** {production:,.1f} tons")
                
                # Show historical context
                st.subheader("Historical Context")
                hist_data = predictor.df[
                    (predictor.df['country'] == country) & 
                    (predictor.df['crop'] == crop)
                ]
                
                if not hist_data.empty:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Yield", f"{hist_data['yield'].mean():,.1f} kg/ha")
                    with col2:
                        st.metric("Avg Area", f"{hist_data['area_harvested'].mean():,.1f} ha")
                    with col3:
                        st.metric("Avg Production", f"{hist_data['production'].mean():,.1f} tons")
                else:
                    st.info("No historical data available for this crop/country combination")

if __name__ == "__main__":
    main()