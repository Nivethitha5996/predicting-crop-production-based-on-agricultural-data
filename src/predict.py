from pathlib import Path
import joblib
import numpy as np
from utils.constants import *

class CropProductionPredictor:
    def __init__(self):
        self.model = self._load_artifact('production_predictor.pkl')
        self.scaler = self._load_artifact('scaler.pkl')
        self.country_encoder = self._load_artifact('country_encoder.pkl')
        self.crop_encoder = self._load_artifact('crop_encoder.pkl')

    def _load_artifact(self, filename):
        path = Path(f'models/{filename}')
        return joblib.load(path) if path.exists() else None

    def predict(self, country, crop, year, area, yield_val=None):
        """Enhanced prediction with auto-scaling and sanity checks"""
        try:
            # Encode categorical features
            country_code = self.country_encoder.transform([country])[0]
            crop_code = self.crop_encoder.transform([crop])[0]
            
            # Use default yield if not provided
            yield_val = yield_val or DEFAULT_YIELD
            
            # Validate numerical inputs
            if any(not isinstance(x, (int, float)) for x in [year, area, yield_val]):
                raise ValueError("Numerical inputs must be numbers")
            
            # Prepare input vector
            features = np.array([[country_code, crop_code, year, area, yield_val]])
            
            # Scale features if scaler exists
            if self.scaler:
                features = self.scaler.transform(features)
            
            # Predict and clip to realistic range
            prediction = self.model.predict(features)[0]
            return self._clip_prediction(prediction, area, crop)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._fallback_prediction(area, crop, yield_val)

    def _clip_prediction(self, prediction, area, crop):
        """Ensure predictions stay within crop-specific bounds"""
        crop_ranges = CROP_YIELD_RANGES.get(crop, (MIN_TON_PER_HA, MAX_TON_PER_HA))
        min_val = area * (crop_ranges[0] / 1000)  # kg/ha → tons/ha
        max_val = area * (crop_ranges[1] / 1000)
        return np.clip(prediction, min_val, max_val)

    def _fallback_prediction(self, area, crop, yield_val):
        """Basic yield-based estimation when model fails"""
        default_yield = CROP_YIELD_RANGES.get(crop, (DEFAULT_YIELD,))[0]
        return area * (default_yield / 1000)  # Convert kg to tons