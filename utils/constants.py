# Crop-specific yield ranges (kg/ha)
CROP_YIELD_RANGES = {
    'Wheat': (2000, 8000),
    'Rice': (3000, 10000),
    'Maize': (3000, 12000),
    # Add other crops
}

# Default fallback yield (kg/ha) when data is missing
DEFAULT_YIELD = 2500  

# Production bounds (tons/hectare)
MIN_TON_PER_HA = 0.1
MAX_TON_PER_HA = 50