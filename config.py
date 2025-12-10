# IQ-FARM Configuration File

# Telegram Bot Configuration
BOT_TOKEN = "8289901238:AAHxIoPDSUgTzrnoAFUBQ1hVFEnfJn3iCpk"
ADMIN_USER_ID = 1113375908 # Your Telegram user ID

# Database paths
SOIL_DATA_PATH = "datasets/soil_data.csv"
CROP_DATA_PATH = "datasets/crop_data.csv"

# Recommendation thresholds
MIN_RECOMMENDATION_SCORE = 40  # Minimum score to recommend a crop (0-100)
TOP_RECOMMENDATIONS = 10  # Number of top crops to show

# Data validation
VALID_REGIONS = [
    'Basra', 'Nasiriyah', 'Baghdad', 'Kirkuk', 'Mosul',
    'Diyala', 'Anbar', 'Sulaymaniyah', 'Erbil', 'Hilla',
    'Karbala', 'Wasit', 'Muthanna', 'Maysan'
]

# Soil parameter ranges
SOIL_PH_MIN = 5.5
SOIL_PH_MAX = 8.5
SOIL_NITROGEN_MIN = 10
SOIL_NITROGEN_MAX = 100
SOIL_MOISTURE_MIN = 10
SOIL_MOISTURE_MAX = 80

# Language (ar = Arabic, en = English)
DEFAULT_LANGUAGE = "en"

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "iq_farm.log"

# Visualization
CHART_DPI = 100
CHART_FIGSIZE_WIDTH = 10
CHART_FIGSIZE_HEIGHT = 6
CHART_COLOR_SUCCESS = "#2ecc71"
CHART_COLOR_WARNING = "#f39c12"
CHART_COLOR_DANGER = "#e74c3c"
CHART_COLOR_PRIMARY = "#3498db"
