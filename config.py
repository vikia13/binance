import os
from dotenv import load_dotenv

# Print the current working directory
print(f"Current directory: {os.getcwd()}")

# Check if .env file exists
env_path = os.path.join(os.getcwd(), '.env')
print(f".env file exists: {os.path.exists(env_path)}")

# Load environment variables
print("Loading environment variables...")
load_dotenv(dotenv_path=env_path, override=True)

# Print all loaded environment variables
print("All environment variables:")
for key, value in os.environ.items():
    if key in ["TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID"]:
        print(f"{key}: {value}")

# WebSocket configuration
BINANCE_WEBSOCKET_URL = "wss://fstream.binance.com/ws/!ticker@arr"

# Alert parameters
TIME_INTERVAL_MINUTES = 3
PRICE_CHANGE_THRESHOLD = 3.0
MAX_SIGNALS_PER_DAY = 50

# Telegram configuration - Use direct access to environment variables
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Print Telegram configuration values for debugging
if TELEGRAM_TOKEN:
    masked_token = TELEGRAM_TOKEN[:5] + "..." + TELEGRAM_TOKEN[-5:] if len(TELEGRAM_TOKEN) > 10 else "***"
    print(f"Using TELEGRAM_TOKEN: {masked_token}")
else:
    print("WARNING: TELEGRAM_TOKEN not found in environment variables")

if TELEGRAM_CHAT_ID:
    print(f"Using TELEGRAM_CHAT_ID: {TELEGRAM_CHAT_ID}")
else:
    print("WARNING: TELEGRAM_CHAT_ID not found in environment variables")

# Database configuration
DB_PATH = "positions.db"

# Technical indicators parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# AI model parameters
AI_MODEL_PATH = "models/trend_detection_model.pkl"