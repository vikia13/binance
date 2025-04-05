import os
import time
import signal
import logging
from dotenv import load_dotenv
from database import Database
from telegram_bot import TelegramBot
from data_processor import DataProcessor
from ai_model import AIModelWrapper  # Renamed to match actual class name
from alert_system import AlertSystem  # Import AlertSystem
from websocket_client import WebSocketClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Main")

# Load environment variables
print("Current directory:", os.getcwd())
env_path = os.path.join(os.getcwd(), '.env')
print(".env file exists:", os.path.exists(env_path))
print("Loading environment variables...")
load_dotenv(dotenv_path=env_path, override=True)

# Print loaded environment variables
print("All environment variables:")
for key, value in os.environ.items():
    if key in ["TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID"]:
        masked_value = value[:5] + "..." + value[-5:] if len(value) > 10 else "***"
        print(f"{key}: {masked_value}")

# Global variables
running = True
components = []
data_processor = None  # Define data_processor at module level
alert_system = None  # Define alert_system at module level


def signal_handler(_, __):
    """Handle termination signals"""
    global running
    logger.info("Received termination signal. Shutting down...")
    running = False


def handle_websocket_message(data):
    """Process incoming WebSocket messages"""
    global data_processor, alert_system

    try:
        # Process each ticker in the data
        for ticker in data:
            if 'e' in ticker and ticker['e'] == '24hrTicker':
                symbol = ticker['s']

                # Only process perpetual futures (USDT pairs)
                if symbol.endswith('USDT'):
                    ticker_data = {
                        'symbol': symbol,
                        'price': float(ticker['c']),
                        'volume': float(ticker['v']),
                        'timestamp': ticker['E']
                    }

                    # Update data processor
                    data_processor.update_data(ticker_data)

                    # Detect trend
                    trend_signal = data_processor.detect_trend(symbol)
                    if trend_signal and alert_system:
                        # Process signal
                        alert_system.process_signal(trend_signal)

                        # Check for exit signals for open positions
                        for position in database.get_open_positions():
                            position_data = {
                                'symbol': position[1],
                                'trend': position[3],
                                'entry_price': position[2]
                            }

                            exit_signal = data_processor.detect_exit_signal(position_data)
                            if exit_signal:
                                alert_system.process_exit_signal(position[0], exit_signal)
    except Exception as e:
        logger.error(f"Error processing WebSocket message: {e}")


def main():
    global running, components, data_processor, alert_system, database

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialize database
        logger.info("Initializing database...")
        database = Database()

        # Add confirmed column to positions table if it doesn't exist
        try:
            conn = database.get_connection()
            cursor = conn.cursor()

            # Check if the confirmed column exists
            cursor.execute("PRAGMA table_info(positions)")
            columns = [column[1] for column in cursor.fetchall()]

            if 'confirmed' not in columns:
                logger.info("Adding 'confirmed' column to positions table")
                cursor.execute("ALTER TABLE positions ADD COLUMN confirmed BOOLEAN DEFAULT 0")
                conn.commit()

            conn.close()
        except Exception as e:
            logger.error(f"Error adding confirmed column: {e}")

        components.append(database)

        # Initialize Telegram bot
        logger.info("Starting Telegram bot...")
        telegram_bot = TelegramBot(database)
        components.append(telegram_bot)

        # Initialize data processor
        logger.info("Initializing data processor...")
        data_processor = DataProcessor()

        # Initialize AI model
        logger.info("Initializing AI model...")
        ai_model = AIModelWrapper()  # Use the correct class name
        components.append(ai_model)

        # Initialize alert system
        logger.info("Setting up alert system...")
        alert_system = AlertSystem(database, telegram_bot, ai_model)

        # Connect to Binance WebSocket
        logger.info("Connecting to Binance WebSocket...")
        websocket_url = "wss://fstream.binance.com/ws/!ticker@arr"
        ws_client = WebSocketClient(websocket_url, handle_websocket_message)
        components.append(ws_client)

        logger.info("System is running. Press Ctrl+C to exit.")

        # Main loop
        while running:
            time.sleep(1)

    except Exception as e:
        logger.error(f"Error in main application: {e}")
    finally:
        cleanup()


def cleanup():
    """Clean up resources before exiting"""
    logger.info("Cleaning up resources...")

    for component in reversed(components):
        try:
            if hasattr(component, 'close'):
                component.close()
            elif hasattr(component, 'stop'):
                component.stop()
        except Exception as e:
            logger.error(f"Error cleaning up component: {e}")

    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()