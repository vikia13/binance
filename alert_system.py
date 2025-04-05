import logging
import time
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("alerts.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AlertSystem")

class AlertSystem:
    def __init__(self, database, telegram_bot, ai_model):
        self.database = database
        self.telegram_bot = telegram_bot
        self.ai_model = ai_model
        self.signal_counter = 0
        logger.info("Alert system initialized")

    def process_signal(self, signal):
        """Process a trading signal and send alerts if appropriate"""
        try:
            symbol = signal['symbol']
            trend = signal['trend']
            price = signal['price']

            # Check if we've already sent too many signals for this symbol today
            if not self.database.increment_signal_count(symbol):
                logger.info(f"Maximum daily signals reached for {symbol}, skipping")
                return False

            # Generate a unique signal ID
            self.signal_counter += 1
            signal_id = int(time.time()) % 10000 + self.signal_counter

            # Store the position in the database
            position_id = self.database.add_position(symbol, price, trend, signal_id)

            if position_id:
                # Format and send the alert message
                emoji = "🟢" if trend == "LONG" else "🔴"
                message = (
                    f"{emoji} *{trend} Signal* - {symbol} #{position_id}\n\n"
                    f"Entry Price: ${price:.4f}\n"
                    f"RSI: {signal.get('rsi', 'N/A'):.1f}\n"
                    f"MACD: {signal.get('macd_diff', 'N/A'):.6f}\n\n"
                    f"Reply with `{position_id}:ok` to confirm entry"
                )

                self.telegram_bot.send_message(message)
                logger.info(f"Signal alert sent for {symbol} {trend} at ${price:.4f}")
                return True
            else:
                logger.error(f"Failed to add position to database for {symbol}")
                return False

        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return False

    def process_exit_signal(self, position_id, exit_signal):
        """Process an exit signal for an open position"""
        try:
            symbol = exit_signal['symbol']
            exit_price = exit_signal['exit_price']
            profit_pct = exit_signal['profit_pct']
            reason = exit_signal.get('reason', 'Technical indicators')

            # Close the position in the database
            if self.database.close_position(position_id, exit_price):
                # Determine emoji based on profit/loss
                if profit_pct > 0:
                    emoji = "✅"
                else:
                    emoji = "❌"

                # Format and send the exit alert
                message = (
                    f"{emoji} *Exit Signal* - {symbol}\n\n"
                    f"Exit Price: ${exit_price:.4f}\n"
                    f"Profit/Loss: {profit_pct:.2f}%\n"
                    f"Reason: {reason}\n"
                )

                self.telegram_bot.send_message(message)
                logger.info(f"Exit alert sent for {symbol} at ${exit_price:.4f} with {profit_pct:.2f}% P/L")
                return True
            else:
                logger.error(f"Failed to close position {position_id} in database")
                return False

        except Exception as e:
            logger.error(f"Error processing exit signal: {e}")
            return False