"""
Zone Tracker Main Scheduler
Orchestrates the entire zone tracking system with market hours awareness
"""

import time
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, Any

from config import SUPPORTED_CURRENCIES, PROCESSING_INTERVAL_SECONDS
from market_hours import is_market_open, get_market_status
from zone_and_indicator_processor import zone_processor
from database_manager import db_manager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('zone_tracker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("zone_tracker_main")

class ZoneTrackerScheduler:
    """Main scheduler for the Zone Tracker system"""
    
    def __init__(self):
        self.running = False
        self.initialization_complete = False
        self.last_status_log = None
        self.error_count = {currency: 0 for currency in SUPPORTED_CURRENCIES}
        self.max_errors_per_currency = 5
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    def start(self):
        """Start the zone tracker system"""
        logger.info("="*80)
        logger.info("ZONE TRACKER SYSTEM STARTUP")
        logger.info("="*80)
        
        try:
            # Initialize the system
            if not self._initialize_system():
                logger.error("System initialization failed, exiting")
                return False
            
            # Start main processing loop
            self.running = True
            self._main_loop()
            
        except Exception as e:
            logger.error(f"Fatal error in main system: {e}", exc_info=True)
            return False
        finally:
            self._cleanup()
        
        return True
    
    def _initialize_system(self) -> bool:
        """Initialize the zone tracking system"""
        logger.info("Initializing Zone Tracker system...")
        
        try:
            # Test database connections
            if not self._test_database_connections():
                return False
            
            # Check if this is first run or incremental
            first_run = self._is_first_run()
            
            if first_run:
                logger.info("First run detected - loading historical data for all currencies")
                return self._initialize_first_run()
            else:
                logger.info("Subsequent run detected - resuming from last processed timestamps")
                return self._initialize_incremental_run()
                
        except Exception as e:
            logger.error(f"Error during system initialization: {e}", exc_info=True)
            return False
    
    def _test_database_connections(self) -> bool:
        """Test database connections"""
        logger.info("Testing database connections...")
        
        # Test FXStrat connection
        if db_manager.get_connection() is None:
            logger.error("Failed to connect to FXStrat database")
            return False
        
        # Test HistoData connection
        if db_manager.get_histodata_connection() is None:
            logger.error("Failed to connect to HistoData database")
            return False
        
        logger.info("Database connections successful")
        return True
    
    def _is_first_run(self) -> bool:
        """Determine if this is the first run"""
        try:
            # Check if any currency has processed data
            for currency in SUPPORTED_CURRENCIES:
                last_timestamp = zone_processor.get_last_processed_timestamp(currency)
                if last_timestamp is not None:
                    return False
            
            # Check if we have any zones or indicators in database
            zone_count = db_manager.execute_query("SELECT COUNT(*) FROM ZoneTracker WHERE IsActive = 1")
            indicator_count = db_manager.execute_query("SELECT COUNT(*) FROM IndicatorTracker")
            
            if zone_count and zone_count[0][0] > 0:
                return False
            if indicator_count and indicator_count[0][0] > 0:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error determining first run status, assuming first run: {e}")
            return True
    
    def _initialize_first_run(self) -> bool:
        """Initialize system for first run with historical data"""
        logger.info("Loading historical data for all currencies...")
        
        success_count = 0
        for currency in SUPPORTED_CURRENCIES:
            try:
                logger.info(f"Loading historical data for {currency}...")
                success = zone_processor.load_initial_data(currency)
                
                if success:
                    success_count += 1
                    logger.info(f"Successfully initialized {currency}")
                else:
                    logger.error(f"Failed to initialize {currency}")
                    self.error_count[currency] += 1
                    
            except Exception as e:
                logger.error(f"Error initializing {currency}: {e}", exc_info=True)
                self.error_count[currency] += 1
        
        if success_count == 0:
            logger.error("Failed to initialize any currencies")
            return False
        
        logger.info(f"Initialization complete: {success_count}/{len(SUPPORTED_CURRENCIES)} currencies successful")
        self.initialization_complete = True
        return True
    
    def _initialize_incremental_run(self) -> bool:
        """Initialize system for incremental run"""
        logger.info("Initializing incremental processing...")
        
        # Log last processed timestamps
        for currency in SUPPORTED_CURRENCIES:
            last_timestamp = zone_processor.get_last_processed_timestamp(currency)
            if last_timestamp:
                logger.info(f"{currency}: Last processed timestamp = {last_timestamp}")
            else:
                logger.warning(f"{currency}: No last timestamp found, will treat as first run")
        
        self.initialization_complete = True
        return True
    
    def _main_loop(self):
        """Main processing loop"""
        logger.info("Starting main processing loop...")
        
        while self.running:
            try:
                # Check market hours
                market_status = get_market_status()
                
                if not market_status['is_open']:
                    self._handle_market_closed(market_status)
                    continue
                
                # Process incremental data for each currency
                self._process_all_currencies()
                
                # Log status periodically
                self._log_periodic_status()
                
                # Sleep until next processing interval
                time.sleep(PROCESSING_INTERVAL_SECONDS)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _handle_market_closed(self, market_status: Dict[str, Any]):
        """Handle market closed periods"""
        current_time = datetime.now()
        
        # Log market status once per hour
        if (self.last_status_log is None or 
            current_time - self.last_status_log > timedelta(hours=1)):
            
            logger.info(f"Market is CLOSED. Next open: {market_status['next_open']}")
            logger.info(f"Time until next open: {market_status['time_until_next_event']}")
            self.last_status_log = current_time
        
        # Sleep for 5 minutes during market closed
        time.sleep(300)
    
    def _process_all_currencies(self):
        """Process incremental data for all currencies"""
        processed_count = 0
        
        for currency in SUPPORTED_CURRENCIES:
            # Skip currencies with too many errors
            if self.error_count[currency] >= self.max_errors_per_currency:
                continue
            
            try:
                success = zone_processor.process_incremental_data(currency)
                
                if success:
                    processed_count += 1
                    # Reset error count on success
                    self.error_count[currency] = 0
                else:
                    self.error_count[currency] += 1
                    logger.warning(f"Failed to process {currency} (errors: {self.error_count[currency]})")
                    
            except Exception as e:
                self.error_count[currency] += 1
                logger.error(f"Error processing {currency}: {e}", exc_info=True)
                
                # If too many errors, disable this currency temporarily
                if self.error_count[currency] >= self.max_errors_per_currency:
                    logger.error(f"Disabling {currency} due to {self.max_errors_per_currency} consecutive errors")
        
        if processed_count == 0:
            logger.warning("No currencies processed successfully this cycle")
    
    def _log_periodic_status(self):
        """Log system status periodically"""
        current_time = datetime.now()
        
        # Log status every 30 minutes
        if (self.last_status_log is None or 
            current_time - self.last_status_log > timedelta(minutes=30)):
            
            self._log_system_status()
            self.last_status_log = current_time
    
    def _log_system_status(self):
        """Log comprehensive system status"""
        logger.info("="*60)
        logger.info("SYSTEM STATUS REPORT")
        logger.info("="*60)
        
        # Memory usage stats
        memory_stats = zone_processor.get_memory_usage_stats()
        for currency, stats in memory_stats.items():
            logger.info(f"{currency}: Bars={stats['bars_15min']}, Zones={stats['zones']}, "
                       f"ValidationRecords={stats['validation_records']}")
        
        # Zone validation stats
        validation_stats = zone_processor.get_zone_validation_stats()
        for currency, stats in validation_stats.items():
            logger.info(f"{currency} Zone Validation: Active={stats['active_zones']}, "
                       f"Losses={stats['zones_with_losses']}, MaxLosses={stats['max_losses_single_zone']}")
        
        # Simulation trade stats
        simulation_stats = zone_processor.get_simulation_trade_stats()
        for currency, stats in simulation_stats.items():
            logger.info(f"{currency} Simulation: Trades={stats['total_trades']}, "
                       f"WinRate={stats['win_rate']:.1f}%, Balance=${stats['current_balance']:,.2f}, "
                       f"P&L=${stats['total_profit_loss']:,.2f}")
        
        # Error counts
        active_currencies = [c for c in SUPPORTED_CURRENCIES if self.error_count[c] < self.max_errors_per_currency]
        logger.info(f"Active currencies: {len(active_currencies)}/{len(SUPPORTED_CURRENCIES)}")
        
        logger.info("="*60)
    
    def _cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        
        try:
            # Close database connections
            db_manager.close_connections()
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        logger.info("Zone Tracker shutdown complete")

def main():
    """Main entry point"""
    scheduler = ZoneTrackerScheduler()
    
    try:
        success = scheduler.start()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 