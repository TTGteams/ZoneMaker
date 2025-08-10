"""
Configuration settings for Zone Tracker
"""

import os
import logging

# Database configuration using environment variables for security
DB_CONFIG = {
    'server': os.getenv('DB_SERVER', '192.168.50.100'),
    'database': os.getenv('DB_DATABASE', 'FXStrat'),  # Main database for writing zones/indicators
    'username': os.getenv('DB_USERNAME', 'djaime'),
    'password': os.getenv('DB_PASSWORD', 'Enrique30072000!3'),
    'driver': os.getenv('DB_DRIVER', 'ODBC Driver 18 for SQL Server')
}

# Historical data database configuration (where HistoData table is located)
HISTODATA_CONFIG = {
    'server': os.getenv('HISTODATA_SERVER', os.getenv('DB_SERVER', '192.168.50.100')),  # Same server by default
    'database': os.getenv('HISTODATA_DATABASE', 'TTG'),  # TTG database for HistoData table
    'username': os.getenv('HISTODATA_USERNAME', os.getenv('DB_USERNAME', 'djaime')),  # Same credentials by default
    'password': os.getenv('HISTODATA_PASSWORD', os.getenv('DB_PASSWORD', 'Enrique30072000!3')),
    'driver': os.getenv('HISTODATA_DRIVER', os.getenv('DB_DRIVER', 'ODBC Driver 18 for SQL Server'))
}

# Backward compatibility - remove this after updating database_manager.py
HISTODATA_DB = HISTODATA_CONFIG['database']

# Supported currencies (reusing from existing system)
SUPPORTED_CURRENCIES = ["EUR.USD", "USD.CAD", "GBP.USD"]

# Zone tracking constants (reusing from algorithm.py)
WINDOW_LENGTH = 10
SUPPORT_RESISTANCE_ALLOWANCE = 0.0011
LIQUIDITY_ZONE_ALERT = 0.002
DEMAND_ZONE_RSI_TOP = 70
SUPPLY_ZONE_RSI_BOTTOM = 30
INVALIDATE_ZONE_LENGTH = 0.0013

# Zone tracker specific settings
INITIAL_HISTORICAL_LIMIT = 150000  # Number of rows to load on first run
PROCESSING_INTERVAL_SECONDS = 300  # 5 minutes
MAX_MEMORY_ROWS = 200000  # Keep rolling window to prevent memory overflow

# Market hours (MST timezone)
MARKET_OPEN_HOUR = 15  # 3 PM MST Sunday
MARKET_CLOSE_HOUR = 15  # 3 PM MST Friday
MARKET_OPEN_DAY = 6  # Sunday (0=Monday, 6=Sunday)
MARKET_CLOSE_DAY = 4  # Friday

# Logging configuration
ZONE_TRACKER_LOG_LEVEL = logging.INFO 
