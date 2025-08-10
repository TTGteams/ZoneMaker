"""
Database Manager for Zone Tracker
Handles connections and queries to both FXStrat and HistoricalDataDB databases
"""

import pyodbc
import pandas as pd
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from config import DB_CONFIG, HISTODATA_CONFIG

# Set up logging
logger = logging.getLogger("zone_tracker")

class DatabaseManager:
    """Manages database connections and operations for Zone Tracker"""
    
    def __init__(self):
        self._connection = None
        self._histodata_connection = None
    
    def get_connection(self, database_override: Optional[str] = None) -> Optional[pyodbc.Connection]:
        """
        Get connection to FXStrat database (or override database)
        Reuses existing connection pattern from algorithm.py
        """
        database_name = database_override or DB_CONFIG['database']
        
        if self._connection is None or not self._test_connection(self._connection):
            conn_str = (
                f"Driver={{{DB_CONFIG['driver']}}};"
                f"Server={DB_CONFIG['server']};"
                f"Database={database_name};"
                f"UID={DB_CONFIG['username']};"
                f"PWD={DB_CONFIG['password']};"
                f"Connection Timeout=30;"
                f"TrustServerCertificate=yes;"
            )
            
            try:
                self._connection = pyodbc.connect(conn_str, autocommit=False)
                self._connection.timeout = 300
                logger.info(f"Successfully connected to {database_name} on {DB_CONFIG['server']}")
            except Exception as e:
                logger.error(f"Database connection error: {e}")
                self._connection = None
                return None
        
        return self._connection
    
    def get_histodata_connection(self) -> Optional[pyodbc.Connection]:
        """
        Get connection to HistoData database (TTG) for HistoData table access
        """
        if self._histodata_connection is None or not self._test_connection(self._histodata_connection):
            conn_str = (
                f"Driver={{{HISTODATA_CONFIG['driver']}}};"
                f"Server={HISTODATA_CONFIG['server']};"
                f"Database={HISTODATA_CONFIG['database']};"
                f"UID={HISTODATA_CONFIG['username']};"
                f"PWD={HISTODATA_CONFIG['password']};"
                f"Connection Timeout=30;"
                f"TrustServerCertificate=yes;"
            )
            
            try:
                self._histodata_connection = pyodbc.connect(conn_str, autocommit=False)
                self._histodata_connection.timeout = 300
                logger.info(f"Successfully connected to {HISTODATA_CONFIG['database']} on {HISTODATA_CONFIG['server']}")
            except Exception as e:
                logger.error(f"HistoData database connection error: {e}")
                self._histodata_connection = None
                return None
        
        return self._histodata_connection
    
    def _test_connection(self, connection: pyodbc.Connection) -> bool:
        """Test if connection is still valid"""
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return True
        except:
            return False
    
    def execute_query(self, query: str, params: Optional[tuple] = None, 
                     use_histodata: bool = False) -> Optional[List[tuple]]:
        """
        Execute a query and return results
        
        Args:
            query: SQL query string
            params: Query parameters (optional)
            use_histodata: If True, use HistoricalDataDB connection
            
        Returns:
            List of result tuples, or None if error
        """
        connection = self.get_histodata_connection() if use_histodata else self.get_connection()
        
        if connection is None:
            logger.error("No database connection available")
            return None
        
        try:
            cursor = connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            results = cursor.fetchall()
            cursor.close()
            return results
            
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            logger.error(f"Query: {query}")
            return None
    
    def execute_query_to_dataframe(self, query: str, params: Optional[tuple] = None,
                                  use_histodata: bool = False) -> Optional[pd.DataFrame]:
        """
        Execute query and return results as pandas DataFrame
        
        Args:
            query: SQL query string
            params: Query parameters (optional)
            use_histodata: If True, use HistoricalDataDB connection
            
        Returns:
            DataFrame with results, or None if error
        """
        connection = self.get_histodata_connection() if use_histodata else self.get_connection()
        
        if connection is None:
            logger.error("No database connection available")
            return None
        
        try:
            df = pd.read_sql(query, connection, params=params)
            return df
        except Exception as e:
            logger.error(f"DataFrame query execution error: {e}")
            logger.error(f"Query: {query}")
            return None
    
    def execute_non_query(self, query: str, params: Optional[tuple] = None,
                         use_histodata: bool = False) -> bool:
        """
        Execute INSERT/UPDATE/DELETE query
        
        Args:
            query: SQL query string
            params: Query parameters (optional)
            use_histodata: If True, use HistoricalDataDB connection
            
        Returns:
            True if successful, False otherwise
        """
        connection = self.get_histodata_connection() if use_histodata else self.get_connection()
        
        if connection is None:
            logger.error("No database connection available")
            return False
        
        try:
            cursor = connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            connection.commit()
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"Non-query execution error: {e}")
            logger.error(f"Query: {query}")
            try:
                connection.rollback()
            except:
                pass
            return False
    
    def batch_insert(self, table_name: str, data: List[Dict[str, Any]], 
                    use_histodata: bool = False) -> bool:
        """
        Batch insert data into table
        
        Args:
            table_name: Target table name
            data: List of dictionaries with column->value mappings
            use_histodata: If True, use HistoricalDataDB connection
            
        Returns:
            True if successful, False otherwise
        """
        if not data:
            return True
        
        connection = self.get_histodata_connection() if use_histodata else self.get_connection()
        
        if connection is None:
            logger.error("No database connection available")
            return False
        
        try:
            # Build INSERT query
            columns = list(data[0].keys())
            placeholders = ', '.join(['?' for _ in columns])
            query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
            
            # Prepare data tuples
            values = [tuple(row[col] for col in columns) for row in data]
            
            cursor = connection.cursor()
            cursor.executemany(query, values)
            connection.commit()
            cursor.close()
            
            logger.info(f"Successfully inserted {len(data)} rows into {table_name}")
            return True
            
        except Exception as e:
            # Check if this is a duplicate key error
            error_msg = str(e)
            if 'duplicate key' in error_msg.lower() or '2601' in error_msg or '2627' in error_msg:
                # Silently fail for duplicates - this is expected behavior
                try:
                    connection.rollback()
                except:
                    pass
                return False
            
            # Log other errors
            logger.error(f"Batch insert error: {e}")
            logger.error(f"Table: {table_name}, Rows: {len(data)}")
            try:
                connection.rollback()
            except:
                pass
            return False
    
    def close_connections(self):
        """Close all database connections"""
        try:
            if self._connection:
                self._connection.close()
                self._connection = None
            if self._histodata_connection:
                self._histodata_connection.close()
                self._histodata_connection = None
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")

# Global database manager instance
db_manager = DatabaseManager() 
