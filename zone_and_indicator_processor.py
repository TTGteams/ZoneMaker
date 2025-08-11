"""
Zone and Indicator Processor for Zone Tracker
Reuses existing functions from algorithm.py for zone identification and indicator calculations
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import sys
import os

# Add parent directory to path to import from algorithm.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database_manager import db_manager
from config import (
    SUPPORTED_CURRENCIES, 
    WINDOW_LENGTH,
    SUPPORT_RESISTANCE_ALLOWANCE,
    LIQUIDITY_ZONE_ALERT,
    DEMAND_ZONE_RSI_TOP,
    SUPPLY_ZONE_RSI_BOTTOM,
    INVALIDATE_ZONE_LENGTH,
    MAX_MEMORY_ROWS
)

# Import functions from central_zone_algorithm.py
try:
    from central_zone_algorithm import (
        identify_liquidity_zones,
        invalidate_zones_via_sup_and_resist,
        calculate_atr_pips_required,
        set_support_resistance_lines,
        STOP_LOSS_PROPORTION,
        TAKE_PROFIT_PROPORTION,
        RISK_PROPORTION,
        COOLDOWN_SECONDS
    )
except ImportError as e:
    logging.error(f"Failed to import from central_zone_algorithm.py: {e}")
    raise

# Set up logging
logger = logging.getLogger("zone_tracker")

class ZoneAndIndicatorProcessor:
    """
    Processes price data to identify zones and calculate indicators
    Reuses existing algorithm.py functions for consistency
    """
    
    def __init__(self):
        # In-memory data storage for each currency
        self.price_data = {currency: pd.DataFrame() for currency in SUPPORTED_CURRENCIES}
        self.current_zones = {currency: {} for currency in SUPPORTED_CURRENCIES}
        
        # Tick-by-tick processing (like algorithm.py)
        self.tick_data = {currency: pd.DataFrame() for currency in SUPPORTED_CURRENCIES}
        self.bars_15min = {currency: pd.DataFrame() for currency in SUPPORTED_CURRENCIES}
        self.current_bar_data = {currency: None for currency in SUPPORTED_CURRENCIES}
        self.current_bar_start = {currency: None for currency in SUPPORTED_CURRENCIES}
        
        # Initialize cumulative zone info (needed for algorithm.py functions)
        self.cumulative_zone_info = {currency: None for currency in SUPPORTED_CURRENCIES}
        
        # Track invalidated zones to prevent immediate recreation
        self.invalidated_zones = {currency: {} for currency in SUPPORTED_CURRENCIES}  # {zone_id: invalidation_time}
        
        # Simple zone validation trades (lightweight - no position sizing)
        self.zone_validation_trades = {currency: [] for currency in SUPPORTED_CURRENCIES}
        
        # Full trade simulation with $100k position sizing
        self.simulation_trades = {currency: [] for currency in SUPPORTED_CURRENCIES}
        self.simulation_balance = {currency: 100000.0 for currency in SUPPORTED_CURRENCIES}  # $100k starting balance
        self.last_simulation_trade_time = {currency: None for currency in SUPPORTED_CURRENCIES}
        
        # Track last processed timestamps for incremental processing
        self.last_processed_timestamp = {currency: None for currency in SUPPORTED_CURRENCIES}
        
        # Cache ATR values per currency (recalculated every 15 minutes)
        self.current_atr = {currency: None for currency in SUPPORTED_CURRENCIES}
    
        # Track bars processed for hourly summaries (4 bars = 1 hour)
        self.bars_processed_count = {currency: 0 for currency in SUPPORTED_CURRENCIES}
        self.last_hourly_summary = {currency: None for currency in SUPPORTED_CURRENCIES}
    
    def load_initial_data(self, currency: str, limit: int = 75000) -> bool:
        """
        Load initial historical data for a currency and process tick-by-tick
        
        Args:
            currency: Currency pair to load data for
            limit: Number of rows to load
            
        Returns:
            True if successful, False otherwise
        """
        try:
            query = f"""
                SELECT [BarDateTime], [Open], [High], [Low], [Close], [ByWhen]
                FROM (
                    SELECT TOP ({limit}) [BarDateTime], [Open], [High], [Low], [Close], [ByWhen]
                    FROM [HistoData] WITH (NOLOCK)
                    WHERE [Identifier] = ?
                    ORDER BY [BarDateTime] DESC
                ) AS recent_data
                ORDER BY [BarDateTime] ASC
            """
            
            df = db_manager.execute_query_to_dataframe(
                query, 
                params=(currency,), 
                use_histodata=True
            )
            
            if df is None or df.empty:
                logger.error(f"No historical data found for {currency}")
                return False
            
            # Process data tick-by-tick (chronological order)
            logger.debug(f"Processing {len(df)} historical bars tick-by-tick for {currency}")
            
            for index, row in df.iterrows():
                # Calculate midpoint as "tick price"
                tick_price = (row['High'] + row['Low']) / 2.0
                tick_time = pd.to_datetime(row['BarDateTime'])
                
                # Process this tick (like algorithm.py does)
                self._process_single_tick(tick_price, tick_time, currency)
            
            # Update last processed timestamp
            if not df.empty:
                self.last_processed_timestamp[currency] = pd.to_datetime(df.iloc[-1]['ByWhen'])
            
            logger.debug(f"Completed tick-by-tick processing for {currency}")
            logger.debug(f"Built {len(self.bars_15min[currency])} 15-minute bars")
            logger.debug(f"Active zones: {len(self.current_zones[currency])}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading initial data for {currency}: {e}")
            return False
    
    def process_incremental_data(self, currency: str) -> bool:
        """
        Process new incremental price data for a currency since last timestamp
        
        Args:
            currency: Currency pair to process
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get new data since last processed timestamp
            last_timestamp = self.last_processed_timestamp[currency]
            
            if last_timestamp is None:
                logger.error(f"No last timestamp found for {currency} - run load_initial_data first")
                return False
            
            query = """
                SELECT [BarDateTime], [Open], [High], [Low], [Close], [ByWhen]
                FROM [HistoData]
                WHERE [Identifier] = ? AND [ByWhen] > ?
                ORDER BY [BarDateTime] ASC
            """
            
            df = db_manager.execute_query_to_dataframe(
                query, 
                params=(currency, last_timestamp), 
                use_histodata=True
            )
            
            if df is None or df.empty:
                logger.debug(f"No new data available for {currency}")
                return True
            
            logger.debug(f"Processing {len(df)} new bars for {currency}")
            
            # Process each new bar as a tick
            for index, row in df.iterrows():
                tick_price = (row['High'] + row['Low']) / 2.0
                tick_time = pd.to_datetime(row['BarDateTime'])
                
                self._process_single_tick(tick_price, tick_time, currency)
            
            # Update last processed timestamp
            self.last_processed_timestamp[currency] = pd.to_datetime(df.iloc[-1]['ByWhen'])
            
            logger.debug(f"Processed {len(df)} new ticks for {currency}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing incremental data for {currency}: {e}")
            return False

    def _process_single_tick(self, tick_price: float, tick_time: datetime, currency: str) -> None:
        """
        Process a single tick like algorithm.py does
        
        Args:
            tick_price: Current tick price (midpoint)
            tick_time: Timestamp of the tick
            currency: Currency pair
        """
        try:
            # 1. Update 15-minute bars (like algorithm.py update_bars_with_tick)
            bar_completed = self._update_bars_with_tick(tick_price, tick_time, currency)
            
            # 2. Simple zone validation on every tick (lightweight)
            self._validate_zones_simple(tick_price, tick_time, currency)
            
            # 3. Support/resistance zone invalidation on every tick (like algorithm.py)
            self._invalidate_zones_support_resistance(tick_price, currency)
            
            # 4. Full trade simulation on every tick (with position sizing)
            self._process_simulation_trades(tick_price, tick_time, currency)
            
            # 5. If a 15-minute bar was completed, do full analysis
            if bar_completed:
                self._process_completed_bar(currency)
            
        except Exception as e:
            logger.error(f"Error processing tick for {currency}: {e}")
    
    def _invalidate_zones_support_resistance(self, tick_price: float, currency: str) -> None:
        """
        Invalidate zones based on support/resistance violations on every tick
        (like algorithm.py invalidate_zones_via_sup_and_resist)
        """
        try:
            if currency not in self.current_zones:
                return
            
            # Use the algorithm.py function directly for consistency
            self.current_zones[currency] = invalidate_zones_via_sup_and_resist(
                tick_price, 
                self.current_zones[currency], 
                currency
            )
            
        except Exception as e:
            logger.error(f"Error invalidating zones via support/resistance for {currency}: {e}")
    
    def _update_bars_with_tick(self, tick_price: float, tick_time: datetime, currency: str) -> bool:
        """
        Update 15-minute bars with new tick data (mirrors algorithm.py logic)
        
        Returns:
            True if a bar was completed, False otherwise
        """
        try:
            # Round tick time to 15-minute intervals
            bar_time = tick_time.replace(minute=(tick_time.minute // 15) * 15, second=0, microsecond=0)
            
            # Check if we need to start a new bar
            if (self.current_bar_start[currency] is None or 
                bar_time > self.current_bar_start[currency]):
                
                # Finalize previous bar if it exists
                finalized_bar = None
                if self.current_bar_data[currency] is not None:
                    finalized_bar = self.current_bar_data[currency].copy()
                    finalized_bar['datetime'] = self.current_bar_start[currency]
                    
                    # Add to 15-minute bars DataFrame
                    new_bar_df = pd.DataFrame([finalized_bar])
                    new_bar_df.set_index('datetime', inplace=True)
                    
                    if self.bars_15min[currency].empty:
                        self.bars_15min[currency] = new_bar_df
                    else:
                        self.bars_15min[currency] = pd.concat([self.bars_15min[currency], new_bar_df])
                    
                    # Trim to prevent memory overflow
                    if len(self.bars_15min[currency]) > MAX_MEMORY_ROWS:
                        self.bars_15min[currency] = self.bars_15min[currency].tail(MAX_MEMORY_ROWS)
                    
                    # Trim other memory structures to prevent unbounded growth
                    self._trim_memory_structures(currency)
                
                # Start new bar
                self.current_bar_start[currency] = bar_time
                self.current_bar_data[currency] = {
                    'open': tick_price,
                    'high': tick_price,
                    'low': tick_price,
                    'close': tick_price
                }
                
                return finalized_bar is not None
            
            else:
                # Update current bar
                if self.current_bar_data[currency] is not None:
                    self.current_bar_data[currency]['high'] = max(self.current_bar_data[currency]['high'], tick_price)
                    self.current_bar_data[currency]['low'] = min(self.current_bar_data[currency]['low'], tick_price)
                    self.current_bar_data[currency]['close'] = tick_price
                
                return False
                
        except Exception as e:
            logger.error(f"Error updating bars for {currency}: {e}")
            return False
    
    def _validate_zones_simple(self, tick_price: float, tick_time: datetime, currency: str) -> None:
        """
        Simple zone validation - just check if zones would cause losses
        Lightweight version for zone quality assessment
        """
        try:
            # Check each zone for potential losses
            zones_to_remove = []
            
            for zone_id, zone_data in self.current_zones[currency].items():
                zone_type = zone_data.get('zone_type')
                zone_start = zone_data.get('start_price')
                zone_end = zone_data.get('end_price')
                
                if not all([zone_type, zone_start, zone_end]):
                    continue
                
                # Simple logic: would this price trigger a stop loss?
                would_cause_loss = False
                
                if zone_type == 'demand':
                    # For demand zones, check if price drops below zone significantly
                    zone_size = abs(zone_end - zone_start)
                    stop_loss_level = zone_start - (STOP_LOSS_PROPORTION * zone_size)
                    if tick_price <= stop_loss_level:
                        would_cause_loss = True
                        
                elif zone_type == 'supply':
                    # For supply zones, check if price rises above zone significantly
                    zone_size = abs(zone_end - zone_start)
                    stop_loss_level = zone_start + (STOP_LOSS_PROPORTION * zone_size)
                    if tick_price >= stop_loss_level:
                        would_cause_loss = True
                
                if would_cause_loss:
                    # Track this as a "theoretical loss" for the zone
                    loss_record = {
                        'zone_id': zone_id,
                        'timestamp': tick_time,
                        'price': tick_price,
                        'zone_type': zone_type,
                        'loss_type': 'stop_loss'
                    }
                    
                    # Add to zone validation trades
                    if currency not in self.zone_validation_trades:
                        self.zone_validation_trades[currency] = []
                    
                    self.zone_validation_trades[currency].append(loss_record)
                    
                    # Check if this zone has too many losses
                    zone_losses = [t for t in self.zone_validation_trades[currency] 
                                 if t['zone_id'] == zone_id and t['loss_type'] == 'stop_loss']
                    
                    if len(zone_losses) >= 2:  # 2 consecutive losses
                        zones_to_remove.append(zone_id)
                        logger.info(f"Marking {currency} zone {zone_id} for removal due to {len(zone_losses)} losses")
            
            # Remove problematic zones and track their invalidation
            for zone_id in zones_to_remove:
                if zone_id in self.current_zones[currency]:
                    # Track invalidation time to prevent immediate recreation
                    self.invalidated_zones[currency][zone_id] = tick_time
                    del self.current_zones[currency][zone_id]
                    logger.info(f"Removed problematic {currency} zone: {zone_id}")
                    
        except Exception as e:
            logger.error(f"Error in simple zone validation for {currency}: {e}")
    
    def _process_simulation_trades(self, tick_price: float, tick_time: datetime, currency: str) -> None:
        """
        Process full trade simulation with $100k position sizing
        """
        try:
            # 1. Manage existing simulation trades (SL/TP checks)
            self._manage_simulation_trades(tick_price, tick_time, currency)
            
            # 2. Check for new trade entries (only if no open trades and we have indicators)
            open_trades = [t for t in self.simulation_trades[currency] if t.get('status') == 'open']
            if not open_trades and not self.bars_15min[currency].empty:
                self._check_simulation_entry(tick_price, tick_time, currency)
                
        except Exception as e:
            logger.error(f"Error processing simulation trades for {currency}: {e}")
    
    def _manage_simulation_trades(self, tick_price: float, tick_time: datetime, currency: str) -> None:
        """
        Manage open simulation trades - check stop loss and take profit
        """
        try:
            trades_to_process = [t for t in self.simulation_trades[currency] if t.get('status') == 'open']
            
            for trade in trades_to_process:
                if trade.get('status') != 'open':
                    continue
                
                trade_direction = trade.get('direction')
                stop_loss = trade.get('stop_loss')
                take_profit = trade.get('take_profit')
                position_size = trade.get('position_size', 100000)  # $100k default
                
                trade_closed = False
                close_reason = None
                exit_price = None
                profit_loss = 0
                
                # Check stop loss and take profit conditions
                if trade_direction == 'long':
                    if tick_price <= stop_loss:
                        # Stop loss hit
                        exit_price = stop_loss
                        close_reason = 'stop_loss'
                        # P&L calculation: (exit_price - entry_price) * position_size
                        profit_loss = (exit_price - trade['entry_price']) * position_size
                        trade_closed = True
                        
                    elif tick_price >= take_profit:
                        # Take profit hit
                        exit_price = take_profit
                        close_reason = 'take_profit'
                        profit_loss = (exit_price - trade['entry_price']) * position_size
                        trade_closed = True
                        
                elif trade_direction == 'short':
                    if tick_price >= stop_loss:
                        # Stop loss hit
                        exit_price = stop_loss
                        close_reason = 'stop_loss'
                        # P&L calculation: (entry_price - exit_price) * position_size
                        profit_loss = (trade['entry_price'] - exit_price) * position_size
                        trade_closed = True
                        
                    elif tick_price <= take_profit:
                        # Take profit hit
                        exit_price = take_profit
                        close_reason = 'take_profit'
                        profit_loss = (trade['entry_price'] - exit_price) * position_size
                        trade_closed = True
                
                if trade_closed:
                    # Update trade record
                    trade.update({
                        'status': 'closed',
                        'exit_time': tick_time,
                        'exit_price': exit_price,
                        'close_reason': close_reason,
                        'profit_loss': profit_loss
                    })
                    
                    # Update balance
                    self.simulation_balance[currency] += profit_loss
                    
                    # Store trade in database
                    self._store_simulation_trade(trade, currency)
                    
                    logger.info(f"Simulation trade closed for {currency}: {trade_direction} "
                              f"P&L: ${profit_loss:.2f}, Balance: ${self.simulation_balance[currency]:.2f}")
                    
        except Exception as e:
            logger.error(f"Error managing simulation trades for {currency}: {e}")
    
    def _check_simulation_entry(self, tick_price: float, tick_time: datetime, currency: str) -> None:
        """
        Check for simulation trade entry conditions with full algorithm.py logic
        """
        try:
            if self.bars_15min[currency].empty or len(self.bars_15min[currency]) < 50:
                return
            
            # Get latest indicators from 15-minute bars
            data = self.bars_15min[currency].copy()
            if 'RSI' not in data.columns or 'MACD' not in data.columns:
                return  # Need indicators for entry decisions
            
            current_rsi = data['RSI'].iloc[-1]
            current_macd = data['MACD'].iloc[-1]
            current_signal = data['MACD_Signal'].iloc[-1]
            
            # Skip if indicators are not available
            if pd.isna(current_rsi) or pd.isna(current_macd) or pd.isna(current_signal):
                return
            
            # Check cooldown period (2 hours like algorithm.py)
            if (self.last_simulation_trade_time[currency] is not None and 
                (tick_time - self.last_simulation_trade_time[currency]).total_seconds() < COOLDOWN_SECONDS):
                return
            
            # Use cached ATR value (calculated every 15 minutes)
            atr_value = self.current_atr[currency]
            if atr_value is None:
                # Fallback: calculate if not cached yet (first run scenario)
                atr_value = calculate_atr_pips_required(data, currency)
                self.current_atr[currency] = atr_value
            
            # Check each zone for entry conditions
            for zone_id, zone_data in self.current_zones[currency].items():
                zone_start = zone_data['start_price']
                zone_end = zone_data['end_price']
                zone_type = zone_data['zone_type']
                zone_size = abs(zone_end - zone_start)
                
                # Check if zone size is sufficient
                if zone_size < atr_value:
                    continue
                
                # Check if price is close to zone
                is_close_to_zone = (
                    (zone_type == 'demand' and zone_start <= tick_price <= zone_end + LIQUIDITY_ZONE_ALERT) or
                    (zone_type == 'supply' and zone_end - LIQUIDITY_ZONE_ALERT <= tick_price <= zone_start)
                )
                
                if not is_close_to_zone:
                    continue
                
                # Check entry conditions based on zone type
                entry_valid = False
                direction = None
                stop_loss = None
                take_profit = None
                
                if zone_type == 'demand' and tick_price < zone_end:
                    # Long entry conditions
                    rsi_condition = current_rsi < DEMAND_ZONE_RSI_TOP
                    macd_condition = current_macd > current_signal
                    
                    if rsi_condition or macd_condition:  # Either condition is sufficient
                        entry_valid = True
                        direction = 'long'
                        stop_loss = tick_price - (STOP_LOSS_PROPORTION * zone_size)
                        take_profit = tick_price + (TAKE_PROFIT_PROPORTION * zone_size)
                        
                elif zone_type == 'supply' and tick_price > zone_end:
                    # Short entry conditions  
                    rsi_condition = current_rsi > SUPPLY_ZONE_RSI_BOTTOM
                    macd_condition = current_macd < current_signal
                    
                    if rsi_condition or macd_condition:  # Either condition is sufficient
                        entry_valid = True
                        direction = 'short'
                        stop_loss = tick_price + (STOP_LOSS_PROPORTION * zone_size)
                        take_profit = tick_price - (TAKE_PROFIT_PROPORTION * zone_size)
                
                if entry_valid:
                    # Create simulation trade with $100k position
                    position_size = 100000.0  # $100k nominal position
                    required_margin = position_size / 33.0  # 33:1 leverage = ~$3,030 margin
                    
                    simulation_trade = {
                        'entry_time': tick_time,
                        'entry_price': tick_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'position_size': position_size,
                        'required_margin': required_margin,
                        'status': 'open',
                        'direction': direction,
                        'zone_start_price': zone_start,
                        'zone_end_price': zone_end,
                        'zone_type': zone_type,
                        'currency': currency
                    }
                    
                    self.simulation_trades[currency].append(simulation_trade)
                    self.last_simulation_trade_time[currency] = tick_time
                    
                    logger.info(f"Simulation trade opened for {currency}: {direction} at {tick_price:.5f} "
                              f"(SL: {stop_loss:.5f}, TP: {take_profit:.5f}, Position: ${position_size:,.0f})")
                    
                    # Only open one trade at a time
                    break
                    
        except Exception as e:
            logger.error(f"Error checking simulation entry for {currency}: {e}")
    
    def _store_simulation_trade(self, trade: Dict[str, Any], currency: str) -> None:
        """
        Store completed simulation trade in database
        """
        try:
            trade_record = {
                'Currency': currency,
                'EntryTime': trade['entry_time'],
                'ExitTime': trade.get('exit_time'),
                'EntryPrice': trade['entry_price'],
                'ExitPrice': trade.get('exit_price'),
                'Direction': trade['direction'],
                'PositionSize': trade['position_size'],
                'RequiredMargin': trade['required_margin'],
                'StopLoss': trade['stop_loss'],
                'TakeProfit': trade['take_profit'],
                'Status': trade['status'],
                'CloseReason': trade.get('close_reason'),
                'ProfitLoss': trade.get('profit_loss'),
                'ZoneStartPrice': trade['zone_start_price'],
                'ZoneEndPrice': trade['zone_end_price'],
                'ZoneType': trade['zone_type']
            }
            
            success = db_manager.batch_insert('HistoDataRunningTradeSimulation', [trade_record])
            
            if success:
                # Update summary statistics
                self._update_simulation_summary(currency)
            else:
                logger.error(f"Failed to store simulation trade for {currency}")
                
        except Exception as e:
            logger.error(f"Error storing simulation trade for {currency}: {e}")
    
    def _update_simulation_summary(self, currency: str) -> None:
        """
        Update TradeSimulationSummary table with latest statistics
        """
        try:
            # Calculate statistics from simulation trades
            trades = self.simulation_trades[currency]
            closed_trades = [t for t in trades if t.get('status') == 'closed']
            open_trades = [t for t in trades if t.get('status') == 'open']
            
            winning_trades = [t for t in closed_trades if t.get('profit_loss', 0) > 0]
            losing_trades = [t for t in closed_trades if t.get('profit_loss', 0) < 0]
            
            total_profit_loss = sum(t.get('profit_loss', 0) for t in closed_trades)
            win_rate = (len(winning_trades) / len(closed_trades)) * 100 if closed_trades else 0
            
            avg_win = sum(t.get('profit_loss', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t.get('profit_loss', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            largest_win = max((t.get('profit_loss', 0) for t in winning_trades), default=0)
            largest_loss = min((t.get('profit_loss', 0) for t in losing_trades), default=0)
            
            # Update summary table
            update_query = """
                UPDATE TradeSimulationSummary 
                SET TotalTrades = ?, OpenTrades = ?, ClosedTrades = ?, 
                    WinningTrades = ?, LosingTrades = ?, TotalProfitLoss = ?,
                    WinRate = ?, AverageWin = ?, AverageLoss = ?,
                    LargestWin = ?, LargestLoss = ?, CurrentBalance = ?,
                    LastUpdated = GETDATE()
                WHERE Currency = ?
            """
            
            params = (
                len(trades), len(open_trades), len(closed_trades),
                len(winning_trades), len(losing_trades), total_profit_loss,
                win_rate, avg_win, avg_loss,
                largest_win, largest_loss, self.simulation_balance[currency],
                currency
            )
            
            db_manager.execute_non_query(update_query, params=params)
            
        except Exception as e:
            logger.error(f"Error updating simulation summary for {currency}: {e}")
    
    def _process_completed_bar(self, currency: str) -> None:
        """
        Process completed 15-minute bar - calculate indicators and identify zones
        """
        try:
            if self.bars_15min[currency].empty or len(self.bars_15min[currency]) < 50:
                return
            
            # Use the existing processing logic but with 15-minute bars
            data = self.bars_15min[currency].copy()
            
            # Calculate indicators
            indicators_data = self._calculate_indicators(data, currency)
            
            # Calculate ATR once per 15-minute bar and cache it
            atr_value = calculate_atr_pips_required(data, currency)
            self.current_atr[currency] = atr_value  # Cache for tick-level usage
            
            processed_data, detected_zones = identify_liquidity_zones(
                data, 
                self.current_zones[currency], 
                currency, 
                atr_value
            )
            
            if detected_zones is not None:
                # Only merge NEW zones that don't already exist (prevents recreation from historical data)
                new_zones_count = 0
                for zone_id, zone_data in detected_zones.items():
                    if zone_id not in self.current_zones[currency]:
                        # Check if this zone was recently invalidated
                        if not self._is_zone_recently_invalidated(zone_id, zone_data, currency):
                            self.current_zones[currency][zone_id] = zone_data
                            new_zones_count += 1
                            logger.warning(f"NEW {currency} {zone_data['zone_type'].upper()} ZONE: "
                                         f"Start={zone_data['start_price']:.5f}, End={zone_data['end_price']:.5f}, "
                                         f"Size={abs(zone_data['end_price'] - zone_data['start_price'])*10000:.1f} pips, "
                                         f"Time={zone_data['confirmation_time']}")
                
                if new_zones_count > 0:
                    logger.info(f"Added {new_zones_count} new zones for {currency}")
                else:
                    logger.debug(f"No new zones detected for {currency}")
            
            # Note: Support/resistance invalidation now happens on every tick
            # in _process_single_tick for realistic trading simulation
            
            # Store indicators in database
            self._store_indicators(indicators_data, currency)
            self._store_zones(currency)
            
            # Increment bar counter and check for hourly summary
            self.bars_processed_count[currency] += 1
            
            # Log hourly summary after every 4 bars (1 hour of data)
            if self.bars_processed_count[currency] >= 4:
                self._log_hourly_summary(currency)
                self.bars_processed_count[currency] = 0  # Reset counter
            
            logger.debug(f"Processed completed bar for {currency}: {len(self.current_zones[currency])} active zones")
            
        except Exception as e:
            logger.error(f"Error processing completed bar for {currency}: {e}")
    
    def _log_hourly_summary(self, currency: str):
        """Log hourly summary after processing 4 bars (1 hour) of data"""
        try:
            # Get current stats
            active_zones = len(self.current_zones[currency])
            total_bars = len(self.bars_15min[currency])
            
            # Zone breakdown
            supply_zones = sum(1 for z in self.current_zones[currency].values() if z.get('zone_type') == 'supply')
            demand_zones = sum(1 for z in self.current_zones[currency].values() if z.get('zone_type') == 'demand')
            
            # Get simulation stats
            sim_trades = self.simulation_trades[currency]
            if sim_trades:
                recent_trades = [t for t in sim_trades[-10:]]  # Last 10 trades
                wins = sum(1 for t in recent_trades if t.get('profit_loss', 0) > 0)
                win_rate = (wins / len(recent_trades) * 100) if recent_trades else 0
            else:
                win_rate = 0
                recent_trades = []
            
            balance = self.simulation_balance[currency]
            
            # Log summary
            logger.info(f"{'='*60}")
            logger.info(f"HOURLY SUMMARY - {currency}")
            logger.info(f"{'='*60}")
            logger.info(f"Active Zones: {active_zones} (Supply: {supply_zones}, Demand: {demand_zones})")
            logger.info(f"Total Bars Processed: {total_bars}")
            logger.info(f"Simulation Balance: ${balance:,.2f}")
            if recent_trades:
                logger.info(f"Recent Win Rate: {win_rate:.1f}% (last {len(recent_trades)} trades)")
            logger.info(f"{'='*60}")
            
        except Exception as e:
            logger.error(f"Error generating hourly summary for {currency}: {e}")
    
    def get_last_processed_timestamp(self, currency: str) -> Optional[datetime]:
        """Get the last processed timestamp for a currency"""
        return self.last_processed_timestamp.get(currency)
    
    def _calculate_indicators(self, data: pd.DataFrame, currency: str) -> List[Dict[str, Any]]:
        """
        Calculate RSI and MACD indicators for the data
        
        Args:
            data: Price data DataFrame
            currency: Currency pair
            
        Returns:
            List of indicator records to store
        """
        try:
            # Calculate RSI (14-period)
            data['RSI'] = ta.rsi(data['close'], length=14)
            
            # Calculate MACD (12, 26, 9)
            macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
            
            if macd is not None and not macd.empty:
                data['MACD'] = macd['MACD_12_26_9']
                data['MACD_Signal'] = macd['MACDs_12_26_9']
                data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            else:
                data['MACD'] = np.nan
                data['MACD_Signal'] = np.nan
                data['MACD_Histogram'] = np.nan
            
            # Prepare indicator records (only for rows with valid data)
            indicator_records = []
            valid_data = data.dropna(subset=['RSI', 'MACD', 'MACD_Signal'])
            
            for index, row in valid_data.iterrows():
                record = {
                    'Currency': currency,
                    'Timestamp': index,
                    'RSI': row['RSI'],
                    'MACD': row['MACD'],
                    'MACDSignal': row['MACD_Signal'],
                    'MACDHistogram': row['MACD_Histogram'],
                    'Price': row['close']
                }
                indicator_records.append(record)
            
            return indicator_records
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {currency}: {e}")
            return []
    
    def _store_indicators(self, indicator_records: List[Dict[str, Any]], currency: str) -> bool:
        """
        Store indicator data with improved duplicate handling
        
        Args:
            indicator_records: List of indicator records
            currency: Currency pair
            
        Returns:
            True if successful, False otherwise
        """
        if not indicator_records:
            return True
        
        try:
            # Try batch insert first
            success = db_manager.batch_insert('IndicatorTracker', indicator_records)
            if success:
                logger.debug(f"Stored {len(indicator_records)} indicator records for {currency}")
                return True
            
            # Batch failed - try individual inserts and count successes
            inserted = 0
            for rec in indicator_records:
                try:
                    if db_manager.batch_insert('IndicatorTracker', [rec]):
                        inserted += 1
                except Exception:
                    # Silently skip duplicates and other non-critical errors
                    continue
            
            # Only log error if NO records were inserted
            if inserted == 0:
                logger.warning(f"No indicator records stored for {currency} (likely all duplicates)")
            else:
                logger.debug(f"Stored {inserted}/{len(indicator_records)} indicators for {currency}")
            
            return inserted > 0
            
        except Exception as e:
            logger.error(f"Critical error storing indicators for {currency}: {e}")
            return False
    
    def _store_zones(self, currency: str) -> bool:
        """
        Store current zones in ZoneTracker table using differential updates
        
        Args:
            currency: Currency pair
            
        Returns:
            True if successful, False otherwise
        """
        try:
            current_zones = self.current_zones[currency]
            
            # Get existing active zones from database
            existing_query = """
                SELECT ZoneStartPrice, ZoneEndPrice, ZoneType, ConfirmationTime 
                FROM ZoneTracker 
                WHERE Currency = ? AND IsActive = 1
            """
            
            existing_df = db_manager.execute_query_to_dataframe(existing_query, params=(currency,))
            existing_zones = set()
            
            if not existing_df.empty:
                for _, row in existing_df.iterrows():
                    zone_key = (
                        float(row['ZoneStartPrice']), 
                        float(row['ZoneEndPrice']), 
                        row['ZoneType'],
                        row['ConfirmationTime']
                    )
                    existing_zones.add(zone_key)
            
            # Build current zones set
            current_zones_set = set()
            new_zone_records = []
            
            for zone_id, zone_data in current_zones.items():
                zone_key = (
                    zone_data['start_price'],
                    zone_data['end_price'], 
                    zone_data['zone_type'],
                    zone_data['confirmation_time']
                )
                current_zones_set.add(zone_key)
                
                # If this zone doesn't exist in DB, prepare to insert it
                if zone_key not in existing_zones:
                    record = {
                        'Currency': currency,
                        'ZoneStartPrice': zone_data['start_price'],
                        'ZoneEndPrice': zone_data['end_price'],
                        'ZoneSize': zone_data.get('zone_size', abs(zone_data['end_price'] - zone_data['start_price'])),
                        'ZoneType': zone_data['zone_type'],
                        'ConfirmationTime': zone_data['confirmation_time'],
                        'IsActive': 1
                    }
                    new_zone_records.append(record)
            
            # Find zones to deactivate (exist in DB but not in current zones)
            zones_to_deactivate = existing_zones - current_zones_set
            
            # Deactivate removed zones
            if zones_to_deactivate:
                for zone_key in zones_to_deactivate:
                    start_price, end_price, zone_type, confirmation_time = zone_key
                    deactivate_query = """
                        UPDATE ZoneTracker 
                        SET IsActive = 0, InvalidationTime = GETDATE() 
                        WHERE Currency = ? AND ZoneStartPrice = ? AND ZoneEndPrice = ? 
                        AND ZoneType = ? AND ConfirmationTime = ? AND IsActive = 1
                    """
                    db_manager.execute_non_query(deactivate_query, 
                                               params=(currency, start_price, end_price, zone_type, confirmation_time))
                
                logger.debug(f"Deactivated {len(zones_to_deactivate)} zones for {currency}")
            
            # Insert new zones
            if new_zone_records:
                success = db_manager.batch_insert('ZoneTracker', new_zone_records)
                
                if success:
                    logger.info(f"Inserted {len(new_zone_records)} new zones for {currency}")
                else:
                    logger.error(f"Failed to insert new zones for {currency}")
                    return False
            
            if not zones_to_deactivate and not new_zone_records:
                logger.debug(f"No zone changes for {currency}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing zones for {currency}: {e}")
            return False
    
    def get_zone_count(self, currency: str) -> int:
        """Get current number of active zones for a currency"""
        return len(self.current_zones.get(currency, {}))
    
    def get_latest_price(self, currency: str) -> Optional[float]:
        """Get latest price for a currency"""
        if currency in self.price_data and not self.price_data[currency].empty:
            return float(self.price_data[currency]['close'].iloc[-1])
        return None
    

    
    def get_memory_usage_stats(self) -> Dict[str, int]:
        """Get memory usage statistics for each currency"""
        stats = {}
        for currency in SUPPORTED_CURRENCIES:
            stats[currency] = {
                'ticks': len(self.tick_data.get(currency, pd.DataFrame())),
                'bars_15min': len(self.bars_15min.get(currency, pd.DataFrame())),
                'zones': len(self.current_zones.get(currency, {})),
                'validation_records': len(self.zone_validation_trades.get(currency, []))
            }
        return stats
    
    def get_zone_validation_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get zone validation statistics for each currency"""
        stats = {}
        for currency in SUPPORTED_CURRENCIES:
            validation_records = self.zone_validation_trades.get(currency, [])
            active_zones = len(self.current_zones.get(currency, {}))
            
            # Count losses by zone
            zone_losses = {}
            for record in validation_records:
                zone_id = record['zone_id']
                if zone_id not in zone_losses:
                    zone_losses[zone_id] = 0
                zone_losses[zone_id] += 1
            
            stats[currency] = {
                'active_zones': active_zones,
                'total_validation_records': len(validation_records),
                'zones_with_losses': len(zone_losses),
                'average_losses_per_zone': sum(zone_losses.values()) / len(zone_losses) if zone_losses else 0,
                'max_losses_single_zone': max(zone_losses.values()) if zone_losses else 0
            }
        
        return stats
    
    def get_simulation_trade_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get full trade simulation statistics for each currency"""
        stats = {}
        for currency in SUPPORTED_CURRENCIES:
            trades = self.simulation_trades[currency]
            closed_trades = [t for t in trades if t.get('status') == 'closed']
            open_trades = [t for t in trades if t.get('status') == 'open']
            
            winning_trades = [t for t in closed_trades if t.get('profit_loss', 0) > 0]
            losing_trades = [t for t in closed_trades if t.get('profit_loss', 0) < 0]
            
            total_profit = sum(t.get('profit_loss', 0) for t in closed_trades)
            
            stats[currency] = {
                'total_trades': len(trades),
                'open_trades': len(open_trades),
                'closed_trades': len(closed_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'total_profit_loss': total_profit,
                'win_rate': (len(winning_trades) / len(closed_trades)) * 100 if closed_trades else 0,
                'current_balance': self.simulation_balance[currency],
                'average_win': sum(t.get('profit_loss', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0,
                'average_loss': sum(t.get('profit_loss', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0,
                'largest_win': max((t.get('profit_loss', 0) for t in winning_trades), default=0),
                'largest_loss': min((t.get('profit_loss', 0) for t in losing_trades), default=0)
            }
        
        return stats
    
    def _trim_memory_structures(self, currency: str) -> None:
        """Trim all memory structures to prevent overflow during long processing runs"""
        try:
            # Trim trade validation records to last 1000 entries
            if len(self.zone_validation_trades[currency]) > 1000:
                self.zone_validation_trades[currency] = self.zone_validation_trades[currency][-1000:]
            
            # Trim simulation trades to last 1000 entries  
            if len(self.simulation_trades[currency]) > 1000:
                self.simulation_trades[currency] = self.simulation_trades[currency][-1000:]
            
            # Trim tick_data if it exists and gets large
            if not self.tick_data[currency].empty and len(self.tick_data[currency]) > MAX_MEMORY_ROWS:
                self.tick_data[currency] = self.tick_data[currency].tail(MAX_MEMORY_ROWS)
                
            # Trim price_data if it exists and gets large
            if not self.price_data[currency].empty and len(self.price_data[currency]) > MAX_MEMORY_ROWS:
                self.price_data[currency] = self.price_data[currency].tail(MAX_MEMORY_ROWS)
            
            # Clean up invalidated zones tracking (keep max 10 entries, remove older than 2 hours)
            self._cleanup_invalidated_zones(currency)
                
        except Exception as e:
            logger.error(f"Error trimming memory structures for {currency}: {e}")
    
    def _is_zone_recently_invalidated(self, zone_id: tuple, zone_data: Dict, currency: str) -> bool:
        """
        Check if a zone was recently invalidated to prevent immediate recreation
        Only blocks recreation if zone confirmation time is NOT after invalidation
        """
        try:
            if zone_id not in self.invalidated_zones[currency]:
                return False  # Zone was never invalidated
                
            invalidation_time = self.invalidated_zones[currency][zone_id]
            zone_confirmation_time = zone_data.get('confirmation_time', datetime.now())
            
            # Allow recreation if zone confirmation is AFTER invalidation (new data)
            if zone_confirmation_time > invalidation_time:
                # Remove from invalidated list since it's now valid with new data
                del self.invalidated_zones[currency][zone_id]
                logger.info(f"Allowing recreation of {currency} zone {zone_id} - confirmed with new data after invalidation")
                return False
            else:
                # Block recreation - same historical data
                logger.debug(f"Blocking recreation of {currency} zone {zone_id} - same historical data")
                return True
                
        except Exception as e:
            logger.error(f"Error checking zone invalidation for {currency}: {e}")
            return False  # Allow on error
    
    def _cleanup_invalidated_zones(self, currency: str) -> None:
        """
        Clean up invalidated zones tracking to prevent memory leaks
        - Remove entries older than 2 hours
        - Keep maximum 10 most recent entries
        """
        try:
            if currency not in self.invalidated_zones or not self.invalidated_zones[currency]:
                return
                
            current_time = datetime.now()
            cutoff_time = current_time - pd.Timedelta(hours=2)
            
            # Remove old entries (older than 2 hours)
            old_zones = [zone_id for zone_id, inv_time in self.invalidated_zones[currency].items() 
                        if inv_time < cutoff_time]
            for zone_id in old_zones:
                del self.invalidated_zones[currency][zone_id]
            
            # Keep only the 10 most recent entries if we still have too many
            if len(self.invalidated_zones[currency]) > 10:
                # Sort by invalidation time (most recent first) and keep top 10
                sorted_zones = sorted(self.invalidated_zones[currency].items(), 
                                    key=lambda x: x[1], reverse=True)
                self.invalidated_zones[currency] = dict(sorted_zones[:10])
                logger.debug(f"Trimmed invalidated zones for {currency} to 10 most recent entries")
                
        except Exception as e:
            logger.error(f"Error cleaning up invalidated zones for {currency}: {e}")

# Global processor instance
zone_processor = ZoneAndIndicatorProcessor() 