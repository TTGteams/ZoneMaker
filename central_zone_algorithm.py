"""
Central Zone Algorithm Functions
Core algorithmic functions for zone identification, validation, and ATR calculations
Extracted from algorithm.py with dependencies removed for clean modular design
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List, Any

# Set up logging
logger = logging.getLogger("zone_tracker")

# Trading constants (extracted from algorithm.py)
STOP_LOSS_PROPORTION = 0.11
TAKE_PROFIT_PROPORTION = 0.65
RISK_PROPORTION = 0.03
COOLDOWN_SECONDS = 7200
SUPPORT_RESISTANCE_ALLOWANCE = 0.0011
LIQUIDITY_ZONE_ALERT = 0.002

def calculate_atr_pips_required(data: pd.DataFrame, currency: str = "EUR.USD", use_cache: bool = True) -> float:
    """
    Calculate the ATR-based pips required using a 4-day window.
    Returns a value between 0.002 (20 pips) and 0.018 (180 pips).
    Uses caching to avoid redundant calculations.
    
    Args:
        data: DataFrame with OHLC data
        currency: Currency pair identifier
        use_cache: Whether to use cached values
        
    Returns:
        float: Pips required based on ATR calculation
    """
    # Use cached value if available and requested
    if use_cache and hasattr(calculate_atr_pips_required, "_cache"):
        cache = calculate_atr_pips_required._cache
        if currency in cache and cache[currency]["timestamp"] >= data.index[-1] - pd.Timedelta(minutes=30):
            return cache[currency]["value"]
    
    try:
        # Calculate True Range
        data = data.copy()
        
        # Ensure data is numeric
        data['high'] = pd.to_numeric(data['high'])
        data['low'] = pd.to_numeric(data['low'])
        data['close'] = pd.to_numeric(data['close'])
        
        data['prev_close'] = data['close'].shift(1)
        
        # True Range calculation
        data['tr1'] = abs(data['high'] - data['low'])
        data['tr2'] = abs(data['high'] - data['prev_close'])
        data['tr3'] = abs(data['low'] - data['prev_close'])
        
        data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Check if we have enough data for ATR
        non_nan_count = data['true_range'].notna().sum()
        if non_nan_count < 14:  # Need at least 14 values for meaningful ATR
            logger.warning(f"Not enough data for ATR calculation ({non_nan_count} values), using default 30 pips")
            return 0.003  # Default 30 pips
            
        # Calculate ATR with appropriate window size
        window_size = min(96, max(14, non_nan_count // 2))  # Adjust window based on available data
        atr = data['true_range'].rolling(window=window_size).mean().iloc[-1]
        
        if pd.isna(atr):
            logger.warning(f"ATR calculation resulted in NaN, using default 30 pips")
            return 0.003  # Default value if ATR is NaN
        
        # Convert ATR to pips required (bounded between 20 and 180 pips)
        pips_required = min(max(atr, 0.002), 0.018)  # 0.002 = 20 pips, 0.018 = 180 pips
        
        # Log less frequently
        logger.info(f"ATR value: {atr:.6f}, Pips required: {pips_required*10000:.1f} pips")
        
        # Cache the result
        if not hasattr(calculate_atr_pips_required, "_cache"):
            calculate_atr_pips_required._cache = {}
        
        calculate_atr_pips_required._cache[currency] = {
            "value": pips_required,
            "timestamp": data.index[-1]
        }
        
        return pips_required
        
    except Exception as e:
        logger.error(f"Error in ATR calculation: {e}", exc_info=True)
        return 0.003  # Default to 30 pips on error

def set_support_resistance_lines(data: pd.DataFrame, currency: str = "EUR.USD") -> pd.DataFrame:
    """
    Sets support and resistance lines based on liquidity zones.
    
    Args:
        data: DataFrame containing zone information
        currency: The currency pair to process
        
    Returns:
        DataFrame with support and resistance lines added
    """
    data = data.copy()
    data['Support_Line'] = np.where(
        data['zone_type'] == 'demand',
        data['Liquidity_Zone'] - SUPPORT_RESISTANCE_ALLOWANCE,
        np.nan
    )
    data['Resistance_Line'] = np.where(
        data['zone_type'] == 'supply',
        data['Liquidity_Zone'] + SUPPORT_RESISTANCE_ALLOWANCE,
        np.nan
    )
    return data

def invalidate_zones_via_sup_and_resist(current_price: float, valid_zones_dict: Dict, currency: str = "EUR.USD") -> Dict:
    """
    Compare the current_price to each zone's threshold. 
    If price has violated it, remove the zone immediately.
    
    Args:
        current_price: The current market price
        valid_zones_dict: Dictionary of current valid zones
        currency: The currency pair to process
        
    Returns:
        Updated zones dictionary with invalidated zones removed
    """
    zones_to_invalidate = []
    
    # Handle both nested (currency-keyed) and flat zone dictionaries
    zones_dict = valid_zones_dict.get(currency, valid_zones_dict) if isinstance(valid_zones_dict, dict) else valid_zones_dict
    
    for zone_id, zone_data in list(zones_dict.items()):
        # Skip zones missing required data
        if 'start_price' not in zone_data or 'zone_type' not in zone_data:
            logger.warning(f"Skipping zone validation for incomplete zone: {zone_id}")
            continue
            
        zone_start = zone_data['start_price']
        zone_type = zone_data['zone_type']

        if zone_type == 'demand':
            support_line = zone_start - SUPPORT_RESISTANCE_ALLOWANCE
            # If price dips far below the zone's support line
            if current_price < (support_line - SUPPORT_RESISTANCE_ALLOWANCE):
                zones_to_invalidate.append((zone_id, "support line violation"))

        elif zone_type == 'supply':
            resistance_line = zone_start + SUPPORT_RESISTANCE_ALLOWANCE
            # If price rises far above the zone's resistance line
            if current_price > (resistance_line + SUPPORT_RESISTANCE_ALLOWANCE):
                zones_to_invalidate.append((zone_id, "resistance line violation"))

    # If any zones were invalidated, log them
    if zones_to_invalidate:
        logger.warning(f"\n\n{'!'*20} {currency} ZONE INVALIDATIONS {'!'*20}")
        for zone_id, reason in zones_to_invalidate:
            zone_data = zones_dict[zone_id]
            zone_type = zone_data.get('zone_type', 'unknown')
            size_pips = zone_data.get('zone_size', 0) * 10000  # Convert to pips
            logger.warning(
                f"Invalidating {zone_type.upper()} zone: "
                f"Start={zone_id[0]:.5f}, End={zone_id[1]:.5f}, "
                f"Size={size_pips:.1f} pips, Reason: {reason}"
            )
        logger.warning(f"{'!'*53}\n")
        
        # Remove the invalidated zones
        for zone_id, _ in zones_to_invalidate:
            del zones_dict[zone_id]

    return zones_dict

def identify_liquidity_zones(data: pd.DataFrame, 
                           current_valid_zones_dict: Dict, 
                           currency: str = "EUR.USD", 
                           pre_calculated_atr: Optional[float] = None,
                           cumulative_zone_info: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Identify liquidity zones in price data using consecutive candle analysis.
    
    Args:
        data: DataFrame with OHLC data
        current_valid_zones_dict: Dictionary of existing valid zones
        currency: Currency pair identifier
        pre_calculated_atr: Pre-calculated ATR value (optional)
        cumulative_zone_info: State information for zone detection across windows
        
    Returns:
        Tuple of (processed_data, updated_zones_dict)
    """
    # Initialize cumulative zone info if not provided
    if cumulative_zone_info is None:
        cumulative_zone_info = {}
    
    logger.debug(f"Identifying zones for {currency}: Processing {len(data)} bars, existing zones: {len(current_valid_zones_dict)}")
    
    # Ensure currency is initialized in cumulative_zone_info
    if currency not in cumulative_zone_info:
        cumulative_zone_info[currency] = None
    
    data = data.copy()
    new_zones_detected = 0
    new_zones_for_storage = []

    # Initialize columns
    data.loc[:, 'Liquidity_Zone'] = np.nan
    data.loc[:, 'Zone_Size'] = np.nan
    data.loc[:, 'Zone_Start_Price'] = np.nan
    data.loc[:, 'Zone_End_Price'] = np.nan
    data.loc[:, 'Zone_Length'] = np.nan
    data.loc[:, 'zone_type'] = ''
    data.loc[:, 'Confirmation_Time'] = pd.NaT

    # Calculate dynamic PIPS_REQUIRED based on ATR (use pre-calculated value if provided)
    if pre_calculated_atr is not None:
        dynamic_pips_required = pre_calculated_atr
    else:
        dynamic_pips_required = calculate_atr_pips_required(data, currency)
    
    logger.debug(f"Dynamic pip threshold: {dynamic_pips_required*10000:.1f} pips")

    # Add candle direction column
    data['candle_direction'] = np.where(data['close'] > data['open'], 'green', 
                                      np.where(data['close'] < data['open'], 'red', 'doji'))

    # If we have a zone in progress from previous window, restore it
    current_run = None
    if cumulative_zone_info[currency] is not None:
        current_run = cumulative_zone_info[currency]
        if current_run['start_index'] < len(data):
            current_run['start_index'] = 0
        else:
            current_run = None

    # Process each candle for zone identification
    for i in range(len(data)):
        # Log first few candles for debugging
        if i < 3:
            logger.info(f"Candle {i}: Open={data.iloc[i]['open']:.5f}, Close={data.iloc[i]['close']:.5f}, Direction={data.iloc[i]['candle_direction']}")
        
        current_candle = data.iloc[i]
        
        if current_run is None:
            # Start new run
            current_run = {
                'start_index': i,
                'start_price': current_candle['open'],
                'direction': current_candle['candle_direction'],
                'high': current_candle['high'],
                'low': current_candle['low'],
                'start_time': data.index[i]
            }
            continue

        # Check if this candle continues the run
        if current_candle['candle_direction'] == current_run['direction'] and \
           current_candle['candle_direction'] != 'doji':
            
            # Update run's high/low
            current_run['high'] = max(current_run['high'], current_candle['high'])
            current_run['low'] = min(current_run['low'], current_candle['low'])
            
            # Calculate total movement
            if current_run['direction'] == 'green':
                total_move = current_run['high'] - current_run['start_price']
            else:  # red
                total_move = current_run['start_price'] - current_run['low']

            # Check if movement meets dynamic pip requirement
            if abs(total_move) >= dynamic_pips_required:
                logger.debug(f"ZONE QUALIFICATION: Move={abs(total_move)*10000:.1f} pips >= Required={dynamic_pips_required*10000:.1f} pips")
                zone_type = 'demand' if current_run['direction'] == 'green' else 'supply'
                
                if zone_type == 'demand':
                    zone_start = current_run['low']
                    zone_end = current_run['high']
                else:
                    zone_start = current_run['high']
                    zone_end = current_run['low']

                # Round zone prices to prevent floating point duplicates
                zone_start = round(zone_start, 6)
                zone_end = round(zone_end, 6)
                
                zone_id = (zone_start, zone_end)
                
                # Only add if zone doesn't exist
                if zone_id not in current_valid_zones_dict:
                    # Create zone object with all required fields
                    zone_object = {
                        'start_price': zone_start,
                        'end_price': zone_end,
                        'zone_size': abs(zone_end - zone_start),
                        'confirmation_time': data.index[i],
                        'zone_type': zone_type,
                        'start_time': current_run['start_time']
                    }
                    
                    # Add to zones dictionary
                    current_valid_zones_dict[zone_id] = zone_object
                    new_zones_detected += 1
                    
                    # Store for potential database insertion (caller's responsibility)
                    new_zones_for_storage.append({
                        'currency': currency,
                        'zone_start_price': zone_start,
                        'zone_end_price': zone_end,
                        'zone_size': abs(zone_end - zone_start),
                        'zone_type': zone_type,
                        'confirmation_time': data.index[i]
                    })

                    # Mark the zone in the data
                    zone_indices = slice(current_run['start_index'], i + 1)
                    data.loc[data.index[zone_indices], 'Liquidity_Zone'] = zone_start
                    data.loc[data.index[zone_indices], 'Zone_Size'] = abs(zone_end - zone_start)
                    data.loc[data.index[zone_indices], 'Zone_Start_Price'] = zone_start
                    data.loc[data.index[zone_indices], 'Zone_End_Price'] = zone_end
                    data.loc[data.index[zone_indices], 'Zone_Length'] = i - current_run['start_index'] + 1
                    data.loc[data.index[zone_indices], 'zone_type'] = zone_type
                    data.loc[data.index[zone_indices], 'Confirmation_Time'] = data.index[i]

                    logger.warning(
                        f"NEW {currency} {zone_type.upper()} ZONE: Start={zone_start:.5f}, End={zone_end:.5f}, "
                        f"Size={abs(zone_end - zone_start)*10000:.1f} pips, Time={data.index[i]}"
                    )

                # Reset run after creating zone
                current_run = None
            elif abs(total_move) >= dynamic_pips_required * 0.7:  # Log rejections close to threshold
                logger.debug(f"ZONE REJECTED: Move={abs(total_move)*10000:.1f} pips < Required={dynamic_pips_required*10000:.1f} pips")

        else:
            # Direction changed, reset the run
            current_run = {
                'start_index': i,
                'start_price': current_candle['open'],
                'direction': current_candle['candle_direction'],
                'high': current_candle['high'],
                'low': current_candle['low'],
                'start_time': data.index[i]
            }

    # Save current run state for next window
    cumulative_zone_info[currency] = current_run
    
    logger.debug(f"ZONE DETECTION RESULTS: New zones: {new_zones_detected}, Total: {len(current_valid_zones_dict)}")
    
    # Log current zones summary at debug level
    if current_valid_zones_dict:
        logger.debug(f"Current zones (count: {len(current_valid_zones_dict)}):")
        for zone_id, zone in current_valid_zones_dict.items():
            logger.debug(f"  {zone['zone_type']} zone: {zone['start_price']:.5f}-{zone['end_price']:.5f}")

    return data, current_valid_zones_dict 
