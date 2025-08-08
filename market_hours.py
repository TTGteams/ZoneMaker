"""
Market Hours Checker for FX Trading
Determines if FX market is open (Sunday 3pm MST to Friday 3pm MST)
"""

import datetime
import pytz
from typing import bool

def is_market_open() -> bool:
    """
    Check if FX market is currently open
    FX Market Hours: Sunday 3:00 PM MST to Friday 3:00 PM MST
    
    Returns:
        True if market is open, False otherwise
    """
    # Get current time in MST timezone
    mst = pytz.timezone('America/Denver')  # MST/MDT timezone
    current_time = datetime.datetime.now(mst)
    
    current_day = current_time.weekday()  # 0=Monday, 6=Sunday
    current_hour = current_time.hour
    
    # Market opens Sunday 3 PM MST (day 6)
    # Market closes Friday 3 PM MST (day 4)
    
    # Check if it's during market days
    if current_day == 6:  # Sunday
        # Market opens at 3 PM on Sunday
        return current_hour >= 15
    elif current_day in [0, 1, 2, 3]:  # Monday through Thursday
        # Market is open all day
        return True
    elif current_day == 4:  # Friday
        # Market closes at 3 PM on Friday
        return current_hour < 15
    else:  # Saturday (day 5)
        # Market is closed on Saturday
        return False

def get_next_market_open() -> datetime.datetime:
    """
    Get the next market open time
    
    Returns:
        DateTime of next market open in MST
    """
    mst = pytz.timezone('America/Denver')
    current_time = datetime.datetime.now(mst)
    current_day = current_time.weekday()
    current_hour = current_time.hour
    
    # If it's Sunday and before 3 PM, next open is today at 3 PM
    if current_day == 6 and current_hour < 15:
        next_open = current_time.replace(hour=15, minute=0, second=0, microsecond=0)
    else:
        # Next open is next Sunday at 3 PM
        days_until_sunday = (6 - current_day) % 7
        if days_until_sunday == 0:  # It's Sunday but after 3 PM
            days_until_sunday = 7
        
        next_open = current_time + datetime.timedelta(days=days_until_sunday)
        next_open = next_open.replace(hour=15, minute=0, second=0, microsecond=0)
    
    return next_open

def get_next_market_close() -> datetime.datetime:
    """
    Get the next market close time
    
    Returns:
        DateTime of next market close in MST
    """
    mst = pytz.timezone('America/Denver')
    current_time = datetime.datetime.now(mst)
    current_day = current_time.weekday()
    current_hour = current_time.hour
    
    # If it's Friday and before 3 PM, next close is today at 3 PM
    if current_day == 4 and current_hour < 15:
        next_close = current_time.replace(hour=15, minute=0, second=0, microsecond=0)
    else:
        # Next close is next Friday at 3 PM
        days_until_friday = (4 - current_day) % 7
        if days_until_friday == 0:  # It's Friday but after 3 PM
            days_until_friday = 7
        
        next_close = current_time + datetime.timedelta(days=days_until_friday)
        next_close = next_close.replace(hour=15, minute=0, second=0, microsecond=0)
    
    return next_close

def get_market_status() -> dict:
    """
    Get comprehensive market status information
    
    Returns:
        Dictionary with market status details
    """
    is_open = is_market_open()
    next_open = get_next_market_open()
    next_close = get_next_market_close()
    
    mst = pytz.timezone('America/Denver')
    current_time = datetime.datetime.now(mst)
    
    return {
        'is_open': is_open,
        'current_time_mst': current_time,
        'next_open': next_open,
        'next_close': next_close,
        'time_until_next_event': (next_close if is_open else next_open) - current_time
    }

# Test function for debugging
def test_market_hours():
    """Test function to verify market hours logic"""
    status = get_market_status()
    print(f"Market Status: {'OPEN' if status['is_open'] else 'CLOSED'}")
    print(f"Current Time (MST): {status['current_time_mst']}")
    print(f"Next Open: {status['next_open']}")
    print(f"Next Close: {status['next_close']}")
    print(f"Time until next event: {status['time_until_next_event']}")

if __name__ == "__main__":
    test_market_hours() 