"""
Risk management utilities: position sizing and leverage calculations.
"""

def position_size(account_equity, risk_per_trade_pct, stop_distance_in_quote, price, max_leverage):
    """
    Calculate position size and leverage for a trade.

    Parameters:
    - account_equity: float, total account equity in base currency (e.g. USD)
    - risk_per_trade_pct: float, risk percent per trade (e.g. 0.01 for 1%)
    - stop_distance_in_quote: float, stop loss distance in quote currency units (e.g. 0.0012)
    - price: float, current price of the instrument (e.g. 1.0834)
    - max_leverage: float, maximum allowed leverage (e.g. 30)

    Returns:
    dict with keys:
    - 'units': position size in base currency lots
    - 'notional': notional exposure value
    - 'used_leverage': effective leverage used
    """

    if account_equity <= 0:
        raise ValueError("Account equity must be positive")
    if risk_per_trade_pct <= 0 or risk_per_trade_pct > 1:
        raise ValueError("Risk per trade percent must be between 0 and 1")
    if stop_distance_in_quote <= 0:
        raise ValueError("Stop distance must be positive")
    if price <= 0:
        raise ValueError("Price must be positive")
    if max_leverage <= 0:
        raise ValueError("Max leverage must be positive")

    risk_amount = account_equity * risk_per_trade_pct
    notional_exposure = risk_amount / stop_distance_in_quote
    max_notional = account_equity * max_leverage
    notional_capped = min(notional_exposure, max_notional)
    used_leverage = notional_capped / account_equity
    units = notional_capped / price

    return {
        'units': units,
        'notional': notional_capped,
        'used_leverage': used_leverage
    }
