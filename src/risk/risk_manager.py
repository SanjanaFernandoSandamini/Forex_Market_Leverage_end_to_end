"""
Risk management utilities: position sizing and leverage calculations.
"""

def position_size(account_equity, risk_per_trade_pct, stop_distance_in_quote, price, max_leverage):
    """
    account_equity: in account currency (e.g. USD)
    risk_per_trade_pct: e.g. 0.01 (1%)
    stop_distance_in_quote: distance between entry and stop in quote currency units (e.g. 0.0012)
    price: current pair price (e.g. 1.0834)
    max_leverage: e.g. 30 (30:1)

    returns dict: units (base currency lots), notional, used_leverage
    """
    risk_amount = account_equity * risk_per_trade_pct
    if stop_distance_in_quote <= 0:
        raise ValueError('stop_distance must be positive')
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
