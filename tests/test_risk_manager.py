from src.risk.risk_manager import position_size

def test_position_size_basic():
    res = position_size(account_equity=1000, risk_per_trade_pct=0.01, stop_distance_in_quote=0.001, price=1.2, max_leverage=30)
    assert res['used_leverage'] <= 30
    assert res['units'] > 0
