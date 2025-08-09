"""
Simple fetcher wrappers for FX data sources (Alpha Vantage & OANDA). Add your API keys as env vars.
"""
import os
import time
import argparse
from typing import List
import requests
import pandas as pd

ALPHA_KEY = os.getenv('ALPHAVANTAGE_KEY')


def _parse_intraday_alpha(json_resp, interval):
    # AlphaVantage returns keys like 'Time Series FX (60min)'
    key = None
    for k in json_resp.keys():
        if 'Time Series' in k or 'Time Series FX' in k:
            key = k
            break
    if key is None:
        raise ValueError('Unexpected AlphaVantage response format')
    data = json_resp[key]
    records = []
    for ts, vals in data.items():
        r = {
            'timestamp': pd.to_datetime(ts),
            'open': float(vals.get('1. open', 0)),
            'high': float(vals.get('2. high', 0)),
            'low': float(vals.get('3. low', 0)),
            'close': float(vals.get('4. close', 0))
        }
        records.append(r)
    df = pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)
    return df


def fetch_alphavantage_fx(from_symbol: str, to_symbol: str, interval: str = '60min', outputsize: str = 'compact'):
    if ALPHA_KEY is None:
        raise EnvironmentError('Set ALPHAVANTAGE_KEY in environment')
    url = 'https://www.alphavantage.co/query'
    if interval in ['1min', '5min', '15min', '30min', '60min']:
        function = 'FX_INTRADAY'
    else:
        function = 'FX_DAILY'
    params = {
        'function': function,
        'from_symbol': from_symbol,
        'to_symbol': to_symbol,
        'interval': interval,
        'outputsize': outputsize,
        'apikey': ALPHA_KEY
    }
    r = requests.get(url, params=params)
    j = r.json()
    if r.status_code != 200:
        raise RuntimeError(f'AlphaVantage error: {r.status_code} {j}')
    df = _parse_intraday_alpha(j, interval)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', type=str, required=True, help='Comma separated list e.g. EURUSD,GBPUSD')
    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--interval', type=str, default='60min')
    args = parser.parse_args()
    pairs = [s.strip().upper() for s in args.symbols.split(',')]
    for p in pairs:
        base = p[:3]
        quote = p[3:]
        print(f'Fetching {base}/{quote}...')
        df = fetch_alphavantage_fx(base, quote, interval=args.interval)
        out = f'data/raw/{p}_{args.interval}.csv'
        df.to_csv(out, index=False)
        print('Saved', out)
        time.sleep(15)  # respect rate limits
