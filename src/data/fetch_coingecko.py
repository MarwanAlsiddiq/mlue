import requests
import pandas as pd
import os
import time

def fetch_coingecko_ohlcv(symbol_id, vs_currency="usd", days="max", data_dir="data/raw"):
    """
    Fetch daily OHLCV and market cap data from CoinGecko for a given coin id (e.g., 'bitcoin', 'gala').
    """
    url = f"https://api.coingecko.com/api/v3/coins/{symbol_id}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": days,
        "interval": "daily"
    }
    headers = {"User-Agent": "Mozilla/5.0 (compatible; CoinGeckoFetcher/1.0)"}
    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    
    # Parse price, market cap, volume
    prices = data.get("prices", [])
    market_caps = data.get("market_caps", [])
    total_volumes = data.get("total_volumes", [])
    
    df = pd.DataFrame(prices, columns=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    
    # Add market cap and volume
    df["market_cap"] = [x[1] for x in market_caps]
    df["volume"] = [x[1] for x in total_volumes]
    
    # CoinGecko does not provide OHLC directly, so use close as proxy
    df["open"] = df["close"]
    df["high"] = df["close"]
    df["low"] = df["close"]
    
    os.makedirs(data_dir, exist_ok=True)
    filename = os.path.join(data_dir, f"{symbol_id}_coingecko_daily.csv")
    df.to_csv(filename)
    print(f"Saved {filename}")
    return df

if __name__ == "__main__":
    coins = ["bitcoin", "gala"]
    for coin in coins:
        fetch_coingecko_ohlcv(coin)
        time.sleep(2)  # Respect API limits
