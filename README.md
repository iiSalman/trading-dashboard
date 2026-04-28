# Trading Dashboard

A small Flask dashboard that scores a watchlist of tickers using live data from Yahoo Finance.

## Quick start

```bash
pip install -r requirements.txt
export TRADIER_TOKEN=your_sandbox_token
python app.py
```

Then open http://localhost:5050.

Get a free sandbox token at https://developer.tradier.com.

## Stack

- Flask
- numpy
- Tradier API (quotes + options chain with greeks)

## Configuration

- `TRADIER_TOKEN` (required): Tradier API token
- `TRADIER_BASE` (optional): defaults to `https://sandbox.tradier.com/v1`
- `WATCHLIST` and `RISK_FREE_RATE`: edit constants at the top of `app.py`
