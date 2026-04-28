# Trading Dashboard

A small Flask dashboard that scores a watchlist of tickers using live data from Yahoo Finance.

## Quick start

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000.

On macOS you can also double-click `run.command`.

## Stack

- Flask
- yfinance
- numpy

## Configuration

Edit the `WATCHLIST` and `RISK_FREE_RATE` constants at the top of `app.py`.
