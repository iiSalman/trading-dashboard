from flask import Flask, render_template, jsonify
import yfinance as yf
import numpy as np
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

app = Flask(__name__)

WATCHLIST = ['NVDA', 'RKLB', 'CRWV', 'MSTR', 'TSLA', 'AAPL', 'AMZN', 'QQQ', 'IREN', 'PLTR',
             'AMD', 'HIMS', 'CRM', 'ARM', 'OKLO', 'META']
RISK_FREE_RATE = 0.053

_cache = {}
_cache_lock = threading.Lock()
CACHE_TTL = 300


def norm_pdf(x):
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)


def calculate_gamma(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = (np.log(S / K) + (RISK_FREE_RATE + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return float(norm_pdf(d1) / (S * sigma * np.sqrt(T)))
    except Exception:
        return 0.0


def analyze_ticker(ticker):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period='5d')
        if hist.empty:
            return {'ticker': ticker, 'error': 'No price data', 'score': 0, 'direction': 'NEUTRAL'}

        spot = float(hist['Close'].iloc[-1])
        prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else spot
        pct_change = round(((spot - prev) / prev) * 100, 2)
        raw_vol = hist['Volume'].iloc[-1]
        volume = int(raw_vol) if raw_vol == raw_vol and raw_vol is not None else 0

        expirations = t.options
        if not expirations:
            return {
                'ticker': ticker, 'price': round(spot, 2), 'change': pct_change,
                'volume': volume, 'score': 0, 'direction': 'NEUTRAL',
                'expiry': None, 'dte': None, 'gex': 0, 'gex_regime': 'N/A',
                'pcr_vol': 1.0, 'pcr_oi': 1.0, 'call_premium': 0, 'put_premium': 0,
                'unusual': False, 'error': 'No options listed',
            }

        today = date.today()
        nearest_expiry = None
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
            if exp_date >= today:
                nearest_expiry = exp
                break

        if not nearest_expiry:
            return {'ticker': ticker, 'error': 'No future expiry', 'score': 0, 'direction': 'NEUTRAL'}

        exp_date = datetime.strptime(nearest_expiry, '%Y-%m-%d').date()
        dte = (exp_date - today).days
        T = max(dte / 365.0, 1 / 365.0)

        chain = t.option_chain(nearest_expiry)
        calls = chain.calls.copy()
        puts = chain.puts.copy()

        # GEX calculation: net dealer gamma exposure
        def safe_int(val):
            try:
                v = float(val)
                return int(v) if v == v else 0
            except (TypeError, ValueError):
                return 0

        def row_gex(row):
            K = float(row['strike'])
            iv = float(row.get('impliedVolatility') or 0.3)
            oi = safe_int(row.get('openInterest', 0))
            g = calculate_gamma(spot, K, T, iv)
            return g * oi * 100 * spot

        call_gex = sum(row_gex(r) for _, r in calls.iterrows())
        put_gex  = sum(row_gex(r) for _, r in puts.iterrows())
        net_gex  = call_gex - put_gex

        # Volume & OI
        call_vol = float(calls['volume'].fillna(0).sum())
        put_vol  = float(puts['volume'].fillna(0).sum())
        call_oi  = float(calls['openInterest'].fillna(0).sum())
        put_oi   = float(puts['openInterest'].fillna(0).sum())

        pcr_vol = round(put_vol / call_vol, 2)  if call_vol > 0 else 1.0
        pcr_oi  = round(put_oi  / call_oi,  2)  if call_oi  > 0 else 1.0

        # Dollar premium
        call_premium = float(((calls['lastPrice'] * calls['volume'].fillna(0)) * 100).sum())
        put_premium  = float(((puts['lastPrice']  * puts['volume'].fillna(0))  * 100).sum())

        # Unusual activity: today vol-to-OI spike
        total_vol = call_vol + put_vol
        total_oi  = call_oi + put_oi
        unusual = (total_oi > 0) and ((total_vol / total_oi) > 0.4)

        # ---- Scoring ----
        score = 0
        direction = 'NEUTRAL'

        # PCR signal
        if pcr_vol <= 0.6:
            score += 2; direction = 'BULLISH'
        elif pcr_vol <= 0.85:
            score += 1; direction = 'BULLISH'
        elif pcr_vol >= 1.6:
            score += 2; direction = 'BEARISH'
        elif pcr_vol >= 1.2:
            score += 1; direction = 'BEARISH'

        # Premium imbalance
        if call_premium > 0 and put_premium > 0:
            ratio = call_premium / put_premium
            if ratio >= 2.0:
                score += 1
                if direction != 'BEARISH': direction = 'BULLISH'
            elif ratio <= 0.5:
                score += 1
                if direction != 'BULLISH': direction = 'BEARISH'

        # Unusual activity bonus
        if unusual:
            score += 1

        # GEX alignment bonus
        if net_gex > 0 and direction == 'BULLISH':
            score += 1
        elif net_gex < 0 and direction == 'BEARISH':
            score += 1

        # Momentum from price action
        if pct_change > 1.5 and direction == 'BULLISH':
            score += 1
        elif pct_change < -1.5 and direction == 'BEARISH':
            score += 1

        score = min(score, 5)

        return {
            'ticker': ticker,
            'price': round(spot, 2),
            'change': pct_change,
            'volume': volume,
            'expiry': nearest_expiry,
            'dte': dte,
            'gex': round(net_gex / 1e6, 2),
            'gex_regime': 'POSITIVE' if net_gex > 0 else 'NEGATIVE',
            'pcr_vol': pcr_vol,
            'pcr_oi': pcr_oi,
            'call_premium': round(call_premium / 1e6, 2),
            'put_premium': round(put_premium / 1e6, 2),
            'unusual': unusual,
            'score': score,
            'direction': direction,
            'call_vol': int(call_vol),
            'put_vol': int(put_vol),
        }

    except Exception as e:
        return {'ticker': ticker, 'error': str(e), 'score': 0, 'direction': 'NEUTRAL'}


def get_all_data():
    all_tickers = WATCHLIST + ['SPY']

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(analyze_ticker, t): t for t in all_tickers}
        raw = {}
        for future in as_completed(futures):
            ticker = futures[future]
            raw[ticker] = future.result()

    spy = raw.get('SPY', {})
    watchlist_results = [raw[t] for t in WATCHLIST if t in raw]
    watchlist_results.sort(key=lambda x: x.get('score', 0), reverse=True)

    # Derive market bias from SPY
    spy_pcr  = spy.get('pcr_vol', 1.0)
    spy_gex  = spy.get('gex_regime', 'POSITIVE')
    spy_chg  = spy.get('change', 0)

    if spy_gex == 'NEGATIVE' and spy_pcr < 0.85:
        market_bias = 'TREND_UP'
        bias_desc = 'Negative GEX + bullish flow — follow momentum longs'
        bias_color = 'green'
    elif spy_gex == 'NEGATIVE' and spy_pcr > 1.2:
        market_bias = 'TREND_DOWN'
        bias_desc = 'Negative GEX + bearish flow — follow momentum shorts'
        bias_color = 'red'
    elif spy_gex == 'POSITIVE' and abs(spy_chg) < 0.5:
        market_bias = 'RANGE'
        bias_desc = 'Positive GEX + low drift — fade moves, trade the range'
        bias_color = 'yellow'
    elif spy_pcr < 0.85:
        market_bias = 'BULLISH'
        bias_desc = 'Call flow dominant — mild upside bias'
        bias_color = 'green'
    elif spy_pcr > 1.2:
        market_bias = 'BEARISH'
        bias_desc = 'Put flow dominant — mild downside bias'
        bias_color = 'red'
    else:
        market_bias = 'NEUTRAL'
        bias_desc = 'Mixed signals — wait for clarity, reduce size'
        bias_color = 'gray'

    top_picks = [r for r in watchlist_results if r.get('score', 0) >= 3][:3]

    now = datetime.now()
    market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    is_weekend   = now.weekday() >= 5
    market_status = 'OPEN' if (not is_weekend and market_open <= now <= market_close) else 'CLOSED'

    return {
        'timestamp': now.isoformat(),
        'date': date.today().strftime('%A, %B %d %Y'),
        'market_status': market_status,
        'market_bias': market_bias,
        'bias_desc': bias_desc,
        'bias_color': bias_color,
        'spy': spy,
        'tickers': watchlist_results,
        'top_picks': top_picks,
    }


def cached_get_all_data():
    with _cache_lock:
        if 'main' in _cache:
            data, ts = _cache['main']
            if time.time() - ts < CACHE_TTL:
                return data
    result = get_all_data()
    with _cache_lock:
        _cache['main'] = (result, time.time())
    return result


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/data')
def api_data():
    return jsonify(cached_get_all_data())


@app.route('/api/refresh')
def api_refresh():
    with _cache_lock:
        _cache.clear()
    return jsonify(cached_get_all_data())


if __name__ == '__main__':
    print("\n  Trading Dashboard running at http://localhost:5050\n")
    app.run(debug=False, port=5050)
