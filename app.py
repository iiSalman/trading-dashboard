import os
import time
import threading
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from flask import Flask, render_template, jsonify

app = Flask(__name__)

WATCHLIST = ['NVDA', 'RKLB', 'CRWV', 'MSTR', 'TSLA', 'AAPL', 'AMZN', 'QQQ', 'IREN', 'PLTR',
             'AMD', 'HIMS', 'CRM', 'ARM', 'OKLO', 'META']
RISK_FREE_RATE = 0.053

TRADIER_TOKEN = os.environ.get('TRADIER_TOKEN', '').strip()
TRADIER_BASE = os.environ.get('TRADIER_BASE', 'https://sandbox.tradier.com/v1').rstrip('/')

_cache = {}
_cache_lock = threading.Lock()
CACHE_TTL = 900


def _tradier_get(path, params=None):
    if not TRADIER_TOKEN:
        raise RuntimeError("TRADIER_TOKEN env var not set")
    headers = {'Authorization': f'Bearer {TRADIER_TOKEN}', 'Accept': 'application/json'}
    r = requests.get(f"{TRADIER_BASE}{path}", headers=headers, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def _retry(fn, retries=3, base_delay=1.0):
    last = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            last = e
            msg = str(e).lower()
            if any(k in msg for k in ("rate", "429", "timeout", "connection")):
                time.sleep(base_delay * (2 ** i))
                continue
            raise
    raise last


def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def get_quotes(symbols):
    data = _retry(lambda: _tradier_get('/markets/quotes', {'symbols': ','.join(symbols)}))
    quotes = _as_list((data.get('quotes') or {}).get('quote'))
    return {q['symbol']: q for q in quotes if isinstance(q, dict) and q.get('symbol')}


def get_expirations(symbol):
    data = _retry(lambda: _tradier_get('/markets/options/expirations', {'symbol': symbol}))
    exps = _as_list((data.get('expirations') or {}).get('date'))
    return [e for e in exps if isinstance(e, str)]


def get_chain(symbol, expiration):
    data = _retry(lambda: _tradier_get(
        '/markets/options/chains',
        {'symbol': symbol, 'expiration': expiration, 'greeks': 'true'},
    ))
    return _as_list((data.get('options') or {}).get('option'))


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


def _f(x, default=0.0):
    try:
        v = float(x)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


def _i(x, default=0):
    try:
        v = float(x)
        return int(v) if v == v else default
    except (TypeError, ValueError):
        return default


def analyze_ticker(ticker, quote):
    try:
        if not quote:
            return {'ticker': ticker, 'error': 'No quote data', 'score': 0, 'direction': 'NEUTRAL'}

        spot = _f(quote.get('last') or quote.get('close'))
        prev = _f(quote.get('prevclose'), spot)
        pct_change = round(((spot - prev) / prev) * 100, 2) if prev else 0.0
        volume = _i(quote.get('volume'))

        if spot <= 0:
            return {'ticker': ticker, 'error': 'No price', 'score': 0, 'direction': 'NEUTRAL'}

        expirations = get_expirations(ticker)
        if not expirations:
            return {
                'ticker': ticker, 'price': round(spot, 2), 'change': pct_change,
                'volume': volume, 'score': 0, 'direction': 'NEUTRAL',
                'expiry': None, 'dte': None, 'gex': 0, 'gex_regime': 'N/A',
                'pcr_vol': 1.0, 'pcr_oi': 1.0, 'call_premium': 0, 'put_premium': 0,
                'unusual': False, 'error': 'No options listed',
            }

        today = date.today()
        nearest_expiry = next(
            (e for e in expirations if datetime.strptime(e, '%Y-%m-%d').date() >= today),
            None,
        )
        if not nearest_expiry:
            return {'ticker': ticker, 'error': 'No future expiry', 'score': 0, 'direction': 'NEUTRAL'}

        exp_date = datetime.strptime(nearest_expiry, '%Y-%m-%d').date()
        dte = (exp_date - today).days
        T = max(dte / 365.0, 1 / 365.0)

        chain = get_chain(ticker, nearest_expiry)
        calls = [o for o in chain if o.get('option_type') == 'call']
        puts = [o for o in chain if o.get('option_type') == 'put']

        def row_gex(opt):
            greeks = opt.get('greeks') or {}
            g = _f(greeks.get('gamma'))
            if g == 0.0:
                K = _f(opt.get('strike'))
                iv = _f(greeks.get('mid_iv') or greeks.get('smv_vol') or 0.3, 0.3)
                g = calculate_gamma(spot, K, T, iv)
            oi = _i(opt.get('open_interest'))
            return g * oi * 100 * spot

        call_gex = sum(row_gex(o) for o in calls)
        put_gex = sum(row_gex(o) for o in puts)
        net_gex = call_gex - put_gex

        call_vol = sum(_i(o.get('volume')) for o in calls)
        put_vol = sum(_i(o.get('volume')) for o in puts)
        call_oi = sum(_i(o.get('open_interest')) for o in calls)
        put_oi = sum(_i(o.get('open_interest')) for o in puts)

        pcr_vol = round(put_vol / call_vol, 2) if call_vol > 0 else 1.0
        pcr_oi = round(put_oi / call_oi, 2) if call_oi > 0 else 1.0

        call_premium = sum(_f(o.get('last')) * _i(o.get('volume')) * 100 for o in calls)
        put_premium = sum(_f(o.get('last')) * _i(o.get('volume')) * 100 for o in puts)

        total_vol = call_vol + put_vol
        total_oi = call_oi + put_oi
        unusual = (total_oi > 0) and ((total_vol / total_oi) > 0.4)

        score = 0
        direction = 'NEUTRAL'

        if pcr_vol <= 0.6:
            score += 2; direction = 'BULLISH'
        elif pcr_vol <= 0.85:
            score += 1; direction = 'BULLISH'
        elif pcr_vol >= 1.6:
            score += 2; direction = 'BEARISH'
        elif pcr_vol >= 1.2:
            score += 1; direction = 'BEARISH'

        if call_premium > 0 and put_premium > 0:
            ratio = call_premium / put_premium
            if ratio >= 2.0:
                score += 1
                if direction != 'BEARISH':
                    direction = 'BULLISH'
            elif ratio <= 0.5:
                score += 1
                if direction != 'BULLISH':
                    direction = 'BEARISH'

        if unusual:
            score += 1

        if net_gex > 0 and direction == 'BULLISH':
            score += 1
        elif net_gex < 0 and direction == 'BEARISH':
            score += 1

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
            'call_vol': call_vol,
            'put_vol': put_vol,
        }

    except Exception as e:
        return {'ticker': ticker, 'error': str(e), 'score': 0, 'direction': 'NEUTRAL'}


def get_all_data():
    all_tickers = WATCHLIST + ['SPY']

    try:
        quotes = get_quotes(all_tickers)
    except Exception as e:
        quotes = {}
        quote_error = str(e)
    else:
        quote_error = None

    raw = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(analyze_ticker, t, quotes.get(t)): t for t in all_tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            raw[ticker] = future.result()

    if quote_error:
        for t in raw:
            if not raw[t].get('error'):
                raw[t]['error'] = f"Quote fetch failed: {quote_error}"

    spy = raw.get('SPY', {})
    watchlist_results = [raw[t] for t in WATCHLIST if t in raw]
    watchlist_results.sort(key=lambda x: x.get('score', 0), reverse=True)

    spy_pcr = spy.get('pcr_vol', 1.0)
    spy_gex = spy.get('gex_regime', 'POSITIVE')
    spy_chg = spy.get('change', 0)

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
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    is_weekend = now.weekday() >= 5
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
