"""Baseline (logistic regression) vs improved (class-balanced, CV-tuned LightGBM + isotonic) on technical indicators."""
import io, json, os, sys, time
import numpy as np, pandas as pd, requests
import matplotlib
matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
SEEDS = [42, 43, 44]  # both models are run with these seeds (LR is deterministic, LightGBM is randomized)
TICKERS = ["aapl", "msft", "jpm", "xom", "jnj"]
URL = "https://stooq.com/q/d/l/?s={}.us&i=d"
YF_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{}?range=max&interval=1d&events=div%7Csplit"
TRAIN_FRAC = 0.70
SOURCE_USED = {}  # ticker -> data source actually used (recorded in results.json)
np.random.seed(SEED)


def _stooq(ticker):
    """Primary source: stooq daily CSV."""
    r = requests.get(URL.format(ticker), timeout=60)
    if r.status_code == 429 or r.status_code >= 500:
        raise requests.HTTPError(f"HTTP {r.status_code}")
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if "Close" not in df.columns or df.empty:
        raise ValueError("no data returned")
    return df


def _yahoo(ticker):
    """Fallback source: Yahoo Finance public chart endpoint (daily OHLCV, adjusted close)."""
    r = requests.get(YF_URL.format(ticker.upper()), headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    if r.status_code == 429 or r.status_code >= 500:
        raise requests.HTTPError(f"HTTP {r.status_code}")
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    adj_block = res["indicators"].get("adjclose")
    adj = adj_block[0]["adjclose"] if adj_block else q["close"]
    df = pd.DataFrame({
        "Date": pd.to_datetime(res["timestamp"], unit="s").normalize(),
        "Open": q["open"], "High": q["high"], "Low": q["low"],
        "Close": adj, "Volume": q["volume"],
    }).dropna(subset=["Close"])
    if df.empty:
        raise ValueError("no data returned")
    return df


def download(ticker):
    """Try stooq, then Yahoo chart API; each with retry/backoff on 429/5xx and network errors."""
    errors = []
    for name, fetch in (("stooq", _stooq), ("yahoo", _yahoo)):
        for attempt in range(3):
            try:
                df = fetch(ticker)
                SOURCE_USED[ticker] = name
                return df
            except Exception as e:  # network, HTTP, parse, or empty-data errors
                errors.append(f"{name}: {e}")
                wait = 2 ** attempt
                print(f"  {ticker}: {name} attempt {attempt + 1} failed ({e}); retrying in {wait}s")
                time.sleep(wait)
                if "404" in str(e):  # not found: no point retrying this source
                    break
    sys.exit(f"ERROR: could not download {ticker} ({'; '.join(errors)})")


def features(raw):
    """Technical indicators computed only from past prices (rolling/ewm windows look backwards)."""
    c = raw["Close"].astype(float)
    r = c.pct_change()
    f = pd.DataFrame(index=raw.index)
    for w in (5, 20, 50):
        f[f"sma{w}"] = c.rolling(w).mean() / c - 1
    f["ema_ratio"] = c.ewm(span=12).mean() / c.ewm(span=26).mean() - 1
    d = c.diff()
    up = d.clip(lower=0).rolling(14).mean()
    dn = (-d.clip(upper=0)).rolling(14).mean()
    f["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    f["macd"] = macd / c
    f["macd_hist"] = (macd - macd.ewm(span=9).mean()) / c
    f["boll"] = (c - c.rolling(20).mean()) / (2 * c.rolling(20).std())
    f["vol10"] = r.rolling(10).std()
    f["vol20"] = r.rolling(20).std()
    for m in (1, 5, 10, 20, 60):
        f[f"mom{m}"] = c.pct_change(m)
    f["next_ret"] = r.shift(-1)                      # next-day return (used for P&L only)
    f["target"] = (f["next_ret"] > 0).astype(int)    # next-day up/down label
    return f


def load():
    """Per ticker: first 70% of rows -> train, last 30% -> test (chronological, no shuffling)."""
    tr_parts, te_parts = [], []
    for t in TICKERS:
        print(f"Loading {t.upper()}")
        raw = download(t).assign(Date=lambda d: pd.to_datetime(d["Date"])).sort_values("Date").reset_index(drop=True)
        f = features(raw).assign(date=raw["Date"].values, ticker=t).dropna()
        cut = int(len(f) * TRAIN_FRAC)
        tr_parts.append(f.iloc[: cut - 1])  # drop last train row: its label uses a test-period price
        te_parts.append(f.iloc[cut:])
    return pd.concat(tr_parts), pd.concat(te_parts)


tr, te = load()
FEAT = [c for c in tr.columns if c not in ("target", "next_ret", "date", "ticker")]
Xtr, ytr = tr[FEAT].values, tr["target"].values
Xte, yte = te[FEAT].values, te["target"].values
print(f"rows train={len(tr)} test={len(te)} features={len(FEAT)}")
print("sources used:", SOURCE_USED)


def run_baseline(seed):
    """White-box baseline: standardised logistic regression, default settings."""
    m = make_pipeline(StandardScaler(), LogisticRegression(random_state=seed))
    return m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def cv_logloss(params):
    """Mean validation log-loss over time-series folds of the TRAINING part only."""
    losses = []
    for a, b in TimeSeriesSplit(3).split(Xtr):
        m = LGBMClassifier(**params, class_weight="balanced", random_state=SEED, verbose=-1).fit(Xtr[a], ytr[a])
        losses.append(log_loss(ytr[b], m.predict_proba(Xtr[b])[:, 1], labels=[0, 1]))
    return np.mean(losses)


grid = list(ParameterGrid(dict(num_leaves=[15, 31], learning_rate=[0.03, 0.1],
                               n_estimators=[200, 400], min_child_samples=[20, 50])))
print(f"CV search over {len(grid)} configs (training folds only)...")
best = min(grid, key=cv_logloss)
print("best params:", best)


def run_improved(seed):
    """Class-balanced LightGBM with CV-chosen params, isotonic-calibrated on time-ordered folds."""
    base = LGBMClassifier(**best, class_weight="balanced", random_state=seed, verbose=-1)
    cal = CalibratedClassifierCV(base, method="isotonic", cv=TimeSeriesSplit(3))
    return cal.fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def evaluate(pred_up):
    """Long-or-flat strategy (hold next day iff P(up)>0.5), equal-weight daily portfolio across tickers."""
    pos = pred_up.astype(int)
    port = pd.Series(pos * te["next_ret"].values, index=te["date"].values).groupby(level=0).mean()
    eq = (1 + port).cumprod()
    return dict(sharpe=port.mean() / port.std(ddof=1) * np.sqrt(252),
                accuracy=accuracy_score(yte, pos),
                f1=f1_score(yte, pos, zero_division=0),
                max_dd=(eq / eq.cummax() - 1).min())


print("Evaluating baseline (logistic regression)...")
base_res = [evaluate(run_baseline(s) > 0.5) for s in SEEDS]
print("Evaluating improved (balanced LightGBM + isotonic)...")
imp_res = [evaluate(run_improved(s) > 0.5) for s in SEEDS]
bh = pd.Series(te["next_ret"].values, index=te["date"].values).groupby(level=0).mean()
bh_sharpe = bh.mean() / bh.std(ddof=1) * np.sqrt(252)


def agg(res, key):
    v = np.array([r[key] for r in res], float)
    return float(v.mean()), float(v.std())


b_s, b_sd = agg(base_res, "sharpe")
i_s, i_sd = agg(imp_res, "sharpe")
if not np.isfinite([b_s, i_s]).all():
    sys.exit("ERROR: non-finite Sharpe ratio; cannot produce meaningful metrics")

metrics = {
    "baseline_score": b_s, "improved_score": i_s, "improvement": i_s - b_s,
    "baseline_score_std": b_sd, "improved_score_std": i_sd,
    "baseline_accuracy": agg(base_res, "accuracy")[0], "improved_accuracy": agg(imp_res, "accuracy")[0],
    "baseline_f1": agg(base_res, "f1")[0], "improved_f1": agg(imp_res, "f1")[0],
    "baseline_max_drawdown": agg(base_res, "max_dd")[0], "improved_max_drawdown": agg(imp_res, "max_dd")[0],
    "buy_hold_sharpe": float(bh_sharpe), "n_seeds": len(SEEDS),
}
n_stooq = sum(v == "stooq" for v in SOURCE_USED.values())
results = {
    "question": "Does a class-balanced, CV-tuned gradient-boosting model on technical indicators give a higher out-of-sample Sharpe ratio than a white-box logistic-regression model on the same indicators for daily stock trading?",
    "method": "Daily OHLCV for AAPL, MSFT, JPM, XOM, JNJ (stooq, with Yahoo Finance chart API as fallback when stooq is unavailable); technical indicators from past prices; target = next-day up/down; per-ticker chronological 70/30 train/test split; long-or-flat strategy Sharpe on the test period.",
    "dataset": {
        "name": "Daily OHLCV: AAPL, MSFT, JPM, XOM, JNJ (US listings)",
        "source_url": ", ".join(URL.format(t) for t in TICKERS) + " (fallback: " + YF_URL.format("<TICKER>") + ")",
        "source_used": dict(SOURCE_USED),
        "n_rows": int(len(tr) + len(te)), "n_train": int(len(tr)), "n_test": int(len(te)),
        "same_as_paper": False, "baseline_is_papers_method": False,
    },
    "metric_name": "annualised Sharpe ratio of long-or-flat strategy (test period)",
    "higher_is_better": True,
    "baseline": "Standardised logistic regression (default settings) on technical indicators; white-box.",
    "improved": "Class-balanced LightGBM, small time-series-CV hyperparameter search on training data only, isotonic calibration.",
    "metrics": metrics,
    "notes": (f"Single chronological 70/30 split per ticker; one test period, so results are sensitive to the market regime. "
              f"Price data: {n_stooq}/{len(TICKERS)} tickers from stooq, the rest from the Yahoo Finance chart API fallback "
              "(stooq returned 404 for some tickers in CI); Yahoo prices use adjusted close. "
              "Hyperparameters chosen by time-series CV on training data only; test set used once. "
              "Logistic regression is deterministic (std 0 across seeds); LightGBM varies with seed. "
              "No transaction costs or slippage. Data (~25k rows) is far below the 200k subsampling threshold, so none applied. "
              "The paper's specific white-box algorithm is not specified, so the baseline is a generic logistic regression."),
}
with open("results.json", "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2)

os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Baseline (LogReg)", "Improved (LGBM)"], [b_s, i_s], yerr=[b_sd, i_sd], capsize=6, color=["grey", "steelblue"])
ax.set_ylabel("Annualised Sharpe (test)")
ax.set_title("Out-of-sample Sharpe, mean ± sd over seeds")
fig.tight_layout()
fig.savefig("figures/sharpe_comparison.png", dpi=120)
print(f"baseline Sharpe={b_s:.3f}  improved Sharpe={i_s:.3f}  -> results.json written")
