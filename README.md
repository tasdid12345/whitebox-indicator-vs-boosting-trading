# Stock direction: logistic regression vs class-balanced LightGBM

Question: does a class-balanced, CV-tuned gradient-boosting model on technical indicators give a higher out-of-sample Sharpe ratio than a standardised logistic regression on the same indicators?

## Data
- Daily OHLCV for AAPL, MSFT, JPM, XOM, JNJ.
- Primary source: stooq (`https://stooq.com/q/d/l/?s={ticker}.us&i=d`).
- Fallback: Yahoo Finance public chart API, used automatically when stooq fails (it returned HTTP 404 for AAPL in CI). Yahoo prices use adjusted close.
- The source actually used for each ticker is recorded in `results.json` under `dataset.source_used`.

## Method
- Features: technical indicators computed only from past prices (SMAs, EMA ratio, RSI, MACD, Bollinger position, volatility, momentum).
- Target: next-day up/down.
- Split: per ticker, first 70% of rows for training and last 30% for testing, in chronological order.
- Strategy: long-or-flat (hold next day iff P(up) > 0.5), equal-weight across tickers; metric is the annualised Sharpe ratio on the test period.
- Baseline: standardised logistic regression with default settings.
- Improved: class-balanced LightGBM, hyperparameters chosen by time-series CV on the training data only, isotonic calibration.
- Seeds 42, 43, 44; mean and standard deviation are reported.

## Run
```
pip install -r requirements.txt
python analysis.py
```
Outputs: `results.json` and `figures/sharpe_comparison.png`.

## Caveats
- One chronological split per ticker, so results depend on the test-period market regime.
- No transaction costs or slippage.
- The baseline is a generic logistic regression, since the source paper's white-box algorithm is not specified.

## Reference

This project is motivated by the following paper:

> F Caro-Reyna, DG Arcos-Bravo (2024). *Investment Portfolio Optimization Using Technical Indicators and White-Box Models*. 2024 IEEE 6th. https://ieeexplore.ieee.org/abstract/document/10885376/

Paper author: Prof. Claudia Nayeli Sánchez Gómez, Universidad Panamericana (cnsanchez@up.edu.mx)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
