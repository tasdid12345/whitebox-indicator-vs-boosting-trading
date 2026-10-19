# Gradient Boosting vs Logistic Regression on Technical Indicators for Daily Stock Trading

This experiment is motivated by "Investment Portfolio Optimization Using Technical Indicators and White-Box Models" (2024), which argues that technical indicators combined with white-box models can support portfolio decisions. We test whether a more flexible but still tunable model on the same indicators gives a better out-of-sample risk-adjusted return than a white-box logistic regression.

**Question:** Does a class-balanced, CV-tuned gradient-boosting model on technical indicators give a higher out-of-sample Sharpe ratio than a white-box logistic-regression model on the same indicators for daily stock trading?

**Dataset:** Daily OHLCV price histories for AAPL, MSFT, JPM, XOM and JNJ, downloaded as CSV from stooq.com. This is not the paper's own dataset (`same_as_paper: false`). Features are SMA/EMA ratios, RSI, MACD, Bollinger position, rolling volatility and multi-window momentum, all computed from past prices. The target is next-day up/down. Each ticker is split chronologically: the first 70% trains and the last 30% tests.

**Baseline vs improved:** The baseline is a standardised logistic regression with default settings. The improved model is a class-balanced LightGBM with a small time-series-CV hyperparameter search on the training part only, followed by isotonic calibration. Both models use the same data, features, test set and evaluation function. The main metric is the annualised Sharpe ratio of a long-or-flat strategy; directional accuracy, F1 and maximum drawdown are secondary.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Results: results are written to `results.json` (charts go to `figures/`).

## Reference

This project is motivated by the following paper:

> F Caro-Reyna, DG Arcos-Bravo (2024). *Investment Portfolio Optimization Using Technical Indicators and White-Box Models*. 2024 IEEE 6th. https://ieeexplore.ieee.org/abstract/document/10885376/

Paper author: Prof. Claudia Nayeli Sánchez Gómez, Universidad Panamericana (cnsanchez@up.edu.mx)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
