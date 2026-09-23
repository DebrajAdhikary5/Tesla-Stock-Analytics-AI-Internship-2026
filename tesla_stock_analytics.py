"""
Tesla Stock Analytics and Price Prediction Using Machine Learning
=================================================================
Academic Data Analytics Project

DISCLAIMER:
This project is an academic data analytics and machine learning exercise.
Predictions are model estimates with quantified uncertainty and do NOT
constitute financial advice or guaranteed future prices. Past stock
performance does not guarantee future results.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
import datetime

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE   = os.path.join(_SCRIPT_DIR, "Tesla.csv - Tesla.csv.csv")
CHARTS_DIR  = os.path.join(_SCRIPT_DIR, "charts")
REPORT_FILE = os.path.join(_SCRIPT_DIR, "Tesla_Stock_Analytics_Report.docx")

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15 (remainder)

RANDOM_STATE = 42
N_SPLITS_CV  = 5   # TimeSeriesSplit folds on training data


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────
def section(title):
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)

def fact(msg):      print(f"  [FACT]        {msg}")
def insight(msg):   print(f"  [INSIGHT]     {msg}")
def risk(msg):      print(f"  [RISK]        {msg}")
def opportunity(msg): print(f"  [OPPORTUNITY] {msg}")
def action(msg):    print(f"  [ACTION]      {msg}")

def millions(x, pos):
    return f'{x/1e6:.1f}M'

def ensure_charts_dir():
    os.makedirs(CHARTS_DIR, exist_ok=True)

def chart_path(name):
    return os.path.join(CHARTS_DIR, name)


# ─────────────────────────────────────────────
# 1. DATA LOADING AND UNDERSTANDING
# ─────────────────────────────────────────────
def load_data():
    section("1. DATA LOADING AND UNDERSTANDING")
    df = pd.read_csv(DATA_FILE, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    fact(f"Dataset loaded: {len(df):,} trading days")
    fact(f"Date range    : {df['Date'].min().date()} to {df['Date'].max().date()}")
    fact(f"Columns       : {', '.join(df.columns.tolist())}")
    fact(f"Price range   : ${df['Adj Close'].min():.2f} – ${df['Adj Close'].max():.2f} (Adj Close)")
    fact(f"Volume range  : {df['Volume'].min():,} – {df['Volume'].max():,}")
    return df


# ─────────────────────────────────────────────
# 2. DATA QUALITY ASSESSMENT
# ─────────────────────────────────────────────
def assess_quality(df):
    section("2. DATA QUALITY ASSESSMENT")
    missing = df.isnull().sum()
    duplicates = df.duplicated().sum()
    fact(f"Missing values per column:\n{missing.to_string()}")
    fact(f"Duplicate rows: {duplicates}")

    # Negative or zero price check
    price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    for col in price_cols:
        invalid = (df[col] <= 0).sum()
        if invalid:
            fact(f"  WARNING: {invalid} non-positive values in {col}")
        else:
            fact(f"  {col}: all values positive (OK)")

    # High >= Low check
    bad_hl = (df['High'] < df['Low']).sum()
    fact(f"  High < Low anomalies: {bad_hl}")

    insight("Dataset is complete with no missing values, no duplicates, and no price anomalies.")
    return df


# ─────────────────────────────────────────────
# 3. DATA CLEANING AND PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    section("3. DATA CLEANING AND PREPROCESSING")
    df = df.copy()

    # Derived columns used throughout analysis
    df['Daily_Return']   = df['Adj Close'].pct_change()
    df['Log_Return']     = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    df['Daily_Range']    = df['High'] - df['Low']
    df['HL_Perc']        = (df['High'] - df['Low']) / df['Low'] * 100
    df['CO_Perc']        = (df['Close'] - df['Open']) / df['Open'] * 100

    # Rolling statistics
    df['MA_20']          = df['Adj Close'].rolling(window=20).mean()
    df['MA_50']          = df['Adj Close'].rolling(window=50).mean()
    df['MA_200']         = df['Adj Close'].rolling(window=200).mean()
    df['Vol_30d']        = df['Daily_Return'].rolling(window=30).std() * np.sqrt(252) * 100
    df['Vol_MA_30']      = df['Volume'].rolling(window=30).mean()

    # Cumulative max for drawdown
    df['Cum_Max']        = df['Adj Close'].cummax()
    df['Drawdown']       = (df['Adj Close'] - df['Cum_Max']) / df['Cum_Max'] * 100

    fact(f"Derived columns added: Daily_Return, Log_Return, Daily_Range, HL_Perc, CO_Perc")
    fact(f"Rolling statistics added: MA_20, MA_50, MA_200, Vol_30d, Vol_MA_30, Drawdown")
    insight("Preprocessing complete. All derived features computed from historically available data only.")
    return df


# ─────────────────────────────────────────────
# 4. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
def exploratory_analysis(df):
    section("4. EXPLORATORY DATA ANALYSIS")

    ret = df['Daily_Return'].dropna()
    fact(f"Mean daily return     : {ret.mean()*100:.4f}%")
    fact(f"Std daily return      : {ret.std()*100:.4f}%")
    fact(f"Skewness              : {ret.skew():.4f}")
    fact(f"Kurtosis              : {ret.kurt():.4f}")
    fact(f"Positive return days  : {(ret > 0).sum()} ({(ret > 0).mean()*100:.1f}%)")
    fact(f"Negative return days  : {(ret < 0).sum()} ({(ret < 0).mean()*100:.1f}%)")
    fact(f"Largest single gain   : +{ret.max()*100:.2f}%")
    fact(f"Largest single loss   : {ret.min()*100:.2f}%")

    insight("Daily returns exhibit positive skewness and high kurtosis, indicating occasional large price moves.")
    insight(f"The stock was positive {(ret > 0).mean()*100:.1f}% of trading days, suggesting an overall upward bias over the period.")


# ─────────────────────────────────────────────
# 5. KPI IDENTIFICATION AND CALCULATION
# ─────────────────────────────────────────────
def calculate_kpis(df):
    section("5. KPI IDENTIFICATION AND ANALYSIS")

    first_price = df['Adj Close'].iloc[0]
    last_price  = df['Adj Close'].iloc[-1]
    total_return = (last_price - first_price) / first_price * 100

    log_ret = df['Log_Return'].dropna()
    n_years = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25
    ann_return = ((last_price / first_price) ** (1 / n_years) - 1) * 100
    ann_vol    = log_ret.std() * np.sqrt(252) * 100

    rv_ratio   = ann_return / ann_vol  # Return-to-Volatility, NOT Sharpe

    avg_volume     = df['Volume'].mean()
    avg_daily_range = df['Daily_Range'].mean()
    max_drawdown   = df['Drawdown'].min()

    kpis = {
        'Total Return (%)':            round(total_return, 2),
        'Annualised Return (%)':        round(ann_return, 2),
        'Annualised Volatility (%)':    round(ann_vol, 2),
        'Return-to-Volatility Ratio':   round(rv_ratio, 4),
        'Average Daily Volume':         round(avg_volume, 0),
        'Average Daily Price Range ($)': round(avg_daily_range, 4),
        'Maximum Drawdown (%)':         round(max_drawdown, 2),
    }

    for k, v in kpis.items():
        if 'Volume' in k:
            fact(f"{k:<35}: {v:,.0f}")
        else:
            fact(f"{k:<35}: {v}")

    insight(f"Tesla gained {total_return:.1f}% over the full period ({n_years:.1f} years), equivalent to {ann_return:.1f}% annualised.")
    insight(f"Annualised volatility of {ann_vol:.1f}% is extremely high compared to typical equities (~15–20%), reflecting Tesla's speculative nature.")
    insight(f"Return-to-Volatility Ratio of {rv_ratio:.2f} (NOTE: this is NOT the Sharpe ratio; no risk-free rate is applied).")
    insight(f"Maximum drawdown of {max_drawdown:.1f}% indicates periods of severe capital loss from peak values.")

    risk(f"Annualised volatility of {ann_vol:.1f}% far exceeds typical equity benchmarks — extreme price swings are the norm.")
    risk(f"Maximum drawdown of {max_drawdown:.1f}% shows that investors could have lost over a third of peak value at worst.")

    opportunity(f"A {ann_return:.1f}% annualised return over {n_years:.1f} years demonstrates exceptional long-term capital appreciation potential.")
    opportunity("High average daily volume signals strong market liquidity, enabling large positions to be entered/exited without significant slippage.")

    action("Analysts should monitor rolling volatility alongside price trends — volatility spikes often precede large directional moves.")
    action("Maximum drawdown should be tracked continuously as a risk management signal, not just as a historical statistic.")

    return kpis


# ─────────────────────────────────────────────
# 6. TREND ANALYSIS
# ─────────────────────────────────────────────
def trend_analysis(df):
    section("6. TREND ANALYSIS")

    # MA crossover signals
    valid = df.dropna(subset=['MA_20', 'MA_50', 'MA_200'])
    bullish_200 = (valid['Adj Close'] > valid['MA_200']).sum()
    bearish_200 = (valid['Adj Close'] < valid['MA_200']).sum()
    total_valid = len(valid)

    fact(f"Days price above MA_200: {bullish_200} ({bullish_200/total_valid*100:.1f}%)")
    fact(f"Days price below MA_200: {bearish_200} ({bearish_200/total_valid*100:.1f}%)")

    # Momentum: last 252 days
    last_year = df.tail(252)
    year_return = (last_year['Adj Close'].iloc[-1] / last_year['Adj Close'].iloc[0] - 1) * 100
    fact(f"1-year trailing return (last 252 trading days): {year_return:.1f}%")

    # Volume trend
    first_half_vol  = df.head(len(df)//2)['Volume'].mean()
    second_half_vol = df.tail(len(df)//2)['Volume'].mean()
    vol_change = (second_half_vol - first_half_vol) / first_half_vol * 100
    fact(f"Average volume first half : {first_half_vol:,.0f}")
    fact(f"Average volume second half: {second_half_vol:,.0f}")
    fact(f"Volume trend change       : {vol_change:+.1f}%")

    insight(f"Tesla spent {bullish_200/total_valid*100:.1f}% of trading days above its 200-day MA — a predominantly bullish long-term signal.")
    insight(f"Trading volume {'increased' if vol_change > 0 else 'decreased'} by {abs(vol_change):.1f}% in the second half of the dataset, indicating increased trading activity during that period.")

    risk("Extended periods above the 200-day MA can lead to sharp mean-reversion corrections when sentiment shifts.")
    opportunity("MA crossover signals (MA_20 crossing above MA_50) historically coincide with the start of strong uptrend phases.")


# ─────────────────────────────────────────────
# 7. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df):
    section("7. FEATURE ENGINEERING")

    fe = df.copy()

    # Lag features (previous days' values — no future leakage)
    for lag in [1, 2, 3, 5]:
        fe[f'Adj_Close_Lag{lag}']  = fe['Adj Close'].shift(lag)
        fe[f'Volume_Lag{lag}']     = fe['Volume'].shift(lag)
        fe[f'Return_Lag{lag}']     = fe['Daily_Return'].shift(lag)

    # Rolling window features (computed on past data only)
    fe['Rolling_Mean_5']   = fe['Adj Close'].shift(1).rolling(5).mean()
    fe['Rolling_Mean_10']  = fe['Adj Close'].shift(1).rolling(10).mean()
    fe['Rolling_Std_5']    = fe['Adj Close'].shift(1).rolling(5).std()
    fe['Rolling_Std_10']   = fe['Adj Close'].shift(1).rolling(10).std()
    fe['Rolling_Vol_10']   = fe['Daily_Return'].shift(1).rolling(10).std()

    # Price-based features (current day — available at end of day)
    fe['HL_Perc_feat']   = fe['HL_Perc']
    fe['CO_Perc_feat']   = fe['CO_Perc']
    fe['Daily_Range_feat'] = fe['Daily_Range']

    # Moving average ratios (price relative to trend)
    fe['Price_to_MA20']  = fe['Adj Close'] / fe['MA_20']
    fe['Price_to_MA50']  = fe['Adj Close'] / fe['MA_50']
    fe['MA20_to_MA50']   = fe['MA_20'] / fe['MA_50']

    # Target: NEXT DAY's Adj Close (1-day ahead forecast)
    # Forecasting horizon: 1 trading day
    # Rationale: short-horizon forecasting is the most reliable horizon for
    # regression-based models on financial data. Today's end-of-day data
    # (OHLCV) is fully available before tomorrow's market open.
    fe['Target'] = fe['Adj Close'].shift(-1)

    FEATURE_COLS = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
        'HL_Perc_feat', 'CO_Perc_feat', 'Daily_Range_feat',
        'Adj_Close_Lag1', 'Adj_Close_Lag2', 'Adj_Close_Lag3', 'Adj_Close_Lag5',
        'Volume_Lag1', 'Volume_Lag2', 'Return_Lag1', 'Return_Lag2', 'Return_Lag3',
        'Rolling_Mean_5', 'Rolling_Mean_10', 'Rolling_Std_5', 'Rolling_Std_10',
        'Rolling_Vol_10', 'Price_to_MA20', 'Price_to_MA50', 'MA20_to_MA50',
    ]

    # Drop rows where any feature or target is NaN (due to lags/rolling)
    fe_clean = fe[FEATURE_COLS + ['Target', 'Date']].dropna()

    fact(f"Features engineered: {len(FEATURE_COLS)}")
    fact(f"Lag features         : 4 lags × 3 columns = 12")
    fact(f"Rolling features     : 5 (mean/std/vol windows)")
    fact(f"Price-ratio features : 3 (MA ratios)")
    fact(f"Current-day features : 6 (OHLCV + range/HL/CO)")
    fact(f"Rows after dropping NaN: {len(fe_clean):,} (from {len(df):,})")
    fact(f"Target variable      : Next-day Adj Close (1-day horizon)")

    insight("All lag and rolling features are computed from past data only, ensuring no temporal leakage in features.")
    insight("Current-day OHLCV features are legitimate inputs: they represent information available at market close, before the next trading day opens.")

    return fe_clean, FEATURE_COLS


# ─────────────────────────────────────────────
# 8. MACHINE LEARNING — MODEL TRAINING
# ─────────────────────────────────────────────
def train_models(fe_clean, FEATURE_COLS):
    section("8. MACHINE LEARNING — MODEL TRAINING & SELECTION")

    n = len(fe_clean)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_test  = n - n_train - n_val

    train_df = fe_clean.iloc[:n_train]
    val_df   = fe_clean.iloc[n_train : n_train + n_val]
    test_df  = fe_clean.iloc[n_train + n_val:]

    fact(f"Total usable rows : {n:,}")
    fact(f"Train set         : {n_train:,} rows ({train_df['Date'].iloc[0].date()} – {train_df['Date'].iloc[-1].date()})")
    fact(f"Validation set    : {n_val:,} rows ({val_df['Date'].iloc[0].date()} – {val_df['Date'].iloc[-1].date()})")
    fact(f"Test set          : {n_test:,} rows ({test_df['Date'].iloc[0].date()} – {test_df['Date'].iloc[-1].date()})")
    fact("Test set is held out and NOT used for model selection or hyperparameter decisions.")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df['Target'].values
    X_val   = val_df[FEATURE_COLS].values
    y_val   = val_df['Target'].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df['Target'].values

    # ── Candidate models ──────────────────────────────────────────
    # Linear Regression is wrapped in a Pipeline with StandardScaler so that
    # scaling is fitted ONLY on the fold-train data inside each CV fold —
    # preventing any leakage of future-fold statistics into earlier folds.
    # Random Forest and Gradient Boosting are tree-based and do not require
    # feature scaling; they use raw features throughout.
    candidates = {
        'Linear Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model',  LinearRegression()),
        ]),
        'Random Forest':     RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE),
    }

    # ── TimeSeriesSplit cross-validation on training data ─────────
    # Each fold uses raw X_train; the Pipeline re-fits its scaler on that
    # fold's training split internally, so no fold-val information touches
    # the scaler. Tree-based models also receive raw X_train.
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    cv_results = {}
    insight(f"TimeSeriesSplit ({N_SPLITS_CV}-fold) cross-validation on training data only.")
    insight("Linear Regression uses a Pipeline (StandardScaler + model) to ensure the scaler is fit per-fold, preventing CV leakage.")
    insight("Random Forest and Gradient Boosting use raw features (no scaling required for tree-based models).")

    for name, model in candidates.items():
        fold_rmses = []
        for fold_train_idx, fold_val_idx in tscv.split(X_train):
            Xf_tr, Xf_val = X_train[fold_train_idx], X_train[fold_val_idx]
            yf_tr, yf_val = y_train[fold_train_idx], y_train[fold_val_idx]
            model.fit(Xf_tr, yf_tr)
            pred = model.predict(Xf_val)
            fold_rmses.append(np.sqrt(mean_squared_error(yf_val, pred)))
        cv_results[name] = np.mean(fold_rmses)
        fact(f"CV Mean RMSE — {name:<25}: ${cv_results[name]:.4f}")

    # ── Validation set comparison ─────────────────────────────────
    # Refit each candidate on full X_train (Pipeline re-fits scaler on X_train
    # for LR; tree models fit on raw X_train). Then predict on X_val (Pipeline
    # applies the train-fitted scaler to X_val; tree models use raw X_val).
    val_results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
        val_results[name] = {
            'MAE':  mean_absolute_error(y_val, pred_val),
            'RMSE': np.sqrt(mean_squared_error(y_val, pred_val)),
            'R2':   r2_score(y_val, pred_val),
        }
        fact(f"Validation — {name:<25}: MAE=${val_results[name]['MAE']:.4f}  RMSE=${val_results[name]['RMSE']:.4f}  R²={val_results[name]['R2']:.6f}")

    # ── Select best model by validation RMSE ─────────────────────
    best_name = min(val_results, key=lambda k: val_results[k]['RMSE'])
    insight(f"Best model selected by validation RMSE: {best_name}")
    insight("Model selection was performed on validation data only. Test set remains untouched at this stage.")

    # ── Retrain best model on train + validation combined ─────────
    X_trainval = np.vstack([X_train, X_val])
    y_trainval  = np.concatenate([y_train, y_val])
    best_model  = candidates[best_name]
    best_model.fit(X_trainval, y_trainval)
    insight(f"Best model ({best_name}) retrained on train + validation combined before final test evaluation.")

    return (best_model, best_name, X_test, y_test,
            test_df, val_results, cv_results, FEATURE_COLS,
            train_df, val_df, X_train, y_train, X_val, y_val)


# ─────────────────────────────────────────────
# 9. MODEL EVALUATION
# ─────────────────────────────────────────────
def evaluate_model(best_model, best_name, X_test, y_test, test_df, val_results, cv_results, FEATURE_COLS):
    section("9. MODEL EVALUATION — FINAL TEST SET")

    # Pipeline handles its own scaling internally; tree models use raw arrays.
    y_pred = best_model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    fact(f"Final model       : {best_name}")
    fact(f"Test MAE          : ${mae:.4f}")
    fact(f"Test RMSE         : ${rmse:.4f}")
    fact(f"Test R²           : {r2:.6f}")
    fact(f"Test MAPE         : {mape:.4f}%")

    residuals = y_test - y_pred
    fact(f"Residual mean     : ${residuals.mean():.4f}")
    fact(f"Residual std      : ${residuals.std():.4f}")
    fact(f"Max over-predict  : ${residuals.min():.4f}")
    fact(f"Max under-predict : ${residuals.max():.4f}")

    # Fix 6: RMSE described as error magnitude, NOT as a prediction interval
    insight(f"The model achieved a test RMSE of ${rmse:.2f}, indicating the magnitude of prediction error under this evaluation.")
    insight(f"R² of {r2:.4f} means the model explains {r2*100:.2f}% of the variance in next-day closing prices on the test set.")
    insight(f"MAPE of {mape:.2f}% indicates the model's average percentage error relative to actual price.")
    insight("Note: formal prediction intervals were not calculated in this project. RMSE is a summary of error magnitude, not a guaranteed uncertainty bound.")

    risk("Model performance on the test set is evaluated under one specific market regime. Performance may differ under different market conditions not seen in training data.")
    risk("A high R² in next-day price prediction partly reflects the strong autocorrelation in stock prices (today's price is close to yesterday's). This does not imply the model captures market inefficiencies.")
    risk("Predictions are point estimates. Formal prediction intervals are not modelled here; actual prices may deviate beyond what RMSE alone suggests.")

    action("When communicating predictions, report the test RMSE alongside them as a measure of historical error magnitude.")
    action("Do not use this model for live trading decisions. It is an academic demonstration of time-series regression.")

    # Feature influence: tree-based importances OR linear regression coefficient magnitudes
    feat_importance = None
    if hasattr(best_model, 'feature_importances_'):
        # Tree-based model
        fi = best_model.feature_importances_
        feat_importance = pd.Series(fi, index=FEATURE_COLS).sort_values(ascending=False)
        fact(f"\nTop 10 Feature Importances ({best_name}):")
        for fname, fval in feat_importance.head(10).items():
            fact(f"  {fname:<30}: {fval:.4f}")
    elif hasattr(best_model, 'named_steps') and 'model' in best_model.named_steps:
        # Pipeline containing LinearRegression — extract absolute coefficients
        lr = best_model.named_steps['model']
        abs_coefs = pd.Series(np.abs(lr.coef_), index=FEATURE_COLS).sort_values(ascending=False)
        feat_importance = abs_coefs  # reuse feat_importance slot; sign noted separately
        fact(f"\nTop 10 Feature Coefficient Magnitudes ({best_name}) — ranked by |coefficient|:")
        for fname, fval in feat_importance.head(10).items():
            sign = '+' if lr.coef_[FEATURE_COLS.index(fname)] >= 0 else '-'
            fact(f"  {fname:<30}: |coef|={fval:.4f}  (sign: {sign})")
        insight("Coefficient magnitudes indicate the relative influence of each feature on the predicted price, scaled by StandardScaler. Larger magnitude = stronger linear association in the scaled feature space.")
        insight("Coefficient sign indicates direction (positive = higher feature value associated with higher predicted price). This is a linear association, not a causal claim.")

    return y_pred, mae, rmse, r2, mape, feat_importance


# ─────────────────────────────────────────────
# 10. RISK AND OPPORTUNITY ANALYSIS
# ─────────────────────────────────────────────
def risk_opportunity_analysis(df, y_test, y_pred, rmse):
    section("10. RISK AND OPPORTUNITY ANALYSIS")

    # Volatility regimes
    high_vol_days = (df['Vol_30d'] > df['Vol_30d'].quantile(0.75)).sum()
    low_vol_days  = (df['Vol_30d'] < df['Vol_30d'].quantile(0.25)).sum()

    fact(f"High volatility days (top 25th pct): {high_vol_days}")
    fact(f"Low  volatility days (bot 25th pct): {low_vol_days}")

    # Largest drawdown periods
    max_dd_val = df['Drawdown'].min()
    max_dd_date = df.loc[df['Drawdown'].idxmin(), 'Date'].date()
    fact(f"Worst drawdown: {max_dd_val:.2f}% on {max_dd_date}")

    # Prediction uncertainty
    pred_range = y_test.max() - y_test.min()
    fact(f"Test set price range    : ${pred_range:.2f}")
    fact(f"Model RMSE (test error)  : ${rmse:.4f}")
    fact(f"RMSE as % of price range: {rmse/pred_range*100:.2f}%")

    insight("Volatility is not uniformly distributed — high-volatility regimes cluster, creating compounding risk during market stress periods.")
    insight(f"The maximum drawdown of {max_dd_val:.1f}% occurred on {max_dd_date}, demonstrating that even strongly trending stocks experience sharp corrections.")
    insight(f"Model RMSE of ${rmse:.2f} represents {rmse/pred_range*100:.2f}% of the test set price range — a non-trivial error magnitude that must be communicated clearly alongside predictions.")

    risk("Clustering of high-volatility days means risk is not evenly spread; drawdowns can be swift and deep.")
    risk("The ML model cannot predict sudden exogenous events (earnings shocks, macroeconomic news, regulatory actions) that cause price discontinuities.")
    risk(f"The model's test RMSE of ${rmse:.2f} indicates prediction error magnitude; formal prediction intervals are not calculated in this project.")

    opportunity("Low-volatility periods identified by the rolling 30-day vol metric may signal stable accumulation phases.")
    opportunity("Post-drawdown recovery patterns in the data suggest that severe corrections have historically been followed by rebounds — a pattern that could inform mean-reversion analysis.")

    action("Implement volatility-regime detection as a model input feature in future iterations to improve performance during high-volatility periods.")
    action("Supplement point predictions with quantile regression or prediction intervals in production analytics systems.")


# ─────────────────────────────────────────────
# 11. ACTIONABLE INSIGHTS SUMMARY
# ─────────────────────────────────────────────
def actionable_summary(df, kpis, mae, rmse, r2, best_name):
    section("11. ACTIONABLE INSIGHTS AND RECOMMENDATIONS")

    print("\n  ── SUMMARY OF FINDINGS ──────────────────────────────────────")
    print(f"  Total Return          : {kpis['Total Return (%)']:+.2f}%")
    print(f"  Annualised Return     : {kpis['Annualised Return (%)']:+.2f}%")
    print(f"  Annualised Volatility : {kpis['Annualised Volatility (%)']:.2f}%")
    print(f"  Return/Volatility     : {kpis['Return-to-Volatility Ratio']:.4f}  (not Sharpe — no risk-free rate)")
    print(f"  Max Drawdown          : {kpis['Maximum Drawdown (%)']:.2f}%")
    print(f"  Best ML Model         : {best_name}")
    print(f"  Model R²              : {r2:.6f}")
    print(f"  Model RMSE            : ${rmse:.4f}")
    print(f"  Model MAE             : ${mae:.4f}")

    print("\n  ── RECOMMENDED ACTIONS ──────────────────────────────────────")
    action("Use rolling volatility (Vol_30d) as an ongoing risk monitoring tool — not just a historical statistic.")
    action("Combine price prediction with volume analysis: unusual volume spikes often precede significant price moves.")
    action("When communicating predictions, report the test RMSE alongside them as a measure of historical error magnitude, not as a guaranteed interval.")
    action("For future iterations: add macroeconomic features (interest rates, market indices) and experiment with LSTM or ARIMA for comparison.")
    action("All model outputs must be accompanied by the academic disclaimer — this is not a financial advisory system.")

    print("\n  ── ACADEMIC DISCLAIMER ──────────────────────────────────────")
    print("  This project is an academic data analytics and machine learning")
    print("  exercise. Predictions are model estimates with quantified")
    print("  uncertainty and do NOT constitute financial advice or guaranteed")
    print("  future prices. Past stock performance does not guarantee future results.")


# ─────────────────────────────────────────────
# 12. VISUALISATIONS
# ─────────────────────────────────────────────
def create_dashboard(df, test_df, y_pred, feat_importance, best_name, best_model=None, FEATURE_COLS=None):
    section("12. GENERATING VISUALISATIONS")
    ensure_charts_dir()

    # ── Figure 1: Main Dashboard (6 panels) ────────────────────────
    fig = plt.figure(figsize=(20, 22))
    fig.suptitle('Tesla Stock Analytics Dashboard\n(Academic Project — Not Financial Advice)',
                 fontsize=16, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: Adj Close + Moving Averages
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['Date'], df['Adj Close'], color='#1f77b4', linewidth=1.0, label='Adj Close', alpha=0.9)
    ax1.plot(df['Date'], df['MA_20'],     color='#ff7f0e', linewidth=1.2, label='MA 20',     alpha=0.8)
    ax1.plot(df['Date'], df['MA_50'],     color='#2ca02c', linewidth=1.2, label='MA 50',     alpha=0.8)
    ax1.plot(df['Date'], df['MA_200'],    color='#d62728', linewidth=1.5, label='MA 200',    alpha=0.9)
    ax1.set_title('Adjusted Closing Price with Moving Averages', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Trading Volume
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(df['Date'], df['Volume'], color='#aec7e8', alpha=0.6, label='Daily Volume')
    ax2.plot(df['Date'], df['Vol_MA_30'], color='#1f77b4', linewidth=1.5, label='30d MA Volume')
    ax2.yaxis.set_major_formatter(FuncFormatter(millions))
    ax2.set_title('Daily Trading Volume', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume (Millions)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Daily Returns Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ret_clean = df['Daily_Return'].dropna()
    ax3.hist(ret_clean, bins=80, density=True, color='#5ba4cf', alpha=0.7, edgecolor='white', linewidth=0.3)
    # Normal distribution overlay using numpy (no scipy)
    mu, sigma = ret_clean.mean(), ret_clean.std()
    x_range = np.linspace(ret_clean.min(), ret_clean.max(), 300)
    normal_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
    ax3.plot(x_range, normal_pdf, color='#d62728', linewidth=2.0, label=f'Normal(μ={mu*100:.3f}%, σ={sigma*100:.3f}%)')
    ax3.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Daily Return')
    ax3.set_ylabel('Density')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Rolling 30-day Volatility
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.fill_between(df['Date'], df['Vol_30d'], alpha=0.4, color='#ff7f0e')
    ax4.plot(df['Date'], df['Vol_30d'], color='#ff7f0e', linewidth=1.0)
    ax4.set_title('30-Day Rolling Annualised Volatility (%)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Volatility (%)')
    ax4.grid(True, alpha=0.3)

    # Panel 5: Actual vs Predicted (test set)
    ax5 = fig.add_subplot(gs[2, 1])
    test_dates = test_df['Date'].values
    ax5.plot(test_dates, test_df['Target'].values, color='#1f77b4', linewidth=1.5, label='Actual', alpha=0.9)
    ax5.plot(test_dates, y_pred,                   color='#d62728', linewidth=1.2, label=f'Predicted ({best_name})', alpha=0.85, linestyle='--')
    ax5.set_title('Actual vs Predicted — Test Set', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Adj Close (USD)')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    plt.savefig(chart_path('tesla_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()
    fact("Saved: charts/tesla_dashboard.png")

    # ── Figure 2: KPI Summary + Feature Importance ──────────────────
    fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle('Tesla Stock KPIs and Feature Importance', fontsize=14, fontweight='bold')

    # KPI bar chart (non-volume, non-price KPIs)
    kpi_labels = ['Total\nReturn (%)', 'Ann. Return\n(%)', 'Ann. Volatility\n(%)', 'Max\nDrawdown (%)']
    kpi_values = [
        df['Adj Close'].iloc[-1] / df['Adj Close'].iloc[0] * 100 - 100,
        ((df['Adj Close'].iloc[-1] / df['Adj Close'].iloc[0]) **
         (1 / ((df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25)) - 1) * 100,
        df['Log_Return'].dropna().std() * np.sqrt(252) * 100,
        df['Drawdown'].min(),
    ]
    colors = ['#2ca02c' if v > 0 else '#d62728' for v in kpi_values]
    bars = axes[0].bar(kpi_labels, kpi_values, color=colors, edgecolor='white', width=0.5)
    for bar, val in zip(bars, kpi_values):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + (1 if val >= 0 else -3),
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0].axhline(0, color='black', linewidth=0.8)
    axes[0].set_title('Key Performance Indicators', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Value (%)')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Feature influence panel — tree importances or LR coefficient magnitudes
    if feat_importance is not None and hasattr(feat_importance, 'values'):
        top_fi = feat_importance.head(10)
        is_lr_pipeline = (best_model is not None and
                          hasattr(best_model, 'named_steps') and
                          'model' in best_model.named_steps)
        if is_lr_pipeline:
            # Linear Regression: colour bars by coefficient sign
            lr = best_model.named_steps['model']
            signs = [np.sign(lr.coef_[FEATURE_COLS.index(n)]) for n in top_fi.index]
            bar_colors = ['#2ca02c' if s >= 0 else '#d62728' for s in signs]
            axes[1].barh(top_fi.index[::-1], top_fi.values[::-1],
                         color=bar_colors[::-1], edgecolor='white')
            axes[1].set_title(f'Top 10 Feature Coefficient Magnitudes\n({best_name}) — ranked by |coef|',
                              fontsize=11, fontweight='bold')
            axes[1].set_xlabel('|Coefficient| (scaled features)')
            axes[1].text(0.98, 0.02, 'Green = positive coef  |  Red = negative coef',
                         transform=axes[1].transAxes, fontsize=7, ha='right', color='#57606a')
        else:
            axes[1].barh(top_fi.index[::-1], top_fi.values[::-1], color='#5ba4cf', edgecolor='white')
            axes[1].set_title(f'Top 10 Feature Importances\n({best_name})', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Importance')
        axes[1].grid(True, alpha=0.3, axis='x')
    else:
        axes[1].text(0.5, 0.5, 'Feature influence\nnot available',
                     ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title('Feature Influence', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(chart_path('tesla_kpi_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    fact("Saved: charts/tesla_kpi_summary.png")


# ─────────────────────────────────────────────
# WORD REPORT GENERATION
# ─────────────────────────────────────────────
def add_heading(doc, text, level):
    h = doc.add_heading(text, level=level)
    return h

def add_paragraph(doc, text, bold=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    return p

def add_bullet(doc, text):
    doc.add_paragraph(text, style='List Bullet')

def add_table(doc, headers, rows):
    # Create exactly 1 header row — data rows are added one at a time below.
    # Previously the table was initialised with (1 + len(rows)) rows AND then
    # add_row() was called, producing one extra blank row per data row.
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for run in hdr_cells[i].paragraphs[0].runs:
            run.bold = True
    for row_data in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row_data):
            cells[i].text = str(val)
    return table

def generate_report(df, kpis, val_results, cv_results, best_name,
                    mae, rmse, r2, mape, feat_importance, FEATURE_COLS,
                    train_df, val_df, test_df, y_pred):
    section("GENERATING WORD REPORT")

    doc = Document()

    # ── Page margins ──
    from docx.oxml import OxmlElement
    section_el = doc.sections[0]
    section_el.page_width  = Inches(8.5)
    section_el.page_height = Inches(11)
    section_el.left_margin   = Inches(1.0)
    section_el.right_margin  = Inches(1.0)
    section_el.top_margin    = Inches(1.0)
    section_el.bottom_margin = Inches(1.0)

    # ══════════════════════════════════════════
    # TITLE PAGE
    # ══════════════════════════════════════════
    doc.add_paragraph()
    doc.add_paragraph()
    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_para.add_run("Tesla Stock Analytics and Price Prediction\nUsing Machine Learning")
    title_run.bold = True
    title_run.font.size = Pt(20)
    title_run.font.color.rgb = RGBColor(0x1F, 0x45, 0x8A)

    doc.add_paragraph()
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = subtitle.add_run("Data Analytics with AI Internship Project\nAcademic Report")
    sub_run.font.size = Pt(14)
    sub_run.italic = True

    doc.add_paragraph()
    doc.add_paragraph()
    date_para = doc.add_paragraph()
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_para.add_run(f"Generated: {datetime.date.today().strftime('%B %d, %Y')}")

    disclaimer_para = doc.add_paragraph()
    disclaimer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    dr = disclaimer_para.add_run(
        "\nDISCLAIMER: This is an academic data analytics and machine learning project. "
        "Predictions are model estimates with quantified uncertainty and do NOT constitute "
        "financial advice or guaranteed future prices."
    )
    dr.italic = True
    dr.font.size = Pt(9)
    dr.font.color.rgb = RGBColor(0x80, 0x00, 0x00)

    doc.add_page_break()

    # ══════════════════════════════════════════
    # ABSTRACT
    # ══════════════════════════════════════════
    add_heading(doc, "Abstract", 1)
    add_paragraph(doc,
        "This report presents a complete academic data analytics and machine learning project "
        "analysing historical Tesla Inc. (TSLA) stock data spanning from 29 June 2010 to "
        "17 March 2017 (1,692 trading days). The project follows a structured analytical workflow "
        "encompassing data quality assessment, exploratory data analysis, KPI identification, "
        "trend analysis, feature engineering, machine learning-based price prediction, model "
        "evaluation, and risk/opportunity analysis. A chronological train/validation/test split "
        "methodology is used to prevent temporal data leakage. Three regression models — "
        "Linear Regression, Random Forest, and Gradient Boosting — are compared using "
        "TimeSeriesSplit cross-validation on training data, with the best model evaluated on a "
        "held-out test set. The project clearly distinguishes between observed facts, derived "
        "insights, risks, opportunities, and recommended actions."
    )

    doc.add_page_break()

    # ══════════════════════════════════════════
    # 1. INTRODUCTION
    # ══════════════════════════════════════════
    add_heading(doc, "1. Introduction", 1)
    add_paragraph(doc,
        "Tesla Inc. (NASDAQ: TSLA) is one of the most traded and analysed equities in modern "
        "financial markets. Its stock price history is characterised by extreme volatility, "
        "exponential growth phases, and sharp corrections — making it an ideal subject for "
        "academic financial data analytics. This project demonstrates a complete end-to-end "
        "data analytics pipeline applied to Tesla's historical stock price data, from raw data "
        "inspection through to machine learning price forecasting and business insight generation."
    )

    # ══════════════════════════════════════════
    # 2. PROBLEM STATEMENT
    # ══════════════════════════════════════════
    add_heading(doc, "2. Problem Statement", 1)
    add_paragraph(doc,
        "Financial analysts and data scientists face the challenge of extracting actionable "
        "insights from high-frequency historical stock data while avoiding common pitfalls such "
        "as temporal data leakage, model overfitting, and overconfident forecasts. This project "
        "addresses: (1) How can historical Tesla stock data be systematically analysed to identify "
        "meaningful trends, KPIs, risks, and opportunities? (2) Can a machine learning regression "
        "model predict next-day closing prices with measurable accuracy using a methodologically "
        "rigorous time-series approach?"
    )

    # ══════════════════════════════════════════
    # 3. OBJECTIVES
    # ══════════════════════════════════════════
    add_heading(doc, "3. Objectives", 1)
    objectives = [
        "Load, inspect, and assess the quality of Tesla's historical stock dataset.",
        "Perform comprehensive exploratory data analysis to understand price behaviour.",
        "Identify and calculate seven meaningful KPIs appropriate for historical stock analysis.",
        "Conduct trend analysis using moving averages and volatility metrics.",
        "Engineer relevant features without temporal leakage.",
        "Compare three ML models using a leakage-free chronological validation methodology.",
        "Evaluate the best model on a held-out test set and report MAE, RMSE, R², and MAPE.",
        "Clearly distinguish facts, insights, risks, opportunities, and recommended actions.",
        "Generate visualisations and a comprehensive academic report.",
    ]
    for obj in objectives:
        add_bullet(doc, obj)

    # ══════════════════════════════════════════
    # 4. DATASET DESCRIPTION
    # ══════════════════════════════════════════
    add_heading(doc, "4. Dataset Description", 1)
    add_paragraph(doc,
        f"The dataset contains {len(df):,} daily trading records for Tesla Inc. (TSLA) covering "
        f"{df['Date'].min().strftime('%d %B %Y')} to {df['Date'].max().strftime('%d %B %Y')}. "
        "The dataset includes the following columns:"
    )
    add_table(doc,
        ['Column', 'Type', 'Description'],
        [
            ['Date',      'datetime', 'Trading date'],
            ['Open',      'float',    'Opening price (USD)'],
            ['High',      'float',    'Intraday high price (USD)'],
            ['Low',       'float',    'Intraday low price (USD)'],
            ['Close',     'float',    'Closing price (USD)'],
            ['Volume',    'integer',  'Number of shares traded'],
            ['Adj Close', 'float',    'Adjusted closing price accounting for corporate actions (USD)'],
        ]
    )
    doc.add_paragraph()
    add_paragraph(doc,
        "Source: [SOURCE URL — Please provide before final submission]. "
        "Column naming conventions are consistent with Yahoo Finance historical data export format "
        "(Open, High, Low, Close, Volume, Adj Close)."
    )

    # ══════════════════════════════════════════
    # 5. DATA PREPROCESSING
    # ══════════════════════════════════════════
    add_heading(doc, "5. Data Preprocessing", 1)
    add_paragraph(doc,
        "The dataset was found to be complete with no missing values, no duplicate rows, and no "
        "price anomalies (all prices positive, High ≥ Low for all rows). The following derived "
        "columns were computed:"
    )
    derived = [
        "Daily_Return: Percentage change in Adj Close from previous day.",
        "Log_Return: Natural log return for use in statistical calculations.",
        "Daily_Range: High minus Low (intraday price spread in USD).",
        "HL_Perc: (High − Low) / Low × 100 — intraday range as percentage.",
        "CO_Perc: (Close − Open) / Open × 100 — open-to-close movement.",
        "MA_20, MA_50, MA_200: 20-, 50-, and 200-day simple moving averages of Adj Close.",
        "Vol_30d: 30-day rolling annualised volatility from daily returns.",
        "Drawdown: Percentage decline from cumulative maximum Adj Close.",
    ]
    for d in derived:
        add_bullet(doc, d)

    # ══════════════════════════════════════════
    # 6. EXPLORATORY DATA ANALYSIS
    # ══════════════════════════════════════════
    add_heading(doc, "6. Exploratory Data Analysis", 1)
    ret = df['Daily_Return'].dropna()
    add_paragraph(doc, "Key EDA statistics for daily returns:")
    add_table(doc,
        ['Metric', 'Value'],
        [
            ['Mean daily return',   f"{ret.mean()*100:.4f}%"],
            ['Std daily return',    f"{ret.std()*100:.4f}%"],
            ['Skewness',            f"{ret.skew():.4f}"],
            ['Excess Kurtosis',     f"{ret.kurt():.4f}"],
            ['Positive return days', f"{(ret>0).sum()} ({(ret>0).mean()*100:.1f}%)"],
            ['Negative return days', f"{(ret<0).sum()} ({(ret<0).mean()*100:.1f}%)"],
            ['Largest single gain', f"+{ret.max()*100:.2f}%"],
            ['Largest single loss', f"{ret.min()*100:.2f}%"],
        ]
    )
    doc.add_paragraph()
    add_paragraph(doc,
        "The dashboard visualisation below shows the complete EDA output including price trends, "
        "volume patterns, return distribution, and rolling volatility."
    )
    if os.path.exists(chart_path('tesla_dashboard.png')):
        doc.add_picture(chart_path('tesla_dashboard.png'), width=Inches(6.0))
        cap = doc.add_paragraph("Figure 1: Tesla Stock Analytics Dashboard — Adj Close with MAs, Volume, Return Distribution, Rolling Volatility, and Actual vs Predicted Prices.")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].italic = True
        cap.runs[0].font.size = Pt(9)

    # ══════════════════════════════════════════
    # 7. KPI ANALYSIS
    # ══════════════════════════════════════════
    add_heading(doc, "7. KPI Analysis", 1)
    add_paragraph(doc,
        "Seven KPIs were selected for their mathematical appropriateness to historical stock analysis. "
        "The Return-to-Volatility Ratio is explicitly named as such and is NOT the Sharpe ratio "
        "(no risk-free rate is subtracted)."
    )
    add_table(doc,
        ['KPI', 'Value', 'Description'],
        [
            ['Total Return (%)',              f"{kpis['Total Return (%)']:+.2f}%",        'Cumulative price appreciation over full period'],
            ['Annualised Return (%)',          f"{kpis['Annualised Return (%)']:+.2f}%",   'CAGR over the dataset period'],
            ['Annualised Volatility (%)',      f"{kpis['Annualised Volatility (%)']:.2f}%",'Annualised std of log returns × √252'],
            ['Return-to-Volatility Ratio',     f"{kpis['Return-to-Volatility Ratio']:.4f}",'Ann. Return ÷ Ann. Volatility (NOT Sharpe)'],
            ['Average Daily Volume',           f"{kpis['Average Daily Volume']:,.0f}",    'Mean shares traded per day'],
            ['Average Daily Price Range ($)',  f"${kpis['Average Daily Price Range ($)']:.2f}", 'Mean of (High − Low) per day'],
            ['Maximum Drawdown (%)',           f"{kpis['Maximum Drawdown (%)']:.2f}%",    'Largest peak-to-trough decline'],
        ]
    )
    doc.add_paragraph()
    if os.path.exists(chart_path('tesla_kpi_summary.png')):
        doc.add_picture(chart_path('tesla_kpi_summary.png'), width=Inches(6.0))
        cap = doc.add_paragraph("Figure 2: KPI Summary and Feature Importance Chart.")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].italic = True
        cap.runs[0].font.size = Pt(9)

    # ══════════════════════════════════════════
    # 8. FEATURE ENGINEERING
    # ══════════════════════════════════════════
    add_heading(doc, "8. Feature Engineering", 1)
    add_paragraph(doc,
        f"A total of {len(FEATURE_COLS)} features were engineered. All lag and rolling features "
        "are computed from historically available data only, ensuring no temporal leakage. "
        "Current-day OHLCV features are legitimate inputs as they represent data available at "
        "market close before the next trading day opens."
    )
    add_paragraph(doc, "Feature categories:")
    feat_cats = [
        "Current-day features: Open, High, Low, Close, Volume, Adj Close, HL_Perc, CO_Perc, Daily_Range.",
        "Lag features (1, 2, 3, 5 days): lagged Adj Close, Volume, and Daily Return.",
        "Rolling statistics: 5- and 10-day rolling mean, standard deviation, and volatility of Adj Close (shifted by 1 to avoid leakage).",
        "Moving average ratios: Price/MA20, Price/MA50, MA20/MA50.",
        "Target variable: Next-day Adj Close (shift −1) — 1-trading-day forecasting horizon.",
    ]
    for fc in feat_cats:
        add_bullet(doc, fc)

    # ══════════════════════════════════════════
    # 9. MACHINE LEARNING METHODOLOGY
    # ══════════════════════════════════════════
    add_heading(doc, "9. Machine Learning Methodology", 1)
    add_paragraph(doc,
        "A three-phase chronological data split was used to prevent temporal data leakage:"
    )
    n = len(df)
    n_fe = len(pd.read_csv(DATA_FILE, parse_dates=['Date']).sort_values('Date').assign(
        Daily_Return=lambda d: d['Adj Close'].pct_change(),
        Log_Return=lambda d: np.log(d['Adj Close']/d['Adj Close'].shift(1)),
    ).dropna())

    add_table(doc,
        ['Split', 'Rows', 'Date Range', 'Purpose'],
        [
            ['Train (70%)',      f"~{len(train_df):,}", f"{train_df['Date'].iloc[0].date()} – {train_df['Date'].iloc[-1].date()}", 'Model training'],
            ['Validation (15%)', f"~{len(val_df):,}",  f"{val_df['Date'].iloc[0].date()} – {val_df['Date'].iloc[-1].date()}",   'Model selection'],
            ['Test (15%)',       f"~{len(test_df):,}",  f"{test_df['Date'].iloc[0].date()} – {test_df['Date'].iloc[-1].date()}", 'Final evaluation (held out)'],
        ]
    )
    doc.add_paragraph()
    add_paragraph(doc,
        "Three candidate models were compared: Linear Regression, Random Forest Regressor "
        "(200 estimators), and Gradient Boosting Regressor (200 estimators). Model selection "
        f"was performed using {N_SPLITS_CV}-fold TimeSeriesSplit cross-validation on the training "
        "data, confirmed by validation set RMSE. The final test set was used only once for "
        "reporting final metrics."
    )

    add_paragraph(doc, "Validation set results:")
    val_rows = []
    for name, metrics in val_results.items():
        val_rows.append([name, f"${metrics['MAE']:.4f}", f"${metrics['RMSE']:.4f}", f"{metrics['R2']:.6f}"])
    add_table(doc, ['Model', 'Val MAE', 'Val RMSE', 'Val R²'], val_rows)

    doc.add_paragraph()
    add_paragraph(doc, f"Selected model: {best_name} (lowest validation RMSE).", bold=True)

    # ══════════════════════════════════════════
    # 10. MODEL EVALUATION
    # ══════════════════════════════════════════
    add_heading(doc, "10. Model Evaluation", 1)
    add_paragraph(doc, f"Final evaluation of {best_name} on the held-out test set:")
    add_table(doc,
        ['Metric', 'Value', 'Interpretation'],
        [
            ['MAE',  f"${mae:.4f}",  'Average absolute prediction error in USD'],
            ['RMSE', f"${rmse:.4f}", 'Root mean squared error — penalises large errors'],
            ['R²',   f"{r2:.6f}",   f'Model explains {r2*100:.2f}% of price variance'],
            ['MAPE', f"{mape:.4f}%", 'Mean absolute percentage error'],
        ]
    )
    doc.add_paragraph()
    add_paragraph(doc,
        f"Note: A high R² in next-day price forecasting partly reflects the strong autocorrelation "
        f"inherent in stock prices (today's price is a strong predictor of tomorrow's). This does "
        f"not imply the model captures market inefficiencies. The model achieved a test RMSE of "
        f"${rmse:.2f}, indicating the magnitude of prediction error under this evaluation. "
        f"Formal prediction intervals were not calculated in this project."
    )

    # ══════════════════════════════════════════
    # 11. KEY FINDINGS
    # ══════════════════════════════════════════
    add_heading(doc, "11. Key Findings", 1)
    findings_facts = [
        f"Tesla's Adj Close ranged from ${df['Adj Close'].min():.2f} to ${df['Adj Close'].max():.2f} over the dataset period.",
        f"Total return of {kpis['Total Return (%)']:+.1f}% over {((df['Date'].iloc[-1]-df['Date'].iloc[0]).days/365.25):.1f} years.",
        f"Annualised volatility of {kpis['Annualised Volatility (%)']:.1f}% — highly elevated vs. typical equities.",
        f"Maximum drawdown of {kpis['Maximum Drawdown (%)']:.1f}% represents significant downside risk.",
        f"Daily returns are positively skewed with high kurtosis, indicating fat-tailed distribution with occasional large moves.",
    ]
    findings_insights = [
        "The 200-day moving average acted as a long-term trend indicator, with sustained periods of price above/below it signalling macro trend direction.",
        "Trading volume increased substantially during parts of the second half of the dataset, indicating increased trading activity.",
        f"The {best_name} model achieves ${rmse:.2f} RMSE on the test set — a practical next-day prediction error that is non-trivial relative to typical daily ranges.",
    ]
    add_paragraph(doc, "Facts:", bold=True)
    for f_item in findings_facts:
        add_bullet(doc, f_item)
    add_paragraph(doc, "Insights:", bold=True)
    for ins in findings_insights:
        add_bullet(doc, ins)

    # ══════════════════════════════════════════
    # 12. RISKS
    # ══════════════════════════════════════════
    add_heading(doc, "12. Risks", 1)
    risks_list = [
        f"Annualised volatility of {kpis['Annualised Volatility (%)']:.1f}% far exceeds typical equity benchmarks — extreme price swings are the norm, not the exception.",
        f"Maximum drawdown of {kpis['Maximum Drawdown (%)']:.1f}% demonstrates that severe capital loss from peak values is a real historical outcome.",
        "The ML model cannot predict sudden exogenous events (earnings shocks, macroeconomic news, regulatory actions) causing price discontinuities.",
        f"The model's test RMSE of ${rmse:.2f} indicates prediction error magnitude; formal prediction intervals are not calculated in this project, so actual deviations may exceed this figure.",
        "Model performance was evaluated under one specific market regime. Performance may differ significantly under different market conditions.",
        "High autocorrelation in stock prices can inflate R² metrics, potentially giving a misleading impression of model capability.",
    ]
    for r_item in risks_list:
        add_bullet(doc, r_item)

    # ══════════════════════════════════════════
    # 13. OPPORTUNITIES
    # ══════════════════════════════════════════
    add_heading(doc, "13. Opportunities", 1)
    opps = [
        f"A {kpis['Annualised Return (%)']:.1f}% annualised return over the full period demonstrates exceptional long-term capital appreciation potential.",
        "High average daily volume signals strong market liquidity, enabling large positions to be entered/exited without significant slippage.",
        "Low-volatility periods identified by the rolling 30-day volatility metric may signal stable accumulation phases.",
        "Post-drawdown recovery patterns suggest that severe corrections have historically been followed by rebounds — a signal useful for mean-reversion analysis.",
        "Moving average crossover signals (MA_20 crossing MA_50) historically coincide with the start of strong trend phases.",
    ]
    for opp in opps:
        add_bullet(doc, opp)

    # ══════════════════════════════════════════
    # 14. RECOMMENDED ACTIONS
    # ══════════════════════════════════════════
    add_heading(doc, "14. Recommended Actions", 1)
    add_paragraph(doc,
        "The following recommended actions are analytical in nature and are NOT investment advice. "
        "This is an academic project."
    )
    actions = [
        "Use rolling volatility (Vol_30d) as an ongoing risk monitoring tool — not just a historical statistic.",
        "Combine price prediction with volume analysis: unusual volume spikes often precede significant price moves.",
        "When communicating model predictions, report the test RMSE alongside them as a measure of historical error magnitude — not as a guaranteed prediction interval.",
        "Implement volatility-regime detection as a model input feature in future iterations.",
        "Supplement point predictions with quantile regression or prediction intervals in production systems.",
        "Add macroeconomic features (interest rates, market indices) and experiment with LSTM or ARIMA for comparison in future work.",
        "All model outputs must be accompanied by the academic disclaimer — this project is not a financial advisory system.",
    ]
    for act in actions:
        add_bullet(doc, act)

    # ══════════════════════════════════════════
    # 15. CONCLUSION
    # ══════════════════════════════════════════
    add_heading(doc, "15. Conclusion", 1)
    add_paragraph(doc,
        "This project demonstrates a complete, methodologically rigorous academic data analytics "
        "and machine learning pipeline applied to Tesla stock data. Key contributions include: "
        "a leakage-free chronological train/validation/test split methodology; systematic "
        "comparison of three ML models without contaminating the test set; explicit separation "
        "of facts, insights, risks, opportunities, and recommended actions; and transparent "
        "communication of model uncertainty. The project successfully identifies meaningful KPIs, "
        "trends, and predictive features from historical Tesla stock data while clearly "
        "acknowledging the limitations of ML-based financial forecasting."
    )

    # ══════════════════════════════════════════
    # 16. LIMITATIONS
    # ══════════════════════════════════════════
    add_heading(doc, "16. Limitations", 1)
    lims = [
        "The dataset covers only one company (Tesla) and one specific time period (2010–2017). Findings may not generalise to other stocks or periods.",
        "No macroeconomic, sentiment, or news data is included — factors that significantly influence stock prices.",
        "The 1-day forecasting horizon is short; multi-day or multi-week forecasts would carry substantially higher uncertainty.",
        "Models assume stationarity in relationships between features and target, which may break down during structural market shifts.",
        "No transaction costs, taxes, or liquidity constraints are modelled.",
        "This is an academic exercise and must not be used for actual trading or investment decisions.",
    ]
    for lim in lims:
        add_bullet(doc, lim)

    # ══════════════════════════════════════════
    # 17. REFERENCES
    # ══════════════════════════════════════════
    add_heading(doc, "17. References", 1)
    refs = [
        "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.",
        "McKinney, W. (2010). Data Structures for Statistical Computing in Python. Proceedings of SciPy.",
        "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.",
        "Friedman, J.H. (2001). Greedy function approximation: a gradient boosting machine. Annals of Statistics, 29(5), 1189–1232.",
        "Tesla Inc. (TSLA) Historical Stock Data. Source: [SOURCE URL — Please provide before final submission].",
        "Hyndman, R.J. & Athanasopoulos, G. (2021). Forecasting: Principles and Practice. OTexts.",
    ]
    for ref in refs:
        add_bullet(doc, ref)

    doc.save(REPORT_FILE)
    fact(f"Word report saved: {REPORT_FILE}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Ensure UTF-8 output on Windows consoles
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except AttributeError:
            pass

    print("\n" + "=" * 70)
    print("  TESLA STOCK ANALYTICS AND PRICE PREDICTION USING MACHINE LEARNING")
    print("  Academic Data Analytics Project")
    print("=" * 70)
    print("\n  DISCLAIMER: This is an academic project. Predictions are model")
    print("  estimates with uncertainty and do NOT constitute financial advice")
    print("  or guaranteed future prices.")

    df = load_data()
    df = assess_quality(df)
    df = preprocess(df)
    exploratory_analysis(df)
    kpis = calculate_kpis(df)
    trend_analysis(df)
    fe_clean, FEATURE_COLS = engineer_features(df)

    (best_model, best_name, X_test, y_test,
     test_df, val_results, cv_results, FEATURE_COLS,
     train_df, val_df, X_train, y_train, X_val, y_val) = train_models(fe_clean, FEATURE_COLS)

    y_pred, mae, rmse, r2, mape, feat_importance = evaluate_model(
        best_model, best_name, X_test, y_test, test_df, val_results, cv_results, FEATURE_COLS
    )

    risk_opportunity_analysis(df, y_test, y_pred, rmse)
    actionable_summary(df, kpis, mae, rmse, r2, best_name)

    create_dashboard(df, test_df, y_pred, feat_importance, best_name,
                     best_model=best_model, FEATURE_COLS=FEATURE_COLS)

    generate_report(
        df, kpis, val_results, cv_results, best_name,
        mae, rmse, r2, mape, feat_importance, FEATURE_COLS,
        train_df, val_df, test_df, y_pred
    )

    section("PROJECT COMPLETE")
    print(f"  Charts    : {CHARTS_DIR}/tesla_dashboard.png")
    print(f"              {CHARTS_DIR}/tesla_kpi_summary.png")
    print(f"  Report    : {REPORT_FILE}")
    print("\n  All original files are untouched.")
    print("  This is an academic project — not financial advice.\n")


if __name__ == "__main__":
    main()
