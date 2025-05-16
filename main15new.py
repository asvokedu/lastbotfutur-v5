import os
import joblib
import logging
import traceback
import pandas as pd
from datetime import datetime, timedelta
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import time
from binance.client import Client
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from dotenv import load_dotenv
from scipy.signal import argrelextrema
from ta.trend import CCIIndicator
import pyodbc
from io import BytesIO
import subprocess
import threading
import gzip
import tempfile
import xgboost as xgb

# Load environment variables
load_dotenv()
model_cache = {}

# Constants
MODEL_DIR = "models"
CACHE_DIR = "cache"

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = Client(api_key, api_secret)

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot_log.txt", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def get_sql_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=gotbai;"
        "UID=sa;PWD=LEtoy_89"
    )

def read_symbols_from_file(filepath="listsyombol.txt"):
    with open(filepath, "r") as f:
        return [line.strip() for line in f if line.strip()]

def fetch_binance_klines(symbol, interval='15m', limit=500):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume",
                                       "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype({
        "open": float, "high": float, "low": float, "close": float, "volume": float
    })
    return df

def fetch_binance_last_price(symbol):
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except Exception as e:
        logging.warning(f"Gagal ambil harga terakhir {symbol}: {e}")
        return None

def wait_until_next_candle(interval='15m'):
    interval_minutes = int(interval.replace('m', ''))
    now = datetime.utcnow()
    minute = (now.minute // interval_minutes) * interval_minutes
    current_candle_time = now.replace(minute=minute, second=0, microsecond=0)
    next_candle_time = current_candle_time + timedelta(minutes=interval_minutes)
    wait_seconds = (next_candle_time - now).total_seconds()

    logging.info(f"[Clock] Menunggu hingga candle {next_candle_time.strftime('%H:%M')} UTC dimulai...")
    while wait_seconds > 0:
        m, s = divmod(int(wait_seconds), 60)
        print(f"\r[Clock] Menuju candle baru ({interval}) dalam {m:02d}:{s:02d} (MM:SS)", end="", flush=True)
        time.sleep(1)
        wait_seconds -= 1
    print(f"\r[Clock] Candle baru dimulai.{ ' ' * 30 }")

def wait_for_all_new_candles(symbols, interval='15m'):
    logging.info("Menunggu semua candle baru terbentuk...")
    while True:
        all_new = True
        now = datetime.utcnow()
        rounded_minute = (now.minute // 15) * 15
        current_time = now.replace(minute=rounded_minute, second=0, microsecond=0)

        for symbol in symbols:
            try:
                df = fetch_binance_klines(symbol, interval, 2)
                latest_time = df.iloc[-1]['timestamp']
                if latest_time < current_time:
                    logging.info(f"{symbol} belum ada candle baru. Last: {latest_time}, Now: {current_time}")
                    all_new = False
                else:
                    logging.info(f"{symbol} candle baru siap: {latest_time}")
            except Exception as e:
                logging.warning(f"Gagal cek candle {symbol}: {e}")
                all_new = False
        if all_new:
            logging.info("Semua candle sudah update.")
            break
        time.sleep(5)


def calculate_technical_indicators(df):
    df["rsi"] = RSIIndicator(close=df["close"], window=4).rsi()
    bb = BollingerBands(close=df["close"], window=10, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    cci = CCIIndicator(high=df['high'], low=df['low'], close=df['close'])
    df['cci'] = cci.cci()
    df['hammer'] = ((df['close'] > df['open']) &
                    ((df['low'] < df['open'] - (df['high'] - df['low']) * 0.5)) &
                    ((df['high'] - df['close']) < 0.2 * (df['high'] - df['low']))).astype(int)
    df['doji'] = (abs(df['close'] - df['open']) <= 0.1 * (df['high'] - df['low'])).astype(int)
    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["signal_line"] = macd.macd_signal()
    df["ema_20"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_100"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_slope"] = df["ema_20"].diff()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df['volatility'] = df['close'].rolling(window=12).std()
    for i in range(1, 4):
        df[f"close_shift_{i}"] = df["close"].shift(i)
        df[f"volume_shift_{i}"] = df["volume"].shift(i)
        df[f"rsi_shift_{i}"] = df["rsi"].shift(i)
        df[f"macd_shift_{i}"] = df["macd"].shift(i)
    if len(df) > 20:
        lows, highs = df["low"].values, df["high"].values
        swing_lows = argrelextrema(lows, np.less_equal, order=3)[0]
        swing_highs = argrelextrema(highs, np.greater_equal, order=3)[0]
        support_levels = [lows[idx] for idx in swing_lows]
        resistance_levels = [highs[idx] for idx in swing_highs]
        df["support"] = np.nan
        df["resistance"] = np.nan
        support_value = support_levels[-1] if support_levels else df["low"].tail(20).min()
        resistance_value = resistance_levels[-1] if resistance_levels else df["high"].tail(20).max()
        df.at[df.index[-1], "support"] = support_value
        df.at[df.index[-1], "resistance"] = resistance_value
    df["trend"] = df.apply(lambda row: 'UPTREND' if row["close"] > row["ema_200"] else 'DOWNTREND', axis=1)
    df["trend_encoded"] = df["trend"].map({"DOWNTREND": 0, "UPTREND": 1})
    df["delta_rsi"] = df["rsi"].diff()

    reward_multiplier = 1.5
    risk_multiplier = 1.0
    n_future = 1

    df['reward_thresh'] = df['volatility'] * reward_multiplier
    df['risk_thresh'] = df['volatility'] * risk_multiplier
    df['future_return'] = df['close'].shift(-n_future) / df['close'] - 1

    # Isi NaN dengan forward fill dan backward fill, lalu isi sisa dengan 0
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)

    return df


def preprocess_data(df):
    df = calculate_technical_indicators(df)
    df = df.dropna().reset_index(drop=True)
    return df

def load_model_from_sql(symbol):
    try:
        conn = get_sql_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT xgb_model, other_data 
            FROM model_storage 
            WHERE symbol = ? AND interval = ?
        """, (symbol, "15m"))
        row = cursor.fetchone()
        conn.close()

        if not row or not row[0] or not row[1]:
            raise ValueError(f"Model untuk simbol {symbol} tidak ditemukan atau data tidak lengkap.")

        xgb_model_binary, other_binary = row

        # Simpan sementara model_binary ke file agar bisa diload oleh Booster
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(xgb_model_binary)
            model_path = tmp_file.name

        booster = xgb.Booster()
        booster.load_model(model_path)
        os.remove(model_path)

        # Load additional metadata
        buffer = BytesIO(other_binary)
        with gzip.GzipFile(fileobj=buffer, mode='rb') as gz:
            other_data = joblib.load(gz)

        model = xgb.XGBClassifier()
        model._Booster = booster
        model._le = None
        model.n_classes_ = other_data.get("n_classes_", 2)
        scaler = other_data.get("scaler")  # Ambil scaler jika ada

        return {
            "model": model,
            "features": other_data.get("features", []),
            "label_encoder": other_data.get("label_encoder"),
            "trend_encoder": other_data.get("trend_encoder"),
            "scaler": scaler
        }

    except Exception as e:
        logging.error(f"Gagal load model dari database untuk {symbol}: {e}\\n{traceback.format_exc()}")
        return None


def analyze_symbol(symbol):
    try:
        logging.info(f"[🔍] Analisis {symbol}...")

        if symbol in model_cache:
            model, features, scaler = model_cache[symbol]
            label_encoder = None
        else:
            model_data = load_model_from_sql(symbol)
            if model_data is None:
                return
            model = model_data["model"]
            features = model_data["features"]
            scaler = model_data.get("scaler")
            label_encoder = model_data.get("label_encoder")
            model_cache[symbol] = (model, features, scaler)

        df = fetch_binance_klines(symbol, '15m', 500)
        if len(df) < 50:
            logging.warning(f"Data kurang dari 50 row untuk {symbol}")
            return

        df = calculate_technical_indicators(df)
        if df.empty:
            logging.warning(f"Tidak ada data valid setelah indikator untuk {symbol}")
            return

        latest = df.iloc[[-2]] if len(df) >= 2 else df.iloc[[-1]]

        missing = [f for f in features if f not in latest.columns]
        if missing:
            logging.error(f"Fitur hilang di {symbol}: {missing}")
            logging.error(f"Contoh fitur yang tersedia: {list(latest.columns)}")
            return

        X = latest[features]
        if scaler is not None:
            X = pd.DataFrame(scaler.transform(X), columns=features, index=X.index)
        
        proba = model.predict_proba(X)[0]
        pred = np.argmax(proba)

        if label_encoder:
            pred_label = label_encoder.inverse_transform([pred])[0]
        else:
            label_map = {0: "BUY", 1: "SELL", 2: "WAIT"}
            pred_label = label_map.get(pred, "UNKNOWN")

        confidence = round(proba[pred] * 100, 2)
        price = float(latest['close'].values[0])
        last_price = fetch_binance_last_price(symbol)

        reason, signal_score = generate_reason(latest)
        tgl = (latest['timestamp'] + pd.Timedelta(minutes=15)).dt.strftime('%Y-%m-%d').values[0]
        jam = (latest['timestamp'] + pd.Timedelta(minutes=15)).dt.strftime('%H:%M').values[0]
        volume = float(latest['volume'].values[0])

        conn = get_sql_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predict_log (tgl, jam, symbol, label, interval, current_price, confidence, volume, reason, score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (tgl, jam, symbol, pred_label, "15m", price, confidence, volume, reason, signal_score))
        conn.commit()
        cursor.close()
        conn.close()

        log_msg = f"✅ Prediksi {symbol}: {pred_label} | Price: {price:.4f} | Confidence: {confidence:.2f}%"
        if last_price:
            log_msg += f" | Last Price: {last_price:.4f}"
        log_msg += f" | Reason: {reason}"
        logging.info(log_msg)

    except Exception as e:
        logging.error(f"Error analisis {symbol}: {e}\n{traceback.format_exc()}")

def generate_reason(latest):
    score = 0
    reasons = []
    if latest['rsi'].values[0] > 70:
        reasons.append("RSI overbought")
        score -= 1
    elif latest['rsi'].values[0] < 30:
        reasons.append("RSI oversold")
        score += 1
    if latest['delta_rsi'].values[0] > 0:
        reasons.append("RSI naik")
        score += 1
    elif latest['delta_rsi'].values[0] < 0:
        reasons.append("RSI turun")
        score -= 1
    if latest['bb_percent'].values[0] > 0.9:
        reasons.append("Price near upper Bollinger Band")
    elif latest['bb_percent'].values[0] < 0.1:
        reasons.append("Price near lower Bollinger Band")
    if latest['cci'].values[0] > 100:
        reasons.append("CCI indicates strong uptrend")
    elif latest['cci'].values[0] < -100:
        reasons.append("CCI indicates strong downtrend")
    if latest['hammer'].values[0]:
        reasons.append("Hammer candlestick detected")
    if latest['doji'].values[0]:
        reasons.append("Doji candlestick detected")
    if latest['macd'].values[0] > latest['signal_line'].values[0]:
        reasons.append("MACD bullish crossover")
        score += 1
    elif latest['macd'].values[0] < latest['signal_line'].values[0]:
        reasons.append("MACD bearish crossover")
        score -= 1
    if latest['ema_20'].values[0] > latest['ema_50'].values[0]:
        reasons.append("EMA20 > EMA50 (bullish slope)")
        score += 1
    else:
        reasons.append("EMA20 < EMA50 (bearish slope)")
        score -= 1
    trend = latest['trend'].values[0]
    reasons.append(f"Trend: {trend}")
    score += 1 if trend == 'UPTREND' else -1
    price = latest['close'].values[0]
    if price <= latest['support'].values[0]:
        reasons.append("Dekat support")
        score += 1
    elif price >= latest['resistance'].values[0]:
        reasons.append("Dekat resistance")
        score -= 1
    volatility = latest['volatility'].values[0]
    if volatility > 0.02:
        reasons.append("Volatilitas tinggi")
        score -= 1
    elif volatility < 0.005:
        reasons.append("Volatilitas rendah")
        score += 0.5
    else:
        reasons.append("Volatilitas moderat")
    return "; ".join(reasons), round(score, 2)

def run_all():
    symbols = read_symbols_from_file()
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(analyze_symbol, symbols)

def main_loop():
    symbols = read_symbols_from_file()
    while True:
        logging.info(f"[Clock] Sinkronisasi waktu {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        wait_until_next_candle()
        wait_for_all_new_candles(symbols)
        logging.info("Mulai analisis prediksi")
        run_all()

if __name__ == "__main__":
    main_loop()

