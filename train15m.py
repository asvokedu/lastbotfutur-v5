import pymssql
import pandas as pd
import os
import joblib
import optuna
import warnings
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import CCIIndicator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from imblearn.over_sampling import RandomOverSampler
import traceback
import gzip
from optuna.pruners import MedianPruner
from collections import Counter


warnings.filterwarnings("ignore")

# ──⚙️ KONFIGURASI DATABASE──
SQL_SERVER   = "34.51.164.80"
SQL_DATABASE = "gotbai"
SQL_USERNAME = "sa"
SQL_PASSWORD = "LEtoy_89"

# Ambil daftar simbol dari file listsyombol.txt di Google Drive
SYMBOL_FILE = "/content/drive/MyDrive/listsyombol.txt"
with open(SYMBOL_FILE, "r") as file:
    SYMBOLS = [line.strip() for line in file if line.strip()]

def get_sql_connection():
    return pymssql.connect(
        server=SQL_SERVER,
        user=SQL_USERNAME,
        password=SQL_PASSWORD,
        database=SQL_DATABASE
    )

def fetch_data_from_sql(symbol):
    try:
        conn = get_sql_connection()
        if symbol not in SYMBOLS:
            return None
        query = f"EXEC sp_asethistori @symbol='{symbol}', @interval = '15m'"
        df = pd.read_sql(query, conn)
        conn.close()
        df.rename(columns={"opened": "open", "closet": "close"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df[["open", "high", "low", "close", "volume"]].astype(float)
    except Exception as e:
        print(f"❌ Gagal ambil data {symbol}: {e}")
        return None

def calculate_indicators(df):
    df = df.copy()
    df['rsi'] = RSIIndicator(df['close']).rsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['signal_line'] = macd.macd_signal()
    df['ema_200'] = df['close'].ewm(span=50).mean()
    # Causal (hanya melihat masa lalu)
    df['support'] = df['low'].rolling(10).min()
    df['resistance'] = df['high'].rolling(10).max()
    df['trend'] = df.apply(lambda x: 'UPTREND' if x['close'] > x['ema_200'] else 'DOWNTREND', axis=1)
    df['volatility'] = df['close'].rolling(window=12).std()
    df['delta_rsi'] = df['rsi'].diff()
    df['ema_slope'] = df['ema_200'].diff()

    for i in range(1, 4):
        df[f'close_shift_{i}'] = df['close'].shift(i)
        df[f'volume_shift_{i}'] = df['volume'].shift(i)
        df[f'rsi_shift_{i}'] = df['rsi'].shift(i)
        df[f'macd_shift_{i}'] = df['macd'].shift(i)

    # Bollinger Bands
    bb = BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # CCI
    cci = CCIIndicator(high=df['high'], low=df['low'], close=df['close'])
    df['cci'] = cci.cci()

    # Candlestick patterns
    df['hammer'] = ((df['close'] > df['open']) &
                    ((df['low'] < df['open'] - (df['high'] - df['low']) * 0.5)) &
                    ((df['high'] - df['close']) < 0.2 * (df['high'] - df['low'])))

    df['doji'] = (abs(df['close'] - df['open']) <= 0.1 * (df['high'] - df['low']))

    df.dropna(inplace=True)
    return df

def generate_dynamic_label(df, reward_multiplier=1.5, risk_multiplier=1.0, n_future=1):
    df = df.copy()
    df.dropna(subset=['volatility'], inplace=True)
    df['reward_thresh'] = df['volatility'] * reward_multiplier
    df['risk_thresh'] = df['volatility'] * risk_multiplier
    df['future_return'] = df['close'].shift(-n_future) / df['close'] - 1

    labels = []
    for i in range(len(df)):
        if pd.isna(df['future_return'].iloc[i]):
            labels.append("WAIT")
            continue
        reward = df['reward_thresh'].iloc[i]
        risk = df['risk_thresh'].iloc[i]
        ret = df['future_return'].iloc[i]
        if ret >= reward:
            labels.append("BUY")
        elif ret <= -risk:
            labels.append("SELL")
        else:
            labels.append("WAIT")
    df['label'] = labels
    return df
def find_best_labeling_threshold(df, reward_grid=[0.0008, 0.09, 0.05, 0.5, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0], risk_grid=[0.0004, 0.05, 0.01, 0.02, 0.3, 0.75, 0.85, 1.0, 1.25, 1.5, 1.75, 2.0]):
    best_score = -1
    best_params = (1.5, 1.0)
    le_label = LabelEncoder()

    for rwd in reward_grid:
        for rsk in risk_grid:
            temp_df = generate_dynamic_label(df, rwd, rsk)
            if temp_df['label'].nunique() < 2:
                continue
            temp_df['label_encoded'] = le_label.fit_transform(temp_df['label'])
            y = temp_df['label_encoded']

            holdout_size = int(len(temp_df) * 0.1)
            y_train = y[:-holdout_size]
            y_holdout = y[-holdout_size:]

            if len(set(y_holdout)) < 2:
                continue

            majority_class = Counter(y_train).most_common(1)[0][0]
            dummy_preds = [majority_class] * len(y_holdout)
            score = f1_score(y_holdout, dummy_preds, average='weighted')

            if score > best_score:
                best_score = score
                best_params = (rwd, rsk)
    return best_params

def save_model_to_sql(symbol, model_obj):
    try:
        buffer_model = BytesIO()
        model_obj['model'].save_model(buffer_model)  # Simpan hanya XGBClassifier ke format JSON
        xgb_model_binary = buffer_model.getvalue()

        # Simpan sisa komponen (features, encoders)
        rest_obj = {
            'features': model_obj['features'],
            'label_encoder': model_obj['label_encoder'],
            'trend_encoder': model_obj['trend_encoder']
        }
        buffer_rest = BytesIO()
        with gzip.GzipFile(fileobj=buffer_rest, mode='wb') as gz:
            joblib.dump(rest_obj, gz)
        other_binary = buffer_rest.getvalue()

        conn = get_sql_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM model_storage WHERE symbol = %s AND interval = %s", (symbol, '15m'))
        exists = cursor.fetchone()[0] > 0

        if exists:
            cursor.execute("""
                UPDATE model_storage
                SET model_binary = %s, other_data = %s, sizemodel = %s, updated_at = GETDATE()
                WHERE symbol = %s AND interval = %s
            """, (xgb_model_binary, other_binary, len(xgb_model_binary), symbol, '15m'))
        else:
            cursor.execute("""
                INSERT INTO model_storage (symbol, model_binary, other_data, sizemodel, interval, updated_at)
                VALUES (%s, %s, %s, %s, %s, GETDATE())
            """, (symbol, xgb_model_binary, other_binary, len(xgb_model_binary), '15m'))

        conn.commit()
        conn.close()
        print(f"✅ Model {symbol} disimpan ke database.")
    except Exception as e:
        print(f"❌ Gagal simpan model {symbol} ke DB: {e}")


def save_model_metrics_to_sql(symbol, f1, acc, total_rows, label_counts):
    try:
        conn = get_sql_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM model_metrics WHERE symbol = %s AND interval = %s", (symbol, '15m'))
        cursor.execute("""
            INSERT INTO model_metrics (symbol, f1_score, accuracy, total_rows, buy_count, sell_count, wait_count, interval, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, GETDATE())
        """, (
            symbol, f1, acc, total_rows,
            label_counts.get("BUY", 0),
            label_counts.get("SELL", 0),
            label_counts.get("WAIT", 0),
            '15m'
        ))

        conn.commit()
        conn.close()
        print(f"📊 Metrics {symbol} disimpan ke database.")
    except Exception as e:
        print(f"❌ Gagal simpan metrics {symbol} ke DB: {e}")

def train_model_for_symbol(symbol):
    print(f"\n🚧 Melatih {symbol}...")

    df = fetch_data_from_sql(symbol)
    if df is None or len(df) < 500:
        print(f"❌ {symbol}: Data kurang, skip.")
        return

    df = calculate_indicators(df)
    # Cari kombinasi reward/risk terbaik
    reward_mul, risk_mul = find_best_labeling_threshold(df)
    print(f"🔎 {symbol} - Best Threshold: reward={reward_mul}, risk={risk_mul}")

    # Buat label menggunakan threshold terbaik
    df = generate_dynamic_label(df, reward_multiplier=reward_mul, risk_multiplier=risk_mul)

    df.dropna(inplace=True)

    if df['label'].nunique() < 2:
        print(f"❌ {symbol}: Label tidak bervariasi, skip.")
        return

    le_trend = LabelEncoder()
    df['trend_encoded'] = le_trend.fit_transform(df['trend'])
    le_label = LabelEncoder()
    df['label_encoded'] = le_label.fit_transform(df['label'])

    features = [c for c in df.columns if c not in ['label','label_encoded', 'open', 'high', 'low', 'close', 'volume',
                                                   'future_return', 'trend', 'reward_thresh', 'risk_thresh']]
    df = df.dropna(subset=features)
    X = df[features]
    y = df['label_encoded']

    # 👉 Split train/test (10% data terakhir = holdout test set)
    holdout_size = int(len(df) * 0.1)
    X_train, X_holdout = X.iloc[:-holdout_size], X.iloc[-holdout_size:]
    y_train, y_holdout = y.iloc[:-holdout_size], y.iloc[-holdout_size:]

    ros = RandomOverSampler(random_state=42)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "n_jobs": 4
        }

        model = XGBClassifier(**params, use_label_encoder=False, eval_metric="mlogloss")
        tscv = TimeSeriesSplit(n_splits=5)
        f1s = []

        for train_idx, test_idx in tscv.split(X_train):
            X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_tr, y_te = y_train.iloc[train_idx], y_train.iloc[test_idx]

            X_tr_bal, y_tr_bal = ros.fit_resample(X_tr, y_tr)
            model.fit(X_tr_bal, y_tr_bal)
            preds = model.predict(X_te)
            f1s.append(f1_score(y_te, preds, average="weighted"))

        return sum(f1s) / len(f1s)

    study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=10))
    study.optimize(objective, n_trials=50, timeout=4500)

    best_params = study.best_params
    best_params["n_jobs"] = 47
    model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="mlogloss")

    # ✅ Final training on full train set (oversampled)
    X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)
    model.fit(X_train_bal, y_train_bal)

    # ✅ Final evaluation on untouched holdout set
    preds_holdout = model.predict(X_holdout)
    f1_holdout = f1_score(y_holdout, preds_holdout, average="weighted")
    acc_holdout = accuracy_score(y_holdout, preds_holdout)

    print(f"✅ {symbol} - Final HOLDOUT F1: {f1_holdout:.4f}, Accuracy: {acc_holdout:.4f}")

    model_obj = {
        'model': model,
        'features': features,
        'label_encoder': le_label,
        'trend_encoder': le_trend
    }

    save_model_to_sql(symbol, model_obj)
    save_model_metrics_to_sql(symbol, f1_holdout, acc_holdout, len(df), df['label'].value_counts().to_dict())


def train_all_symbols():
    workers = min(os.cpu_count() - 4, 96)
    print(f"🚀 Mulai pelatihan paralel ({workers} proses)...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        executor.map(train_model_for_symbol, SYMBOLS)

if __name__ == "__main__":
    train_all_symbols()
