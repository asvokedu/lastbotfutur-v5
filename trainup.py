import pymssql
import pandas as pd
import os
import joblib
import optuna
import warnings
from ta.trend import MACD
from ta.momentum import RSIIndicator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings("ignore")

SQL_SERVER   = "34.51.181.101"
SQL_DATABASE = "gotbai"
SQL_USERNAME = "sa"
SQL_PASSWORD = "LEtoy_89"

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
        query = f"EXEC sp_asethistori @symbol='{symbol}'"
        df = pd.read_sql(query, conn)
        conn.close()
        df.rename(columns={"opened": "open", "closet": "close"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
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
    df['ema_200'] = df['close'].ewm(span=200).mean()
    df['support'] = df['low'][::-1].rolling(10).min()[::-1]
    df['resistance'] = df['high'][::-1].rolling(10).max()[::-1]
    df['trend'] = (df['close'].shift(1) > df['ema_200'].shift(1)).map({True: 'UPTREND', False: 'DOWNTREND'})
    df['volatility'] = df['close'].rolling(window=10).std()
    df['delta_rsi'] = df['rsi'].diff()
    df['ema_slope'] = df['ema_200'].diff()

    for i in range(1, 4):
        df[f'close_shift_{i}'] = df['close'].shift(i)
        df[f'volume_shift_{i}'] = df['volume'].shift(i)
        df[f'rsi_shift_{i}'] = df['rsi'].shift(i)
        df[f'macd_shift_{i}'] = df['macd'].shift(i)

    df.dropna(inplace=True)
    return df

def generate_dynamic_label(df, n_future=1):
    df = df.copy()
    df['volatility'] = df['close'].rolling(window=10).std()
    df['reward_thresh'] = df['volatility'] * 1.5
    df['risk_thresh'] = df['volatility'] * 1.0
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

def save_model_to_sql(symbol, model_obj):
    try:
        buffer = BytesIO()
        joblib.dump(model_obj, buffer)
        model_binary = buffer.getvalue()
        model_size_kb = len(model_binary) / 1024

        if model_size_kb > 5000:
            print(f"⚠️ Model {symbol} terlalu besar ({model_size_kb:.2f} KB), tidak disimpan.")
            return

        conn = get_sql_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM model_storage WHERE symbol = %s", (symbol,))
        exists = cursor.fetchone()[0] > 0

        if exists:
            cursor.execute("UPDATE model_storage SET model_binary = %s, updated_at = GETDATE() WHERE symbol = %s",
                           (model_binary, symbol))
        else:
            cursor.execute("INSERT INTO model_storage (symbol, model_binary, updated_at) VALUES (%s, %s, GETDATE())",
                           (symbol, model_binary))

        conn.commit()
        conn.close()
        print(f"✅ Model {symbol} disimpan ke database.")
    except Exception as e:
        print(f"❌ Gagal simpan model {symbol} ke DB: {e}")

def save_model_metrics_to_sql(symbol, f1, acc, total_rows, label_counts):
    try:
        conn = get_sql_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM model_metrics WHERE symbol = %s", (symbol,))
        cursor.execute("""
            INSERT INTO model_metrics (symbol, f1_score, accuracy, total_rows, buy_count, sell_count, wait_count, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, GETDATE())
        """, (
            symbol, f1, acc, total_rows,
            label_counts.get("BUY", 0),
            label_counts.get("SELL", 0),
            label_counts.get("WAIT", 0)
        ))

        conn.commit()
        conn.close()
        print(f"📊 Metrics {symbol} disimpan ke database.")
    except Exception as e:
        print(f"❌ Gagal simpan metrics {symbol} ke DB: {e}")

def train_model_for_symbol(symbol):
    print(f"\n🚧 Melatih {symbol}...")
    df = fetch_data_from_sql(symbol)
    if df is None or len(df) < 100:
        print("❌ Data kurang, skip."); return

    df = calculate_indicators(df)
    df = generate_dynamic_label(df)
    df.dropna(inplace=True)

    if df['label'].nunique() < 2:
        print("❌ Label tidak bervariasi, skip."); return

    le_trend = LabelEncoder()
    df['trend_encoded'] = le_trend.fit_transform(df['trend'])

    features = [c for c in df.columns if c not in ['label', 'open', 'high', 'low', 'close', 'volume',
                                                   'future_return', 'trend', 'reward_thresh', 'risk_thresh']]
    X = df[features]
    y = df['label']
    le_label = LabelEncoder()
    y_enc = le_label.fit_transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=0.2, random_state=42)
    ros = RandomOverSampler(random_state=42)
    X_tr_bal, y_tr_bal = ros.fit_resample(X_tr, y_tr)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "n_jobs": 2
        }
        m = XGBClassifier(**params, use_label_encoder=False, eval_metric="mlogloss")
        m.fit(X_tr_bal, y_tr_bal)
        preds = m.predict(X_te)
        return f1_score(y_te, preds, average="weighted")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)
    best = study.best_params
    best["n_jobs"] = 2
    model = XGBClassifier(**best, use_label_encoder=False, eval_metric="mlogloss")
    model.fit(X_tr_bal, y_tr_bal)

    preds = model.predict(X_te)
    f1 = f1_score(y_te, preds, average="weighted")
    acc = accuracy_score(y_te, preds)
    label_counts = df['label'].value_counts().to_dict()

    model_obj = {
        'model': model,
        'features': features,
        'label_encoder': le_label,
        'trend_encoder': le_trend
    }

    save_model_to_sql(symbol, model_obj)
    save_model_metrics_to_sql(symbol, f1, acc, len(df), label_counts)

def train_all_symbols():
    cpu_total = os.cpu_count() or 8
    workers = max(1, min(cpu_total - 2, 96))
    print(f"🚀 Mulai pelatihan paralel ({workers} proses)...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        executor.map(train_model_for_symbol, SYMBOLS)

if __name__ == "__main__":
    train_all_symbols()
