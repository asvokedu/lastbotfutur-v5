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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from xgboost import XGBClassifier
from concurrent.futures import ProcessPoolExecutor
from imblearn.over_sampling import SMOTE
from io import BytesIO
from imblearn.over_sampling import RandomOverSampler
import traceback
import gzip
import tempfile
import numpy as np
from collections import Counter

warnings.filterwarnings("ignore")

# ──⚙️ KONFIGURASI DATABASE──
SQL_SERVER   = "34.51.175.118"
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

    bb = BollingerBands(df['close'])
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

    # Tambahan fitur baru
    df['price_change_pct'] = df['close'].pct_change()
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['candle_body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)
    df['upper_shadow_ratio'] = (df['high'] - df[['close', 'open']].max(axis=1)) / (df['high'] - df['low'] + 1e-9)
    df['lower_shadow_ratio'] = (df[['close', 'open']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-9)
    df['breakout_high_20'] = (df['close'] > df['high'].rolling(20).max()).astype(int)
    df['breakout_low_20'] = (df['close'] < df['low'].rolling(20).min()).astype(int)

    df.dropna(inplace=True)
    return df

def select_top_features(model, X, top_n=30):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    print("🔍 Top features:")
    print(feature_importance_df.head(top_n))

    return feature_importance_df['feature'].head(top_n).tolist()


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

def estimate_threshold_range(df, low_mult=0.5, high_mult=2.0):
    """
    Menentukan rentang threshold reward berdasarkan karakteristik volatilitas simbol tertentu.
    """
    vol = df['volatility'].dropna()
    if len(vol) == 0:
        return [0.5, 1.0, 1.5]
    
    avg_vol = vol.mean()
    min_threshold = max(round(avg_vol * low_mult, 4), 0.0005)
    max_threshold = max(round(avg_vol * high_mult, 4), min_threshold + 0.001)

    
    steps = 6
    thresholds = [round(min_threshold + i*(max_threshold-min_threshold)/steps, 4) for i in range(steps+1)]
    return thresholds


def find_best_labeling_threshold(df, thresholds, min_count=5, n_splits=3):
    best_score = -1
    best_threshold = None
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for threshold in thresholds:
        df_labeled = generate_dynamic_label(df.copy(), reward_multiplier=threshold, risk_multiplier=1.0)
        label_col = 'label'
        df_labeled = df_labeled.dropna()

        if df_labeled[label_col].nunique() < 2:
            print(f"⚠️ Threshold {threshold:.3f} dilewati: hanya satu kelas tersedia.")
            continue

        # Hitung distribusi dan proporsi dinamis berdasarkan jumlah kelas
        expected = 1.0 / df_labeled[label_col].nunique()
        min_prop = expected * 0.09
        max_prop = expected * 2.0

        label_counts = df_labeled[label_col].value_counts()
        label_props = label_counts / label_counts.sum()

        if (label_counts < min_count).any():
            print(f"❌ Threshold {threshold:.3f} dilewati: kelas dengan sampel kurang dari {min_count} ditemukan {label_counts.to_dict()}")
            continue
        if (label_props < min_prop).any() or (label_props > max_prop).any():
            print(f"❌ Threshold {threshold:.3f} dilewati: distribusi tidak seimbang {label_props.to_dict()}")
            continue

        feature_cols = [col for col in df_labeled.columns if col not in ['symbol', 'open_time', label_col, 'trend', 'future_return', 'reward_thresh', 'risk_thresh']]
        X = df_labeled[feature_cols]
        y = df_labeled[label_col]

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        f1_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

            ros = RandomOverSampler()
            X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

            model = XGBClassifier(
                objective='multi:softmax',
                num_class=len(le.classes_),
                eval_metric='mlogloss',
                n_jobs=4,
                use_label_encoder=False,
                verbosity=0
            )
            model.fit(X_train_res, y_train_res)
            preds = model.predict(X_val)
            score = f1_score(y_val, preds, average="weighted")
            f1_scores.append(score)

        avg_score = np.mean(f1_scores)
        print(f"✅ Threshold {threshold:.4f} → Avg F1: {avg_score:.4f} | Distribusi: {label_props.to_dict()}")

        if avg_score > best_score:
            best_score = avg_score
            best_threshold = threshold

    if best_threshold is not None:
        print(f"\n🎯 Best threshold: {best_threshold:.4f} dengan Avg F1: {best_score:.4f}")
    else:
        print("❌ Tidak ada threshold valid yang memenuhi kriteria distribusi dan jumlah sampel.")

    return best_threshold


def objective(trial, X_train, y_train, X_val, y_val, num_classes):
    param = {
        'verbosity': 0,
        'objective': 'multi:softmax',
        'num_class': num_classes,
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'n_jobs': 4,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }
    model = XGBClassifier(**param)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return f1_score(y_val, preds, average='weighted')

def tune_hyperparameters(X, y, num_classes, timeout=300):
    tscv = TimeSeriesSplit(n_splits=3)
    train_idx, val_idx = list(tscv.split(X))[-1]

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    ros = RandomOverSampler()
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    study = optuna.create_study(direction='maximize')
    func = lambda trial: objective(trial, X_train_res, y_train_res, X_val, y_val, num_classes)
    study.optimize(func, timeout=timeout, n_trials=50)

    print(f"Best trial params: {study.best_trial.params}")
    print(f"Best F1 score in tuning: {study.best_value:.4f}")
    return study.best_trial.params

def save_model_to_sql(symbol, model_obj):
    try:
        booster = model_obj['model'].get_booster()
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            temp_path = tmp_file.name
        booster.save_model(temp_path)
        with open(temp_path, 'rb') as f:
            xgb_model_binary = f.read()
        os.remove(temp_path)

        rest_obj = {
            'features': model_obj['features'],
            'label_encoder': model_obj['label_encoder'],
            'trend_encoder': model_obj['trend_encoder'],
            'best_threshold': model_obj.get('best_threshold', None),
            'f1_score': model_obj.get('f1_score', None),
            'n_classes_': len(model_obj['label_encoder'].classes_)
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
                UPDATE model_storage SET
                xgb_model = %s,
                other_data = %s,
                updated_at = GETDATE()
                WHERE symbol = %s AND interval = %s
            """, (xgb_model_binary, other_binary, symbol, '15m'))
        else:
            cursor.execute("""
                INSERT INTO model_storage(symbol, interval, xgb_model, other_data, updated_at)
                VALUES (%s, %s, %s, %s, GETDATE())
            """, (symbol, '15m', xgb_model_binary, other_binary))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Model for {symbol} saved to database")
    except Exception as e:
        print(f"❌ Failed saving model for {symbol}: {e}")
        traceback.print_exc()

def save_model_metrics_to_sql(symbol, f1, acc, total_rows, label_counts):
    try:
        conn = get_sql_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM model_metrics WHERE symbol = %s AND interval = %s", (symbol, '15m'))
        cursor.execute("""
            INSERT INTO model_metrics (symbol, f1_score, accuracy, total_rows, buy_count, sell_count, wait_count, interval, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, GETDATE())
        """, (
            symbol,
            float(f1),
            float(acc),
            int(total_rows),
            int(label_counts.get("BUY", 0)),
            int(label_counts.get("SELL", 0)),
            int(label_counts.get("WAIT", 0)),
            '15m'
        ))

        conn.commit()
        conn.close()
        print(f"📊 Metrics {symbol} disimpan ke database.")
    except Exception as e:
        print(f"❌ Gagal simpan metrics {symbol} ke DB: {e}")


def train_model_for_symbol(symbol):
    try:
        print(f"▶️ Starting training for {symbol}...")
        df_raw = fetch_data_from_sql(symbol)
        if df_raw is None or len(df_raw) < 500:
            print(f"⚠️ Data untuk {symbol} kurang, skip training.")
            return

        df_feat = calculate_indicators(df_raw)

        thresholds = estimate_threshold_range(df_feat)
        best_threshold = find_best_labeling_threshold(
        df_feat,
        thresholds=thresholds,
        min_count=5,
        n_splits=3
        )


        if best_threshold is None:
            print(f"❌ Tidak ada threshold valid untuk {symbol}, skip training.")
            return  # keluar dari fungsi


        df_labeled = generate_dynamic_label(df_feat.copy(), reward_multiplier=best_threshold, risk_multiplier=1.0)
        label_col = 'label'
        df_labeled.dropna(subset=[label_col], inplace=True)

        # Encode label dan trend sekali saja
        label_encoder = LabelEncoder()
        trend_encoder = LabelEncoder()
        df_labeled['label_encoded'] = label_encoder.fit_transform(df_labeled[label_col])
        df_labeled['trend_encoded'] = trend_encoder.fit_transform(df_labeled['trend'])

        exclude_cols = ['symbol', label_col, 'trend', 'future_return', 'reward_thresh', 'risk_thresh']
        feature_cols = [col for col in df_labeled.columns if col not in exclude_cols and col != 'label_encoded']


        X = df_labeled[feature_cols]
        y = df_labeled['label_encoded']

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

        # Split data dengan stratifikasi
        label_counts = y.value_counts()
        print(f"Distribusi label: \n{label_counts}")

        if label_counts.min() < 2:
            print("⚠️ Ada kelas dengan kurang dari 2 sampel, skip training untuk simbol ini.")
            return  # Hentikan proses training supaya tidak error

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

        
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        num_classes = len(label_encoder.classes_)
        best_params = tune_hyperparameters(X_train_res, y_train_res, num_classes)

        # Latih model sementara untuk feature importance
        temp_model = XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_jobs=4,
            **best_params
        )
        temp_model.fit(X_train_res, y_train_res)

        # Seleksi fitur terbaik
        top_features = select_top_features(temp_model, X_train_res)

        # Filter data dengan fitur terpilih
        X_train_res = X_train_res[top_features]
        X_test = X_test[top_features]

        # Latih ulang model FINAL dengan fitur terpilih
        model = XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_jobs=4,
            **best_params
        )
        model.fit(X_train_res, y_train_res)

        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds, average='weighted')
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average='weighted')
        recall = recall_score(y_test, preds, average='weighted')

        print(f"\n📊 Evaluation Metrics for {symbol}:")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, preds, target_names=label_encoder.classes_))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))

        model_obj = {
            'model': model,
            'features': top_features,
            'label_encoder': label_encoder,
            'trend_encoder': trend_encoder,
            'best_threshold': best_threshold,
            'f1_score': f1,
            'scaler': scaler
        }

        save_model_to_sql(symbol, model_obj)

        label_counts = df_labeled['label'].value_counts().to_dict()
        save_model_metrics_to_sql(symbol, f1, acc, len(df_labeled), label_counts)

    except Exception as e:
        print(f"❌ Error training model for {symbol}: {e}")
        traceback.print_exc()


def main():
    with ProcessPoolExecutor(max_workers=6) as executor:
        executor.map(train_model_for_symbol, SYMBOLS)

if __name__ == "__main__":
    main()

