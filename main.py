import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
import random
import tensorflow as tf
import time
import joblib

TRAIN_ANN = False  # True = trainiere neu, False = lade gespeichertes Modell
TRAIN_RF = False  # True = trainiere neu, False = lade gespeichertes Modell

start_time = time.time()

os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Daten laden
data = pd.read_csv("oxfordmanrealizedvolatilityindices.csv")
data["Date"] = pd.to_datetime(data["Date"], utc=True)
data.set_index("Date", inplace=True)

# Symbol auswählen
symbol = ".GDAXI"
print("Tracker:", symbol)
symbol_data = data[data["Symbol"] == symbol].copy()

# Feature Engineering
symbol_data["daily_RV"] = symbol_data["rv5"]
symbol_data["weekly_rv"] = symbol_data["daily_RV"].rolling(window=5).mean()
symbol_data["monthly_rv"] = symbol_data["daily_RV"].rolling(window=22).mean()
symbol_data["rsv_plus"] = symbol_data["rv5"] - symbol_data["rsv"]
symbol_data.dropna(inplace=True)

# Datenaufteilung
train_size = int(len(symbol_data) * 0.8)
validation_size = int(len(symbol_data) * 0.1)
train_data = symbol_data.iloc[:train_size]
validation_data = symbol_data.iloc[train_size : train_size + validation_size]
test_data = symbol_data.iloc[train_size + validation_size :]
combined_train_val_data = pd.concat([train_data, validation_data])

# Modelle trainieren (HAR, SHAR)
def train_model(X, y):
    X = sm.add_constant(X.dropna())
    y = y.loc[X.index]
    return sm.OLS(y, X).fit()

# HAR(3)
X_train_1 = combined_train_val_data[["daily_RV", "weekly_rv", "monthly_rv"]].shift(1)
y_train_1 = combined_train_val_data["daily_RV"]
model1 = train_model(X_train_1, y_train_1)

# SHAR(4)
X_train_2 = combined_train_val_data[["rsv_plus", "rsv", "weekly_rv", "monthly_rv"]].shift(1)
y_train_2 = combined_train_val_data["daily_RV"]
model2 = train_model(X_train_2, y_train_2)

# Random Forest
features_rf = ["weekly_rv", "monthly_rv", "rsv_plus", "rsv"]
X_train_rf = train_data[features_rf].shift(1).dropna()
y_train_rf = train_data["daily_RV"].loc[X_train_rf.index]
param_grid_rf = {
        "n_estimators": [500],
        "max_features": [1, "sqrt", "log2"],
        "max_depth": [5, 10, 20],
    }
if TRAIN_RF:
    results_rf = []
    for params in ParameterGrid(param_grid_rf):
        rf = RandomForestRegressor(**params, random_state=42)
        rf.fit(X_train_rf, y_train_rf)
        X_val_rf = validation_data[features_rf].shift(1).dropna()
        y_val_rf = validation_data["daily_RV"].loc[X_val_rf.index]
        preds = rf.predict(X_val_rf)
        mse = mean_squared_error(y_val_rf, preds) * 1e6
        ql = np.mean((y_val_rf / preds) - np.log(y_val_rf / preds) - 1)
        ss_res = np.sum((y_val_rf - preds) ** 2)
        ss_tot = np.sum((y_val_rf - np.mean(y_val_rf)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        results_rf.append({**params, "MSE": mse, "QLIKE": ql, "R^2": r2})
    best_rf_params = min(results_rf, key=lambda x: (x["MSE"], x["QLIKE"], -x["R^2"]))
    best_rf = RandomForestRegressor(
        n_estimators=best_rf_params["n_estimators"],
        max_features=best_rf_params["max_features"],
        max_depth=best_rf_params["max_depth"],
        random_state=42,
    )
    best_rf.fit(X_train_rf, y_train_rf)
    joblib.dump(best_rf, "best_rf_model.joblib")
    print("Bestes RandomForest-Modell wurde als 'best_rf_model.joblib' gespeichert.")
else:
    best_rf = joblib.load("best_rf_model.joblib")
    print("RandomForest-Modell aus 'best_rf_model.joblib' geladen.")

# ANN (Keras) mit Dropout
features_ann = ["weekly_rv", "monthly_rv", "rsv_plus", "rsv"]
scaler = StandardScaler()
X_train_ann = train_data[features_ann].shift(1).dropna()
y_train_ann = train_data["daily_RV"].loc[X_train_ann.index]
X_val_ann = validation_data[features_ann].shift(1).dropna()
y_val_ann = validation_data["daily_RV"].loc[X_val_ann.index]
X_train_ann_scaled = scaler.fit_transform(X_train_ann)
X_val_ann_scaled = scaler.transform(X_val_ann)
param_grid_ann = {
        "dropout_rate": [0.0, 0.1, 0.2],
        "learning_rate": [0.001, 0.0005],
        "batch_size": [16, 32],
        "epochs": [100, 200]
    }

if TRAIN_ANN:
    def build_ann(input_dim, dropout_rate=0.2):
        model = Sequential()
        model.add(Dense(32, activation="relu", input_dim=input_dim))
        model.add(Dropout(dropout_rate))
        model.add(Dense(16, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation="linear"))
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        return model

    best_val_loss = float('inf')
    best_params = None
    best_model = None

    for dropout in param_grid_ann["dropout_rate"]:
        for lr in param_grid_ann["learning_rate"]:
            for batch in param_grid_ann["batch_size"]:
                for ep in param_grid_ann["epochs"]:
                    model = build_ann(X_train_ann_scaled.shape[1], dropout_rate=dropout)
                    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
                    history = model.fit(
                        X_train_ann_scaled, y_train_ann,
                        epochs=ep, batch_size=batch, verbose=0,
                        validation_data=(X_val_ann_scaled, y_val_ann),
                        callbacks=[EarlyStopping(patience=20, restore_best_weights=True)]
                    )
                    val_loss = min(history.history['val_loss'])
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = {
                            "dropout_rate": dropout,
                            "learning_rate": lr,
                            "batch_size": batch,
                            "epochs": ep
                        }
                        best_model = model

    print("Best ANN params:", best_params)
    ann = best_model
    # Speichere das beste Keras-ANN Modell
    ann.save("best_ann_model.keras")
    print("Bestes ANN-Modell wurde als 'best_ann_model.keras' gespeichert.")

else:
    from tensorflow.keras.models import load_model
    features_ann = ["weekly_rv", "monthly_rv", "rsv_plus", "rsv"]
    scaler = StandardScaler()
    X_train_ann = train_data[features_ann].shift(1).dropna()
    scaler.fit(X_train_ann)  # Damit der Scaler für spätere Transformationen bereit ist
    ann = load_model("best_ann_model.keras")
    print("ANN-Modell aus 'best_ann_model.keras' geladen.")

# Modelle validieren
def validate_model(model, X, y, is_rf=False, is_ann=False):
    X = X.dropna()
    y = y.loc[X.index]
    if is_ann:
        X = scaler.transform(X)
        preds = model.predict(X).flatten()
    elif is_rf:
        preds = model.predict(X)
    else:
        X = sm.add_constant(X)
        preds = model.predict(X)
    mse = mean_squared_error(y, preds) * 1e6
    ql = np.mean((y / preds) - np.log(y / preds) - 1)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return mse, ql, r2

# Validation
mse_val_1, ql_val_1, r2_val_1 = validate_model(
    model1, validation_data[["daily_RV", "weekly_rv", "monthly_rv"]].shift(1), validation_data["daily_RV"]
)
mse_val_2, ql_val_2, r2_val_2 = validate_model(
    model2, validation_data[["rsv_plus", "rsv", "weekly_rv", "monthly_rv"]].shift(1), validation_data["daily_RV"]
)
mse_val_rf, ql_val_rf, r2_val_rf = validate_model(
    best_rf, validation_data[features_rf].shift(1).dropna(), validation_data["daily_RV"], is_rf=True
)
mse_val_ann, ql_val_ann, r2_val_ann = validate_model(
    ann, validation_data[features_ann].shift(1).dropna(), validation_data["daily_RV"], is_ann=True
)

print("Validation Results:")
print(f"HAR(3):     MSE={mse_val_1:.7f}, QLIKE={ql_val_1:.5f}, R^2={r2_val_1:.7f}")
print(f"SHAR(4):    MSE={mse_val_2:.7f}, QLIKE={ql_val_2:.5f}, R^2={r2_val_2:.7f}")
print(f"RandomForest: MSE={mse_val_rf:.7f}, QLIKE={ql_val_rf:.5f}, R^2={r2_val_rf:.7f}")
print(f"ANN (Keras): MSE={mse_val_ann:.7f}, QLIKE={ql_val_ann:.5f}, R^2={r2_val_ann:.7f}")

# Testdaten vorbereiten
X_test_1 = test_data[["daily_RV", "weekly_rv", "monthly_rv"]].shift(1)
y_test_1 = test_data["daily_RV"]
X_test_2 = test_data[["rsv_plus", "rsv", "weekly_rv", "monthly_rv"]].shift(1)
y_test_2 = test_data["daily_RV"]
X_test_rf = test_data[features_rf].shift(1).dropna()
y_test_rf = test_data["daily_RV"].loc[X_test_rf.index]
X_test_ann = test_data[features_ann].shift(1).dropna()
y_test_ann = test_data["daily_RV"].loc[X_test_ann.index]

# Test
mse_test_1, ql_test_1, r2_test_1 = validate_model(model1, X_test_1, y_test_1)
mse_test_2, ql_test_2, r2_test_2 = validate_model(model2, X_test_2, y_test_2)
mse_test_rf, ql_test_rf, r2_test_rf = validate_model(best_rf, X_test_rf, y_test_rf, is_rf=True)
mse_test_ann, ql_test_ann, r2_test_ann = validate_model(ann, X_test_ann, y_test_ann, is_ann=True)

print("\nTest Results:")
print(f"HAR(3):     MSE={mse_test_1:.7f}, QLIKE={ql_test_1:.5f}, R^2={r2_test_1:.7f}")
print(f"SHAR(4):    MSE={mse_test_2:.7f}, QLIKE={ql_test_2:.5f}, R^2={r2_test_2:.7f}")
print(f"RandomForest: MSE={mse_test_rf:.7f}, QLIKE={ql_test_rf:.5f}, R^2={r2_test_rf:.7f}")
print(f"ANN (Keras): MSE={mse_test_ann:.7f}, QLIKE={ql_test_ann:.5f}, R^2={r2_test_ann:.7f}")

end_time = time.time()
print(f"\nLaufzeit des Codes: {end_time - start_time:.2f} Sekunden")