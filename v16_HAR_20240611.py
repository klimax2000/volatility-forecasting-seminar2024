import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
import time

start_time = time.time()

# Daten laden
data = pd.read_csv("C:\\users\\oxfordmanrealizedvolatilityindices.csv")
data["Date"] = pd.to_datetime(data["Date"], utc=True)
data.set_index("Date", inplace=True)

# Available Ticker Symbols:
# .AEX .AORD .BFX .BSESN .BVLG .BVSP .DJI .FCHI .FTMIB .FTSE .GDAXI .GSPTSE .HSI .IBEX .IXIC .KS11 .KSE .MXX
# .N225 .NSEI .OMXC20 .OMXHPI .OMXSPI .OSEAX .RUT .SMSI .SPX .SSEC .SSMI .STI .STOXX50E
# Symbol auswählen, z.B. '.AEX'

# Symbol auswählen
symbol = ".GDAXI"
print("Tracker:", symbol)
symbol_data = data[data["Symbol"] == symbol].copy()

# Berechnungen
symbol_data["daily_RV"] = symbol_data["rv5"]
symbol_data["weekly_rv"] = symbol_data["daily_RV"].rolling(window=5).mean()
symbol_data["monthly_rv"] = symbol_data["daily_RV"].rolling(window=22).mean()
symbol_data["rsv_plus"] = symbol_data["rv5"] - symbol_data["rsv"]
symbol_data.dropna(inplace=True)

# Datenaufteilung
train_size = int(len(symbol_data) * 0.8)
validation_size = int(len(symbol_data) * 0.1)
test_size = len(symbol_data) - train_size - validation_size

train_data = symbol_data.iloc[:train_size]
validation_data = symbol_data.iloc[train_size : train_size + validation_size]
test_data = symbol_data.iloc[train_size + validation_size :]
# Kombinieren von Trainings- und Validierungsdatensatz
combined_train_val_data = pd.concat([train_data, validation_data])


# Modelle trainieren
def train_model(X, y):
    X = sm.add_constant(X.dropna())
    y = y.loc[X.index]
    return sm.OLS(y, X).fit()


# Modell 1: HAR(3)
X_train_1 = combined_train_val_data[["daily_RV", "weekly_rv", "monthly_rv"]].shift(1)
y_train_1 = combined_train_val_data["daily_RV"]
model1 = train_model(X_train_1, y_train_1)


# Modell 2 - SHAR(4)
X_train_2 = combined_train_val_data[
    ["rsv_plus", "rsv", "weekly_rv", "monthly_rv"]
].shift(1)
y_train_2 = combined_train_val_data["daily_RV"]
model2 = train_model(X_train_2, y_train_2)

# Random Forest Hyperparameter Tuning
features_rf = ["weekly_rv", "monthly_rv", "rsv_plus", "rsv"]
X_train_rf = train_data[features_rf].shift(1).dropna()
y_train_rf = train_data["daily_RV"].loc[X_train_rf.index]
param_grid = {
    "n_estimators": [500],
    "max_features": [1, "sqrt", "log2"],
    "max_depth": [5, 10, 20],
}

results = []

for params in ParameterGrid(param_grid):
    rf = RandomForestRegressor(**params, random_state=42)
    rf.fit(X_train_rf, y_train_rf)

    validation_predictions = rf.predict(validation_data[features_rf].shift(1).dropna())
    y_val_true = validation_data["daily_RV"].loc[
        validation_data[features_rf].shift(1).dropna().index
    ]

    mse = mean_squared_error(y_val_true, validation_predictions) * 1000000
    ql = np.mean(
        (y_val_true / validation_predictions)
        - np.log(y_val_true / validation_predictions)
        - 1
    )
    ss_res = np.sum((y_val_true - validation_predictions) ** 2)
    ss_tot = np.sum((y_val_true - np.mean(y_val_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    results.append({**params, "MSE": mse, "QLIKE": ql, "R^2": r_squared})

# Ergebnisse in eine CSV-Datei schreiben
results_df = pd.DataFrame(results)
results_df.to_csv(
    "C:\\Users\\random_forest_hyperparameter_tuning_results.csv",
    index=False,
)

print(
    "Ergebnisse wurden in random_forest_hyperparameter_tuning_results.csv gespeichert."
)


# Bestes RF Modell auswählen basierend auf min. MSE, min. QLIKE und max. R^2
best_result = min(results, key=lambda x: (x["MSE"], x["QLIKE"], -x["R^2"]))
print(best_result)
best_rf_params = {k: best_result[k] for k in param_grid.keys()}
best_rf = RandomForestRegressor(**best_rf_params, random_state=42)
best_rf.fit(X_train_rf, y_train_rf)


# ANN Hyperparameter Tuning
features_ann = ["weekly_rv", "monthly_rv", "rsv_plus", "rsv"]
X_train_ann = train_data[features_ann].shift(1).dropna()
y_train_ann = train_data["daily_RV"].loc[X_train_ann.index]

param_grid = {  # Igor Honig
    "hidden_layer_sizes": [(5, 2), (5,)],
    "activation": ["relu"],
    "solver": ["adam"],
    "learning_rate": [
        "constant",
        "adaptive",
    ],
}

results = []

for params in ParameterGrid(param_grid):
    ann = MLPRegressor(**params, random_state=42, max_iter=1000)
    ann.fit(X_train_ann, y_train_ann)

    validation_predictions = ann.predict(
        validation_data[features_ann].shift(1).dropna()
    )
    y_val_true = validation_data["daily_RV"].loc[
        validation_data[features_ann].shift(1).dropna().index
    ]

    mse = mean_squared_error(y_val_true, validation_predictions) * 1000000
    ql = np.mean(
        (y_val_true / validation_predictions)
        - np.log(y_val_true / validation_predictions)
        - 1
    )
    ss_res = np.sum((y_val_true - validation_predictions) ** 2)
    ss_tot = np.sum((y_val_true - np.mean(y_val_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    results.append({**params, "MSE": mse, "QLIKE": ql, "R^2": r_squared})

# Ergebnisse in eine CSV-Datei schreiben
results_df = pd.DataFrame(results)
results_df.to_csv(
    "C:\\Users\\ann_hyperparameter_tuning_results.csv",
    index=False,
)

print("Ergebnisse wurden in ann_hyperparameter_tuning_results.csv gespeichert.")

# Bestes ANN Modell auswählen basierend auf min. MSE, min. QLIKE und max. R^2
best_result = min(results, key=lambda x: (x["MSE"], x["QLIKE"], -x["R^2"]))
print(best_result)
best_ann_params = {k: best_result[k] for k in param_grid.keys()}
best_ann = MLPRegressor(**best_ann_params, random_state=42, max_iter=1000)
best_ann.fit(X_train_ann, y_train_ann)


# Modelle validieren
def validate_model(model, X, y, is_rf=False, is_ann=False):
    X = X.dropna()
    y = y.loc[X.index]
    if not is_rf and not is_ann:
        X = sm.add_constant(X)
    predictions = model.predict(X)
    mse = (np.mean((y - predictions) ** 2)) * 1000000
    ql = np.mean((y / predictions) - np.log(y / predictions) - 1)
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return mse, ql, r_squared


# Validierung von jedem Modell
mse_val_1, ql_val_1, r_squared_val_1 = validate_model(
    model1,
    combined_train_val_data[["daily_RV", "weekly_rv", "monthly_rv"]].shift(1),
    combined_train_val_data["daily_RV"],
)
mse_val_2, ql_val_2, r_squared_val_2 = validate_model(
    model2,
    combined_train_val_data[["rsv_plus", "rsv", "weekly_rv", "monthly_rv"]].shift(1),
    combined_train_val_data["daily_RV"],
)
mse_val_rf, ql_val_rf, r_squared_val_rf = validate_model(
    best_rf,
    combined_train_val_data[features_rf].shift(1),
    combined_train_val_data["daily_RV"],
    is_rf=True,
)

# Validierung des besten Modells
mse_val_ann, ql_val_ann, r_squared_val_ann = validate_model(
    best_ann,
    validation_data[features_ann].shift(1),
    validation_data["daily_RV"],
    is_ann=True,
)


# Ergebnisse Validation ausgeben
print(
    "Model 1 - Validation MSE: {:.7f}, QLIKE: {:.5f}, R^2: {:.7f}".format(
        mse_val_1, ql_val_1, r_squared_val_1
    )
)
print(
    "Model 2 - Validation MSE: {:.7f}, QLIKE: {:.5f}, R^2: {:.7f}".format(
        mse_val_2, ql_val_2, r_squared_val_2
    )
)
print(
    "Random Forest - Validation MSE: {:.7f}, QLIKE: {:.5f}, R^2: {:.7f}".format(
        mse_val_rf, ql_val_rf, r_squared_val_rf
    )
)
print(
    "Best ANN - Validation MSE: {:.7f}, QLIKE: {:.5f}, R^2: {:.7f}".format(
        mse_val_ann, ql_val_ann, r_squared_val_ann
    )
)


# Modelle Testen
def test_model(model, X, y, is_rf=False, is_ann=False):
    X = X.dropna()
    y = y.loc[X.index]
    if not is_rf and not is_ann:
        X = sm.add_constant(X)
    predictions = model.predict(X)
    mse = (np.mean((y - predictions) ** 2)) * 1000000
    ql = np.mean((y / predictions) - np.log(y / predictions) - 1)
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return mse, ql, r_squared


# Test-MSE und QLIKE für jedes Modell berechnen
mse_test_1, ql_test_1, r_squared_test_1 = test_model(
    model1,
    test_data[["daily_RV", "weekly_rv", "monthly_rv"]].shift(1),
    test_data["daily_RV"],
)
mse_test_2, ql_test_2, r_squared_test_2 = test_model(
    model2,
    test_data[["rsv_plus", "rsv", "weekly_rv", "monthly_rv"]].shift(1),
    test_data["daily_RV"],
)
mse_test_rf, ql_test_rf, r_squared_test_rf = test_model(
    best_rf, test_data[features_rf].shift(1), test_data["daily_RV"], is_rf=True
)
# Test-MSE und QLIKE für ANN berechnen
mse_test_ann, ql_test_ann, r_squared_test_ann = test_model(
    best_ann,
    test_data[features_ann].shift(1),
    test_data["daily_RV"],
    is_ann=True,
)

# Testergebnisse anzeigen
print(
    "Model 1 - Test MSE: {:.7f}, QLIKE: {:.5f}, R^2: {:.7f}".format(
        mse_test_1, ql_test_1, r_squared_test_1
    )
)
print(
    "Model 2 - Test MSE: {:.7f}, QLIKE: {:.5f}, R^2: {:.7f}".format(
        mse_test_2, ql_test_2, r_squared_test_2
    )
)
print(
    "Random Forest - Test MSE: {:.7f}, QLIKE: {:.5f}, R^2: {:.7f}".format(
        mse_test_rf, ql_test_rf, r_squared_test_rf
    )
)
# Testergebnisse anzeigen
print(
    "ANN - Test MSE: {:.7f}, QLIKE: {:.5f}, R^2: {:.7f}".format(
        mse_test_ann, ql_test_ann, r_squared_test_ann
    )
)

# print(model1.summary())
# print(model2.summary())

end_time = time.time()
print(f"Laufzeit des Codes: {end_time - start_time} Sekunden")
