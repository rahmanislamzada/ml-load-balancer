"""Train a regression model that predicts request latency from the backend state.

Input : ml/data.csv  (backend, active_connections, avg_latency, error_rate,
                      request_rate, latency)
Output: ml/model.joblib  (loaded by MLStrategy in the proxy)

Run from the project root:  python train_model.py
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from load_balancer.strategies import FEATURE_NAMES, DEFAULT_MODEL_PATH

DATA_FILE = os.path.join("ml", "data.csv")

df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} samples")
print(df.groupby("backend")["latency"].agg(["count", "mean", "std"]).round(3))

X = df[FEATURE_NAMES]          # same features, same order as in the proxy
y = df["latency"]              # seconds

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Small model on purpose: it is called on EVERY request, so prediction must be fast.
model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

# Baselines: is the model better than trivial guesses?
mean_mae = mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))
lat_mask = X_test["avg_latency"] > 0
avglat_mae = mean_absolute_error(y_test[lat_mask], X_test["avg_latency"][lat_mask])

print("\n=== Evaluation on the test set (20%) ===")
print(f"Model            MAE: {mae*1000:.1f} ms | RMSE: {rmse*1000:.1f} ms | R2: {r2:.3f}")
print(f"Baseline 'always the mean'      MAE: {mean_mae*1000:.1f} ms")
print(f"Baseline 'use avg_latency'      MAE: {avglat_mae*1000:.1f} ms")

print("\n=== Feature importance ===")
for name, imp in sorted(zip(FEATURE_NAMES, model.feature_importances_),
                        key=lambda t: -t[1]):
    print(f"{name:20s} {imp:.3f}")

os.makedirs(os.path.dirname(DEFAULT_MODEL_PATH), exist_ok=True)
joblib.dump(model, DEFAULT_MODEL_PATH)
print(f"\nModel saved to {DEFAULT_MODEL_PATH}")