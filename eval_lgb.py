import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

print('Loading Data...', flush=True)
df = pd.read_parquet('data/processed/features.parquet')
X = df[[c for c in df.columns if 'lag' in c or 'rolling' in c]].fillna(0)
y = df['AQI'].fillna(0)

split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = lgb.LGBMRegressor(n_estimators=500, max_depth=10, learning_rate=0.05, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'R2: {r2:.4f}')
print(f'RMSE: {rmse:.4f}')
