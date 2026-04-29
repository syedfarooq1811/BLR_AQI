import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_parquet('data/processed/features.parquet').head(5000)
X = df[[c for c in df.columns if 'lag' in c or 'rolling' in c]].fillna(0)
y = df['AQI'].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'R2: {r2:.4f}')
print(f'RMSE: {rmse:.4f}')
