import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_parquet('data/processed/features.parquet')
X = df[[c for c in df.columns if 'lag' in c or 'rolling' in c]].fillna(0)
y = df['AQI'].fillna(0)

y_pred = X['AQI_lag_1h']
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f'R2: {r2:.4f}')
print(f'RMSE: {rmse:.4f}')
