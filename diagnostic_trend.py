import pandas as pd
import numpy as np
import os
from sklearn.linear_model import Ridge

# Set pythonpath for imports to work
import sys

sys.path.append(os.getcwd())

from src.features.level import LevelTransformer
from src.features.dates import DatePartTransformer

# 1. Mock data (growing trend)
dates = pd.date_range("2013-01-01", periods=100)
df = pd.DataFrame(
    {
        "date": dates,
        "store_nbr": [1] * 100,
        "family": ["FOOD"] * 100,
        "sales": 100 + np.arange(100) * 10,  # 100, 110, 120...
    }
)

# 2. Transformers
df = DatePartTransformer(parts=["time_idx"], column="date").fit_transform(df)
df = LevelTransformer(window=30, target_col="sales", log_transform=True).fit_transform(
    df
)

# Drop rows where Level is invalid (shift 16)
df = df.dropna(subset=["level_sales_30"])

# 3. Fit Ridge
X = df[["time_idx", "level_sales_30"]]
y = np.log1p(df["sales"])

model = Ridge()
model.fit(X, y)

print("\nRidge Coefficients:", model.coef_)
print("Ridge Intercept:", model.intercept_)

y_pred = model.predict(X)
print("\nLog Y (Actual) vs Log Y (Trend Pred):")
for act, pred in zip(y.iloc[16:21], y_pred[16:21]):
    print(f"Act: {act:.4f}, Pred: {pred:.4f}, Res: {act - pred:.4f}")
