data_collection.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Simulate Raw Logistics Dataset with Data Quality Issues
np.random.seed(42)
n_samples = 1000

data = {
    "shipment_id": [f"SHP_{i:04d}" for i in range(n_samples)],
    "distance_km": np.random.exponential(scale=10, size=n_samples),
    "package_weight_kg": np.random.normal(loc=5, scale=2, size=n_samples),
    "transit_time_mins": np.random.normal(loc=35, scale=10, size=n_samples),
    "traffic_index": np.random.randint(1, 10, size=n_samples),
}
df = pd.DataFrame(data)

# Introduce synthetic missing values and outliers
df.loc[np.random.choice(n_samples, 25), "distance_km"] = np.nan
df.loc[np.random.choice(n_samples, 15), "transit_time_mins"] = np.nan
df.loc[10, "distance_km"] = 550.0  # Extreme outlier

# 2. Data Cleaning: Handling Missing Values
df["distance_km"] = df["distance_km"].fillna(df["distance_km"].median())
df["transit_time_mins"] = df["transit_time_mins"].fillna(
    df["transit_time_mins"].median()
)

# 3. Outlier Treatment (IQR Method for distance_km)
Q1 = df["distance_km"].quantile(0.25)
Q3 = df["distance_km"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df["distance_km"] >= lower_bound) & (df["distance_km"] <= upper_bound)]

# 4. Normalization / Feature Scaling
scaler = StandardScaler()
features_to_scale = ["distance_km", "package_weight_kg", "transit_time_mins"]
df[
    [f"{col}_scaled" for col in features_to_scale]
] = scaler.fit_transform(df[features_to_scale])

print("Preprocessing complete. Cleaned shape:", df.shape)
print(df.head(3))
