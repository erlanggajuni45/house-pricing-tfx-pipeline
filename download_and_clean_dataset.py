import kagglehub
import pandas as pd

path = kagglehub.dataset_download("debayank2024/house-price-prediction")

df = pd.read_csv(f"{path}/modified_data.csv")

cols_to_drop = ['date', 'street', 'price_per_sqft']
df_clean = df.drop(columns=cols_to_drop)

df_clean.to_csv("data/housing_clean.csv", index=False)

print(f"Data tersimpan: {df_clean.shape}")