import pandas as pd
import numpy as np
import os

# Load Dataset
df = pd.read_csv("data/raw/bangalore.csv")

print("Dataset Loaded:", df.shape)

# Keep useful columns
df = df[['location', 'size', 'total_sqft', 'bath', 'price']]

# Remove missing values
df.dropna(inplace=True)

# Extract BHK
df['bhk'] = df['size'].str.extract('(\d+)').astype(float)

# Convert sqft
def convert_sqft(x):

    try:

        if '-' in str(x):
            a, b = x.split('-')
            return (float(a) + float(b)) / 2

        return float(x)

    except:
        return np.nan

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)

# Remove nulls
df.dropna(inplace=True)

# Price per sqft
df['price_per_sqft'] = (
    df['price'] * 100000
) / df['total_sqft']

# Remove outliers
df = df[
    (df['price_per_sqft'] > 1000) &
    (df['price_per_sqft'] < 30000)
]

# Remove tiny properties
df = df[df['total_sqft'] > 300]

# Remove huge properties
df = df[df['total_sqft'] < 10000]

# Luxury score
df['luxury_score'] = (
    df['total_sqft'] / df['bhk']
)

# Bath ratio
df['bath_per_bhk'] = (
    df['bath'] / df['bhk']
)

# Save
os.makedirs("data", exist_ok=True)

df.to_csv("data/cleaned_data.csv", index=False)

print("Cleaned Data Saved")
print(df.shape)