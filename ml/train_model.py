import pandas as pd
import numpy as np
import pickle
import os

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ---------------------------------------
# CREATE MODELS DIRECTORY
# ---------------------------------------

os.makedirs("models", exist_ok=True)

# ---------------------------------------
# LOAD DATASET
# ---------------------------------------

df = pd.read_csv("data/geo_data.csv")

print("✅ Dataset Loaded:", df.shape)

# ---------------------------------------
# REMOVE BAD DATA
# ---------------------------------------

df = df[df['total_sqft'] > 300]

df = df[df['total_sqft'] < 10000]

df = df[df['bath'] <= 10]

df = df[df['bhk'] <= 10]

# ---------------------------------------
# REMOVE NULL VALUES
# ---------------------------------------

df.dropna(inplace=True)

# ---------------------------------------
# PRICE PER SQFT
# ---------------------------------------

df['price_per_sqft'] = (
    df['price'] * 100000
) / df['total_sqft']

# ---------------------------------------
# REMOVE OUTLIERS
# ---------------------------------------

df = df[
    (df['price_per_sqft'] > 1000) &
    (df['price_per_sqft'] < 30000)
]

print("✅ Outliers Removed")

# ---------------------------------------
# LUXURY FEATURES
# ---------------------------------------

df['luxury_score'] = (
    df['total_sqft'] / df['bhk']
)

df['bath_per_bhk'] = (
    df['bath'] / df['bhk']
)

# ---------------------------------------
# GEO FEATURES
# ---------------------------------------

CENTER_LAT = 12.9716
CENTER_LON = 77.5946

# Distance From Center

df['distance_center'] = np.sqrt(

    (df['latitude'] - CENTER_LAT) ** 2 +

    (df['longitude'] - CENTER_LON) ** 2
)

# Trend Score

df['trend_score'] = 10 - (
    df['distance_center'] * 10
)

df['trend_score'] = df['trend_score'].clip(1, 10)

# Schools

df['schools'] = (
    10 - (df['distance_center'] * 5)
).clip(1, 10).astype(int)

# Hospitals

df['hospitals'] = (
    6 - (df['distance_center'] * 3)
).clip(1, 6).astype(int)

print("✅ Geo Features Added")

# ---------------------------------------
# LOCATION DENSITY
# ---------------------------------------

location_counts = df['location'].value_counts()

df['location_density'] = df['location'].map(
    location_counts
)

# ---------------------------------------
# PREMIUM LOCATION SCORE
# ---------------------------------------

premium_locations = [

    'Indiranagar',

    'Koramangala',

    'Whitefield',

    'HSR Layout',

    'Electronic City',

    'Marathahalli',

    'Sarjapur Road',

    'JP Nagar',

    'Jayanagar'
]

df['premium_location'] = df['location'].apply(

    lambda x: 1 if x in premium_locations else 0
)

# ---------------------------------------
# LOG SQFT
# ---------------------------------------

df['log_sqft'] = np.log(
    df['total_sqft']
)

# ---------------------------------------
# BHK DENSITY
# ---------------------------------------

df['bhk_density'] = (
    df['total_sqft'] / df['bhk']
)

# ---------------------------------------
# BATH LUXURY SCORE
# ---------------------------------------

df['bath_luxury_score'] = (
    df['bath'] * df['bhk']
)

# ---------------------------------------
# ROOM DENSITY
# ---------------------------------------

df['room_density'] = (

    df['total_sqft']

    /

    (df['bhk'] + df['bath'])
)

# ---------------------------------------
# FAMILY SCORE
# ---------------------------------------

df['family_score'] = (

    (df['bhk'] * 2)

    +

    df['bath']
)

# ---------------------------------------
# LOCATION PRICE MAP
# ---------------------------------------

location_price_map = (

    df.groupby('location')['price_per_sqft']
    .mean()
)

# ---------------------------------------
# SAVE LOCATION PRICE MAP
# ---------------------------------------

location_price_map.to_csv(
    "models/location_price_map.csv"
)

print("✅ Location Price Map Saved")

# ---------------------------------------
# AREA AVG PRICE
# ---------------------------------------

df['area_avg_price'] = (
    df['location']
    .map(location_price_map)
)

# ---------------------------------------
# INVESTMENT SCORE
# ---------------------------------------

df['investment_score'] = (

    df['trend_score']

    *

    df['area_avg_price']

) / 1000

# ---------------------------------------
# LOCATION ENCODING
# ---------------------------------------

df = pd.get_dummies(

    df,

    columns=['location']
)

print("✅ Location Encoding Done")

# ---------------------------------------
# BASE FEATURES
# ---------------------------------------

base_features = [

    'total_sqft',

    'bath',

    'bhk',

    'luxury_score',

    'bath_per_bhk',

    'latitude',

    'longitude',

    'area_avg_price',

    'schools',

    'hospitals',

    'distance_center',

    'trend_score',

    'location_density',

    'premium_location',

    'log_sqft',

    'bhk_density',

    'bath_luxury_score',

    'room_density',

    'family_score',

    'investment_score'
]

# ---------------------------------------
# LOCATION COLUMNS
# ---------------------------------------

location_columns = [

    col for col in df.columns

    if col.startswith("location_")
]

# ---------------------------------------
# FINAL FEATURES
# ---------------------------------------

features = list(

    dict.fromkeys(

        base_features +

        location_columns
    )
)

print()
print("🔥 TOTAL FEATURES:", len(features))
print()

# ---------------------------------------
# CHECK DUPLICATES
# ---------------------------------------

duplicates = [

    x for x in features

    if features.count(x) > 1
]

print("✅ DUPLICATES:", duplicates)

# ---------------------------------------
# CREATE X AND Y
# ---------------------------------------

X = df[features]

y = df['price']

# ---------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)

# ---------------------------------------
# CATBOOST MODEL
# ---------------------------------------

model = CatBoostRegressor(

    iterations=3000,

    learning_rate=0.03,

    depth=10,

    loss_function='RMSE',

    eval_metric='R2',

    verbose=200
)

# ---------------------------------------
# TRAIN MODEL
# ---------------------------------------

print()
print("🚀 Training Started...")
print()

model.fit(
    X_train,
    y_train
)

print()
print("✅ Training Completed")
print()

# ---------------------------------------
# PREDICTION
# ---------------------------------------

preds = model.predict(X_test)

# ---------------------------------------
# ACCURACY
# ---------------------------------------

score = r2_score(
    y_test,
    preds
)

print()
print("🔥 MODEL ACCURACY (R2 SCORE):", score)
print()

# ---------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------

importance_df = pd.DataFrame({

    'Feature': features,

    'Importance': model.feature_importances_
})

importance_df = importance_df.sort_values(

    by='Importance',

    ascending=False
)

print("🔥 TOP IMPORTANT FEATURES")
print()

print(
    importance_df.head(20)
)

# ---------------------------------------
# SAVE MODEL
# ---------------------------------------

pickle.dump(

    model,

    open(
        "models/house_price_model.pkl",
        "wb"
    )
)

# ---------------------------------------
# SAVE COLUMNS
# ---------------------------------------

pickle.dump(

    features,

    open(
        "models/columns.pkl",
        "wb"
    )
)

print()
print("✅ MODEL SAVED")
print("✅ COLUMNS SAVED")