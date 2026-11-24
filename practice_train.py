import pandas as pd
import numpy as np
import pickle

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ---------------------------------------
# LOAD DATASET
# ---------------------------------------

df = pd.read_csv("data/geo_data.csv")

print("Dataset Loaded:", df.shape)

# ---------------------------------------
# REMOVE BAD DATA
# ---------------------------------------

df = df[df['total_sqft'] > 300]

df = df[df['total_sqft'] < 10000]

df = df[df['bath'] <= 10]

df = df[df['bhk'] <= 10]

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
# LOCATION PRICE MAP
# ---------------------------------------

location_price_map = (

    df.groupby('location')['price_per_sqft']
    .mean()
)

# SAVE LOCATION PRICE MAP

location_price_map.to_csv(
    "models/location_price_map.csv"
)

# ---------------------------------------
# AREA AVG PRICE
# ---------------------------------------

df['area_avg_price'] = (
    df['location']
    .map(location_price_map)
)

# ---------------------------------------
# FEATURES
# ---------------------------------------

features = [

    'total_sqft',
    'bath',
    'bhk',
    'price_per_sqft',
    'luxury_score',
    'bath_per_bhk',
    'latitude',
    'longitude',
    'area_avg_price'
]

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

    iterations=5000,

    learning_rate=0.03,

    depth=10,

    loss_function='RMSE',

    eval_metric='R2',

    verbose=200
)

# ---------------------------------------
# TRAIN MODEL
# ---------------------------------------

print("Training Started...")

model.fit(X_train, y_train)

print("Training Completed")

# ---------------------------------------
# PREDICTION
# ---------------------------------------

preds = model.predict(X_test)

# ---------------------------------------
# ACCURACY
# ---------------------------------------

score = r2_score(y_test, preds)

print()

print("🔥 MODEL ACCURACY (R2 SCORE):", score)

print()

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

print("✅ MODEL SAVED")