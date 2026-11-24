import pandas as pd
from geopy.geocoders import Nominatim
import time

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv("data/cleaned_data.csv")

print("✅ Dataset Loaded")

# -----------------------------
# UNIQUE LOCATIONS ONLY
# -----------------------------

unique_locations = df['location'].unique()

print("Total Unique Locations:", len(unique_locations))

# -----------------------------
# GEOLOCATOR
# -----------------------------

geolocator = Nominatim(user_agent="real_estate_ai")

# -----------------------------
# LOCATION CACHE
# -----------------------------

location_dict = {}

# -----------------------------
# FETCH LAT/LON
# -----------------------------

for loc in unique_locations:

    try:

        geo = geolocator.geocode(f"{loc}, Bangalore")

        if geo:

            location_dict[loc] = {
                "latitude": geo.latitude,
                "longitude": geo.longitude
            }

            print(f"✅ {loc}")

        else:

            location_dict[loc] = {
                "latitude": None,
                "longitude": None
            }

        # avoid rate limit
        time.sleep(1)

    except Exception as e:

        print(f"❌ Error: {loc}")

        location_dict[loc] = {
            "latitude": None,
            "longitude": None
        }

# -----------------------------
# MAP VALUES BACK
# -----------------------------

df['latitude'] = df['location'].map(
    lambda x: location_dict[x]['latitude']
)

df['longitude'] = df['location'].map(
    lambda x: location_dict[x]['longitude']
)

# -----------------------------
# REMOVE NULLS
# -----------------------------

df.dropna(inplace=True)

# -----------------------------
# SAVE
# -----------------------------

df.to_csv("data/geo_data.csv", index=False)

print("✅ Geo dataset created!")
print("Final Shape:", df.shape)