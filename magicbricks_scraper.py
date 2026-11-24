import requests
from bs4 import BeautifulSoup
import pandas as pd
from fake_useragent import UserAgent
import time

ua = UserAgent()

headers = {
    "User-Agent": ua.random
}

url = "https://www.magicbricks.com/property-for-sale/residential-real-estate?cityName=Bangalore"

response = requests.get(url, headers=headers)

print(response.status_code)

soup = BeautifulSoup(response.text, "lxml")

properties = []

cards = soup.find_all("div", class_="mb-srp__card")

for card in cards:
    try:
        title = card.find("h2").text.strip()

        price = card.find("div", class_="mb-srp__card__price--amount").text.strip()

        area = card.find("div", class_="mb-srp__card__summary--value").text.strip()

        properties.append({
            "title": title,
            "price": price,
            "area": area
        })

    except:
        pass

df = pd.DataFrame(properties)

print(df.head())

df.to_csv("magicbricks_data.csv", index=False)

print("Saved Successfully")