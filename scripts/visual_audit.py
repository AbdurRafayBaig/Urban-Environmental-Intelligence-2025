import pandas as pd
import matplotlib.pyplot as plt

from utils import load_openaq

print("Loading dataset from OpenAQ (preprocessing)...")
df = load_openaq("../data/openaq_2025.csv")

# average pollution by zone
zone_avg = df.groupby("zone")[["pm25","pm10","no2","ozone"]].mean()

print(zone_avg)

# -----------------------------
# SMALL MULTIPLES
# -----------------------------
pollutants = ["pm25","pm10","no2","ozone"]

for pollutant in pollutants:
    plt.figure()
    zone_avg[pollutant].plot(kind="bar")
    plt.title(f"{pollutant.upper()} Levels by Zone")
    plt.xlabel("Zone")
    plt.ylabel("Average Level")
    plt.show()
