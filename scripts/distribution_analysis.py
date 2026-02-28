import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# shared loader handles pivot and cleaning of raw OpenAQ file
from utils import load_openaq

print("Loading dataset from OpenAQ (this may take a moment)...")
df = load_openaq("../data/openaq_2025.csv")

# -----------------------------
# SELECT INDUSTRIAL ZONE
# -----------------------------
industrial = df[df["zone"] == "Industrial"]

pm25 = industrial["pm25"].dropna()

# -----------------------------
# EXTREME HAZARD EVENTS
# -----------------------------
extreme_events = pm25[pm25 > 200]

print("🚨 Extreme Hazard Events (>200):", len(extreme_events))

# -----------------------------
# 99th PERCENTILE
# -----------------------------
p99 = np.percentile(pm25, 99)
print("📊 99th Percentile:", round(p99,2))

# -----------------------------
# PLOT 1: HISTOGRAM (PEAK VIEW)
# -----------------------------
plt.figure()
plt.hist(pm25, bins=50)
plt.title("PM2.5 Distribution (Peak View)")
plt.xlabel("PM2.5")
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# PLOT 2: LOG SCALE HISTOGRAM (TAIL VIEW)
# -----------------------------
plt.figure()
plt.hist(pm25, bins=50, log=True)
plt.title("PM2.5 Distribution (Tail View - Log Scale)")
plt.xlabel("PM2.5")
plt.ylabel("Log Frequency")
plt.show()
