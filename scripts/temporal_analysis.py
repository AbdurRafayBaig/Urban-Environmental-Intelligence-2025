import pandas as pd
import matplotlib.pyplot as plt

# shared loader for preprocessing raw OpenAQ data
from utils import load_openaq

print("Loading dataset from OpenAQ (preprocessing)...")
df = load_openaq("../data/openaq_2025.csv")

# convert datetime
df["datetime"] = pd.to_datetime(df["datetime"])

# -----------------------------
# HEALTH THRESHOLD VIOLATION
# -----------------------------
print("🚨 Detecting PM2.5 violations...")

threshold = 35

# drop rows where pm25 is missing before computing violation flag
df = df.dropna(subset=["pm25"])
df["violation"] = df["pm25"] > threshold

print("Total violations:", df["violation"].sum())

# -----------------------------
# HOURLY VIOLATIONS (ALL STATIONS)
# -----------------------------
hourly_violations = (
    df.groupby("datetime")["violation"]
    .sum()
    .reset_index()
)

# -----------------------------
# CREATE HEATMAP DATA
# rows = day
# columns = hour
# -----------------------------
heatmap_data = (
    df.groupby(["day","hour"])["pm25"]
    .mean()
    .unstack()
)

# -----------------------------
# VISUALIZATION: HEATMAP
# -----------------------------
print("📊 Creating heatmap...")

plt.figure(figsize=(12,6))
plt.imshow(heatmap_data, aspect='auto')

plt.colorbar(label="Average PM2.5")
plt.xlabel("Hour of Day")
plt.ylabel("Day of Month")
plt.title("PM2.5 Daily Pattern Heatmap")

plt.show()

# -----------------------------
# MONTHLY PATTERN
# -----------------------------
monthly_avg = df.groupby("month")["pm25"].mean()

plt.figure()
monthly_avg.plot(marker='o')
plt.title("Monthly PM2.5 Pattern")
plt.xlabel("Month")
plt.ylabel("Average PM2.5")
plt.show()
