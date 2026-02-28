import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from utils import load_openaq

print("Loading dataset from OpenAQ (preprocessing)...")
df = load_openaq("../data/openaq_2025.csv")

# -----------------------------
# SELECT FEATURES FOR PCA
# -----------------------------
features = ["pm25","pm10","no2","ozone","temperature","humidity"]

# select feature matrix and drop rows with any missing values
X = df[features].dropna()  # drop rows with missing feature values


# -----------------------------
# STANDARDIZE DATA
# -----------------------------
print("⚙ Standardizing variables...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# APPLY PCA
# -----------------------------
print("🔬 Applying PCA...")

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    data=principal_components,
    columns=["PC1","PC2"]
)

# add zone for clustering
pca_df["zone"] = df["zone"]

# -----------------------------
# EXPLAINED VARIANCE
# -----------------------------
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

# -----------------------------
# LOADINGS (IMPORTANT)
# -----------------------------
loadings = pd.DataFrame(
    pca.components_.T,
    columns=["PC1","PC2"],
    index=features
)

print("\nPCA Loadings:")
print(loadings)

# -----------------------------
# VISUALIZATION
# -----------------------------
print("📊 Creating PCA plot...")

plt.figure(figsize=(8,6))

for zone in pca_df["zone"].unique():
    subset = pca_df[pca_df["zone"] == zone]
    plt.scatter(subset["PC1"], subset["PC2"], label=zone, alpha=0.5)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: Industrial vs Residential Zones")
plt.legend()

plt.show()
