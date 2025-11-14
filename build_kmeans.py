# build_kmeans_india.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA = Path("data/processed")
OUT  = Path("data/clustered")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "india_daily.csv")
pollutants = ['PM2.5','PM10','NO2','SO2','CO','O3']
avail = [c for c in pollutants if c in df.columns]

city_means = df.groupby('City')[avail].mean().fillna(0).reset_index()

scaler = StandardScaler()
X = scaler.fit_transform(city_means[avail])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
city_means['cluster'] = kmeans.fit_predict(X)
city_means['avg_pollution'] = city_means[avail].mean(axis=1)

city_means = city_means.sort_values('cluster')
city_means.to_csv(OUT / "india_city_clusters.csv", index=False)

print("Nation-wide city clusters:")
print(city_means[['City','cluster','avg_pollution']].round(2))

# Plot
plt.figure(figsize=(10,6))
sns.scatterplot(data=city_means, x='PM2.5', y='PM10',
                hue='cluster', palette='viridis', s=120)
plt.title('K-Means Clusters of Indian Cities (PM2.5 vs PM10)')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig(OUT / "india_clusters.png")
plt.show()