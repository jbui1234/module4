import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

df = pd.read_csv("module_4.csv")

features = df[[
    "Gross_Tertiary_Education_Enrollment", 
    "Youth_15_24_Literacy_Rate_Male", 
    "Youth_15_24_Literacy_Rate_Female", 
    "Unemployment_Rate"
]]

features_clean = features.dropna()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_clean)

inertia = []
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig("elbow_plot.png")
plt.show()
plt.close()

k_optimal = 3
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
df_cleaned = df.loc[features_clean.index]  
df_cleaned['Cluster'] = kmeans.fit_predict(scaled_features)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=scaled_features[:, 0],
    y=scaled_features[:, 1],
    hue=df_cleaned['Cluster'],
    palette='Set1'
)
plt.title("Cluster Visualization: Tertiary Enrollment vs Male Literacy")
plt.xlabel("Gross Tertiary Enrollment (scaled)")
plt.ylabel("Male Literacy Rate (scaled)")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("cluster_visualization.png")
plt.show()
plt.close()

cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_df = pd.DataFrame(cluster_centers, columns=features.columns)
cluster_df['Cluster'] = cluster_df.index
cluster_summary = cluster_df.round(2)
cluster_summary.to_csv("cluster_summary.csv", index=False)

df_cleaned.to_csv("clustered_global_education.csv", index=False)

print("Cluster summary:\n", cluster_summary)

score = silhouette_score(scaled_features, df_cleaned['Cluster'])
print(f"Silhouette Score: {score:.2f}")

from sklearn.metrics import silhouette_samples
import numpy as np

sil_vals = silhouette_samples(scaled_features, df_cleaned['Cluster'])

plt.figure(figsize=(10, 6))
y_lower = 10
for i in range(k_optimal):
    ith_cluster_sil_vals = sil_vals[df_cleaned['Cluster'] == i]
    ith_cluster_sil_vals.sort()
    size_cluster_i = ith_cluster_sil_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_vals)
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {i}')
    y_lower = y_upper + 10

plt.axvline(x=sil_vals.mean(), color="red", linestyle="--")
plt.title("Silhouette Plot for Each Cluster")
plt.xlabel("Silhouette Coefficient")
plt.ylabel("Cluster")
plt.tight_layout()
plt.savefig("silhouette_plot.png")
plt.show()
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(cluster_summary.set_index("Cluster"), annot=True, cmap="coolwarm")
plt.title("Cluster Centers by Feature")
plt.tight_layout()
plt.savefig("cluster_centers_heatmap.png")
plt.show()
plt.close()

cluster_counts = df_cleaned['Cluster'].value_counts().sort_index()
print("Countries per Cluster:\n", cluster_counts)
plt.figure(figsize=(6, 4))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="Set1")
plt.title("Number of Countries in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("cluster_sizes.png")
plt.show()
plt.close()

pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=pca_components[:, 0], y=pca_components[:, 1],
    hue=df_cleaned['Cluster'], palette="Set1"
)
plt.title("PCA Projection of Countries by Cluster")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("pca_clusters.png")
plt.show()
plt.close()
