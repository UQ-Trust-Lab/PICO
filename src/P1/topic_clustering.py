from bertopic import BERTopic
from hdbscan import HDBSCAN
import numpy as np
import pandas as pd
import hdbscan
from cuml.cluster import HDBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# load data
datafile_path = "../../Dataset/all_skill_with_openai_embedding.csv"

df = pd.read_csv(datafile_path)
df["embeddings"] = df.embeddings.apply(eval).apply(np.array)  # convert string to numpy array
matrix = np.vstack(df.embeddings.values)
matrix.shape

hdbscan_cluster = HDBSCAN(min_cluster_size=23, min_samples=2)

# fit the clusterer to the data
cluster_labels = hdbscan_cluster.fit_predict(matrix)

df["cluster_hdbscan"] = cluster_labels
print(df["cluster_hdbscan"].value_counts())

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(n_clusters)


# Run HDBSCAN
hdbscan_instance = HDBSCAN(min_cluster_size=23,min_samples=2)
cluster_labels = hdbscan_cluster.fit_predict(matrix)
hdbscan_instance.fit(matrix)

df["cluster_labels_previous"] = cluster_labels
df["cluster_labels_reassigned"] = cluster_labels

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

file_path = "../../Result/P1/topic_clustering_result.csv"
df.to_csv(file_path)


distance_matrix = pairwise_distances(matrix)

mask = df['cluster_labels_reassigned'] == -1

row_indices = np.where(mask)[0]

# Calculate the nearest cluster labels for the selected rows
nearest_cluster_indices = np.argmin(distance_matrix[row_indices][:, hdbscan_instance.labels_ != -1], axis=1)
nearest_cluster_labels = hdbscan_instance.labels_[hdbscan_instance.labels_ != -1][nearest_cluster_indices]


df.loc[mask, 'cluster_labels_reassigned'] = nearest_cluster_labels

print(df)
df.to_csv(file_path)
