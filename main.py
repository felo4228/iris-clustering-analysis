"""
Iris Clustering Project
-----------------------

Project goal:
1. Load and explore the Iris dataset.
2. Standardize all numerical features.
3. Estimate the number of clusters using DBSCAN.
4. Estimate the number of clusters using the Elbow Method with K-Means.
5. Apply a final K-Means model using the selected K value.
6. Evaluate the final clustering with the Silhouette Score.
7. Visualize the clusters in 2D.

Note:
The original Iris class labels are intentionally ignored because this project
simulates an unsupervised learning problem.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score


# -------------------------------------------------------------------
# 1. Data Exploration & Preprocessing


def load_and_prepare_data():
    """Load the Iris dataset, create a DataFrame, and standardize the features."""
    iris = load_iris()

    # Create a DataFrame using the original feature names from the dataset.
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    print("\n==============================")
    print("1. DATA EXPLORATION")
    print("==============================")

    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    print("\nGeneral information:")
    print(df.info())

    print("\nDescriptive statistics:")
    print(df.describe())

    print("\nMissing values by column:")
    print(df.isnull().sum())

    # Standardization transforms the features so that they have mean 0
    # and standard deviation 1. This is important for distance-based models.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    print("\nData successfully standardized.")

    return df, X_scaled, iris.feature_names


# -------------------------------------------------------------------
# 2. Estimate K with DBSCAN


def estimate_k_with_dbscan(X_scaled, eps_values=None, min_samples=5):
    """
    Run DBSCAN with different eps values.

    DBSCAN labels noisy points as -1.

    """
    if eps_values is None:
        eps_values = [0.35, 0.45, 0.55]

    results = []

    print("\n==============================")
    print("2. DBSCAN RESULTS")
    print("==============================")

    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)

        unique_labels = set(labels)

        # If -1 exists, it represents noise and should not be counted as a cluster.
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)

        results.append({
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters": n_clusters,
            "n_noise_points": n_noise
        })

        print(f"\neps = {eps}")
        print(f"min_samples = {min_samples}")
        print(f"Detected clusters: {n_clusters}")
        print(f"Noise points: {n_noise}")

    results_df = pd.DataFrame(results)

    print("\nDBSCAN summary:")
    print(results_df)

    return results_df


# -------------------------------------------------------------------
# 3. Estimate K with the Elbow Method


def elbow_method(X_scaled, k_range=range(1, 11)):
    """
    Calculate K-Means inertia/WCSS for different K values.

    Inertia measures how compact the clusters are.
    The goal is to find the 'elbow point', where the improvement starts
    decreasing more slowly.
    """
    inertias = []

    print("\n==============================")
    print("3. ELBOW METHOD")
    print("==============================")

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        print(f"K = {k} | Inertia/WCSS = {kmeans.inertia_:.2f}")

    # Plot the Elbow Method chart.
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, marker="o")
    plt.xlabel("Number of clusters K")
    plt.ylabel("Inertia / WCSS")
    plt.title("Elbow Method for Selecting K")
    plt.xticks(list(k_range))
    plt.grid(True)

    # In the Iris dataset, the elbow is commonly observed around K = 3.
    selected_k = 3
    plt.axvline(x=selected_k, linestyle="--", label=f"Selected K = {selected_k}")
    plt.legend()
    plt.show()

    return inertias, selected_k


# -------------------------------------------------------------------
# 4. Final Application & Visualization


def final_kmeans_clustering(X_scaled, feature_names, selected_k=3):
    """Apply final K-Means, calculate Silhouette Score, and visualize results."""
    print("\n==============================")
    print("4. FINAL K-MEANS CLUSTERING")
    print("==============================")

    kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_

    silhouette = silhouette_score(X_scaled, labels)

    print(f"\nFinal selected K: {selected_k}")
    print(f"Silhouette Score: {silhouette:.4f}")

    #  DataFrame for 2D visualization using the first two standardized features.
    plot_df = pd.DataFrame({
        feature_names[0]: X_scaled[:, 0],
        feature_names[1]: X_scaled[:, 1],
        "cluster": labels
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x=feature_names[0],
        y=feature_names[1],
        hue="cluster",
        palette="viridis",
        s=80
    )

    # Plot the centroids projected onto the first two standardized features.
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=250,
        c="red",
        label="Centroids"
    )

    plt.title("K-Means Clusters on the Iris Dataset")
    plt.xlabel(f"Standardized {feature_names[0]}")
    plt.ylabel(f"Standardized {feature_names[1]}")
    plt.legend()
    plt.grid(True)
    plt.show()

    return kmeans, labels, silhouette


# -------------------------------------------------------------------
# Main execution


def main():
    """Run the complete Iris clustering workflow."""
    df, X_scaled, feature_names = load_and_prepare_data()

    dbscan_results = estimate_k_with_dbscan(
        X_scaled,
        eps_values=[0.35, 0.45, 0.55],
        min_samples=5
    )

    inertias, selected_k = elbow_method(
        X_scaled,
        k_range=range(1, 11)
    )

    kmeans_model, labels, silhouette = final_kmeans_clustering(
        X_scaled,
        feature_names,
        selected_k=selected_k
    )

    print("\n==============================")
    print("PROJECT COMPLETED")
    print("==============================")
    print("Final model: K-Means")
    print(f"Selected K: {selected_k}")
    print(f"Final Silhouette Score: {silhouette:.4f}")


if __name__ == "__main__":
    main()
