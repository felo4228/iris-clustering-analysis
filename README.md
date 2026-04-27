# Iris Clustering Project: DBSCAN + Elbow Method + K-Means

This is an unsupervised Machine Learning project using the Iris dataset.

The goal is to explore two different methods for estimating the number of clusters `K`:

1. DBSCAN, by testing different `eps` values and counting the detected clusters and noise points.
2. Elbow Method, by using K-Means and the inertia/WCSS metric to select a plausible value of `K`.

Finally, the project applies K-Means with the selected `K`, calculates the Silhouette Score, and visualizes the clusters in 2D using the first two standardized features.

## Installation

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
```

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

```bash
python main.py
```

The script displays:

- Initial descriptive analysis of the dataset.
- DBSCAN results for `eps = 0.35, 0.45, 0.55` and `min_samples = 5`.
- Elbow Method chart for `K = 1` to `10`.
- Final K-Means clustering.
- Silhouette Score of the final model.
- 2D visualization of the clusters and their centroids.

---

## Dataset

This project uses the Iris dataset included in `scikit-learn`.

The original class labels are ignored in order to simulate a real unsupervised clustering problem.

---

## Main Concepts

### DBSCAN

DBSCAN is a density-based clustering algorithm. It groups points that are close to each other and identifies isolated points as noise.

Important parameters:

- `eps`: maximum distance between two points to be considered neighbors.
- `min_samples`: minimum number of points required to form a dense region.

### Elbow Method

The Elbow Method helps estimate a good value for `K` in K-Means.

It plots the inertia/WCSS for different values of `K`. The best value is usually around the point where the curve starts to bend, known as the elbow point.

### K-Means

K-Means is a centroid-based clustering algorithm. It divides the data into `K` groups by minimizing the distance between each point and its assigned cluster centroid.

### Silhouette Score

The Silhouette Score evaluates how well-separated and internally cohesive the clusters are.

The score ranges from `-1` to `1`:

- Close to `1`: well-separated clusters.
- Around `0`: overlapping clusters.
- Below `0`: points may be assigned to the wrong cluster.

---

## Author

Felipe Gonzalez del Solar.
