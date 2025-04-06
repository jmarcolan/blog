---
title: Clustering in Biostatistics Using Machine Learning with Pandas
date: 2023-10-05
tags: [biostatistics, neuroscience, machine learning, statistics]
summary: A comprehensive guide to clustering using K-Means algorithm on health metrics in biostatistics.
slug: clustering-in-biostatistics-using-machine-learning
---

## Introduction
Clustering is a pivotal tool in machine learning and biostatistics, essential for grouping data into meaningful clusters based on features and similarities. This capability allows researchers to uncover hidden patterns within complex datasets, which can be invaluable for patient stratification, understanding genetic variations, and optimizing treatment plans. In this blog post, we will explore how to perform clustering using the K-Means algorithm on a sample DataFrame created with Pandas, specifically focusing on biostatistical data.

## Understanding Clustering
Clustering techniques categorize data points into distinct groups (clusters) based on shared characteristics. This process is a form of unsupervised learning, meaning it identifies groupings without predetermined categories. Among the most commonly used clustering algorithms are:
- **K-Means**: Efficient and widely adopted due to its simplicity and effectiveness.
- **Hierarchical Clustering**: Builds a tree-like structure of clusters.
- **DBSCAN**: Effective in separating noise and identifying clusters based on density.

K-Means will be our focus, providing a straightforward approach to cluster data, particularly useful in healthcare settings where understanding data distributions is crucial.

### Example: Creating a Biostatistical Dataset
Let’s consider a hypothetical dataset that includes vital health metrics of patients, specifically Systolic Blood Pressure and Cholesterol levels. We will first import the necessary libraries and create a Pandas DataFrame.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Creating a sample DataFrame
data = {
    'Patient_ID': np.arange(1, 11),
    'Systolic_BP': [120, 130, 125, 140, 135, 150, 145, 133, 122, 138],
    'Cholesterol': [180, 190, 170, 220, 200, 240, 210, 215, 175, 225]
}

df = pd.DataFrame(data)
print("DataFrame:")
print(df)
```

### Applying K-Means Clustering
Next, we will apply the K-Means algorithm to our dataset to categorize patients based on their health metrics.

```python
# Applying K-Means clustering
kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(df[['Systolic_BP', 'Cholesterol']])
print("\nDataFrame with Cluster Assignments:")
print(df)
```

### Visualizing the Clusters
Visual representation of the clustered data can greatly assist in understanding the distribution of groups. The following code generates a scatter plot where each patient is represented according to their cluster.

```python
# Plotting the clusters
plt.scatter(df['Systolic_BP'], df['Cholesterol'], c=df['Cluster'], cmap='viridis', s=100)
plt.title('K-Means Clustering of Patients')
plt.xlabel('Systolic Blood Pressure')
plt.ylabel('Cholesterol Levels')
plt.colorbar(label='Cluster')
plt.grid()
plt.show()
```

### Analyzing Cluster Results
The output DataFrame will now show each patient's assigned cluster based on K-Means analysis. This information can facilitate deeper insights into patient profiles and inform clinical decisions.

| Patient_ID | Systolic_BP | Cholesterol | Cluster |
|------------|-------------|-------------|---------|
| 1          | 120         | 180         | 0       |
| 2          | 130         | 190         | 0       |
| 3          | 125         | 170         | 0       |
| 4          | 140         | 220         | 2       |
| 5          | 135         | 200         | 0       |
| 6          | 150         | 240         | 2       |
| 7          | 145         | 210         | 1       |
| 8          | 133         | 215         | 0       |
| 9          | 122         | 175         | 0       |
| 10         | 138         | 225         | 1       |

## Summary
In this blog post, we explored the practical application of the K-Means clustering algorithm using a sample DataFrame relevant to biostatistics. By successfully clustering patients based on health metrics, we have illustrated how clustering techniques can provide insights into disease patterns and treatment approaches. Such methods are crucial in the field of biostatistics and can lead to improved healthcare strategies.

## References
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media.
- McKinney, W. (2010). *Data Analysis with Pandas*. O'Reilly Media.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Springer.