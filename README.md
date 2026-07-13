Feature Standardization Requirement for Clustering
Standardizing features using `StandardScaler` is a critical pre-processing constraint for K-Means. K-Means functions by calculating Euclidean distance metric spaces across feature values. Without standardization, features with significantly higher absolute scale ranges (such as `Price` scaling in the hundreds of thousands or `Square_Footage` scaling in thousands) completely dominate metrics compared to smaller range values like `Year_Built`. Scaling transforms feature spaces to have a mean of 0 and variance of 1 (mu=0, sigma^2=1), allowing all features to influence cluster boundaries equally.

Elbow Method vs. Silhouette Score Analysis
* Elbow Method (WCSS*: Measures internal cluster compactness. We search for the "elbow point" or inflection curve where adding more centroids results in diminishing returns in variance reduction.
* Silhouette Analysis: Gauges structural separation boundaries by calculating how close an instance is to its neighboring clusters versus its assigned cluster. A score approaching indicates instances are far away from neighboring clusters, confirming well-defined, distinct property segments.
