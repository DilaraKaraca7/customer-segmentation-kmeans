import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample data (fake customer data)
data = {
    'Annual Income': [15, 16, 17, 18, 19, 40, 42, 44, 46, 48],
    'Spending Score': [39, 81, 6, 77, 40, 60, 61, 62, 63, 64]
}

df = pd.DataFrame(data)

# K-Means model
kmeans = KMeans(n_clusters=2)
df['Cluster'] = kmeans.fit_predict(df)

# Plot
plt.scatter(df['Annual Income'], df['Spending Score'], c=df['Cluster'])
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.show()
