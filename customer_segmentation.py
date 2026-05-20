import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("Mall_Customers.csv")
X = df[['Age', 'Spending Score (1-100)']]

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

plt.scatter(df['Age'], df['Spending Score (1-100)'], c=df['Cluster'])
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")
plt.show()
