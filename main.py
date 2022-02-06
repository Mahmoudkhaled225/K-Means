"""
THE SPARKS FOUNDATION
DATA SCIENCE and BUSINESS ANALYTICS INTERNSHIP (CURRENT PATCH)
Task 2 --> Prediction Number Of Clusters And Visualised
Note
i did all the analysisphase and i got my insights and started perform the algorithm
"""

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


# Importing the dataset locally from my machine
iris = pd.read_csv(r"C:\Users\Eltaysser\Downloads\Iris.csv")

x = iris.iloc[:, :-1].values  # last column values excluded
y = iris.iloc[:, -1].values  # last column value

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=100, n_init=10, random_state=0)  # Applying Kmeans classifier
y_kmeans = kmeans.fit_predict(x)
print(kmeans.cluster_centers_)  # display cluster centers

# Visualising the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c="red", label="Iris-setosa")
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c="blue", label="Iris-versicolour")
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c="green", label="Iris-virginica")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Centroids')
plt.legend()
plt.show()
