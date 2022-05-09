import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv('data/Mall_Customers.csv')
sns.set_theme(style='whitegrid')

# Visualization of gender distribution
genders = dataset.Gender.value_counts()

# Plot data on scatter plot
sns.scatterplot(x='Age', y='Spending Score (1-100)', data=dataset)

# Plot gender visualization result on bar chart
plt.figure(figsize=(7, 6))

sns.barplot(x=genders.index, y=genders.values)

# plt.show()

# Visualize customers by age groups
age18_25 = dataset.Age[(dataset.Age >= 18) & (dataset.Age <= 25)]
age26_35 = dataset.Age[(dataset.Age >= 26) & (dataset.Age <= 35)]
age36_45 = dataset.Age[(dataset.Age >= 36) & (dataset.Age <= 45)]
age46_55 = dataset.Age[(dataset.Age >= 46) & (dataset.Age <= 55)]
ageAbove55 = dataset.Age[(dataset.Age >= 56)]

x = ['18-25', '26-35', '36-45', '46-55', 'Above 55']
y = [len(age18_25.values), len(age26_35.values), len(age36_45.values), len(age46_55.values), len(ageAbove55.values)]

plt.figure(figsize=(7, 6))
plt.title = 'Customers by age groups'
plt.xlabel('Age groups')
plt.ylabel('Number of customers')
sns.barplot(x=x, y=y)
# plt.show()

# Visualize customers by spending score
ss = 'Spending Score (1-100)'

# ss1_20 = []
# ss = dataset.loc[:, 'Spending Score (1-100)'].values
# for i in range(ss.shape[0]):
#     val = ss[i]
#     if 1 <= val <= 20:
#         ss1_20.append(val)
#
#     i += 1
#
# ss1_20 = ((ss >= 1) & (ss <= 20))

ss1_20 = dataset[ss][(dataset[ss] >= 1) & (dataset[ss] <= 20)]
ss21_40 = dataset[ss][(dataset[ss] >= 21) & (dataset[ss] <= 40)]
ss41_60 = dataset[ss][(dataset[ss] >= 41) & (dataset[ss] <= 60)]
ss61_80 = dataset[ss][(dataset[ss] >= 61) & (dataset[ss] <= 80)]
ss81_100 = dataset[ss][(dataset[ss] >= 81) & (dataset[ss] <= 100)]

x = ['1-20', '21-40', '41-60', '61-80', '81-100']
y = [len(ss1_20.values), len(ss21_40.values), len(ss41_60.values), len(ss61_80.values), len(ss81_100.values)]

plt.figure(figsize=(7, 6))
plt.title = 'Customers by spending score'
plt.xlabel('Spending score')
plt.ylabel('Number of customers')
sns.barplot(x=x, y=y)
# plt.show()

# Visualize customers by annual income
ai = 'Annual Income (k$)'

ai0_30 = dataset[ai][(dataset[ai] >= 0) & (dataset[ai] <= 30)]
ai31_60 = dataset[ai][(dataset[ai] >= 31) & (dataset[ai] <= 60)]
ai61_90 = dataset[ai][(dataset[ai] >= 61) & (dataset[ai] <= 90)]
ai91_120 = dataset[ai][(dataset[ai] >= 91) & (dataset[ai] <= 120)]
ai121_150 = dataset[ai][(dataset[ai] >= 121) & (dataset[ai] <= 150)]

x = ['0-30', '31-60', '61-90', '91-120', '121-150']
y = [len(ai0_30.values), len(ai31_60.values), len(ai61_90.values), len(ai91_120.values), len(ai121_150.values)]

plt.figure(figsize=(7, 6))
plt.title = 'Customers by annual income'
plt.xlabel('Annual income (k$)')
plt.ylabel('Number of customers')
sns.barplot(x=x, y=y)
# plt.show()

# Elbow Method (used to find the value of k for KMeans Clustering)
"""
Elbow is the common method used to determine the best value of K. 
This method calculates the variance between data points within a cluster using the Sum of Squared Error. 
The best value of k to select is the point of inflection on the curve.
"""
x = dataset.drop(columns=['CustomerID', 'Gender', 'Age'])
print(x.shape)

wcss = []  # within cluster sum of squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(1, 11), wcss)
plt.xlabel('KValue')
plt.xticks(np.arange(1, 11, 1))
plt.ylabel('WCSS')
plt.show()

# KMeans Algorithm
df = pd.read_csv('data/Mall_Customers.csv')
X = df.drop(columns=['CustomerID'])

# Convert Gender column to binary
X['Gender'].replace({'Female': 0, 'Male': 1}, inplace=True)

kmeansModel = KMeans(n_clusters=5, init='k-means++', random_state=0)
y_prediction = kmeansModel.fit_predict(X)

fig = px.scatter_3d(X, x=X['Age'], y=X['Annual Income (k$)'], z=X['Spending Score (1-100)'],
                    color=y_prediction, labels={'x': 'Age', 'y': 'Annual Income (k$)', 'z': 'Spending Score (1-100)'})
fig.show()

"""
# One-Hot Encoding on Gender
encoder = OneHotEncoder(handle_unknown="ignore")

# Perform one-hot encoding on 'team' column
encoder_df = pd.DataFrame(encoder.fit_transform(df[['Gender']]).toarray())

# Merge one-hot encoded columns back with original DataFrame
final_df = df.join(encoder_df)

final_df = final_df.drop(columns=['Gender'])
# plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=80, c='red', label='customer1')
# plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=80, c='blue', label='customer2')
# plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=80, c='green', label='customer3')
# plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s=80, c='orange', label='customer4')
# plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s=80, c='pink', label='customer5')
# plt.scatter(kmeansModel.cluster_centers_[:, 0], kmeansModel.cluster_centers_[:, 1], s=100, c='black', label='centroids')
# plt.title = 'Cluster of customers'
# plt.xlabel('Annual Income')
# plt.ylabel('Spending Scores')
# plt.legend()
# plt.show()
"""