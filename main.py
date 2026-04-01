import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# 1. Setup paths
DATA_PATH = r'C:\Users\Hemant\OneDrive\Documents\car_prices.csv'
IMAGE_DIR = 'images'
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# 2. Data Preparation (Unit I)
df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
df = df.sample(n=10000, random_state=42)

# Convert only necessary columns to numeric, bad data becomes NaN
numeric_cols = ['year', 'condition', 'odometer', 'mmr', 'sellingprice']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN in numeric columns
df = df.dropna(subset=numeric_cols)

# Simple encoding
le = LabelEncoder()
df['make_encoded'] = le.fit_transform(df['make'].astype(str))

# Create correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Map")
plt.savefig('images/correlation.png')
plt.close()

# 3. Regression Models (Unit II)
X = df[['year', 'condition', 'odometer', 'mmr']]
y = df['sellingprice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple & Multiple Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
print(f"Regression Accuracy (R2): {model_lr.score(X_test, y_test):.2f}")

# 4. Classification (Unit III)
# Is price > median? (Binary Target)
df['high_price'] = (df['sellingprice'] > df['sellingprice'].median()).astype(int)
yc = df['high_price']
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, yc, test_size=0.2, random_state=42)

# Using a Decision Tree
model_dt = DecisionTreeClassifier(max_depth=5)
model_dt.fit(Xc_train, yc_train)
print(f"Classification Accuracy: {accuracy_score(yc_test, model_dt.predict(Xc_test)):.2f}")

# Confusion Matrix Image
cm = confusion_matrix(yc_test, model_dt.predict(Xc_test))
sns.heatmap(cm, annot=True, fmt='d')
plt.savefig('images/confusion_matrix.png')
plt.close()

# 5. Clustering (Unit IV)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Association Rules
df_rules = pd.get_dummies(df[['make', 'high_price']])
freq_sets = apriori(df_rules, min_support=0.1, use_colnames=True)
res_rules = association_rules(freq_sets, metric="lift", min_threshold=1)
print(res_rules.head(3))

# 6. Dimensions and Neural Networks (Unit V)
# PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=df['sellingprice'])
plt.savefig('images/pca.png')
plt.close()

# Neural Network (MLP)
mlp = MLPRegressor(hidden_layer_sizes=(10,), max_iter=500)
mlp.fit(X_train, y_train)

# 7. Model Performance (Unit VI)
# Random Forest Ensemble
rf = RandomForestRegressor(n_estimators=50)
rf.fit(X_train, y_train)
print(f"Random Forest Result: {rf.score(X_test, y_test):.2f}")

print("Work Completed! Images saved in 'images' folder.")
