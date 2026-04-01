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
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

# 1. SETUP: Paths and Samples
DATA_PATH = r'C:\Users\Hemant\OneDrive\Documents\car_prices.csv'
IMAGE_DIR = 'images'
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# LOAD DATA
df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
df = df.sample(n=10000, random_state=42)

# 2. UNIT I: Data Preparation & Preprocessing
print("Unit I: Preparing Data...")
numeric_cols = ['year', 'condition', 'odometer', 'mmr', 'sellingprice']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=numeric_cols)

# EDA VISUALS
plt.figure(figsize=(10, 6))
sns.distplot(df['sellingprice'], color='blue')
plt.title("Car Price Distribution")
plt.savefig('images/price_distribution.png')
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.savefig('images/correlation_heatmap.png')
plt.close()

# 3. UNIT II: Supervised Learning - Regression
print("Unit II: Training Regression Models...")
X = df[['year', 'condition', 'odometer', 'mmr']]
y = df['sellingprice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multiple Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
preds_lr = model_lr.predict(X_test)

# REGRESSION PLOT
plt.figure(figsize=(10, 6))
plt.scatter(y_test, preds_lr, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title("Regression: Actual vs Predicted Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.savefig('images/regression_results.png')
plt.close()

# 4. UNIT III: Supervised Learning - Classification
print("Unit III: Training Classification Models...")
df['high_price'] = (df['sellingprice'] > df['sellingprice'].median()).astype(int)
yc = df['high_price']
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, yc, test_size=0.2, random_state=42)

# Decision Tree Classifier
model_dt = DecisionTreeClassifier(max_depth=3)
model_dt.fit(Xc_train, yc_train)

# CLASSIFICATION VISUALS
plt.figure(figsize=(10, 8))
cm = confusion_matrix(yc_test, model_dt.predict(Xc_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Classification Confusion Matrix")
plt.savefig('images/confusion_matrix.png')
plt.close()

plt.figure(figsize=(15, 10))
plot_tree(model_dt, feature_names=X.columns, class_names=['Normal', 'High'], filled=True)
plt.title("Decision Tree Visualization")
plt.savefig('images/decision_tree_viz.png')
plt.close()

# 5. UNIT IV: Unsupervised Learning - Clustering
print("Unit IV: Unsupervised Learning...")
# K-Means Elbow Method
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for K-Means")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.savefig('images/elbow_method.png')
plt.close()

# Hierarchical Dendrogram (Small Sample)
X_small = X.iloc[:50, :]
Z = linkage(X_small, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.savefig('images/dendrogram.png')
plt.close()

# 6. UNIT V: Dimensionality Reduction & NNs
print("Unit V: Dimensions & Neural Networks...")
# PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(X)
plt.figure(figsize=(10, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=df['sellingprice'], cmap='viridis', alpha=0.5)
plt.title("PCA Projection of Car Features")
plt.colorbar(label="Price")
plt.savefig('images/pca_projection.png')
plt.close()

# Neural Network (MLP)
mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
mlp.fit(X_train, y_train)

# 7. UNIT VI: Model Performance & Ensembles
print("Unit VI: Performance Analysis...")
# Random Forest importance
rf = RandomForestRegressor(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)

plt.figure(figsize=(10, 6))
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.nlargest(4).plot(kind='barh', color='purple')
plt.title("Random Forest - Feature Importance")
plt.savefig('images/feature_importance.png')
plt.close()

print(f"Final Random Forest Score: {rf.score(X_test, y_test):.2f}")
print("ALL UNITS COMPLETED AND 10+ VISUALS SAVED IN 'images' FOLDER.")
