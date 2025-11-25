import numpy as np
from allyouneed.linear_model import LogisticRegression, LinearRegression
from allyouneed.neighbors import KNeighborsClassifier, KNeighborsRegressor
from allyouneed.tree import DecisionTreeClassifier, DecisionTreeRegressor
from allyouneed.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from allyouneed.svm import SVC, SVR
from allyouneed.cluster import KMeans
from allyouneed.decomposition import PCA, TruncatedSVD
from allyouneed.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

CLASSIFIER = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "DecisionTree": DecisionTreeClassifier(max_depth=5),
    "RandomForest": RandomForestClassifier(n_estimators=10, max_depth=5),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=10, max_depth=5),
    "SVC": SVC(kernel='linear', C=1.0),
    "LogisticRegression": LogisticRegression(max_iter=200),
}

REGRESSOR = {
    "KNN": KNeighborsRegressor(n_neighbors=3),
    "DecisionTree": DecisionTreeRegressor(max_depth=5),
    "RandomForest": RandomForestRegressor(n_estimators=10, max_depth=5),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=10, max_depth=5),
    "SVR": SVR(kernel='linear', C=1.0),
    "LinearRegression": LinearRegression(),
}

CLUSTER = {
    "KMeans": KMeans(n_clusters=2),
}

DECOMPOSITION = {
    "PCA": PCA(n_components=2),
    "TruncatedSVD": TruncatedSVD(n_components=2),
}

PREPROCESSING = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
}

def main():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([0, 1, 0, 1])

    print("=== CLASSIFIERS ===")
    for name, model in CLASSIFIER.items():
        model.fit(X, y)
        predictions = model.predict(X)
        score = model.score(X, y)
        print(f"{name} - predictions: {predictions}, score: {score:.4f}")

    print("\n=== REGRESSORS ===")
    for name, model in REGRESSOR.items():
        model.fit(X, y)
        predictions = model.predict(X)
        score = model.score(X, y)
        print(f"{name} - predictions: {predictions}, score: {score:.4f}")

    print("\n=== CLUSTERING ===")
    for name, model in CLUSTER.items():
        labels = model.fit_predict(X)
        score = model.score(X)
        print(f"{name} - labels: {labels}, score: {score:.4f}")

    print("\n=== DECOMPOSITION ===")
    for name, model in DECOMPOSITION.items():
        X_transformed = model.fit_transform(X)
        print(f"{name} - shape: {X_transformed.shape}, variance ratio: {model.explained_variance_ratio_}")

    print("\n=== PREPROCESSING ===")
    for name, scaler in PREPROCESSING.items():
        X_scaled = scaler.fit_transform(X)
        print(f"{name} - first row: {X_scaled[0]}")

    print("\n=== ENCODERS ===")
    labels = np.array(['cat', 'dog', 'cat', 'bird'])
    le = LabelEncoder()
    encoded = le.fit_transform(labels)
    print(f"LabelEncoder - encoded: {encoded}, classes: {le.classes_}")

    ohe = OneHotEncoder()
    onehot = ohe.fit_transform(labels)
    print(f"OneHotEncoder - shape: {onehot.shape}")

    print("\nAll models executed successfully.")

if __name__ == "__main__":
    main()