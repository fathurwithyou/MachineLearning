import numpy as np
from allyouneed.linear_model import LogisticRegression, LinearRegression
from allyouneed.neighbors import KNeighborsClassifier, KNeighborsRegressor
from allyouneed.tree import DecisionTreeClassifier, DecisionTreeRegressor
from allyouneed.svm import SVC, SVR

CLASSIFIER = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "DecisionTree": DecisionTreeClassifier(max_depth=5),
    "SVC": SVC(kernel='linear', C=1.0),
    "LogisticRegression": LogisticRegression(max_iter=200),
}

REGRESSOR = {
    "KNN": KNeighborsRegressor(n_neighbors=3),
    "DecisionTree": DecisionTreeRegressor(max_depth=5),
    "SVR": SVR(kernel='linear', C=1.0),
    "LinearRegression": LinearRegression(),
}

def main():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([0, 1])

    for name, model in CLASSIFIER.items():
        model.fit(X, y)
        predictions = model.predict(X)
        print(f"{name} predictions: {predictions}")

    for name, model in REGRESSOR.items():
        model.fit(X, y)
        predictions = model.predict(X)
        print(f"{name} predictions: {predictions}")

    print("All models executed successfully.")

if __name__ == "__main__":
    main()