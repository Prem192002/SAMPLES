import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Select the k best features using chi-squared test
selector = SelectKBest(chi2, k=2)
X_new = selector.fit_transform(X, y)

# Print the indices of the selected features
mask = selector.get_support()   # get a boolean mask of the selected features
selected_features = np.arange(X.shape[1])[mask]   # get the indices of the selected features
print(f"Selected features: {selected_features}")