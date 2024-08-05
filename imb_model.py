# Logistic Regression
def logistic_regression_model(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)
    return model

# Random Forest
def random_forest_model(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# SVM
def svm_model(X_train, y_train):
    from sklearn.svm import SVC
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model

# knn
def knn_model(X_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    return model