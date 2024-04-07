import numpy as np
import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Machine Learning Classification Explorer")

st.write("""
# Select Classifier Of Your Choice
""")

selected_dataset = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine dataset", "Diabetes"))
st.write(f"Your Dataset :  {selected_dataset}")

selected_classifier = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))
st.write(f"Your Classifier : {selected_classifier}")


def get_data(selected_dataset):
    if selected_dataset == "Iris":
        data = datasets.load_iris()
    elif selected_dataset == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif selected_dataset == "Diabetes":
        data = datasets.load_diabetes()
    else:
        data = datasets.load_wine()
    x = data.data
    y = data.target
    return x, y


x, y = get_data(selected_dataset)
st.write("Shape of dataset ", x.shape)
st.write("Number of classes ", len(np.unique(y)))


def parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K_val = st.sidebar.slider("K", 1, 15)
        params["K"] = K_val
    elif clf_name == "SVM":
        C_val = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C_val

    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimate = st.sidebar.slider("n_estimator", 1, 100)
        rand_state = st.sidebar.slider("rand_state", 1, 200)
        params["max_depth"] = max_depth
        params["n_estimator"] = n_estimate
        params["rand_state"] = rand_state

    return params


params = parameter_ui(selected_classifier)


def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        # max_depth = st.sidebar.slider("max_depth", 2, 15)
        # n_estimate = st.sidebar.slider("n_estimator", 1, 100)
        clf = RandomForestClassifier(n_estimators=params["n_estimator"], max_depth=params["max_depth"],
                                     random_state=params["rand_state"])
    return clf


clf = get_classifier(selected_classifier, params)

# Classification

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

acc = accuracy_score(y_test, y_pred) * 100

st.write(f"Classifier = {selected_classifier}")
st.write(f"Accuracy = {acc:.3f} %")

# PLOT

pca = PCA(2)
x_projected = pca.fit_transform(x)

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]
fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.colorbar()
st.pyplot(fig)