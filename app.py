import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris

st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸")
st.title("🌸 Interactive ML Web App - Iris Classifier")
st.write("Adjust the flower measurements using the sliders below to predict the Iris species in real time.")

model = joblib.load("iris_model.pkl")
iris = load_iris()

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(features)[0]
probabilities = model.predict_proba(features)[0]
predicted_species = iris.target_names[prediction].capitalize()

st.subheader("Prediction Result")
st.success(f"Predicted Species: **{predicted_species}**")

st.write("Prediction Confidence:")
for name, prob in zip(iris.target_names, probabilities):
    st.write(f"- **{name.capitalize()}**: {prob * 100:.2f}%")