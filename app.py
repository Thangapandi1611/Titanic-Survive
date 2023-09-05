import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model_path = "model/titanic.pkl"
with open(model_path, "rb") as model_file:
    model1 = pickle.load(model_file)

# Streamlit UI
st.title("Titanic Survival Prediction")

# Create input fields for user input
st.sidebar.header("User Input")

# Input: Pclass
pclass = st.sidebar.selectbox("Passenger Class (Pclass)", [1, 2, 3])

# Input: Sex
sex = st.sidebar.selectbox("Sex", ["Male","Female"])
if sex=="Male":
    sex=1
else:
    sex=0

# Input: Age
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)


# Create a dictionary for user input
input_data = {
    "Age": age,
    "Sex": sex,
    "Pclass": pclass
}

# Convert user input into a DataFrame
input_df = pd.DataFrame([input_data])

# Make predictions
if st.sidebar.button("Predict"):
    if input_df is None:
        st.error("Input data")
    pred=model1.predict(input_df)[0]
    pred=pred>0.5

    if not pred:
        st.error("Sorry, the passenger did not survive.")
    else:
        st.success("The passenger survived!")

# Display the input data
st.subheader("User Input Data")
st.write(input_df)
