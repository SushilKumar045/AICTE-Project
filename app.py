import streamlit as st
from PIL import Image
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Load dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, -1].values

# Encode categorical data
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Standardize features
sc = StandardScaler()
X = sc.fit_transform(X)

# Instantiate and train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the model
joblib.dump(model, 'Project3_Decision_Tree_model.pkl')

# Load the model
model = joblib.load('Project3_Decision_Tree_model.pkl')

def predict_note_authentication(UserID, Gender, Age, EstimatedSalary):
    input_data = sc.transform([[Gender, Age, EstimatedSalary]])
    output = model.predict(input_data)
    print("Purchased", output)
    if output == [1]:
        prediction = "Item will be purchased"
    else:
        prediction = "Item will not be purchased"
    print(prediction)
    return prediction

def main():
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Poornima Institute of Engineering & Technology</h2>
    <h3 style="color:white;text-align:center;">Department of Computer Engineering</h3>
    <h4 style="color:white;text-align:center;">Internship Project Deployment</h4>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.header("Item Purchase Prediction")
    UserID = st.text_input("UserID", "")
    Gender = st.selectbox('Gender', ('Male', 'Female'))
    Gender = 1 if Gender == 'Male' else 0
    Age = st.number_input("Insert Age", 18, 60)
    EstimatedSalary = st.number_input("Insert salary", 15000, 150000)
    result = ""
    if st.button("Predict"):
        result = predict_note_authentication(UserID, Gender, Age, EstimatedSalary)
        st.success('Model has predicted: {}'.format(result))
    if st.button("About"):
        st.subheader("Developed by Sushil Kumar")
        st.subheader("Student, Department of Artificial Intelligence and Data Science")

if __name__ == '__main__':
    main()
