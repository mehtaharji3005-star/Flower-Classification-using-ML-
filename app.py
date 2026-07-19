import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
import os
import altair as alt

from sklearn.datasets import load_iris

data = load_iris()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target

classes = data.target_names

X = df.iloc[:, :-1]

all_model_name = [
    "Logistic Regression",
    "Naive Bayes",
    "Decision Tree",
    "Random Forest",
    "SVM",
    "KNN"
]

all_models = []

for model_name in all_model_name:

    file_name = model_name + ".pkl"

    if os.path.exists(file_name):

        with open(file_name, "rb") as f:
            model = pickle.load(f)

        all_models.append(model)

    else:

        st.error(f"{file_name} not found.")
        st.stop()

        st.title("🌸 ML Flower Classification Project")

url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQF2roQNP1rPFtklA8xgZt76jyhj6x2BUjVe6gxwxJ53pI0_TYfQLRZh8oZ&s=10"

st.image(url)

st.dataframe(df.sample(5))

st.sidebar.title("Select Iris Features")
st.sidebar.image(url)

user_input = []

for feature in X.columns:

    value = st.sidebar.slider(
        feature,
        min_value=float(X[feature].min()),
        max_value=float(X[feature].max()),
        value=float(X[feature].mean()),
        step=0.1
    )

    user_input.append(value)

    st.subheader("User Input")

st.write(user_input)

if st.button("Predict"):

    input_df = pd.DataFrame(
        [user_input],
        columns=X.columns
    )

    model_ans = []

    model_prob = []

    for model in all_models:

        prediction = model.predict(input_df)[0]

        try:
            probability = model.predict_proba(input_df).max()

        except:

            probability = 1.0

        model_ans.append(classes[prediction])

        model_prob.append(float(probability))

comp_df = pd.DataFrame({

    "Model": all_model_name,

    "Probability": model_prob,

    "Prediction": model_ans

})

st.dataframe(comp_df)

chart = alt.Chart(comp_df).mark_bar().encode(

    x="Model",

    y="Probability",

    tooltip=["Model", "Probability", "Prediction"]

)

st.altair_chart(chart, use_container_width=True)

final_prediction = pd.Series(model_ans).mode()[0]

st.success(f"Predicted Flower : {final_prediction}")

footer = """
<style>
.footer{
position:fixed;
left:0;
bottom:0;
width:100%;
text-align:center;
color:gray;
padding:10px;
}
</style>

<div class="footer">
Made with ❤️ using Streamlit
</div>
"""

st.markdown(footer, unsafe_allow_html=True)