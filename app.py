# app.py

import streamlit as st
import numpy as np
import pickle

# ---------------- PAGE ----------------
st.set_page_config(
    page_title="Pokemon Legendary Predictor",
    page_icon="⚡",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
with open("pokemon_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
type1_encoder = data["type1_encoder"]
type2_encoder = data["type2_encoder"]
threshold = data["threshold"]

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}
h1 {
    text-align:center;
}
.stButton>button {
    width:100%;
    height:55px;
    font-size:20px;
    border-radius:12px;
}
.result-box {
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("⚡ Pokemon Legendary Predictor")
st.write("Fill Pokemon stats below and check if it is Legendary.")

# ---------------- INPUT LAYOUT ----------------
col1, col2 = st.columns(2)

with col1:
    total = st.number_input("Total Stats", min_value=1, value=500)
    hp = st.number_input("HP", min_value=1, value=70)
    attack = st.number_input("Attack", min_value=1, value=80)
    defense = st.number_input("Defense", min_value=1, value=70)
    sp_atk = st.number_input("Sp. Atk", min_value=1, value=90)

with col2:
    sp_def = st.number_input("Sp. Def", min_value=1, value=80)
    speed = st.number_input("Speed", min_value=1, value=100)
    generation = st.selectbox("Generation", [1,2,3,4,5,6])
    type1 = st.selectbox("Type 1", type1_encoder.classes_)
    type2 = st.selectbox("Type 2", type2_encoder.classes_)

# ---------------- BUTTON ----------------
if st.button("Predict Now"):

    type1_val = type1_encoder.transform([type1])[0]
    type2_val = type2_encoder.transform([type2])[0]

    input_data = np.array([[
        total,
        hp,
        attack,
        defense,
        sp_atk,
        sp_def,
        speed,
        generation,
        type1_val,
        type2_val
    ]])

    input_data = scaler.transform(input_data)

    prob = model.predict_proba(input_data)[:,1][0]
    pred = 1 if prob >= threshold else 0

    st.divider()

    if pred == 1:
        st.markdown(
            f"<div class='result-box' style='background:#14532d;'>🔥 Legendary Pokemon<br>{round(prob*100,2)}%</div>",
            unsafe_allow_html=True
        )
        st.balloons()

    else:
        st.markdown(
            f"<div class='result-box' style='background:#7f1d1d;'>❌ Not Legendary<br>{round(prob*100,2)}%</div>",
            unsafe_allow_html=True
        )

# ---------------- FOOTER ----------------
st.divider()
st.caption("Built with Streamlit + Machine Learning")