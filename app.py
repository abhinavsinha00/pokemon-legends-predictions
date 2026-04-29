import streamlit as st
import numpy as np
import pickle

# ---------------- LOAD ----------------
with open("pokemon_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
type1_encoder = data["type1_encoder"]
type2_encoder = data["type2_encoder"]
threshold = data["threshold"]

# ---------------- PAGE ----------------
st.set_page_config(page_title="Pokemon Predictor", page_icon="⚡", layout="centered")

st.title("⚡ Pokemon Legendary Predictor")

st.write("Enter Pokemon stats")

# ---------- INPUTS ----------
col1, col2 = st.columns(2)

with col1:
    total = st.number_input("Total", min_value=1, value=500)
    hp = st.number_input("HP", min_value=1, value=70)
    attack = st.number_input("Attack", min_value=1, value=80)
    defense = st.number_input("Defense", min_value=1, value=70)
    sp_atk = st.number_input("Sp. Atk", min_value=1, value=90)

with col2:
    sp_def = st.number_input("Sp. Def", min_value=1, value=80)
    speed = st.number_input("Speed", min_value=1, value=100)
    generation = st.selectbox("Generation", [1,2,3,4,5,6])

    # IMPORTANT: names not numbers
    type1 = st.selectbox("Type 1", type1_encoder.classes_)
    type2 = st.selectbox("Type 2", type2_encoder.classes_)

# ---------- PREDICT ----------
if st.button("Predict"):

    # Convert names -> encoded numbers
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

    # scale
    input_data = scaler.transform(input_data)

    # probability
    prob = model.predict_proba(input_data)[:,1][0]

    pred = 1 if prob >= threshold else 0

    if pred == 1:
        st.success("🔥 Legendary Pokemon")
    else:
        st.error("❌ Not Legendary")

    st.write("Probability:", round(prob * 100, 2), "%")