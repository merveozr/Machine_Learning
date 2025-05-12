import streamlit as st
import pandas as pd
import pickle

# Modeli yükle
model = pickle.load(open("model.pkl", "rb"))

st.title("Meme Kanseri Durumu Tahmin Uygulaması")

# Kullanıcıdan giriş al (örnek alanlar)
age = st.slider("Age", 20, 100, 50)
tumor_size = st.slider("Tumor Size", 0, 100, 20)
inv_nodes = st.slider("Invaded Lymph Nodes", 0, 40, 4)

# Kullanıcıdan alınan verilerle tahmin
if st.button("Tahmin Et"):
    input_data = pd.DataFrame([[age, tumor_size, inv_nodes]], columns=["Age", "Tumor Size", "Inv_nodes"])
    prediction = model.predict(input_data)
    st.success(f"Tahmin Sonucu: {'Ölmüş' if prediction[0]==1 else 'Hayatta'}")
