import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import base64

st.set_page_config(
    page_title="Meme Kanseri Tahmin Aracı",
    page_icon="🎗️",
    layout="wide"
)

def add_bg_color():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #FFE3F8;
        }}
        .stButton>button {{
            background-color: #FF69B4;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
            transition: all 0.3s;
        }}
        .stButton>button:hover {{
            background-color: #FF1493;
            transform: translateY(-2px);
            box-shadow: 0 5px 10px rgba(0,0,0,0.2);
        }}
        .result-box {{
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-top: 20px;
        }}
        .header {{
            color: #9C27B0;
            font-family: 'Arial', sans-serif;
        }}
        .subheader {{
            color: #E91E63;
            font-family: 'Arial', sans-serif;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_color()

df = pd.read_csv("Breast_Cancer.csv")

df_encoded = df.copy()
label_encoders = {}
for column in df_encoded.columns:
    if df_encoded[column].dtype == "object":
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le

X = df_encoded.drop("Status", axis=1)
y = label_encoders["Status"].transform(df["Status"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

input_features = [
    "Survival Months",
    "Age",
    "Regional Node Examined",
    "Tumor Size",
    "Reginol Node Positive"
]

feature_labels = {
    "Survival Months": "Hayatta Kalma Ayları",
    "Age": "Yaş",
    "Regional Node Examined": "İncelenen Bölgesel Nodül Sayısı",
    "Tumor Size": "Tümör Boyutu (mm)",
    "Reginol Node Positive": "Pozitif Bölgesel Nodül Sayısı"
}

st.markdown('<h1 class="header">🎗️ Meme Kanseri Durum Tahmini</h1>', unsafe_allow_html=True)
st.markdown("""
Bu uygulama, belirli hasta verilerine dayanarak meme kanseri durumunu tahmin etmek için makine öğrenimi kullanmaktadır.
Lütfen aşağıdaki alanları uygun değerlerle doldurunuz ve sonuçları görmek için 'Tahmin Et' butonuna tıklayınız.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<h3 class="subheader">📝 Hasta Bilgileri</h3>', unsafe_allow_html=True)
    
    user_input = {}
    for col in X.columns:
        if col in input_features:
            min_val = float(df_encoded[col].min())
            max_val = float(df_encoded[col].max())
            avg_val = float(df_encoded[col].mean())
            
            if col == "Age":
                step = 1.0
                format = "%d"
            elif col == "Survival Months":
                step = 1.0
                format = "%d"
            else:
                step = 0.1
                format = "%.1f"
            
            val = st.slider(
                feature_labels.get(col, col),
                min_value=min_val,
                max_value=max_val,
                value=avg_val,
                step=step,
                format=format,
                help=f"Bu değer için ortalama: {avg_val:.2f}"
            )
            user_input[col] = val
        else:
            user_input[col] = float(df_encoded[col].mean())
    
    predict_btn = st.button("🔍 Tahmin Et", use_container_width=True)

with col2:
    if predict_btn:
        st.markdown('<h3 class="subheader">🧪 Tahmin Sonuçları</h3>', unsafe_allow_html=True)
        
        with st.spinner('Tahmin yapılıyor...'):
            input_df = pd.DataFrame([user_input])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            status_label = label_encoders["Status"].inverse_transform([prediction])[0]
        
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        
        if status_label == "Alive":
            st.success(f"### 🌱 Tahmin Edilen Durum: **Hayatta**")
            emoji = "🌟"
        else:
            st.error(f"### 🍂 Tahmin Edilen Durum: **Vefat Etmiş**")
            emoji = "⚠️"
        
        status_options = label_encoders["Status"].classes_
        for i, label in enumerate(status_options):
            if label == "Alive":
                display_label = "Hayatta"
            else:
                display_label = "Vefat Etmiş"
            
            prob = prediction_proba[i]
            st.write(f"{emoji} **{display_label}** olasılığı: **{prob*100:.2f}%**")
            st.progress(prob)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        
    else:
        st.markdown('<h3 class="subheader">🧪 Tahmin Sonuçları</h3>', unsafe_allow_html=True)
        st.info("👈 Lütfen hasta bilgilerini girin ve 'Tahmin Et' butonuna tıklayın.")

st.markdown("---")
st.markdown("**Meme Kanseri Tahmin Aracı** | Bu uygulama, eğitilmiş bir makine öğrenimi modeli kullanır ve yalnızca bilgi amaçlıdır. Tıbbi tanı ve tedavi için kullanılamaz. Merve Özer")
