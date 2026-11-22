import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AgriYield AI | Indian Crop Yield Predictor",
    page_icon="🌾",
    layout="centered",
)

# AMOLED Black + Emerald & Gold Glassmorphism Theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000;
        color: #f3f4f6;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .hero-card {
        background: linear-gradient(135deg, #041f14 0%, #050505 65%, #072919 100%);
        border: 1px solid #10b98155;
        border-radius: 20px;
        padding: 24px 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 32px rgba(16, 185, 129, 0.16);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.16);
        color: #34d399;
        border: 1px solid #10b98166;
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.6px;
        margin-bottom: 10px;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 8px 0;
        line-height: 1.2;
    }
    .hero-sub {
        color: #9ca3af;
        font-size: 0.92rem;
        margin: 0;
        line-height: 1.5;
    }

    div.stButton > button:first-child {
        width: 100%;
        background: linear-gradient(90deg, #059669 0%, #10b981 100%);
        color: #000000;
        font-weight: 800;
        font-size: 1.05rem;
        border: none;
        border-radius: 14px;
        padding: 0.75rem 1.2rem;
        box-shadow: 0 0 22px rgba(16, 185, 129, 0.4);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
        box-shadow: 0 0 30px rgba(52, 211, 153, 0.6);
        transform: translateY(-1px);
    }

    .metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin: 16px 0;
    }
    .metric-card {
        background: #080d0b;
        border: 1px solid #10b98155;
        border-radius: 16px;
        padding: 16px 14px;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.1);
    }
    .metric-card-gold {
        background: #0d0b07;
        border: 1px solid #f59e0b55;
        border-radius: 16px;
        padding: 16px 14px;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.1);
    }
    .metric-label {
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        color: #9ca3af;
        margin-bottom: 6px;
        font-weight: 700;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #34d399;
        line-height: 1.15;
    }
    .metric-value-gold {
        font-size: 1.6rem;
        font-weight: 800;
        color: #fbbf24;
        line-height: 1.15;
    }
    .metric-unit {
        font-size: 0.82rem;
        color: #6b7280;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load("yield_model.pkl")


model = load_model()

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">⚡ RANDOM FOREST ML • R² 0.9719</div>
        <h1 class="hero-title">🌾 Indian Crop Yield Predictor</h1>
        <p class="hero-sub">
            Forecast productivity in <b>Tonnes/Hectare</b> &amp; <b>Punjab Quintals/Ekad</b>
            using state, season, area, rainfall, and agro-input data.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


states = [
    "Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh", "Maharashtra",
    "Rajasthan", "Gujarat", "Bihar", "West Bengal", "Andhra Pradesh",
    "Telangana", "Karnataka", "Tamil Nadu", "Kerala", "Odisha", "Assam",
    "Chhattisgarh", "Jharkhand", "Uttarakhand", "Himachal Pradesh",
    "Jammu and Kashmir", "Delhi", "Goa", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Puducherry", "Sikkim", "Tripura",
]

crops = [
    "Wheat", "Rice", "Potato", "Sugarcane", "Maize", "Cotton(lint)",
    "Barley", "Bajra", "Jowar", "Gram", "Arhar/Tur", "Moong(Green Gram)",
    "Urad", "Masoor", "Mustard", "Groundnut", "Soyabean", "Sunflower",
    "Onion", "Garlic", "Ginger", "Turmeric", "Banana", "Coconut",
    "Jute", "Mesta", "Ragi", "Small millets", "Tobacco", "Sweet potato",
]

seasons = ["Rabi", "Kharif", "Whole Year", "Summer", "Winter", "Autumn"]

with st.container(border=True):
    st.markdown("#### 🌱 Farm & Regional Parameters")
    col1, col2 = st.columns(2)
    with col1:
        state = st.selectbox("State / UT", states, index=0)
        season = st.selectbox("Cropping Season", seasons, index=0)
        rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=650.0, step=10.0)
    with col2:
        crop = st.selectbox("Crop Variety", crops, index=0)
        area = st.number_input("Cultivated Area (Hectares)", min_value=0.01, value=5.0, step=0.5)
        fertilizer = st.number_input("Fertilizer Used (kg)", min_value=0.0, value=1500.0, step=50.0)

    pesticide = st.number_input("Pesticide Used (kg)", min_value=0.0, value=15.0, step=1.0)
    st.caption(f"📏 Equivalent Punjab Land Area: **{area * 2.471:.2f} Ekad (Killa)**")

    predict_btn = st.button("🚀 Predict Crop Yield")

if predict_btn:
    input_df = pd.DataFrame(
        [
            {
                "Crop": crop,
                "Season": season,
                "State": state,
                "Area": area,
                "Annual_Rainfall": rainfall,
                "Fertilizer": fertilizer,
                "Pesticide": pesticide,
            }
        ]
    )

    pred_yield = float(model.predict(input_df)[0])
    total_production = pred_yield * area
    ekad_area = area * 2.471
    quintals_per_ekad = (pred_yield * 10.0) / 2.471
    total_quintals = total_production * 10.0

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Standard Yield</div>
                <div class="metric-value">{pred_yield:.2f}</div>
                <div class="metric-unit">Tonnes / Hectare</div>
            </div>
            <div class="metric-card-gold">
                <div class="metric-label">Punjab Mandi Yield</div>
                <div class="metric-value-gold">{quintals_per_ekad:.2f}</div>
                <div class="metric-unit">Quintals / Ekad</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Production</div>
                <div class="metric-value">{total_production:,.2f}</div>
                <div class="metric-unit">Metric Tonnes ({area:.2f} ha)</div>
            </div>
            <div class="metric-card-gold">
                <div class="metric-label">Total Mandi Weight</div>
                <div class="metric-value-gold">{total_quintals:,.1f}</div>
                <div class="metric-unit">Quintals ({ekad_area:.2f} Ekad)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 📊 Input Intensity Breakdown (Per Hectare)")
    intensity_df = pd.DataFrame(
        {
            "Metric": ["Fertilizer (kg/ha)", "Annual Rainfall (mm)", "Pesticide (g/ha)"],
            "Value": [fertilizer / area, rainfall, (pesticide * 1000) / area],
        }
    ).set_index("Metric")
    st.bar_chart(intensity_df, color="#10b981")



