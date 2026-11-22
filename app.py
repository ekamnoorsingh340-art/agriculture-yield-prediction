import joblib
import pandas as pd
import streamlit as st

# Page setup
st.set_page_config(
    page_title="Crop Yield Predictor", page_icon="🌾", layout="centered"
)


# Load the trained pipeline
@st.cache_resource
def load_model():
  return joblib.load("yield_model.pkl")


model = load_model()

# App Header
st.title("🌾 Indian Crop Yield Prediction")
st.markdown(
    "Predict expected crop yield (**tonnes per hectare**) using state, season,"
    " area, rainfall, and input usage."
)
st.divider()

# Dropdown options
states = [
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Delhi",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jammu and Kashmir",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Puducherry",
    "Punjab",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
]

crops = [
    "Rice",
    "Wheat",
    "Maize",
    "Bajra",
    "Jowar",
    "Barley",
    "Ragi",
    "Cotton(lint)",
    "Sugarcane",
    "Potato",
    "Onion",
    "Moong(Green Gram)",
    "Urad",
    "Arhar/Tur",
    "Gram",
    "Masoor",
    "Groundnut",
    "Mustard",
    "Rapeseed &Mustard",
    "Soyabean",
    "Sunflower",
    "Sesamum",
    "Jute",
    "Tobacco",
    "Turmeric",
    "Ginger",
    "Garlic",
    "Dry chillies",
    "Banana",
    "Coconut",
]

seasons = ["Kharif", "Rabi", "Whole Year", "Summer", "Winter", "Autumn"]

# Input Form
with st.form("prediction_form"):
  col1, col2 = st.columns(2)

  with col1:
    selected_state = st.selectbox("State", states, index=states.index("Punjab"))
    selected_crop = st.selectbox("Crop", crops, index=crops.index("Wheat"))
    selected_season = st.selectbox(
        "Season", seasons, index=seasons.index("Rabi")
    )

  with col2:
    area = st.number_input(
        "Cultivated Area (Hectares)", min_value=0.1, value=100.0, step=10.0
    )
    rainfall = st.number_input(
        "Annual Rainfall (mm)", min_value=0.0, value=650.0, step=25.0
    )
    fertilizer = st.number_input(
        "Fertilizer Used (kg)", min_value=0.0, value=15000.0, step=500.0
    )
    pesticide = st.number_input(
        "Pesticide Used (kg)", min_value=0.0, value=30.0, step=5.0
    )

  submitted = st.form_submit_button("Predict Yield", use_container_width=True)

# Prediction Output
if submitted:
  input_data = pd.DataFrame([{
      "Crop": selected_crop,
      "Season": selected_season,
      "State": selected_state,
      "Area": area,
      "Annual_Rainfall": rainfall,
      "Fertilizer": fertilizer,
      "Pesticide": pesticide,
  }])

  predicted_yield = model.predict(input_data)[0]
  total_production = predicted_yield * area

  st.success("Prediction Complete!")
  res_col1, res_col2 = st.columns(2)
  res_col1.metric("Predicted Yield", f"{predicted_yield:.2f} tonnes / ha")
  res_col2.metric("Estimated Total Production", f"{total_production:,.2f} tonnes")
