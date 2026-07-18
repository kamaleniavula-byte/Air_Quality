import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="🌍 AI Air Quality Analyzer",
    page_icon="🌿",
    layout="wide"
)

st.title("🌍 AI Air Quality Analyzer")
st.write("Analyze air quality and receive AI-powered health recommendations.")

# -----------------------------
# Sidebar
# -----------------------------
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password"
)

client = OpenAI(api_key=api_key) if api_key else None

# -----------------------------
# Inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    city = st.text_input("City", "Hyderabad")
    temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 30.0)
    humidity = st.slider("Humidity (%)", 0, 100, 60)
    wind = st.number_input("Wind Speed (km/h)", 0.0, 100.0, 8.0)

with col2:
    pm25 = st.number_input("PM2.5", 0.0, 500.0, 55.0)
    pm10 = st.number_input("PM10", 0.0, 500.0, 80.0)
    no2 = st.number_input("NO₂", 0.0, 500.0, 25.0)
    co = st.number_input("CO", 0.0, 50.0, 1.2)

# -----------------------------
# AQI Category
# -----------------------------
def get_aqi_category(pm25):
    if pm25 <= 12:
        return "Good 😊", 25
    elif pm25 <= 35.4:
        return "Moderate 🙂", 75
    elif pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups 😐", 125
    elif pm25 <= 150.4:
        return "Unhealthy 😷", 180
    elif pm25 <= 250.4:
        return "Very Unhealthy 🚨", 260
    else:
        return "Hazardous ☠️", 350

# -----------------------------
# Analyze Button
# -----------------------------
if st.button("🌿 Analyze Air Quality"):

    category, aqi = get_aqi_category(pm25)

    st.success(f"Estimated AQI: {aqi}")

    st.metric("Air Quality", category)

    st.progress(min(aqi/350,1.0))

    if client:

        prompt = f"""
You are an environmental expert.

City: {city}

Weather
Temperature: {temperature} °C
Humidity: {humidity}%
Wind Speed: {wind} km/h

Pollutants
PM2.5: {pm25}
PM10: {pm10}
NO2: {no2}
CO: {co}

Estimated AQI: {aqi}

Provide:

1. Air Quality Summary
2. Health Effects
3. Outdoor Activity Advice
4. Safety Tips
5. Pollution Reduction Tips
6. Final Recommendation

Use simple language.
"""

        with st.spinner("Generating AI Analysis..."):

            response = client.responses.create(
                model="gpt-5-mini",
                input=prompt
            )

        st.subheader("🤖 AI Analysis")
        st.write(response.output_text)

    else:
        st.info("Enter your OpenAI API Key to receive AI-generated analysis.")