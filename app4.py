import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="🌍 AI Air Quality Analyzer",
    page_icon="🌿",
    layout="wide"
)

st.title("🌍 AI Air Quality Prediction & Analysis System")
st.write("Analyze air quality using your dataset and receive AI-powered insights.")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("Settings")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password"
)

client = OpenAI(api_key=api_key) if api_key else None

# -------------------------------------------------
# User Inputs
# -------------------------------------------------
st.header("Enter Air Quality Details")

col1, col2 = st.columns(2)

with col1:
    city = st.text_input("City", "Hyderabad")
    temperature = st.number_input(
        "Temperature (°C)",
        value=30.0
    )
    humidity = st.slider(
        "Humidity (%)",
        0,
        100,
        60
    )
    wind = st.number_input(
        "Wind Speed (km/h)",
        value=8.0
    )

with col2:
    pm25 = st.number_input(
        "PM2.5",
        value=55.0
    )

    pm10 = st.number_input(
        "PM10",
        value=80.0
    )

    no2 = st.number_input(
        "NO₂",
        value=25.0
    )

    co = st.number_input(
        "CO",
        value=1.2
    )

# -------------------------------------------------
# AQI Category
# -------------------------------------------------
def get_aqi(pm25):

    if pm25 <= 12:
        return 25, "Good 😊"

    elif pm25 <= 35.4:
        return 75, "Moderate 🙂"

    elif pm25 <= 55.4:
        return 125, "Unhealthy for Sensitive Groups 😐"

    elif pm25 <= 150.4:
        return 180, "Unhealthy 😷"

    elif pm25 <= 250.4:
        return 260, "Very Unhealthy 🚨"

    else:
        return 350, "Hazardous ☠️"

# -------------------------------------------------
# Analyze Button
# -------------------------------------------------
if st.button("🌿 Analyze Air Quality"):

    aqi, category = get_aqi(pm25)

    st.success(f"Estimated AQI : {aqi}")

    st.metric(
        "Air Quality",
        category
    )

    st.progress(min(aqi / 350, 1.0))

    if client:

        prompt = f"""
You are an environmental scientist.

City : {city}

Temperature : {temperature}

Humidity : {humidity}

Wind Speed : {wind}

PM2.5 : {pm25}

PM10 : {pm10}

NO2 : {no2}

CO : {co}

AQI : {aqi}

Explain:

1. Air Quality Summary

2. Health Effects

3. Safety Tips

4. Outdoor Activities

5. Pollution Reduction Tips

6. Final Recommendation
"""

        with st.spinner("Generating AI Analysis..."):

            response = client.responses.create(
                model="gpt-5-mini",
                input=prompt
            )

        st.subheader("🤖 AI Analysis")

        st.write(response.output_text)

    else:
        st.warning("Please enter your OpenAI API Key.")

    st.divider()

    st.header("📊 Air Quality Dashboard")

    try:

        df = pd.read_csv(
            "Air Quality.csv",
            sep=";",
            decimal=","
        )

        df.drop(
            columns=[
                "Unnamed: 15",
                "Unnamed: 16"
            ],
            inplace=True,
            errors="ignore"
        )

        df.replace(-200, pd.NA, inplace=True)

        numeric_columns = [
            "CO(GT)",
            "NO2(GT)",
            "T",
            "RH"
        ]

        for col in numeric_columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Average Temperature",
                round(df["T"].mean(), 2)
            )

        with c2:
            st.metric(
                "Average Humidity",
                round(df["RH"].mean(), 2)
            )

        with c3:
            st.metric(
                "Average CO",
                round(df["CO(GT)"].mean(), 2)
            )

        with c4:
            st.metric(
                "Average NO₂",
                round(df["NO2(GT)"].mean(), 2)
            )
        # ---------------------------------------
        # Temperature Chart
        # ---------------------------------------
        st.subheader("🌡 Temperature Trend")

        fig = px.line(
            df,
            y="T",
            title="Temperature"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------
        # Humidity Chart
        # ---------------------------------------
        st.subheader("💧 Humidity Trend")

        fig = px.line(
            df,
            y="RH",
            title="Relative Humidity"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------
        # Carbon Monoxide Chart
        # ---------------------------------------
        st.subheader("🟠 CO(GT) Levels")

        fig = px.line(
            df,
            y="CO(GT)",
            title="Carbon Monoxide"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------
        # NO2 Chart
        # ---------------------------------------
        st.subheader("🔵 NO₂ Levels")

        fig = px.bar(
            df.head(200),
            y="NO2(GT)",
            title="Nitrogen Dioxide"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------
        # Temperature Distribution
        # ---------------------------------------
        st.subheader("📊 Temperature Distribution")

        fig = px.histogram(
            df,
            x="T",
            nbins=30,
            title="Temperature Histogram"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------
        # Humidity Distribution
        # ---------------------------------------
        st.subheader("📊 Humidity Distribution")

        fig = px.histogram(
            df,
            x="RH",
            nbins=30,
            title="Humidity Histogram"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------
        # Scatter Plot
        # ---------------------------------------
        st.subheader("📈 CO vs NO₂")

        fig = px.scatter(
            df,
            x="CO(GT)",
            y="NO2(GT)",
            color="T",
            title="CO vs NO₂"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------
        # Correlation Heatmap
        # ---------------------------------------
        st.subheader("🔥 Correlation Matrix")

        corr = df.select_dtypes(include="number").corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            title="Correlation Heatmap"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------
        # Dataset Summary
        # ---------------------------------------
        st.subheader("📋 Dataset Summary")

        st.write(df.describe())

        # ---------------------------------------
        # Download Dataset
        # ---------------------------------------
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Cleaned Dataset",
            data=csv,
            file_name="clean_air_quality.csv",
            mime="text/csv"
        )

        # ---------------------------------------
        # AI Dataset Analysis
        # ---------------------------------------
        if client:

            summary_prompt = f"""
Analyze this air quality dataset.

Average Temperature: {df['T'].mean():.2f}

Average Humidity: {df['RH'].mean():.2f}

Average CO: {df['CO(GT)'].mean():.2f}

Average NO2: {df['NO2(GT)'].mean():.2f}

Maximum Temperature: {df['T'].max()}

Maximum CO: {df['CO(GT)'].max()}

Give:

1. Overall Air Quality

2. Major Pollution Sources

3. Health Risks

4. Suggestions

5. Environmental Improvements

6. Final Conclusion
"""

            with st.spinner("Generating Dataset Insights..."):

                ai_summary = client.responses.create(
                    model="gpt-5-mini",
                    input=summary_prompt
                )

            st.subheader("🤖 AI Dataset Insights")

            st.write(ai_summary.output_text)

        st.success("✅ Dashboard Loaded Successfully!")

    except FileNotFoundError:
        st.error("❌ Air Quality.csv not found. Place it in the same folder as app.py.")

    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.markdown(
    "<center><h4>🌍 AI Air Quality Prediction & Analysis System</h4>"
    "<p>Built with ❤️ using Streamlit, OpenAI, Pandas and Plotly</p></center>",
    unsafe_allow_html=True
)
