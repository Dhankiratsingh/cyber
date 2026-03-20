import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

st.set_page_config(page_title="IDS with MAP", layout="wide")
st.title("🌍 Intrusion Detection System - Live Attack Map")

# Sample data WITH GEO-LOCATIONS (like video)
sample_data = {
    'Flow Duration': [100, 5000, 300, 8000, 200, 10000],
    'Tot Fwd Pkts': [10, 1, 20, 2, 15, 5],
    'Tot Bwd Pkts': [5, 50, 10, 100, 8, 200],
    'TotLen Fwd Pkts': [1000, 100, 2000, 200, 1500, 500],
    'TotLen Bwd Pkts': [500, 5000, 1000, 8000, 750, 10000],
    'Label': ['BENIGN', 'DoS Hulk', 'BENIGN', 'PortScan', 'BENIGN', 'DDoS'],
    'Latitude': [28.61, 40.71, -33.87, 35.68, 51.51, 22.57],  # Delhi, NYC, Sydney, Moscow, London, Mumbai
    'Longitude': [77.23, -74.01, 151.21, 139.77, -0.13, 77.20],
    'Country': ['India', 'USA', 'Australia', 'Russia', 'UK', 'India']
}
df = pd.DataFrame(sample_data)
st.session_state.df = df

# Sidebar controls
st.sidebar.header("⚙️ Controls")
simulate_attacks = st.sidebar.checkbox("Simulate Live Attacks", True)

# Train model (same as before)
if st.button("🚀 Train Model"):
    numeric_cols = ['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts']
    X = df[numeric_cols].fillna(0)
    le = LabelEncoder()
    y = le.fit_transform(df['Label'])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)
    
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.le = le
    st.session_state.acc = acc
    st.success(f"✅ Trained! Accuracy: {acc:.2%}")

# MAP SECTION (EXACTLY LIKE VIDEO)
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🗺️ Live Attack Map")
    
    # Filter attacks only
    attacks_df = df[df['Label'].str.contains('DoS|DDoS|PortScan|Hulk', na=False)].copy()
    
    if simulate_attacks:
        # Animate new attacks appearing
        for i in range(len(attacks_df)):
            time.sleep(0.5)
            attack_map = px.scatter_mapbox(
                attacks_df.iloc[:i+1], 
                lat="Latitude", 
                lon="Longitude",
                color="Label",
                hover_name="Country",
                hover_data=["Label"],
                mapbox_style="carto-positron",
                zoom=2,
                height=500,
                title=f"🟥 {i+1} Attacks Detected"
            )
            attack_map.update_layout(mapbox=dict(center=dict(lat=20, lon=0)))
            st.plotly_chart(attack_map, use_container_width=True)
            st.empty()
    else:
        # Static map
        attack_map = px.scatter_mapbox(
            attacks_df, 
            lat="Latitude", 
            lon="Longitude",
            color="Label",
            size_max=15,
            hover_name="Country",
            mapbox_style="carto-positron",
            zoom=2,
            height=500
        )
        st.plotly_chart(attack_map, use_container_width=True)

with col2:
    st.subheader("📊 Stats")
    if 'acc' in st.session_state:
        st.metric("Accuracy", f"{st.session_state.acc:.2%}")
    
    st.metric("Total Attacks", len(df[df['Label'].str.contains('DoS|DDoS|PortScan', na=False)]))
    st.metric("Normal Traffic", len(df[df['Label'] == 'BENIGN']))
    
    # Attack locations list
    st.subheader("🎯 Attack Sources")
    attack_countries = df[df['Label'].str.contains('DoS|DDoS|PortScan')]['Country'].value_counts()
    st.write(attack_countries)

# Live Packet Scanner (bottom)
st.subheader("🔴 Live Packet Scanner")
col1, col2, col3 = st.columns(3)
with col1:
    flow_dur = st.number_input("Flow Duration", 0, 10000, 500)
with col2:
    fwd_pkts = st.number_input("Fwd Packets", 1, 100, 10)
with col3:
    bwd_pkts = st.number_input("Bwd Packets", 1, 100, 5)

if st.button("🔍 SCAN PACKET") and 'model' in st.session_state:
    test_sample = st.session_state.scaler.transform([[flow_dur, fwd_pkts, bwd_pkts, 1000, 500]])
    pred = st.session_state.model.predict(test_sample)[0]
    prob = st.session_state.model.predict_proba(test_sample)[0].max()
    
    if pred == 1:  # Attack
        st.error(f"🚨 **ATTACK DETECTED!** Confidence: {prob:.1%}")
        st.balloons()  # Video-style celebration
    else:
        st.success(f"✅ **NORMAL** Confidence: {prob:.1%}")

st.info("🎥 **MAP EXACTLY LIKE VIDEO**: Red points show attack locations. Check 'Simulate Live Attacks' for animation!")

