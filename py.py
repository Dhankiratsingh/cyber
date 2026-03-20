import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="IDS MAP FIXED", layout="wide")
st.title("🌍 IDS - Attack Map (100% Working)")

# Data with attack locations (India + global)
data = {
    'Flow Duration': [100, 5000, 300, 8000, 200, 10000, 2500],
    'Tot Fwd Pkts': [10, 1, 20, 2, 15, 5, 3],
    'Tot Bwd Pkts': [5, 50, 10, 100, 8, 200, 75],
    'TotLen Fwd Pkts': [1000, 100, 2000, 200, 1500, 500, 1200],
    'TotLen Bwd Pkts': [500, 5000, 1000, 8000, 750, 10000, 6000],
    'Label': ['BENIGN', 'DoS Hulk', 'BENIGN', 'PortScan', 'BENIGN', 'DDoS', 'DoS slowloris'],
    'lat': [28.61, 40.71, -33.87, 55.75, 51.51, 22.57, 19.07],  # Delhi, NYC, Sydney, Moscow, London, Mumbai, Bangalore
    'lon': [77.23, -74.01, 151.21, 37.62, -0.13, 77.20, 72.84],
    'Country': ['India', 'USA', 'Australia', 'Russia', 'UK', 'India', 'India']
}
df = pd.DataFrame(data)

# MAP THAT ALWAYS WORKS (pydeck)
st.subheader("🗺️ Live Attack Map")
attack_data = df[df['Label'].str.contains('DoS|DDoS|PortScan|Hulk|slowloris', na=False)]

# Pydeck layer for attacks (RED points)
attack_layer = pdk.Layer(
    'ScatterplotLayer',
    data=attack_data,
    get_position=['lon', 'lat'],
    get_radius=200000,  # Size
    get_fill_color=[255, 0, 0, 180],  # Red with transparency
    get_line_color=[0, 0, 0, 255],
    line_width_min_pixels=1
)

# Normal traffic (GREEN points)
normal_layer = pdk.Layer(
    'ScatterplotLayer',
    data=df[df['Label'] == 'BENIGN'],
    get_position=['lon', 'lat'],
    get_radius=150000,
    get_fill_color=[0, 255, 0, 120],  # Green
    get_line_color=[0, 0, 0, 255]
)

# MAP VIEW
view_state = pdk.ViewState(latitude=20, longitude=0, zoom=1.5)
map_view = pdk.Deck(layers=[attack_layer, normal_layer], initial_view_state=view_state, height=500)
st.pydeck_chart(map_view)

# Train model
if st.button("🚀 Train IDS Model"):
    X = df[['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts']].fillna(0)
    le = LabelEncoder()
    y = le.fit_transform(df['Label'])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)
    
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.success(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    st.session_state.model = model
    st.session_state.scaler = scaler

# Live scanner
col1, col2, col3 = st.columns(3)
flow = col1.number_input("Flow Duration", 0, 10000, 1000)
fwd = col2.number_input("Fwd Packets", 1, 100, 10)
bwd = col3.number_input("Bwd Packets", 1, 200, 50)

if st.button("🔍 SCAN") and 'model' in st.session_state:
    sample = st.session_state.scaler.transform([[flow, fwd, bwd, 1000, 5000]])
    pred = st.session_state.model.predict(sample)[0]
    prob = st.session_state.model.predict_proba(sample).max()
    
    if pred != 0:  # Attack
        st.error(f"🚨 ATTACK! {prob:.1%} confidence")
        st.balloons()
    else:
        st.success(f"✅ Normal traffic")

st.info("✅ **MAP SHOWS**: Red=Attacks (Moscow, NYC, India), Green=Normal")
