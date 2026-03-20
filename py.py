import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import os

# Cache model training
@st.cache_resource
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Select key features from CIC-IDS (adjust columns as per your CSV)
    features = ['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 
                'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 
                'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Flow Pkts/s', 'Flow Bytes/s',
                'Label']  # Add more if available
    df = df[features].dropna()
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    return df, le

@st.cache_resource
def train_model(df):
    X = df.drop('Label', axis=1)
    y = df['Label']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    return model, scaler, acc, explainer, X_test, y_test, y_pred

st.set_page_config(page_title="IDS Streamlit", layout="wide")

st.title("🚨 Intrusion Detection System (Streamlit Version)")
st.markdown("Replicating video project: Random Forest on CIC-IDS2017 for real-time cyber threat detection [page:0].")

uploaded_file = st.file_uploader("Upload CIC-IDS CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset preview:")
    st.dataframe(df.head())

    if st.button("Load & Train Model"):
        df, le = load_data(uploaded_file.name)  # Save temp for cache
        model, scaler, acc, explainer, X_test, y_test, y_pred = train_model(df)
        
        joblib.dump(model, 'rf_model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        joblib.dump(le, 'le.joblib')
        
        st.success(f"✅ Model trained! Accuracy: {acc:.4f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))
        
        # Live prediction simulation (mimic video's real-time feed)
        st.subheader("🔴 Live Prediction (Simulate Attacks)")
        threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.5)
        
        # Sample new data (simulate incoming traffic)
        num_samples = st.slider("Packets to predict", 1, 100, 10)
        sample_idx = np.random.choice(len(X_test), num_samples)
        X_sample = X_test[sample_idx]
        
        if st.button("Predict Now"):
            probs = model.predict_proba(X_sample)
            preds = model.predict(X_sample)
            risky = np.mean(preds == 1) * 100  # % attacks
            
            fig = px.bar(x=['Normal', 'Attack'], y=[(1-np.mean(preds)), np.mean(preds)], 
                         title=f"Prediction Results ({risky:.1f}% Risk)")
            st.plotly_chart(fig)
            
            # SHAP explanation for top risky sample
            shap_values = explainer.shap_values(X_sample[0:1])[1]
            st.subheader("SHAP: Why this is an Attack?")
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot()
            
            st.metric("Attack Rate", f"{risky:.1f}%", delta=f"{risky-50:.1f}%")

# Load saved model for quick demo
if os.path.exists('rf_model.joblib'):
    st.info("📁 Using saved model for faster demo.")
    model = joblib.load('rf_model.joblib')
    scaler = joblib.load('scaler.joblib')
    # Predict on manual input
    st.subheader("Manual Packet Input")
    flow_duration = st.number_input("Flow Duration", 0, 1000000)
    tot_fwd_pkts = st.number_input("Tot Fwd Pkts", 1, 100)
    # Add more inputs matching features...
    if st.button("Classify Packet"):
        input_data = scaler.transform(np.array([[flow_duration, tot_fwd_pkts, 0,0,0,0,0,0,0,0,0]]))  # Dummy
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0].max()
        st.warning("🚨 ATTACK!" if pred == 1 else "✅ Normal", icon="🚨" if pred == 1 else "✅")
        st.write(f"Confidence: {prob:.2f}")
