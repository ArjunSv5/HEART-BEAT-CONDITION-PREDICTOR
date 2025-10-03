import streamlit as st
import pandas as pd
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ========================
#  FEATURE EXTRACTION
# ========================
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfccs, axis=1)
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# ========================
#  TRAIN MODEL
# ========================
@st.cache_resource
def train_model(csv_file, base_path):
    df = pd.read_csv(csv_file)

    features, labels = [], []
    for index, row in df.iterrows():
        file_name = row['fname'].strip()
        file_path = os.path.join(base_path, os.path.basename(file_name))
        label = row['label']
        if pd.notna(label):
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(label)

    X = np.array(features)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, acc, conf_matrix, report

# ========================
#  STREAMLIT FRONTEND
# ========================
st.title("💓 Heartbeat Condition Predictor")

# Train model once
csv_file = "set_a.csv"
base_path = "set_a"

with st.spinner("Training model... Please wait ⏳"):
    model, acc, conf_matrix, report = train_model(csv_file, base_path)

st.success(f"✅ Model trained with accuracy: {acc*100:.2f}%")

# Upload audio file
uploaded_file = st.file_uploader("Upload a heartbeat audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    features = extract_features("temp_audio.wav")

    if features is not None:
        prediction = model.predict([features])[0]
        st.subheader("🔎 Prediction Result:")
        st.success(f"Condition: **{prediction}**")

# Show classification report
if st.checkbox("Show Classification Report"):
    st.write(pd.DataFrame(report).transpose())
