import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("Mini-pjt.pkl")


st.title("🔥 Burnout Risk Predictor")
st.write("An AI-powered demo for HRs & CEOs to simulate employee burnout risk.")

# Input sliders
meetings = st.slider("Number of Meetings", 0, 20, 5)
meeting_minutes = st.slider("Total Meeting Minutes", 0, 500, 120)
back_to_back = st.slider("Back-to-Back Meetings", 0, 10, 2)
sleep = st.slider("Sleep Hours (Last Night)", 0, 10, 6)
exercise = st.slider("Exercise Minutes", 0, 120, 30)
after_hours_work = st.slider("After-Hours Work Minutes", 0, 300, 60)

# Create input vector (match training order!)
X_input = np.array([[meetings, meeting_minutes, back_to_back, 0,  # avg_meeting_gap placeholder
                     0, 0, 0, after_hours_work, sleep, exercise,
                     0, 0, 0, 0, 0, 0, 0, 0]])

# Prediction
pred = model.predict(X_input)[0]
proba = model.predict_proba(X_input)[0]

st.subheader(f"Predicted Burnout Risk: **{pred}**")
st.write("Probabilities:")

# Add labels to probabilities
labels = ["Low", "Moderate", "High"]
prob_df = pd.DataFrame({
    "Risk Level": labels,
    "Probability": proba[0]
})

# Find the max probability index
max_idx = prob_df["Probability"].idxmax()

# Apply styling → highlight max row
def highlight_max(s):
    return ['background-color: lightgreen' if i == max_idx and s.name == "Risk Level" 
            else 'background-color: lightgreen' if i == max_idx and s.name == "Probability" 
            else '' for i in range(len(s))]

st.dataframe(prob_df.style.apply(highlight_max))

