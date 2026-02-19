import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Fake Account Detector", page_icon="🔍")

st.title("🔍 Fake Instagram Account Detector")
st.markdown("Enter basic account details to analyze authenticity.")

st.divider()

# ==========================
# USER INPUTS
# ==========================

username = st.text_input("Enter Username")

profile_pic = st.selectbox("Has Profile Picture?", ["Yes", "No"])
followers = st.number_input("Number of Followers", 0, 1000000, 0)
follows = st.number_input("Number Following", 0, 1000000, 0)
posts = st.number_input("Number of Posts", 0, 100000, 0)
description_length = st.number_input("Bio Length", 0, 500, 0)

st.divider()

# ==========================
# PROCESSING INPUTS
# ==========================

if st.button("Analyze Account"):

    profile_pic = 1 if profile_pic == "Yes" else 0

    # 🔥 Extract features from username
    username_length = len(username)
    digit_count = sum(c.isdigit() for c in username)
    nums_username = digit_count / (username_length + 1)

    # Default values (for removed features)
    fullname_words = 1
    nums_fullname = 0
    name_eq_username = 0
    external_url = 0
    private = 0

    # 🔥 Feature Engineering
    ff_ratio = followers / (follows + 1)
    posts_per_follower = posts / (followers + 1)
    has_description = 1 if description_length > 0 else 0

    # Feature order MUST match training
    features = np.array([[profile_pic,
                          nums_username,
                          fullname_words,
                          nums_fullname,
                          name_eq_username,
                          description_length,
                          external_url,
                          private,
                          posts,
                          followers,
                          follows,
                          ff_ratio,
                          posts_per_follower,
                          has_description]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Likely FAKE Account (Confidence: {probability:.2f})")
    else:
        st.success(f"✅ Likely REAL Account (Confidence: {1 - probability:.2f})")
