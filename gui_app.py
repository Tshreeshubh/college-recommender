# gui_app.py

import streamlit as st
import pandas as pd
from recommender import recommend_colleges

st.set_page_config(page_title="College Recommendation System", layout="wide")

st.title("🎓 College Recommendation System")

# User input form
with st.form("user_input_form"):
    user_course = st.text_input("Enter your preferred course (e.g., Computer Science, BBA)")
    user_location = st.text_input("Preferred location (e.g., Kathmandu)")
    user_university = st.text_input("Preferred university (e.g., Tribhuvan University)")
    user_ownership = st.selectbox("Ownership Type", ["private Institution", "community Institution"])
    top_n = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
    
    submitted = st.form_submit_button("Recommend Colleges")

# Handle submission
if submitted:
    if not user_course or not user_location or not user_university:
        st.warning("❗ Please fill all the fields before submitting.")
    else:
        st.info("🔍 Searching for best matches...")
        recommendations = recommend_colleges(
            user_course=user_course,
            user_location=user_location,
            user_university=user_university,
            user_ownership=user_ownership,
            top_n=top_n
        )

        if recommendations.empty:
            st.error("No recommendations found. Try different inputs.")
        else:
            st.success("✅ Here are your recommended colleges:")
            st.dataframe(recommendations, use_container_width=True)