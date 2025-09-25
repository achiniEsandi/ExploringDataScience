import streamlit as st
from openai import OpenAI
import pandas as pd
import datetime
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
print(os.getenv("OPENAI_API_KEY"))

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Visionary – Career Path Finder")
st.title("🎯 Visionary – AI Career Path Finder")
st.write("Get personalized career suggestions based on your skills and interests.")

# User inputs
skills = st.text_input("🧠 Enter your skills (comma-separated)")
interests = st.text_input("💡 Enter your interests")
goals = st.text_area("🌟 What's your ideal job or work lifestyle?")

# Optional resume upload
uploaded_file = st.file_uploader("📄 Upload Resume (PDF - Optional)", type="pdf")
resume_text = ""

if uploaded_file is not None:
    import pdfplumber
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                resume_text += text

if st.button("🚀 Find My Career Path"):
    with st.spinner("AI is analyzing your inputs..."):
        # Combine resume text and user inputs for better results
        combined_info = f"""
        Skills: {skills}
        Interests: {interests}
        Career Goals: {goals}
        Resume: {resume_text[:1000]}...
        """

        prompt = f"""
        You are an experienced career advisor.
        Based on the following user details:
        {combined_info}

        Suggest 3 modern career paths.
        For each, include:
        1. Job Title
        2. 1-2 sentence description
        3. 3 skills they should improve
        4. Why this matches the user
        """

        result = ""  # Initialize result
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )

            result = response.choices[0].message.content
        except Exception as e:
            st.error(f"An error occurred: {e}")

        st.markdown("### 🧭 AI Suggestions")
        if result:
            st.success(result)

        # Save to history
        data = {
            "timestamp": [datetime.datetime.now()],
            "skills": [skills],
            "interests": [interests],
            "goals": [goals],
            "result": [result]
        }

        df = pd.DataFrame(data)
        if not os.path.exists("career_history.csv"):
            df.to_csv("career_history.csv", index=False)
        else:
            df.to_csv("career_history.csv", mode='a', header=False, index=False)

        # Download button
        st.download_button(
            label="📥 Download Suggestions",
            data=result,
            file_name="career_suggestions.txt",
            mime="text/plain"
        )
