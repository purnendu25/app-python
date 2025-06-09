import streamlit as st
import pickle
import re
import nltk
import pandas as pd
import requests
from docx import Document
import fitz  # PyMuPDF
from streamlit_lottie import st_lottie
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time
import random

# Setup
st.set_page_config(
    page_title="🚀 Resume Screener AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(to right, #fceabb, #f8b500);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 10px 24px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #388e3c;
        color: white;
    }
    .footer {
        background-color: #fafafa;
        text-align: center;
        color: #666;
        padding: 15px;
        position: fixed;
        bottom: 0;
        width: 100%;
        font-size: 13px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <h1 style="text-align: center; color: #4a90e2;">🚀 Dynamic Resume Screening App</h1>
    <p style="text-align: center; font-size: 1.2rem;">Upload your resume to get an AI-based career prediction!</p>
""", unsafe_allow_html=True)

# Lottie animation in sidebar


def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


with st.sidebar:
    st_lottie(load_lottie_url(
        "https://assets1.lottiefiles.com/packages/lf20_jcikwtux.json"), height=200)
    st.markdown("### 🧠 How It Works")
    st.write("We use ML and NLP to predict your domain from your resume.")

# Load models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

category_mapping = {
    6: "Data Science", 12: "HR", 0: 'Advocate', 1: 'Arts', 24: 'Web Designing',
    16: 'Mechanical Engineer', 22: 'Sales', 14: 'Health and fitness', 5: 'Civil Engineer',
    15: 'Java Developer', 4: 'Business Analyst', 21: 'SAP Developer', 2: 'Automation Testing',
    11: 'Electrical Engineering', 18: 'Operations Manager', 20: 'Python Developer',
    8: 'DevOps Engineer', 17: 'Network Security Engineer', 19: 'PMO', 7: 'Database',
    13: 'Hadoop', 10: 'ETL Developer', 9: 'DotNet Developer', 3: 'Blockchain', 23: "Testing"
}

emoji_map = {
    "Data Science": "📊", "HR": "🧑‍💼", "Advocate": "⚖️", "Arts": "🎨", "Web Designing": "🖥️",
    "Mechanical Engineer": "⚙️", "Sales": "💼", "Health and fitness": "🏋️‍♂️", "Civil Engineer": "🏗️",
    "Java Developer": "☕", "Business Analyst": "📈", "SAP Developer": "🗃️", "Automation Testing": "🤖",
    "Electrical Engineering": "🔌", "Operations Manager": "🗂️", "Python Developer": "🐍",
    "DevOps Engineer": "🚀", "Network Security Engineer": "🛡️", "PMO": "📅", "Database": "🗄️",
    "Hadoop": "☁️", "ETL Developer": "🔄", "DotNet Developer": "🖥️", "Blockchain": "⛓️", "Testing": "🧪",
}

# Preload NLTK data
for pkg in ["punkt", "stopwords"]:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg ==
                       "punkt" else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

# Helper Functions


def clean_resume(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            return "".join(page.get_text() for page in doc)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        try:
            return uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            return uploaded_file.read().decode("latin-1")


def generate_wordcloud(text):
    return WordCloud(background_color='white', max_words=100,
                     max_font_size=60, random_state=42).generate(text)

# Main Function


def main():
    uploaded_file = st.file_uploader(
        "📤 Upload your resume (PDF/DOCX/TXT)", type=['pdf', 'docx', 'txt'])

    if uploaded_file:
        resume_text = extract_text_from_file(uploaded_file)

        if not resume_text.strip():
            st.error("Couldn't extract any text. Try another resume file.")
            return

        with st.spinner("⏳ Analyzing your resume..."):
            time.sleep(1.5)

        cleaned_resume = clean_resume(resume_text)
        try:
            input_feature = tfidf.transform([cleaned_resume])
            prediction_id = clf.predict(input_feature)[0]
            category_name = category_mapping.get(prediction_id, "Unknown")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        st.balloons()

        tab1, tab2, tab3 = st.tabs(["📝 Summary", "📊 Details", "☁️ Word Cloud"])

        with tab1:
            st.success(f"**Predicted Category:** {category_name}")
            emoji = emoji_map.get(category_name, "🔎")
            st.markdown(
                f"### {emoji} You are a great fit for **{category_name}**!")
            st.write("Here's a snippet of your resume text:")
            st.code(
                resume_text[:500] + ("..." if len(resume_text) > 500 else ""), language="text")

        with tab2:
            st.markdown("### 📈 Prediction Confidence Scores")
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(input_feature)[0]
                prob_df = pd.DataFrame({
                    "Category": [category_mapping.get(i, "Unknown") for i in range(len(probs))],
                    "Probability": probs
                }).sort_values(by="Probability", ascending=False)
                st.bar_chart(prob_df.set_index("Category")["Probability"])
            else:
                st.warning(
                    "Prediction probabilities not available for this model.")

            st.markdown("### 📄 Full Extracted Resume Text")
            st.text_area("", resume_text, height=300)

        with tab3:
            st.markdown("### ☁️ Word Cloud of Resume")
            wc = generate_wordcloud(cleaned_resume)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

    else:
        st.info("👈 Upload a resume file from the left sidebar to get started!")

    st.markdown("""
        <div class="footer">
            &copy; 2025 Made with ❤️ by <strong>Purnendu Tiwari</strong>, <strong>Neha Sharma</strong>, and <strong>Anshul Sharma</strong>.
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
