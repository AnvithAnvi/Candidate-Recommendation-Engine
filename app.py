import streamlit as st
import fitz  # PyMuPDF
import docx
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch

# ----------------------------
# File Parsing Functions
# ----------------------------

def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join(para.text for para in doc.paragraphs)

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

def extract_resume_text(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file)
    elif file.type == "text/plain":
        return extract_text_from_txt(file)
    else:
        return ""

# ----------------------------
# Streamlit App UI
# ----------------------------

st.set_page_config(page_title="Candidate Recommender", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextArea textarea {
        background-color: #ffffff;
    }
    .stFileUploader {
        border: 2px dashed #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 Candidate Recommendation Engine (AI Enhanced)")
st.caption("Match resumes to job descriptions using embeddings and AI-generated justifications.")

# Two-column layout
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📄 Job Description")
    job_desc = st.text_area("Paste the job description:", height=250)

with col2:
    st.subheader("📤 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Supported formats: PDF, DOCX, TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="You can upload multiple resumes. Recommended: max 5 for speed."
    )

submit_btn = st.button("🚀 Find Top Candidates", use_container_width=True)

st.markdown("---")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

model = load_model()
summarizer = load_summarizer()

# ----------------------------
# Matching Logic
# ----------------------------

if submit_btn:
    if not job_desc:
        st.warning("⚠️ Please enter a job description.")
    elif not uploaded_files:
        st.warning("⚠️ Please upload at least one resume.")
    else:
        with st.spinner("🤖 Crunching through resumes..."):
            job_embedding = model.encode(job_desc, convert_to_tensor=True)
            results = []

            for file in uploaded_files[:5]:  # Limit to 5 resumes
                text = extract_resume_text(file)
                if not text.strip():
                    continue

                resume_embedding = model.encode(text, convert_to_tensor=True)
                score = util.cos_sim(job_embedding, resume_embedding).item()

                # Match Level Tag
                if score >= 0.75:
                    tag = "🟢 High Match"
                elif score >= 0.5:
                    tag = "🟡 Medium Match"
                else:
                    tag = "🔴 Low Match"

                # Better AI Summary Prompt
                summary_prompt = f"Summarize this candidate’s professional experience and education:\n{text[:2000]}"
                short_summary = summarizer(summary_prompt, max_length=120, min_length=40, do_sample=False)[0]['summary_text']

                # Better AI Reasoning Prompt
                reasoning_prompt = (
                    f"You are a hiring assistant.\n\n"
                    f"Here is a job description:\n{job_desc}\n\n"
                    f"Here is a summary of a candidate’s resume:\n{short_summary}\n\n"
                    f"Explain in 3-4 clear sentences why this candidate would be a strong match for this job."
                )
                reasoning = summarizer(reasoning_prompt, max_length=100, min_length=40, do_sample=False)[0]['summary_text']

                results.append({
                    "name": file.name,
                    "score": round(score, 3),
                    "text": text[:800],
                    "tag": tag,
                    "summary": short_summary,
                    "reasoning": reasoning
                })

            top_candidates = sorted(results, key=lambda x: x['score'], reverse=True)[:10]

        st.subheader("📊 Top Candidate Matches")

        if top_candidates:
            for i, res in enumerate(top_candidates, 1):
                with st.expander(f"{i}. {res['name']}  —  Score: {res['score']}  {res['tag']}", expanded=(i==1)):
                    st.progress(res['score'])
                    st.markdown(f"**Match Level:** {res['tag']}")
                    st.markdown(f"**🧠 Resume Summary:** {res['summary']}")
                    st.markdown(f"**🤖 Why a Great Fit:** {res['reasoning']}")
                    st.text_area("📝 Resume Preview", res['text'], height=180)
        else:
            st.warning("No valid resumes with content found.")

        st.success("✅ Done! Candidates ranked by similarity score.")
