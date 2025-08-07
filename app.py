# Optimized app.py for Streamlit Community Cloud

import streamlit as st
import fitz  # PyMuPDF
import docx
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ----------------------------
# Load Lightweight Summarizer Model
# ----------------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

# ----------------------------
# Utility Functions
# ----------------------------

def chunk_text(text, max_chunk=1000):
    paragraphs = text.split('\n')
    chunks = []
    chunk = ''
    for para in paragraphs:
        if len(chunk) + len(para) < max_chunk:
            chunk += ' ' + para
        else:
            chunks.append(chunk.strip())
            chunk = para
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def summarize(text):
    try:
        trimmed = text[:1000]
        result = summarizer(trimmed, max_length=120, min_length=40, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        return f"⚠️ Summary Error: {e}"

def explain_fit(resume_text, job_desc):
    try:
        prompt = f"""
        Here is a candidate resume:
        {resume_text[:1000]}

        And here is a job description:
        {job_desc[:1000]}

        Why is this candidate a strong match? Provide a concise 3-4 sentence explanation.
        """
        result = summarizer(prompt, max_length=120, min_length=50, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        return f"⚠️ Reasoning Error: {e}"

# ----------------------------
# Resume Extraction
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
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Candidate Recommender", layout="wide")
st.title("🔍 Candidate Recommendation Engine")
st.caption("Match resumes to job descriptions using efficient AI summarization.")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📄 Job Description")
    job_desc = st.text_area("Paste the job description:", height=250)

with col2:
    st.subheader("📤 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT resumes (Max 5 resumes recommended)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

submit_btn = st.button("🚀 Find Top Candidates", use_container_width=True)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

if submit_btn:
    if not job_desc:
        st.warning("⚠️ Please enter a job description.")
    elif not uploaded_files:
        st.warning("⚠️ Please upload at least one resume.")
    else:
        with st.spinner("Analyzing resumes..."):
            job_embedding = model.encode(job_desc, convert_to_tensor=True)
            results = []

            for file in uploaded_files[:5]:  # Limit for performance
                text = extract_resume_text(file)
                if not text.strip():
                    continue

                resume_embedding = model.encode(text, convert_to_tensor=True)
                score = util.cos_sim(job_embedding, resume_embedding).item()

                tag = "🟢 High Match" if score >= 0.75 else "🟡 Medium Match" if score >= 0.5 else "🔴 Low Match"
                summary = summarize(text)
                reasoning = explain_fit(text, job_desc)

                results.append({
                    "name": file.name,
                    "score": round(score, 3),
                    "summary": summary,
                    "reasoning": reasoning,
                    "tag": tag,
                    "text": text[:1000]
                })

            top_candidates = sorted(results, key=lambda x: x['score'], reverse=True)

        st.subheader("📊 Top Matches")
        for i, res in enumerate(top_candidates, 1):
            with st.expander(f"{i}. {res['name']} — Score: {res['score']} {res['tag']}", expanded=(i==1)):
                st.markdown(f"**🧠 Summary:** {res['summary']}")
                st.markdown(f"**🤖 Why a Great Fit:** {res['reasoning']}")
                st.text_area("📄 Resume Preview", res['text'], height=160)

        st.success("✅ Done! Candidates ranked by relevance.")
