import streamlit as st
import fitz  # PyMuPDF
import docx
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ----------------------------
# Load Summarizer Model (BART)
# ----------------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# ----------------------------
# Elaborate + Short Summarization Logic
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

def summarize_resume(text, summary_type="Elaborate (paragraph)"):
    try:
        if summary_type == "Short (bullet points)":
            trimmed = text[:1000]
            result = summarizer(
                f"Summarize this resume into 3-5 bullet points: {trimmed}",
                max_length=150, min_length=60, do_sample=False
            )
            return result[0]['summary_text']
        else:
            chunks = chunk_text(text, max_chunk=1000)
            summaries = []
            for chunk in chunks:
                result = summarizer(chunk, max_length=250, min_length=100, do_sample=False)
                summaries.append(result[0]['summary_text'])
            return "\n\n".join(summaries)
    except Exception as e:
        return f"⚠️ Error generating summary: {e}"

# ----------------------------
# Resume Parsing Functions
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

st.title("🔍 Candidate Recommendation Engine")
st.caption("Match resumes to job descriptions")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📄 Job Description")
    job_desc = st.text_area("Paste the job description:", height=250)

with col2:
    st.subheader("📄 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Supported formats: PDF, DOCX, TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="You can upload multiple resumes."
    )

summary_type = st.selectbox(
    "Choose summary type:",
    options=["Elaborate (paragraph)", "Short (bullet points)"],
    index=0,
    help="Select how detailed the AI-generated summary should be."
)

submit_btn = st.button("🚀 Find Top Candidates", use_container_width=True)

st.markdown("---")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ----------------------------
# Matching Logic
# ----------------------------

if submit_btn:
    if not job_desc:
        st.warning("⚠️ Please enter a job description.")
    elif not uploaded_files:
        st.warning("⚠️ Please upload at least one resume.")
    else:
        st.success("✅ Analyzing resumes against the job description...")
        job_embedding = model.encode(job_desc, convert_to_tensor=True)
        results = []

        for file in uploaded_files:
            text = extract_resume_text(file)
            if not text.strip():
                continue

            resume_embedding = model.encode(text, convert_to_tensor=True)
            score = util.cos_sim(job_embedding, resume_embedding).item()

            if score >= 0.75:
                tag = "🟢 High Match"
            elif score >= 0.5:
                tag = "🟡 Medium Match"
            else:
                tag = "🔴 Low Match"

            summary = summarize_resume(text, summary_type)

            # AI Justification Summary
            reasoning_prompt = f"""Given the following resume, explain why this candidate should be chosen over others for the job description.

Resume:
{text[:1500]}

Job Description:
{job_desc[:1000]}
"""
            justification = summarize_resume(reasoning_prompt, summary_type)

            results.append({
                "name": file.name,
                "score": round(score, 3),
                "text": text[:1000],
                "tag": tag,
                "summary": summary,
                "justification": justification
            })

        top_candidates = sorted(results, key=lambda x: x['score'], reverse=True)[:10]

        st.subheader("📊 Top Candidate Matches")

        if top_candidates:
            for i, res in enumerate(top_candidates, 1):
                with st.expander(f"{i}. {res['name']}  —  Score: {res['score']}  {res['tag']}", expanded=(i==1)):
                    st.progress(res['score'])
                    st.markdown(f"**Match Level:** {res['tag']}")
                    st.markdown("**🧠 Resume Summary :**")
                    st.write(res['summary'])
                    st.markdown("**🤖 Why Choose This Candidate :**")
                    st.write(res['justification'])
                    st.markdown("**📄 Resume Preview:**")
                    st.text_area("Preview", res['text'], height=180)
        else:
            st.warning("No valid resumes with content found.")

        st.markdown("---")
        
        st.success("🎉 Done! Candidates ranked by similarity score.")
