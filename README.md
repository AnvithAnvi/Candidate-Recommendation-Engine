# Candidate-Recommendation-Engine

This web app helps match candidates to a job description based on semantic similarity. It allows the user to upload multiple resumes and receive ranked recommendations along with AI-generated summaries.

---

## ✅ How It Works

### 1. **Input**
- **Job Description**: A text box to paste the job role.
- **Candidate Resumes**: Supports file uploads in PDF, DOCX, and TXT format.

### 2. **Processing**
- Extracts and cleans text from each resume.
- Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to generate embeddings for both the job description and resumes.
- Computes cosine similarity between job and each resume embedding.

### 3. **Output**
- Displays the top 5–10 candidates with:
  - File name (ID)
  - Similarity score
  - AI-generated summary (local model)
  - Resume preview

---

## ⚙️ Tech Stack

- **Framework**: Streamlit (for rapid prototyping and interactive UI)
- **Embedding Model**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Summarizer Model**: `facebook/bart-large-cnn` via Hugging Face Transformers (no API key required)
- **Text Extraction**: PyMuPDF (`fitz`), python-docx

---

## 🧪 Assumptions

- Resumes are expected to be in English and in standard formats (PDF/DOCX/TXT).
- Resume filenames are used as candidate IDs.
- AI summaries are generated from the first ~1000 characters of each resume to stay within model limits.
- The similarity threshold for tagging High/Medium/Low matches is arbitrarily chosen for demonstration.

---

## 💡 Bonus Implementation

- ✅ **AI-based summaries** are generated using a local transformer model (no OpenAI or external API required).
- ✅ The app is self-contained and works without internet after model download.
- ✅ Highlights top candidates with a clear visual cue and ranking.

---

## 📂 Files Included

- `app.py` – Main Streamlit app
- `README.md` – This file
- `requirements.txt` – All Python dependencies

---

## 🚀 To Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
