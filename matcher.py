from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess_text
from page_extraction import page_extract

def calculate_similarity(resume_text, job_text):

    if not resume_text.strip() or not job_text.strip():
        raise ValueError(f"Fields cannot be empty")
    
    vec=TfidfVectorizer()

    resume_text=preprocess_text(resume_text)
    job_text=preprocess_text(job_text)

    doc=[resume_text, job_text]
    matrix=vec.fit_transform(doc)

    sim=cosine_similarity(matrix[0:1],matrix[1:2])[0][0]

    return int(sim*100)

    
