import streamlit as st
from page_extraction import page_extract
from preprocessing import preprocess_text
from skills_extraction import extract_skills
from matcher import calculate_similarity
import html

st.set_page_config(
    page_title="Resume Matcher",
    layout="wide")

st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    text-align: center;
    padding: 35px 10px 30px 10px;
}

.hero h1 {
    font-size: 46px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 12px;
}

.hero p {
    color: #a5a5b5;
    font-size: 17px;
}

.section-heading {
    font-size: 22px;
    font-weight: 700;
    color: white;
    margin-bottom: 15px;
}

div.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #a855f7);
    color: white;
    border: none;
    border-radius: 10px;
    height: 3.2em;
    font-size: 17px;
    font-weight: 700;
    width: 100%;
}

div.stButton > button:hover {
    border: 1px solid #c4b5fd;
    color: white;
}

.skill-tag {
    display: inline-block;
    padding: 7px 13px;
    margin: 5px 5px 5px 0;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 500;
}

.matched {
    background-color: #123b2b;
    color: #86efac;
    border: 1px solid #166534;
}

.missing {
    background-color: #431e25;
    color: #fca5a5;
    border: 1px solid #991b1b;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <h2>Match Your Resume</h2>
    <h4>Find the skills you’re missing before recruiters do.</h4>
    <p>
        Analyze your resume against any job description.
        Discover your strengths and identify missing skills.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container(border=True):
        st.subheader("Your Resume")
        st.caption("Upload your resume in PDF format.")

        resume_file = st.file_uploader(
            "Choose your resume",
            type=["pdf"],
            label_visibility="collapsed"
        )

with col2:
    with st.container(border=True):
        st.subheader("Job Description")
        st.caption("Paste the job description below.")

        job_text = st.text_area(
            "Job Description",
            placeholder="Paste the complete job description here...",
            height=68,
            label_visibility="collapsed"
        )

st.write("")

analyze = st.button(
    "Analyse My Resume",
    use_container_width=False
)

if analyze:

    if resume_file is None:
        st.warning("Please upload your resume first.")

    elif not job_text.strip():
        st.warning("Please paste a job description.")

    else:
        try:
            with st.spinner("Analyzing your resume..."):

                resume_text=page_extract(resume_file)

                if not resume_text.strip():
                    st.error("Extraction failed")
                    st.stop()

                resume_skills=set(extract_skills(resume_text))
                job_skills=set(extract_skills(job_text))
                
                resume_text=preprocess_text(resume_text)
                job_text=preprocess_text(job_text)

                sim=calculate_similarity(resume_text, job_text)

                matched=sorted(resume_skills.intersection(job_skills))
                missing=sorted(job_skills-resume_skills)

                st.divider()

            st.markdown(
                '<div class="section-heading">Your Match Results</div>',
                unsafe_allow_html=True
            )

            # Similarity score
            with st.container(border=True):

                st.subheader("Resume–Job Similarity")

                st.metric(
                    "Text Similarity Score",
                    f"{sim}%"
                )

                st.progress(
                    max(0.0, min(float(sim) / 100, 1.0))
                )

                st.caption(
                    "This measures textual similarity, not the "
                    "probability of getting hired."
                )

            # Matched and missing skills
            col3, col4 = st.columns(2, gap="large")

            with col3:
                with st.container(border=True):

                    st.subheader("Matched Skills")

                    st.metric("Skills found", len(matched))

                    if matched:
                        tags = "".join(
                            f'<span class="skill-tag matched">'
                            f'{html.escape(skill)}</span>'
                            for skill in matched
                        )

                        st.markdown(tags, unsafe_allow_html=True)

                    else:
                        st.info("No matching skills detected.")

            with col4:
                with st.container(border=True):

                    st.subheader("Missing Skills")

                    st.metric("Skills to work on", len(missing))

                    if missing:
                        tags = "".join(
                            f'<span class="skill-tag missing">'
                            f'{html.escape(skill)}</span>'
                            for skill in missing
                        )

                        st.markdown(tags, unsafe_allow_html=True)

                    else:
                        st.success("No missing skills detected!")

            st.divider()

            st.caption(
                "Resume analysis powered by NLP"
            )


        except Exception as e:
            st.exception(e)