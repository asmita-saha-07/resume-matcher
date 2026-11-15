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

resume_text = '''Michael Anderson

**Machine Learning Engineer**

**Professional Summary:**
Computer Science graduate with 3 years of experience in machine learning, data analysis, and software development. Experienced in building predictive models, developing NLP applications, and deploying machine learning solutions.

**Technical Skills:**
Python, Machine Learning, Deep Learning, Natural Language Processing, TensorFlow, Keras, Scikit-learn, Pandas, NumPy, Matplotlib, SQL, Flask, Streamlit, Git, Data Preprocessing, Feature Engineering, Model Evaluation, Classification, Regression, Neural Networks.

**Work Experience:**

**Machine Learning Engineer – DataTech Solutions (2022–2025)**
- Developed machine learning models for customer segmentation and sales forecasting.
- Built classification models using Random Forest, Logistic Regression, and Support Vector Machines.
- Developed NLP-based applications for text classification and sentiment analysis.
- Improved model accuracy by 18% through feature engineering and hyperparameter tuning.
- Deployed machine learning models using Flask and Streamlit.
- Collaborated with software engineers to integrate predictive models into applications.

**Data Analyst – Insight Analytics (2021–2022)**
- Performed exploratory data analysis using Python, Pandas, and NumPy.
- Created visualizations and reports using Matplotlib and Power BI.
- Used SQL queries to extract, clean, and analyze business data.
- Assisted in developing predictive analytics solutions.

**Education:**
Bachelor of Technology in Computer Science – 2021

**Projects:**
- Customer Churn Prediction using Artificial Neural Networks.
- Spam Email Classification using NLP and Naive Bayes.
- House Price Prediction using Random Forest Regression.'''

job_text="""Machine Learning Engineer

We are looking for a skilled Machine Learning Engineer to design, develop, and deploy machine learning solutions that solve real-world business problems.

**Responsibilities:**
- Design and implement machine learning algorithms and predictive models.
- Collect, clean, preprocess, and analyze large datasets.
- Perform feature engineering and feature selection.
- Develop classification, regression, and clustering models.
- Build and optimize deep learning models using neural networks.
- Develop NLP applications for text classification and language processing.
- Evaluate model performance using appropriate metrics.
- Optimize models through hyperparameter tuning.
- Deploy machine learning models using APIs and web frameworks.
- Collaborate with data scientists and software engineers.

**Required Qualifications:**
- Bachelor's degree in Computer Science, Artificial Intelligence, Data Science, or a related field.
- 2+ years of experience in machine learning or data science.
- Strong programming skills in Python.
- Knowledge of supervised and unsupervised machine learning algorithms.
- Experience with data preprocessing, feature engineering, and model evaluation.
- Understanding of deep learning concepts and neural networks.
- Familiarity with NLP techniques.

**Technical Skills:**
Python, Scikit-learn, TensorFlow, Keras, PyTorch, Pandas, NumPy, SQL, Machine Learning, Deep Learning, Natural Language Processing, Neural Networks, Feature Engineering, Model Deployment, Flask, FastAPI, REST APIs, Git.

**Preferred Qualifications:**
- Experience deploying machine learning models in production environments.
- Knowledge of cloud platforms such as AWS, Azure, or Google Cloud.
- Experience building scalable machine learning applications.
- Familiarity with Docker and CI/CD pipelines."""
print(calculate_similarity(resume_text, job_text))
    
    