import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv(r'C:\Users\USER\OneDrive\Desktop\compres\[3] DATA_FINAL (1).csv')
file_path = 'comppress file2.zip'

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['Skill'])

def get_job_recommendations(skills, top_n=5):
    skills = skills.lower()
    skills_vector = vectorizer.transform([skills])
    similarity_scores = cosine_similarity(skills_vector, X_tfidf).flatten()
    df['Similarity Score'] = similarity_scores
    recommendations = df.sort_values(by='Similarity Score', ascending=False).head(top_n)
    jobs = recommendations[['Company', 'Job','Job Description',  'Skill', 'Minimum Salary', 'Maximum Salary','Payment Periode', 'Similarity Score']]
    return jobs


st.title('Job Recommender System')

user_name = st.text_input('Enter your name')

user_skills = st.text_input('Enter your skill (only 1 skill):')

if st.button('Get Recommendations'):
    if user_skills:
        recommendations = get_job_recommendations(user_skills)
        st.write("Job Recommendations:")
        st.write(recommendations)
    else:
        st.write("Please enter your skills.")




