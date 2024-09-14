import streamlit as st
import PyPDF2
import docx
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import StringIO
import string

nltk.download('punkt')


cv = CountVectorizer()



# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text
# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for para in doc.paragraphs:
        text += para.text
    return text
# Function to preprocess text

def preprocess_text(text):
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove any extra spaces between words
    text = ' '.join(text.split())
    
    return text






def removing_stopwords(texts):
    sw= set(stopwords.words("english"))
    tokenize_word_resume = word_tokenize(texts)
    text = []
    for word in tokenize_word_resume:
        if word not in sw :
            text.append(word)
            text = " ".join(texts)
            return text


# Function to calculate similarity using TF-IDF and cosine similarity
def calculate_similarity(resume_text, job_description_text):
    # Preprocess the text
    resume_texts = preprocess_text(resume_text)
    job_description_texts = preprocess_text(job_description_text)
    
    
    resume_stopwords =removing_stopwords(resume_texts)
    desc_stopwords =removing_stopwords(job_description_texts)


    content = [resume_stopwords,desc_stopwords]
    matrix = cv.fit_transform(content)
    
    # Compute cosine similarity
    similarity_mat = cosine_similarity(matrix)
    print("Resume matches by :" + str(similarity_mat[1][0]* 100)+ "%")
    return similarity_mat
# Streamlit UI
st.title("ATS Resume Checker with Machine Learning")
st.write("Upload your resume and the job description to check the ATS score using a machine learning model.")
# Upload resume file
resume_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])
job_description = st.text_area("Paste the job description")
if resume_file is not None and job_description:
    # Extract text from the uploaded resume file
    if resume_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(resume_file)
    elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume_text = extract_text_from_docx(resume_file)
    else:
        stringio = StringIO(resume_file.getvalue().decode("utf-8"))
        resume_text = stringio.read()
    # Calculate similarity score using the model
    similarity_mat = calculate_similarity(resume_text, job_description)


    
    st.write(f"**ATS Score: {str(similarity_mat[1][0]* 100)}%**")

   


    
