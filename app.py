import os
import streamlit as st
import fitz  # PyMuPDF
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# ---- Setup Gemini API ----
GOOGLE_API_KEY = "AIzaSyA02m9uCQYMlftxAsejMTOas8wQCc2DYHY"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
output_parser = StrOutputParser()

# ---- Functions ----
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def analyze_cv(cv_text, job_desc=None):
    prompt_template = "Analyze the following CV and extract key information such as skills, experience, education, and strengths."
    if job_desc:
        prompt_template += f" Then match it against this job description: {job_desc}. Provide a job match percentage and missing skills."

    prompt = ChatPromptTemplate.from_template(prompt_template + "\n\nCV Text:\n{cv}")
    chain = prompt | llm | output_parser
    return chain.invoke({"cv": cv_text})

def get_improvement_tips(cv_text):
    prompt = ChatPromptTemplate.from_template(
        "Act as a professional resume reviewer. Give actionable improvement tips for this CV:\n\n{cv}"
    )
    chain = prompt | llm | output_parser
    return chain.invoke({"cv": cv_text})

def chat_about_cv(cv_text, user_question):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant helping the user understand their resume."),
        ("human", "CV: {cv}\n\nQuestion: {question}")
    ])
    chain = prompt | llm | output_parser
    return chain.invoke({"cv": cv_text, "question": user_question})

# ---- Streamlit App ----
st.set_page_config(page_title="CV Analyzer Chatbot", layout="centered")
st.title("📄🤖 CV Analyzer Chatbot with Gemini")

# Upload CV
cv_file = st.file_uploader("Upload your CV (PDF only):", type=["pdf"])
job_desc = st.text_area("Optional: Paste Job Description to match against CV", height=150)

if cv_file:
    cv_text = extract_text_from_pdf(cv_file)
    st.success("✅ CV uploaded and parsed successfully!")

    # Analyze CV & Match
    if st.button("🔍 Analyze CV"):
        with st.spinner("Analyzing..."):
            result = analyze_cv(cv_text, job_desc)
            st.subheader("📊 CV Analysis Result:")
            st.write(result)

    # Improvement Tips
    if st.button("💡 Get Resume Improvement Tips"):
        with st.spinner("Reviewing CV..."):
            tips = get_improvement_tips(cv_text)
            st.subheader("🛠️ Improvement Suggestions:")
            st.write(tips)

    # Chat About Your CV
    user_question = st.text_input("💬 Ask something about your CV:")
    if st.button("Ask"):
        if user_question:
            with st.spinner("Thinking..."):
                reply = chat_about_cv(cv_text, user_question)
                st.subheader("🤖 Gemini Says:")
                st.write(reply)
        else:
            st.warning("Please enter a question.")
else:
    st.info("Please upload a CV to begin.")
