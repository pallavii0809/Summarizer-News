import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import os

# Load environment variables
load_dotenv()

# Set up Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# Prompt template for summarization
summarize_prompt = PromptTemplate(
    template="Summarize the following news article:\n\n{article}\n\nSummary:",
    input_variables=["article"]
)

# Create LLMChain
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

# News extraction logic
def extract_news(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        return f"Failed to fetch news from {url}: {e}"

# Summarization logic
def summarize_news(url):
    article = extract_news(url)
    if article.startswith("Failed to fetch"):
        return article, None
    summary = summarize_chain.run(article=article)
    return summary, article

# Streamlit UI
st.set_page_config(page_title="🗞️ News Summarizer with Gemini", layout="centered")

st.title("🗞️ News Summarizer")
st.caption("Paste a URL from any news article, and get a concise summary powered by Google's Gemini.")

url = st.text_input("🔗 Enter News Article URL")

if st.button("Summarize"):
    if url:
        with st.spinner("Fetching and summarizing..."):
            summary, article = summarize_news(url)
        if "Failed to fetch" in summary:
            st.error(summary)
        else:
            st.subheader("✅ Summary")
            st.write(summary)

            with st.expander("📄 View Full Article Text"):
                st.write(article)
    else:
        st.warning("Please enter a valid URL.")

# Footer
st.markdown("---")
st.markdown("Created with 💡 using LangChain & Gemini")
