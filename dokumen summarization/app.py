from flask import Flask, render_template, request
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.microsoft import EdgeChromiumDriverManager

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

def web_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    service = Service(EdgeChromiumDriverManager().install())
    driver = webdriver.Edge(service=service, options=options)
    return driver

def extract_article_content(driver, article_url):
    driver.get(article_url)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//h1'))
    )
    title = driver.find_element(By.XPATH, './/h1').text.strip()
    content_elements = driver.find_elements(By.XPATH, './/div[@class="news-text"]/p')
    content = " ".join(p.text for p in content_elements)
    driver.quit()
    return {"Title": title, "Content": content}

def preprocess_content(content):
    # Lowercase
    text = content.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('indonesian'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Rejoin tokens
    return " ".join(filtered_tokens)

def summarize_content(content):
    sentences = sent_tokenize(content)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    cossim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Graph representation
    G = nx.DiGraph()
    for i in range(len(cossim_matrix)):
        G.add_node(i)

    for i in range(len(cossim_matrix)):
        for j in range(len(cossim_matrix)):
            if cossim_matrix[i][j] > 0.1 and i != j:
                G.add_edge(i, j)

    closeness = nx.closeness_centrality(G)
    sorted_nodes = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
    summary = " ".join(sentences[node] for node, _ in sorted_nodes[:3])
    return summary

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        article_url = request.form['article_url']
        driver = web_driver()
        article_data = extract_article_content(driver, article_url)
        preprocessed_content = preprocess_content(article_data["Content"])
        summary = summarize_content(article_data["Content"])
        return render_template('result.html', title=article_data["Title"], summary=summary, content=article_data["Content"])
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
