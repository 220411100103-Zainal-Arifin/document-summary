import streamlit as st
import pandas as pd
import re
import networkx as nx
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.microsoft import EdgeChromiumDriverManager
import nltk
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Fungsi untuk menjalankan WebDriver Edge dengan webdriver_manager
def web_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    service = Service(EdgeChromiumDriverManager().install())
    driver = webdriver.Edge(service=service, options=options)
    return driver

# Fungsi untuk scraping data dari artikel
def extract_article_content(driver, article_url):
    driver.get(article_url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//h1')))
    title = driver.find_element(By.XPATH, './/h1').text.strip()
    date = driver.find_element(By.XPATH, './/p[@class="pt-20 date"]').text.strip()
    content_elements = driver.find_elements(By.XPATH, './/div[@class="news-text"]/p')
    content = " ".join(p.text for p in content_elements)
    driver.quit()
    return title, date, content

# Fungsi untuk preprocessing teks
def preprocess_text(content):
    content = content.lower()
    content = re.sub(r'[0-9]|[/(){}\[\]\|@,;_]|[^a-z .]+', ' ', content)
    content = re.sub(r'\s+', ' ', content).strip()
    tokens = word_tokenize(content)
    stopword = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word not in stopword]
    return ' '.join(tokens)

# Fungsi untuk membuat ringkasan dan visualisasi graf
def summarize_and_visualize(content):
    kalimat = sent_tokenize(content)
    preprocessed_text = preprocess_text(content)
    kalimat_preprocessing = sent_tokenize(preprocessed_text)
    
    # TF-IDF dan cosine similarity
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(kalimat_preprocessing)
    cossim_prep = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Analisis jaringan dengan NetworkX
    G = nx.DiGraph()
    for i in range(len(cossim_prep)):
        G.add_node(i)
        for j in range(len(cossim_prep)):
            if cossim_prep[i][j] > 0.1 and i != j:
                G.add_edge(i, j)
                
    # Hitung closeness centrality dan buat ringkasan
    closeness_scores = nx.closeness_centrality(G)
    sorted_closeness = sorted(closeness_scores.items(), key=lambda x: x[1], reverse=True)
    ringkasan = " ".join(kalimat[node] for node, _ in sorted_closeness[:3])

    # Visualisasi graf
    pos = nx.spring_layout(G, k=2)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='b')
    nx.draw_networkx_edges(G, pos, edge_color='red', arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title("Graph Representation of Sentence Similarity")

    # Tampilkan grafik di Streamlit
    st.pyplot(plt)

    return ringkasan

# Streamlit interface
st.title("Web Article Summarizer with Graph Visualization")

# Input URL
url_input = st.text_input("Masukkan URL artikel:")
if st.button("Generate Summary"):
    if url_input:
        driver = web_driver()
        with st.spinner("Mengambil konten artikel..."):
            title, date, content = extract_article_content(driver, url_input)
        
        if content:
            st.write("**Konten:**", content[:1000] + "...")  # Menampilkan 500 karakter pertama
            with st.spinner("Menganalisis dan membuat ringkasan..."):
                ringkasan = summarize_and_visualize(content)
            st.subheader("Ringkasan Artikel:")
            st.write(ringkasan)
        else:
            st.error("Gagal mengambil konten artikel.")
