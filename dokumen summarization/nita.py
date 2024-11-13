import streamlit as st
import pandas as pd
import re
import networkx as nx
import nltk
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Fungsi untuk scraping data dari artikel menggunakan BeautifulSoup
def crawl_article(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Pastikan permintaan berhasil
        soup = BeautifulSoup(response.content, 'html.parser')

        # Mengambil judul
        title_element = soup.find('h1', class_='text-cnn_black')
        title = title_element.get_text().strip() if title_element else 'Judul tidak ditemukan'

        # Mengambil Isi
        content_div = soup.find('div', class_='detail-text')
        content = "\n".join([p.get_text().strip() for p in content_div.find_all('p')]) if content_div else 'Isi artikel tidak ditemukan'

        # Mengambil tanggal
        date_div = soup.find('div', class_='text-cnn_grey text-sm mb-4')
        date_text = date_div.text.strip() if date_div else 'Tanggal tidak ditemukan'

        return title, date_text, content
    except requests.RequestException as e:
        st.error(f"Error fetching article: {e}")
        return None, None, None

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
        with st.spinner("Mengambil konten artikel..."):
            title, date, content = crawl_article(url_input)
        
        if content:
            st.write("**Judul:**", title)
            st.write("**Tanggal:**", date)
            st.write("**Konten:**", content[:1000] + "...")  # Menampilkan 1000 karakter pertama
            with st.spinner("Menganalisis dan membuat ringkasan..."):
                ringkasan = summarize_and_visualize(content)
            st.subheader("Ringkasan Artikel:")
            st.write(ringkasan)
        else:
            st.error("Gagal mengambil konten artikel.")
