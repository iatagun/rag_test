# test_and_txt_input.py
# Eğitilen modelin test edilmesi ve eğitim verisinin bir .txt dosyasından alınması

import torch
import pickle
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt')

# Eğitimde kullanılan ayarları tekrar tanımla
embedding_dim = 100

# Model sınıfı tekrar tanımlanmalı
class WordEmbeddingModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embed_dim)

    def forward(self, inputs):
        return self.embeddings(inputs)

# Vocab ve model yükle
with open("word_to_idx.pkl", "rb") as f:
    word_to_idx = pickle.load(f)
with open("idx_to_word.pkl", "rb") as f:
    idx_to_word = pickle.load(f)

model = WordEmbeddingModel(len(word_to_idx), embedding_dim)
model.load_state_dict(torch.load("word_embedding_model.pt", map_location=torch.device('cpu')))
model.eval()

# Anlamsal benzerlik testi fonksiyonu
with torch.no_grad():
    embeddings = model.embeddings.weight.cpu().numpy()
    similarity_matrix = cosine_similarity(embeddings)

    def get_similar_words(word, topn=5):
        if word not in word_to_idx:
            print(f"'{word}' kelimesi modelde bulunamadı.")
            return
        idx = word_to_idx[word]
        similarities = similarity_matrix[idx]
        top_indices = np.argsort(similarities)[::-1][1:topn+1]
        return [(idx_to_word[i], similarities[i]) for i in top_indices]

# Test örneği
kelime = "sistem"
sonuclar = get_similar_words(kelime)
if sonuclar:
    print(f"'{kelime}' kelimesine en yakın {len(sonuclar)} kelime:")
    for s, score in sonuclar:
        print(f"  {s} (benzerlik: {score:.4f})")
