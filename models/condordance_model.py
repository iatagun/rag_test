# Concordance + Semantic Relation Model
# Bu örnek: bir metinde sözcüklerin bağlam içerisindeki ilişkilerini tespit eder
# ve bir basit sinir ağı modeli ile anlamsal benzerlikleri öğrenir.

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
from nltk.corpus import stopwords
import pickle

nltk.download('punkt')
nltk.download('stopwords')

# Örnek metin (kendi metninizi buraya koyabilirsiniz)
text = "C://Users//user//OneDrive//Belgeler//GitHub//rag_test//models//data.txt"
with open(text, 'r', encoding='utf-8') as file:
    text = file.read()

# Ön işleme
tokens = word_tokenize(text.lower())
stop_words = set(stopwords.words('turkish'))
tokens = [t for t in tokens if t.isalnum() and t not in stop_words]

# Concordance haritası: kelime etrafındaki bağlamları toplar
window_size = 8
concordance_dict = defaultdict(list)
for i in range(len(tokens)):
    word = tokens[i]
    left = tokens[max(i - window_size, 0):i]
    right = tokens[i+1:i+1+window_size]
    context = left + right
    concordance_dict[word].append(context)

# Bağlam temelli vektörler oluştur (word embedding benzeri)
unique_words = list(set(tokens))
word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

embedding_dim = 100
context_vectors = np.zeros((len(unique_words), embedding_dim))

# Eğitim için bağlam çiftleri oluştur
pairs = []
for word, contexts in concordance_dict.items():
    for context in contexts:
        for ctx_word in context:
            if ctx_word in word_to_idx:
                pairs.append((word_to_idx[word], word_to_idx[ctx_word]))

# Basit Embedding modeli
class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)

    def forward(self, inputs):
        return self.embeddings(inputs)

# Model ve eğitim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WordEmbeddingModel(len(unique_words), embedding_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CosineEmbeddingLoss()

# Eğitim verisini Tensor'a çevir
x_data = torch.tensor([pair[0] for pair in pairs], dtype=torch.long).to(device)
y_data = torch.tensor([pair[1] for pair in pairs], dtype=torch.long).to(device)
labels = torch.tensor([1.0] * len(pairs), dtype=torch.float).to(device)

# Eğitim
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    embed_x = model(x_data)
    embed_y = model(y_data)
    loss = loss_fn(embed_x, embed_y, labels)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Modeli ve indeksleri kaydet
torch.save(model.state_dict(), "word_embedding_model.pt")
with open("word_to_idx.pkl", "wb") as f:
    pickle.dump(word_to_idx, f)
with open("idx_to_word.pkl", "wb") as f:
    pickle.dump(idx_to_word, f)

# Anlamsal benzerliği test et (sadece ilk 10 kelime için)
with torch.no_grad():
    embeddings = model.embeddings.weight.cpu().numpy()
    similarity_matrix = cosine_similarity(embeddings)
    for i, word in enumerate(unique_words[:10]):  # Sadece ilk 10 kelimeyi göster
        top_idx = np.argsort(similarity_matrix[i])[::-1][1:4]  # En benzer 3 kelime
        print(f"{word}: {[unique_words[j] for j in top_idx]}")
