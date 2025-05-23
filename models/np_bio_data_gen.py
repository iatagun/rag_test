# BiLSTM ile BIO Etiketli Ad Öbeği Tanıma Modeli (spaCy olmadan)
# Elle hazırlanmış BIO verisi üzerinden model eğitimi

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pickle

etiketli_veri_dosyasi = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\rag_test\\turkce_np_source.txt"  # hazır etiketli veri (BIO formatlı)

class NERDataset(Dataset):
    def __init__(self, filepath, word2idx=None, tag2idx=None):
        with open(filepath, encoding="utf-8") as f:
            lines = f.read().strip().split("\n\n")
        self.sentences = []
        self.labels = []
        for block in lines:
            tokens = []
            tags = []
            for line in block.strip().split("\n"):
                if line:
                    word, tag = line.split("\t")
                    tokens.append(word)
                    tags.append(tag)
            if tokens:
                self.sentences.append(tokens)
                self.labels.append(tags)

        words = set(w for s in self.sentences for w in s)
        tags = set(t for l in self.labels for t in l)

        self.word2idx = word2idx or {w: i+2 for i, w in enumerate(words)}
        self.word2idx["PAD"] = 0
        self.word2idx["UNK"] = 1
        self.tag2idx = tag2idx or {t: i for i, t in enumerate(tags)}

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        tags = self.labels[idx]
        x = [self.word2idx.get(w, self.word2idx["UNK"]) for w in words]
        y = [self.tag2idx[t] for t in tags]
        return torch.tensor(x), torch.tensor(y)

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=64, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        mask = x != 0
        lengths = mask.sum(dim=1)
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        out = self.fc(unpacked)
        return out

full_dataset = NERDataset(etiketli_veri_dosyasi)
train_data, val_data = train_test_split(list(range(len(full_dataset))), test_size=0.2, random_state=42)
train_loader = DataLoader([full_dataset[i] for i in train_data], batch_size=1, shuffle=True)
val_loader = DataLoader([full_dataset[i] for i in val_data], batch_size=1)

model = BiLSTMTagger(len(full_dataset.word2idx), len(full_dataset.tag2idx))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(30):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.view(-1, out.shape[-1]), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "bilstm_np_model.pt")
with open("word2idx.pkl", "wb") as f:
    pickle.dump(full_dataset.word2idx, f)
with open("tag2idx.pkl", "wb") as f:
    pickle.dump(full_dataset.tag2idx, f)

print("Model ve sözlükler başarıyla kaydedildi. Artık inference'a hazırsın!")
