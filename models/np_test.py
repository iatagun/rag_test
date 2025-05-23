# BILSTM NP Model ile test
import torch
import pickle
from np_bio_data_gen import BiLSTMTagger  # Aynı klasörde olması gerekir

# 1. Model ve sözlükleri yükle
with open("word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)
with open("tag2idx.pkl", "rb") as f:
    tag2idx = pickle.load(f)
idx2tag = {i: t for t, i in tag2idx.items()}

model = BiLSTMTagger(vocab_size=len(word2idx), tagset_size=len(tag2idx))
model.load_state_dict(torch.load("bilstm_np_model.pt", map_location=torch.device('cpu')))
model.eval()

# 2. Test cümlesi (tokenize edilmiş şekilde verilmeli)
test_sentence = "İnsanlık tarihine kısa bir göz atıldığında bugüne kadar kaba bir tanımlama ile, ilkel toplum düzeninden tarım toplumuna, tarım toplumundan sanayi toplumuna geçişi yaşayarak ve nihayet tanımlayıcı terim olarak üzerinde henüz tam bir görüş birliğine varamadığımız sanayi ötesi toplum aşamasına gelinmiştir.".split()
inputs = torch.tensor([[word2idx.get(w, word2idx['UNK']) for w in test_sentence]])

# 3. Tahmin
torch.no_grad().__enter__()
outputs = model(inputs)
predictions = torch.argmax(outputs, dim=-1).squeeze().tolist()

# 4. Sonuçları yazdır
print("Test cümlesi etiketlemesi:")
for word, tag_id in zip(test_sentence, predictions):
    print(f"{word}\t{idx2tag[tag_id]}")