# np_inference_only.py – Sadece inference yapan, eğitimsiz script
import torch
import pickle
from np_bio_data_gen import BiLSTMTagger

def extract_strongest_np(sentence: str) -> str:
    """
    Verilen cümlede en uzun NP (noun phrase) öbeğini çıkarır.
    """
    # Sözlükleri yükle
    with open("word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)
    with open("tag2idx.pkl", "rb") as f:
        tag2idx = pickle.load(f)
    idx2tag = {i: t for t, i in tag2idx.items()}

    # Modeli yükle
    model = BiLSTMTagger(vocab_size=len(word2idx), tagset_size=len(tag2idx))
    model.load_state_dict(torch.load("bilstm_np_model.pt", map_location=torch.device("cpu")))
    model.eval()

    # Cümleyi işle
    tokens = sentence.strip().split()
    indexed = [word2idx.get(w, word2idx['UNK']) for w in tokens]
    inputs = torch.tensor([indexed])

    with torch.no_grad():
        logits = model(inputs)
        preds = torch.argmax(logits, dim=-1).squeeze().tolist()

    # NP zincirlerini çıkar
    current = []
    all_nps = []
    for token, tag_id in zip(tokens, preds):
        tag = idx2tag[tag_id]
        if tag == "B-NP":
            if current: all_nps.append(current)
            current = [token]
        elif tag == "I-NP" and current:
            current.append(token)
        else:
            if current: all_nps.append(current)
            current = []
    if current:
        all_nps.append(current)

    return " ".join(max(all_nps, key=len)) if all_nps else ""

if __name__ == "__main__":
    text = input("Cümleyi girin: ")
    result = extract_strongest_np(text)
    print("En güçlü NP öbeği:", result)
