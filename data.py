# data.py
import pandas as pd
import nltk
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader
import torch

nltk.download('punkt')
tokenizer = get_tokenizer("basic_english")

class SentimentDataset(Dataset):
    def __init__(self, csv_file, max_len=50, pretrained=True):
        self.data = pd.read_csv(csv_file)
        self.max_len = max_len
        self.label_map = {"Positive": 0, "Negative": 1, "Neutral": 2}
        self.pretrained = pretrained
        self.vocab = GloVe(name='6B', dim=100) if pretrained else None
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.build_vocab()
        # Tạo embedding matrix nếu dùng pretrained
        if pretrained:
            self.embedding_matrix = self.build_embedding_matrix()

    def build_vocab(self):
        idx = len(self.word2idx)
        for text in self.data['text']:
            tokens = tokenizer(text.lower())
            for token in tokens:
                if token not in self.word2idx:
                    self.word2idx[token] = idx
                    idx += 1

    def build_embedding_matrix(self):
        # Tạo embedding matrix với kích thước [vocab_size, embedding_dim]
        embedding_matrix = torch.zeros(len(self.word2idx), 100)  # 100 là dim của GloVe
        for word, idx in self.word2idx.items():
            if word in ["<PAD>", "<UNK>"]:
                continue  # Bỏ qua PAD và UNK, để mặc định là vector 0
            try:
                embedding_matrix[idx] = self.vocab[word]
            except KeyError:
                # Nếu từ không có trong GloVe, để vector 0
                pass
        return embedding_matrix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text'].lower()
        label = self.label_map[self.data.iloc[idx]['label']]
        tokens = tokenizer(text)
        indices = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices += [self.word2idx["<PAD>"]] * (self.max_len - len(indices))
        indices = torch.tensor(indices, dtype=torch.long)
        if self.pretrained:
            embeddings = torch.stack([self.vocab[token] for token in tokens], dim=0)
            if embeddings.size(0) > self.max_len:
                embeddings = embeddings[:self.max_len, :]
            else:
                padding = torch.zeros(self.max_len - embeddings.size(0), 100)
                embeddings = torch.cat([embeddings, padding], dim=0)
            return embeddings, torch.tensor(label, dtype=torch.long)
        else:
            return indices, torch.tensor(label, dtype=torch.long)  # Trả về indices cho Scratch

def get_dataloader(csv_file, batch_size=32, pretrained=True):
    dataset = SentimentDataset(csv_file, pretrained=pretrained)
    if pretrained:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True), dataset.word2idx, dataset.embedding_matrix
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), dataset.word2idx, None