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
        # Kiểm tra cột cần thiết
        required_columns = ['text', 'label']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"CSV file must contain {required_columns} columns")
        self.max_len = max_len
        self.label_map = {"Positive": 0, "Negative": 1, "Neutral": 2}
        self.pretrained = pretrained
        self.vocab = GloVe(name='6B', dim=100) if pretrained else None
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.build_vocab()
        if pretrained:
            self.embedding_matrix = self.build_embedding_matrix()

    def build_vocab(self):
        idx = len(self.word2idx)
        for text in self.data['text']:
            tokens = tokenizer(str(text).lower())  # Chuyển text thành chuỗi để tránh lỗi
            for token in tokens:
                if token not in self.word2idx:
                    self.word2idx[token] = idx
                    idx += 1

    def build_embedding_matrix(self):
        embedding_matrix = torch.zeros(len(self.word2idx), 100)
        for word, idx in self.word2idx.items():
            if word in ["<PAD>", "<UNK>"]:
                continue
            try:
                embedding_matrix[idx] = self.vocab[word]
            except KeyError:
                pass
        return embedding_matrix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['text']).lower()
        label = self.label_map[self.data.iloc[idx]['label']]
        tokens = tokenizer(text)
        # Xử lý trường hợp tokens rỗng
        if not tokens:
            tokens = ["<UNK>"]
        indices = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices += [self.word2idx["<PAD>"]] * (self.max_len - len(indices))
        indices = torch.tensor(indices, dtype=torch.long)
        if self.pretrained:
            # Sử dụng embedding_matrix thay vì tra cứu lại GloVe
            embeddings = torch.zeros(self.max_len, 100)
            for i, idx in enumerate(indices):
                if i >= self.max_len:
                    break
                embeddings[i] = self.embedding_matrix[idx]
            return embeddings, torch.tensor(label, dtype=torch.long)
        else:
            return indices, torch.tensor(label, dtype=torch.long)

def get_dataloader(csv_file, batch_size=32, pretrained=True):
    dataset = SentimentDataset(csv_file, pretrained=pretrained)
    if pretrained:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True), dataset.word2idx, dataset.embedding_matrix
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), dataset.word2idx, None