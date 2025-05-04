import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from data import get_dataloader
from model import SentimentRNN
import json

def train_model(model, train_loader, val_loader, epochs=10, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_accuracy": [], "val_f1": []}
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        
        val_acc, val_f1 = evaluate_model(model, val_loader)
        history["val_accuracy"].append(val_acc)
        history["val_f1"].append(val_f1)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}, Val Accuracy: {val_acc}, Val F1: {val_f1}")
    return history

def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return accuracy, f1

if __name__ == "__main__":

    train_loader_pre, word2idx, embedding_matrix = get_dataloader("sentiment_data.csv", pretrained=True)
    test_loader_pre, _, _ = get_dataloader("sentiment_data.csv", pretrained=True)
    train_loader_scratch, word2idx_scratch, _ = get_dataloader("sentiment_data.csv", pretrained=False)
    test_loader_scratch, _, _ = get_dataloader("sentiment_data.csv", pretrained=False)


    pretrained_model = SentimentRNN(vocab_size=len(word2idx), pretrained=True, embedding_matrix=embedding_matrix)
    pretrained_history = train_model(pretrained_model, train_loader_pre, test_loader_pre)
    pretrained_acc, pretrained_f1 = evaluate_model(pretrained_model, test_loader_pre)
    print(f"Pretrained - Accuracy: {pretrained_acc}, F1-score: {pretrained_f1}")


    scratch_model = SentimentRNN(vocab_size=len(word2idx_scratch), pretrained=False, embedding_matrix=None)
    scratch_history = train_model(scratch_model, train_loader_scratch, test_loader_scratch)
    scratch_acc, scratch_f1 = evaluate_model(scratch_model, test_loader_scratch)
    print(f"Scratch - Accuracy: {scratch_acc}, F1-score: {scratch_f1}")


    results = {
        "Pretrained": {
            "Accuracy": pretrained_acc,
            "F1-score": pretrained_f1,
            "History": pretrained_history
        },
        "Scratch": {
            "Accuracy": scratch_acc,
            "F1-score": scratch_f1,
            "History": scratch_history
        }
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)