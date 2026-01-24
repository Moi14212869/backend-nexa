import json
import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleNN

with open("data.json", encoding="utf-8") as f:
    data = json.load(f)

words = sorted({w for phrases in data.values() for p in phrases for w in p.lower().split()})
labels = list(data.keys())

def vectorize(sentence):
    return torch.tensor([1 if w in sentence else 0 for w in words], dtype=torch.float32)

X, y = [], []

for label, phrases in data.items():
    for phrase in phrases:
        X.append(vectorize(phrase.lower()))
        y.append(labels.index(label))

X = torch.stack(X)
y = torch.tensor(y)

model = SimpleNN(len(words), 8, len(labels))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for _ in range(500):
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()

torch.save({
    "state_dict": model.state_dict(),
    "words": words,
    "labels": labels
}, "model.pth")

print("✅ Modèle entraîné → model.pth")
