import torch
from model import SimpleNN
import random

checkpoint = torch.load("model.pth", map_location=torch.device("cpu"))

words = checkpoint["words"]
labels = checkpoint["labels"]

model = SimpleNN(len(words), 8, len(labels))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

responses = {
    "salutation": ["Salut 😄", "Hello 👋", "Coucou !"],
    "question": ["Je suis NEXA, une IA 🤖", "Bonne question 🤔"],
    "au_revoir": ["À bientôt 👋", "À très vite !"],
    "autre": ["Intéressant 👀", "Dis-m'en plus."]
}

def vectorize(sentence):
    sentence = sentence.lower()
    return torch.tensor([1 if w in sentence else 0 for w in words], dtype=torch.float32)

def mon_ia(message: str) -> str:
    with torch.no_grad():
        x = vectorize(message)
        output = model(x)
        label = labels[torch.argmax(output).item()]
        return random.choice(responses[label])
