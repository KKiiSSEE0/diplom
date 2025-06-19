import torch
import torch.nn as nn
import re
import pickle

class CodeLSTM(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=64, output_dim=1):
        super(CodeLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, return_sequences=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim//2, batch_first=True)
        self.fc = nn.Linear(hidden_dim//2, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# Загрузка модели и словаря
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CodeLSTM().to(device)
model.load_state_dict(torch.load("code_lstm_model.pth"))
model.eval()

with open("token_to_idx.pkl", "rb") as f:
    token_to_idx = pickle.load(f)

max_length = 200

def check_code_snippet(snippet):
    tokens = re.findall(r'\w+|[^\s\w]', snippet)
    seq = [token_to_idx.get(token, token_to_idx['<UNK>']) for token in tokens]
    if len(seq) > max_length:
        seq = seq[:max_length]
    else:
        seq = seq + [token_to_idx['<PAD>']] * (max_length - len(seq))
    X = torch.tensor([seq], dtype=torch.long).to(device)
    with torch.no_grad():
        pred = model(X).squeeze().cpu().numpy()
    return "Buggy" if pred > 0.5 else "Stable"

if __name__ == "__main__":
    while True:
        print("Введите сниппет кода (или 'exit' для выхода):")
        snippet = input()
        if snippet.lower() == 'exit':
            break
        print(f"Prediction: {check_code_snippet(snippet)}\n")