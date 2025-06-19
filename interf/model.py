import torch
from transformers import AutoTokenizer

class BugDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        from model_architecture import MultiTaskModel  # обязательно добавь свой класс модели
        # Заменить на реальные числа классов!
        self.model = MultiTaskModel(n_tb_classes=93, n_bt_classes=79).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, code_snippet):
        inputs = self.tokenizer(code_snippet, return_tensors="pt", padding='max_length',
                                truncation=True, max_length=256).to(self.device)
        with torch.no_grad():
            tb_logits, bt_logits = self.model(**inputs)
        tb_pred = torch.argmax(tb_logits, dim=1).item()
        bt_pred = torch.argmax(bt_logits, dim=1).item()
        return tb_pred, bt_pred
