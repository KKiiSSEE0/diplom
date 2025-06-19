import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# === Константы ===
MODEL_NAME = 'microsoft/codebert-base'
MAX_LEN = 256
BATCH_SIZE = 8
TOP_K_CLASSES = 30
MIN_LEN = 5
MAX_LEN_TOKENS = 1024
TEST_SAMPLE_SIZE = 1000
HIDDEN_SIZE = 768

# === Очистка и фильтрация датасета ===
def clean_dataset(df):
    df = df[df['traceback_type'].notna()]
    df['text'] = df['before_merge'].apply(lambda x: x.decode('utf-8', errors='replace') if isinstance(x, bytes) else str(x))
    df['code_len'] = df['text'].apply(lambda x: len(str(x).split()))
    df = df[df['code_len'] >= MIN_LEN]
    df = df[df['code_len'] <= MAX_LEN_TOKENS]

    top_classes = df['traceback_type'].value_counts().nlargest(TOP_K_CLASSES).index
    df['traceback_type'] = df['traceback_type'].apply(lambda x: x if x in top_classes else 'OTHER')

    print("\n[INFO] Классы после фильтрации:")
    print(df['traceback_type'].value_counts())
    return df

# === Набор данных ===
class CodeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }, torch.tensor(self.labels[idx], dtype=torch.long)

# === Модель ===
class CodeBERTClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        for layer in self.bert.encoder.layer[:2]:
            for param in layer.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        return self.classifier(x)

# === Обучение модели на тестовом сете ===
def run_small_training():
    df = pd.read_pickle("D:/diplom/dataset/buggy_dataset/bugfixes_train.pickle")
    df = clean_dataset(df)

    df_sampled = df.sample(n=min(TEST_SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)

    # Удаляем классы, где <2 примеров (после выборки!)
    df_sampled = df_sampled.groupby('traceback_type').filter(lambda x: len(x) >= 2)

    texts = df_sampled['text'].tolist()
    labels, uniques = pd.factorize(df_sampled['traceback_type'])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42)

    train_dataset = CodeDataset(X_train, y_train, tokenizer)
    val_dataset = CodeDataset(X_val, y_val, tokenizer)

    sample_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_tensor = torch.tensor(sample_weights)[y_train]
    sampler = WeightedRandomSampler(weights=sample_tensor, num_samples=len(sample_tensor), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = CodeBERTClassifier(n_classes=len(uniques))
    print(f"\n✅ Модель готова. Классов: {len(uniques)}, обучающая выборка: {len(X_train)}, валидационная: {len(X_val)}")
    return model, train_loader, val_loader

if __name__ == "__main__":
    model, train_loader, val_loader = run_small_training()


from train_loop_small_set import train_loop

train_loop(model, train_loader, val_loader, device='cuda', epochs=10)
