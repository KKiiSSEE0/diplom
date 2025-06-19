import os
import re
import time
import random
import warnings
import json
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import GradScaler
from torch.nn import GRU
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from pathlib import Path

matplotlib.use('Agg')
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Конфигурация
MAX_LEN = 256
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 2e-5
MODEL_NAME = 'microsoft/codebert-base'
MIN_SAMPLES_PER_CLASS = 200
PATIENCE = 7
HIDDEN_SIZE = 768
SAVE_PATH = Path("last_variant")
SAVE_PATH.mkdir(exist_ok=True)

def setup_environment():
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA доступен: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Количество GPU: {torch.cuda.device_count()}")
        print(f"Название GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 50)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_buggy_vs_clean(df):
    print("\n📈 Статистика по багам:")
    print(df['traceback_type'].value_counts().sort_values(ascending=False))
    print("\nВсего багованных файлов:", (df['traceback_type'] != 'NO_BUG').sum())
    print("Всего корректных файлов:", (df['traceback_type'] == 'NO_BUG').sum())

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, target):
        logprobs = self.log_softmax(x)
        with torch.no_grad():
            true_dist = torch.zeros_like(logprobs)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * logprobs, dim=-1))

class CodeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }, torch.tensor(self.labels[idx], dtype=torch.long)

class CodeBERTClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        for layer in self.bert.encoder.layer[:2]:
            for param in layer.parameters():
                param.requires_grad = False

        self.gru = GRU(input_size=768, hidden_size=384, num_layers=1,
                       batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.batchnorm = nn.BatchNorm1d(768)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        gru_output, _ = self.gru(sequence_output)
        cls_output = gru_output[:, 0, :]
        x = self.dropout(cls_output)
        x = self.batchnorm(x)
        return self.classifier(x)

def compute_sample_weights(labels):
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float32)[labels]

def load_and_filter_data(base_path, min_samples_per_class=200, stable_sample_size=50000):
    buggy_path = base_path / "buggy_dataset/bugfixes_train.pickle"
    stable_path = base_path / "stable_dataset/stable_code_train.pickle"

    buggy = pd.read_pickle(buggy_path)
    stable = pd.read_pickle(stable_path)

    buggy = buggy[buggy['traceback_type'].notna()]
    buggy['text'] = buggy['before_merge'].apply(lambda x: x.decode('utf-8', errors='replace') if isinstance(x, bytes) else str(x))
    stable = stable.sample(n=10407, random_state=42)
    stable['text'] = stable['before_merge'].apply(lambda x: x.decode('utf-8', errors='replace') if isinstance(x, bytes) else str(x))
    stable['traceback_type'] = 'NO_BUG'

    full_df = pd.concat([buggy, stable], ignore_index=True)
    class_counts = full_df['traceback_type'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index.tolist()
    if 'NO_BUG' not in valid_classes:
        valid_classes.append('NO_BUG')

    full_df = full_df[full_df['traceback_type'].isin(valid_classes)]

    print("\n📊 Классы после фильтрации:")
    print(full_df['traceback_type'].value_counts())

    labels, class_names = pd.factorize(full_df['traceback_type'])
    texts = full_df['text'].tolist()

    plt.figure(figsize=(15, 6))
    full_df['traceback_type'].value_counts().plot(kind='bar')
    plt.title("Распределение классов")
    plt.savefig("class_distribution.png")
    plt.close()

    return texts, labels, class_names

def train():
    device = setup_environment()
    writer = SummaryWriter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    base_path = Path("D:/diplom/dataset")
    texts, labels, classes = load_and_filter_data(base_path, MIN_SAMPLES_PER_CLASS)

    df_temp = pd.DataFrame({'label': labels, 'text': texts})
    df_temp['traceback_type'] = [classes[l] for l in labels]
    count_buggy_vs_clean(df_temp)

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_losses, val_accuracies, val_f1s = [], [], []

    sample_weights = compute_sample_weights(y_train)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(CodeDataset(X_train, y_train, tokenizer),
                              batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(CodeDataset(X_val, y_val, tokenizer),
                            batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    model = CodeBERTClassifier(len(classes)).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100,
                                                num_training_steps=len(train_loader) * EPOCHS)

    criterion = LabelSmoothingLoss(classes=len(classes), smoothing=0.1).to(device)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val_acc = 0.0
    epochs_no_improve = 0
    start_time = time.time()
    train_losses, val_accuracies, val_f1s = [], [], []

    history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{EPOCHS}")

        for inputs, labels_batch in progress_bar:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(**inputs)
            loss = criterion(outputs, labels_batch)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels_batch).sum().item()
            total += labels_batch.size(0)

            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{correct / total:.2%}",
                'LR': f"{scheduler.get_last_lr()[0]:.2e}"
            })

        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels_batch in val_loader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels_batch = labels_batch.to(device)
                outputs = model(**inputs)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels_batch).sum().item()
                val_total += labels_batch.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())

        train_loss = total_loss / len(train_loader)
        val_acc = val_correct / val_total
        f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f"\n📊 Эпоха {epoch + 1} завершена.")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Validation F1-score: {f1:.4f}")

        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        val_f1s.append(f1)

        writer.add_scalars('Loss', {'train': train_loss, 'val': val_acc}, epoch)
        writer.add_scalar('F1/val', f1, epoch)

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_acc': val_acc,
            'f1': f1
        }, SAVE_PATH / f"epoch_{epoch+1}.pth")

        history.append({'epoch': epoch+1, 'train_loss': train_loss, 'val_acc': val_acc, 'f1': f1})

        if val_acc > best_val_acc:
            torch.save(model.state_dict(), "best_model.pth")
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == PATIENCE:
                print(f"🛑 Ранняя остановка на эпохе {epoch + 1}")
                break

    with open(SAVE_PATH / "metrics.json", "w") as f:
        json.dump(history, f, indent=4)

    writer.close()
    print(f"\nОбучение завершено за {(time.time() - start_time) / 60:.2f} минут")
    print(f"Лучшая точность: {best_val_acc:.2%} | F1: {f1:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.plot(val_f1s, label='Validation F1-score')
    plt.title("Процесс обучения")
    plt.xlabel("Эпоха")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid()
    plt.savefig("training_curves.png")
    plt.close()


if __name__ == "__main__":
    train()
