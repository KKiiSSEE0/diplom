# Модуль 1: Импорт библиотек
# Назначение: Подключение необходимых библиотек для работы с данными, машинным обучением, визуализацией и утилитами
import os
import re
import time
import random
import warnings
import json
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
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
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm
from pathlib import Path

# Модуль 2: Настройка окружения
# Назначение: Отключение графического backend для matplotlib и предупреждений Hugging Face
matplotlib.use('Agg')
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Модуль 3: Конфигурация
# Назначение: Определение гиперпараметров модели и пути для сохранения результатов
MAX_LEN = 256
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 2e-5
MODEL_NAME = 'microsoft/codebert-base'
MIN_SAMPLES_PER_CLASS = 200  # Минимальное количество примеров для класса
PATIENCE = 7
HIDDEN_SIZE = 768
SAVE_PATH = Path("last_variant")
SAVE_PATH.mkdir(exist_ok=True)

# Модуль 4: Функция для получения путей к файлам
# Назначение: Обеспечение корректного доступа к файлам при запуске через PyInstaller
def resource_path(relative_path):
    """Возвращает абсолютный путь к файлу, учитывая PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Модуль 5: Настройка окружения
# Назначение: Проверка версии PyTorch и доступности CUDA
def setup_environment():
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA доступен: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Количество GPU: {torch.cuda.device_count()}")
        print(f"Название GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 50)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Модуль 6: Класс сглаживания меток
# Назначение: Реализация функции потерь с Label Smoothing для повышения устойчивости модели
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

# Модуль 7: Класс датасета
# Назначение: Создание пользовательского датасета для обработки текстов кода и меток
class CodeDataset(Dataset):
    def __init__(self, texts, tb_labels, bt_labels, tokenizer):
        self.texts = texts
        self.tb_labels = tb_labels
        self.bt_labels = bt_labels
        self.tokenizer = tokenizer

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
        }, torch.tensor(self.tb_labels[idx], dtype=torch.long), torch.tensor(self.bt_labels[idx], dtype=torch.long)

# Модуль 8: Модель для многозадачной классификации
# Назначение: Определение архитектуры модели с использованием CodeBERT и GRU
class MultiTaskModel(nn.Module):
    def __init__(self, n_tb_classes, n_bt_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        for layer in self.bert.encoder.layer[:2]:
            for param in layer.parameters():
                param.requires_grad = False
        self.gru = GRU(input_size=768, hidden_size=384, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.batchnorm = nn.BatchNorm1d(768)
        self.traceback_head = nn.Linear(768, n_tb_classes)
        self.bugtype_head = nn.Linear(768, n_bt_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        gru_output, _ = self.gru(sequence_output)
        cls_output = gru_output[:, 0, :]
        x = self.dropout(cls_output)
        x = self.batchnorm(x)
        return self.traceback_head(x), self.bugtype_head(x)

# Модуль 9: Вычисление весов для классов
# Назначение: Создание весов для балансировки классов при обучении
def compute_sample_weights(labels):
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float32)[labels]

# Модуль 10: Загрузка и фильтрация данных
# Назначение: Загрузка датасетов, фильтрация редких классов и подготовка данных
def load_and_filter_data(base_path):
    buggy = pd.read_pickle(base_path / "buggy_dataset/bugfixes_train_with_bugtype.pickle")
    stable = pd.read_pickle(base_path / "stable_dataset/stable_code_train.pickle")

    buggy = buggy[buggy['traceback_type'].notna() & buggy['bug_type'].notna()]

    tb_counts = buggy['traceback_type'].value_counts()
    buggy['traceback_type'] = buggy['traceback_type'].apply(
        lambda x: x if tb_counts.get(x, 0) >= MIN_SAMPLES_PER_CLASS else 'OTHER')

    bt_counts = buggy['bug_type'].value_counts()
    buggy['bug_type'] = buggy['bug_type'].apply(
        lambda x: x if bt_counts.get(x, 0) >= MIN_SAMPLES_PER_CLASS else 'OTHER')

    buggy['text'] = buggy['before_merge'].apply(
        lambda x: x.decode('utf-8', errors='replace') if isinstance(x, bytes) else str(x))

    stable = stable.sample(n=len(buggy), random_state=42)
    stable['text'] = stable['before_merge'].apply(
        lambda x: x.decode('utf-8', errors='replace') if isinstance(x, bytes) else str(x))
    stable['traceback_type'] = 'NO_BUG'
    stable['bug_type'] = 'NO_BUG'

    full_df = pd.concat([buggy, stable], ignore_index=True)

    tb_filtered = tb_counts[tb_counts < MIN_SAMPLES_PER_CLASS]
    bt_filtered = bt_counts[bt_counts < MIN_SAMPLES_PER_CLASS]
    print("\n Отфильтрованные классы traceback:")
    print(tb_filtered)
    print("\n Отфильтрованные классы bug_type:")
    print(bt_filtered)
    print(f"\n Классов traceback после фильтрации: {buggy['traceback_type'].nunique()}")
    print(f" Классов bug_type после фильтрации: {buggy['bug_type'].nunique()}")

    tb_labels, tb_classes = pd.factorize(full_df['traceback_type'])
    bt_labels, bt_classes = pd.factorize(full_df['bug_type'])
    texts = full_df['text'].tolist()

    print(f"\n Используется {len(texts)} примеров для обучения и классификации по {len(tb_classes)} traceback и {len(bt_classes)} bug классов.")
    return texts, tb_labels, bt_labels, tb_classes, bt_classes

# Модуль 11: Основная функция обучения
# Назначение: Обучение модели, валидация, сохранение результатов и визуализация метрик
def train():
    device = setup_environment()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_path = Path("D:/diplom/dataset")
    texts, tb_labels, bt_labels, tb_classes, bt_classes = load_and_filter_data(base_path)

    X_train, X_val, y_tb_train, y_tb_val, y_bt_train, y_bt_val = train_test_split(
        texts, tb_labels, bt_labels, test_size=0.2, stratify=tb_labels, random_state=42
    )

    print(f"\n Размеры выборок: Train={len(X_train)}, Validation={len(X_val)}")

    train_loader = DataLoader(CodeDataset(X_train, y_tb_train, y_bt_train, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CodeDataset(X_val, y_tb_val, y_bt_val, tokenizer), batch_size=BATCH_SIZE)

    model = MultiTaskModel(len(tb_classes), len(bt_classes)).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100,
                                                num_training_steps=len(train_loader) * EPOCHS)

    tb_loss_fn = LabelSmoothingLoss(len(tb_classes)).to(device)
    bt_loss_fn = LabelSmoothingLoss(len(bt_classes)).to(device)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    writer = SummaryWriter()

    best_tb_f1 = 0.0
    all_tb_f1, all_bt_f1 = [], []

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct_tb, correct_bt, total = 0, 0, 0, 0
        progress_bar = tqdm(train_loader, desc=f" Обучение | Эпоха {epoch + 1}/{EPOCHS}")

        for inputs, tb_targets, bt_targets in progress_bar:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            tb_targets, bt_targets = tb_targets.to(device), bt_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            tb_logits, bt_logits = model(**inputs)
            loss = tb_loss_fn(tb_logits, tb_targets) + bt_loss_fn(bt_logits, bt_targets)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            correct_tb += (tb_logits.argmax(1) == tb_targets).sum().item()
            correct_bt += (bt_logits.argmax(1) == bt_targets).sum().item()
            total += tb_targets.size(0)

            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'TB Acc': f"{correct_tb / total:.2%}",
                'BT Acc': f"{correct_bt / total:.2%}"
            })

        model.eval()
        val_correct_tb, val_correct_bt, val_total = 0, 0, 0
        tb_preds, tb_true, bt_preds, bt_true = [], [], [], []
        with torch.no_grad():
            for inputs, tb_targets, bt_targets in val_loader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                tb_targets, bt_targets = tb_targets.to(device), bt_targets.to(device)
                tb_logits, bt_logits = model(**inputs)

                tb_preds.extend(tb_logits.argmax(1).cpu().numpy())
                tb_true.extend(tb_targets.cpu().numpy())
                bt_preds.extend(bt_logits.argmax(1).cpu().numpy())
                bt_true.extend(bt_targets.cpu().numpy())

                val_correct_tb += (tb_logits.argmax(1) == tb_targets).sum().item()
                val_correct_bt += (bt_logits.argmax(1) == bt_targets).sum().item()
                val_total += tb_targets.size(0)

        tb_f1 = f1_score(tb_true, tb_preds, average='weighted')
        bt_f1 = f1_score(bt_true, bt_preds, average='weighted')

        print(f"\n Эпоха {epoch + 1} завершена.")
        print(f"Traceback Acc: {val_correct_tb / val_total:.4f}, F1: {tb_f1:.4f}")
        print(f"BugType Acc: {val_correct_bt / val_total:.4f}, F1: {bt_f1:.4f}")

        all_tb_f1.append(tb_f1)
        all_bt_f1.append(bt_f1)

        writer.add_scalars('F1', {'traceback': tb_f1, 'bugtype': bt_f1}, epoch)

        torch.save(model.state_dict(), SAVE_PATH / f"epoch_{epoch+1}.pth")

        if tb_f1 > best_tb_f1:
            best_tb_f1 = tb_f1
            torch.save(model.state_dict(), SAVE_PATH / "best_model.pth")

    # Модуль 12: Визуализация результатов
    # Назначение: Создание графиков F1-метрик и матриц ошибок
    plt.figure(figsize=(8, 5))
    plt.plot(all_tb_f1, label='Traceback F1')
    plt.plot(all_bt_f1, label='BugType F1')
    plt.title('F1 по эпохам')
    plt.xlabel('Эпоха')
    plt.ylabel('F1-метрика')
    plt.legend()
    plt.grid()
    plt.savefig(SAVE_PATH / "f1_metrics.png")

    tb_cm = confusion_matrix(tb_true, tb_preds)
    bt_cm = confusion_matrix(bt_true, bt_preds)
    plt.figure(figsize=(12, 5))
    sns.heatmap(tb_cm, cmap="Blues", square=True, xticklabels=False, yticklabels=False)
    plt.title("Матрица ошибок traceback")
    plt.savefig(SAVE_PATH / "confusion_traceback.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    sns.heatmap(bt_cm, cmap="Greens", square=True, xticklabels=False, yticklabels=False)
    plt.title("Матрица ошибок bug_type")
    plt.savefig(SAVE_PATH / "confusion_bugtype.png")
    plt.close()

    writer.close()
    print("\nОбучение завершено.")

# Модуль 13: Точка входа
# Назначение: Запуск функции обучения
if __name__ == "__main__":
    train()