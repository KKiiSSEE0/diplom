import pandas as pd
import json
import os

# Пути к pickle-файлам вашей обучающей выборки
BUGGY_PATH  = "D:/diplom/dataset/buggy_dataset/bugfixes_train_with_bugtype.pickle"
STABLE_PATH = "D:/diplom/dataset/stable_dataset/stable_code_train.pickle"

# 1. Считаем данные
buggy  = pd.read_pickle(BUGGY_PATH)
stable = pd.read_pickle(STABLE_PATH)

# Только непустые
buggy = buggy.dropna(subset=["traceback_type", "bug_type"])
# Текст нам не нужен — только метки
buggy["traceback_type"] = buggy["traceback_type"].astype(str)
buggy["bug_type"]       = buggy["bug_type"].astype(str)

# Анало­гично stable (все у него NO_BUG)
stable = stable.sample(n=len(buggy), random_state=42)
stable["traceback_type"] = "NO_BUG"
stable["bug_type"]       = "NO_BUG"

# 2. Объединяем
df = pd.concat([buggy, stable], ignore_index=True)

# 3. Факторизуем
tb_labels, tb_classes = pd.factorize(df["traceback_type"])
bt_labels, bt_classes = pd.factorize(df["bug_type"])

# 4. Сохраняем в JSON
os.makedirs(".", exist_ok=True)
with open("tb_classes.json", "w", encoding="utf-8") as f:
    json.dump(list(tb_classes), f, ensure_ascii=False, indent=2)

with open("bt_classes.json", "w", encoding="utf-8") as f:
    json.dump(list(bt_classes), f, ensure_ascii=False, indent=2)

print("✔ Сохранены tb_classes.json и bt_classes.json")
