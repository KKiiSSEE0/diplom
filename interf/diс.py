# extract_classes.py

import pandas as pd
import json
import glob
import os

# 1) Укажите корневую папку с dataset
DATA_ROOT = r"D:/diplom/dataset"

# 2) Собираем пути к нужным pickle-файлам
buggy_files  = glob.glob(os.path.join(DATA_ROOT, "buggy_dataset", "*.pickle"))
stable_files = glob.glob(os.path.join(DATA_ROOT, "stable_dataset", "*.pickle"))

# Попробуем найти явные трен/вал/тестые наборы
def pick_file(files, keyword):
    for p in files:
        if keyword in os.path.basename(p):
            return p
    return files[0] if files else None

BUGGY_PICKLE  = pick_file(buggy_files, "train")
STABLE_PICKLE = pick_file(stable_files, "train")

print("→ Buggy pickle :", BUGGY_PICKLE)
print("→ Stable pickle:", STABLE_PICKLE)

# 3) Загружаем buggy
if not BUGGY_PICKLE:
    raise FileNotFoundError("Не найден ни один pickle в buggy_dataset/")
buggy = pd.read_pickle(BUGGY_PICKLE)
buggy = buggy[buggy['traceback_type'].notna()]

# 4) Загружаем stable, если есть
if STABLE_PICKLE:
    stable = pd.read_pickle(STABLE_PICKLE)
    stable['traceback_type'] = 'NO_BUG'
    df = pd.concat([buggy, stable], ignore_index=True)
    print(f"Loaded buggy({len(buggy)}) + stable({len(stable)}) → total {len(df)}")
else:
    df = buggy.copy()
    print(f"Loaded buggy only ({len(buggy)})")

# 5) Фильтрация по MIN_COUNT (должно совпадать с вашей константой при тренировке)
MIN_COUNT = 200
cnts = df['traceback_type'].value_counts()
valid = cnts[cnts >= MIN_COUNT].index
df = df[df['traceback_type'].isin(valid)]
print("Classes kept (count>=200):", valid.tolist())

# 6) factorize → tb_classes
_, tb_classes = pd.factorize(df['traceback_type'])
tb_list = tb_classes.tolist()

# 7) Сохраняем JSON
os.makedirs("classes_json", exist_ok=True)
with open("classes_json/tb_classes.json", "w", encoding="utf-8") as f:
    json.dump(tb_list, f, ensure_ascii=False, indent=2)

print(f"✅ Сохранено {len(tb_list)} TB_CLASSES в classes_json/tb_classes.json")
