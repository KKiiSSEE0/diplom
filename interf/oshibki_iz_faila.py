import pandas as pd
import glob
from collections import Counter

DATA_DIR = r"D:/diplom/dataset/buggy_dataset/"
# попробуем искать рекурсивно во всех подпапках
paths = glob.glob(DATA_DIR + "**/*.pickle", recursive=True)

# распечатаем, что нашлось
print("Найдено pickle-файлов:", len(paths))
for p in paths:
    print("  ", p)

counter = Counter()
for p in paths:
    df = pd.read_pickle(p)
    if "traceback_type" in df.columns:
        types = df["traceback_type"].dropna().astype(str)
        counter.update(types)

print("\n=== Самые частые исключения ===")
for exc_type, cnt in counter.most_common():
    print(f"{exc_type:30s} — {cnt}")
