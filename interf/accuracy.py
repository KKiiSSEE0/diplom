import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# === 1. Фейковые метрики по эпохам ===
epochs = list(range(1, 11))

train_accuracy = [0.72, 0.75, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88]
val_accuracy   = [0.70, 0.73, 0.76, 0.78, 0.79, 0.81, 0.82, 0.82, 0.83, 0.83]

train_f1 = [0.71, 0.74, 0.77, 0.79, 0.81, 0.83, 0.84, 0.85, 0.86, 0.87]
val_f1   = [0.69, 0.72, 0.75, 0.77, 0.78, 0.80, 0.81, 0.81, 0.82, 0.82]

# === 2. Строим графики ===
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracy, label='Train Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.title('Точность обучения и валидации по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.ylim(0.6, 0.9)
plt.grid(True)
plt.legend()
plt.savefig('accuracy_plot.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_f1, label='Train F1')
plt.plot(epochs, val_f1, label='Validation F1')
plt.title('F1-мера обучения и валидации по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('F1-мера')
plt.ylim(0.6, 0.9)
plt.grid(True)
plt.legend()
plt.savefig('f1_plot.png')
plt.show()

# === 3. Создаём таблицу метрик ===
df = pd.DataFrame({
    'Эпоха': epochs,
    'Train Accuracy': train_accuracy,
    'Val Accuracy': val_accuracy,
    'Train F1': train_f1,
    'Val F1': val_f1
})

print("\n📊 Таблица метрик по эпохам:")
print(df.round(4))

# Сохраняем в CSV для вставки в отчет
df.to_csv("metrics_table.csv", index=False)
