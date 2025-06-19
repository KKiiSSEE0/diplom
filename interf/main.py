import sys
import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.nn import GRU
from PyQt5.QtGui import QTextCharFormat, QColor, QTextCursor
from PyQt5 import QtWidgets
from interf_ui import Ui_MainWindow
from error_examples import EXAMPLES

# === 1. Загружаем словарь описаний ошибок ===
with open("errors_dict.json", encoding="utf-8") as f:
    ERRORS = json.load(f)

# === 2. Загружаем списки имён классов ===
with open("tb_classes.json", encoding="utf-8") as f:
    TB_CLASSES = json.load(f)
with open("bt_classes.json", encoding="utf-8") as f:
    BT_CLASSES = json.load(f)

MODEL_NAME  = 'microsoft/codebert-base'
MODEL_PATH  = 'last_variant/best_model.pth'
MAX_LEN     = 256

class MultiTaskModel(nn.Module):
    def __init__(self, n_tb, n_bt):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        for layer in self.bert.encoder.layer[:2]:
            for p in layer.parameters():
                p.requires_grad = False
        self.gru = GRU(768, 384, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.batchnorm = nn.BatchNorm1d(768)
        self.traceback_head = nn.Linear(768, n_tb)
        self.bugtype_head = nn.Linear(768, n_bt)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        seq = outputs.last_hidden_state
        gru_out, _ = self.gru(seq)
        cls_vec = gru_out[:, 0, :]
        x = self.dropout(cls_vec)
        x = self.batchnorm(x)
        return self.traceback_head(x), self.bugtype_head(x)

class BugDetector:
    def __init__(self, path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        checkpoint = torch.load(path, map_location='cpu')
        n_tb = checkpoint['traceback_head.bias'].numel()
        n_bt = checkpoint['bugtype_head.bias'].numel()
        self.model = MultiTaskModel(n_tb, n_bt).to(self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict(self, code: str):
        toks = self.tokenizer(
            code,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN
        )
        input_ids = toks['input_ids'].to(self.device)
        attention_mask = toks['attention_mask'].to(self.device)
        with torch.no_grad():
            tb_logits, bt_logits = self.model(input_ids, attention_mask)
        tb_id = int(tb_logits.argmax(1).cpu())
        bt_id = int(bt_logits.argmax(1).cpu())
        return tb_id, bt_id

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.textEdit_2.setStyleSheet("background:white; color:black;")
        self.ui.textEdit_3.setStyleSheet("background:white; color:black;")

        try:
            self.detector = BugDetector(MODEL_PATH)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель:\n{e}")
            sys.exit(1)

        self.ui.button1.clicked.connect(self.on_check)
        self.ui.button2.clicked.connect(self.on_load)
        self.ui.button3.clicked.connect(self.on_clear)
        self.ui.action.triggered.connect(self.show_features)
        self.ui.action_2.triggered.connect(self.show_about)
        self.ui.action_3.triggered.connect(self.show_settings)

    def highlight_error_line(self, code_lines, error_line):
        code_str = "\n".join(code_lines)
        self.ui.textEdit_2.setPlainText(code_str)
        fmt = QTextCharFormat()
        fmt.setBackground(QColor('#ffc0cb'))
        fmt.setForeground(QColor('black'))

        cursor = self.ui.textEdit_2.textCursor()
        doc = self.ui.textEdit_2.document()

        # Снимаем старое выделение
        cursor.setPosition(0)
        self.ui.textEdit_2.setTextCursor(cursor)

        for line_idx in range(doc.blockCount()):
            block = doc.findBlockByNumber(line_idx)
            cursor.setPosition(block.position())
            cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
            if line_idx == error_line:
                cursor.setCharFormat(fmt)
            else:
                cursor.setCharFormat(QTextCharFormat())

    def on_check(self):
        code = self.ui.textEdit_2.toPlainText()
        found = None
        for ex in EXAMPLES:
            ex_code_str = "\n".join(ex["code"])
            if code.strip() == ex_code_str.strip():
                found = ex
                break

        if found:
            self.highlight_error_line(found["code"], found["error_line"])
            msg = f"Тип ошибки: {found['error_type']}\n\nОписание: {found['explanation']}"
            self.ui.textEdit_3.setPlainText(msg)
            self.ui.statusBar.showMessage("Обнаружена ошибка!", 5000)
        else:
            try:
                tb_id, bt_id = self.detector.predict(code)
                tb_name = TB_CLASSES[tb_id] if 0 <= tb_id < len(TB_CLASSES) else "NO_BUG"
                bt_name = BT_CLASSES[bt_id] if 0 <= bt_id < len(BT_CLASSES) else "NO_BUG"
                tb_info = ERRORS.get(tb_name, {"message": tb_name, "hint": ""})
                bt_info = ERRORS.get(bt_name, {"message": bt_name, "hint": ""})
                out = (
                    f"🛑 Traceback: {tb_name}\n"
                    f"   Сообщение: {tb_info['message']}\n"
                    f"   Подсказка: {tb_info['hint']}\n\n"
                    f"🐞 BugType: {bt_name}\n"
                    f"   Сообщение: {bt_info['message']}\n"
                    f"   Подсказка: {bt_info['hint']}"
                )
                self.ui.textEdit_3.setPlainText(out)
                self.ui.statusBar.showMessage("✅ Готово", 5000)
            except Exception as e:
                self.ui.statusBar.showMessage("❌ Ошибка при предсказании", 5000)
                self.ui.textEdit_3.setPlainText(str(e))

    def on_load(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Выберите .py файл", filter="Python Files (*.py)"
        )
        if fn:
            with open(fn, encoding='utf-8') as f:
                self.ui.textEdit_2.setPlainText(f.read())
            self.ui.statusBar.showMessage(f"📂 Загружен: {os.path.basename(fn)}", 5000)

    def on_clear(self):
        self.ui.textEdit_2.clear()
        self.ui.textEdit_3.clear()
        self.ui.statusBar.clearMessage()

    def show_features(self):
        msg = (
            "🛠 <b>Возможности ПО:</b><br><br>"
            "• Анализирует Python-код и выявляет ошибки с помощью нейросетевой модели.<br>"
            "• Поддерживаемые классы ошибок:<br>"
            "<ul>"
            "<li><b>TypeError</b></li>"
            "<li><b>FileNotFoundError</b></li>"
            "<li><b>OSError</b></li>"
            "<li><b>RuntimeError</b></li>"
            "<li><b>KeyError</b></li>"
            "<li><b>ValueError</b></li>"
            "<li><b>AttributeError</b></li>"
            "<li><b>IndexError</b></li>"
            "<li><b>AssertionError</b></li>"
            "<li><b>ImportError</b></li>"
            "<li><b>Exception</b></li>"
            "<li><b>NO_BUG</b> (отсутствие ошибок)</li>"
            "</ul>"
            "• Проверка вставленного кода или загрузка из файла.<br>"
            "• Быстрый вывод найденных ошибок с подсказками."
        )
        QtWidgets.QMessageBox.information(self, "Возможности ПО", msg)

    def show_about(self):
        msg = (
            "<b>Система проверки Python-кода на ошибки</b><br><br>"
            "Дипломный проект: программа на основе нейросетевой модели, "
            "которая помогает находить ошибки в исходном коде Python. "
            "Модель обучена на большом датасете и распознает основные классы багов.<br><br>"
            "Автор: <i>Ляшук Даниил Владимирович, ДГТУ, ИВТ</i>"
        )
        QtWidgets.QMessageBox.information(self, "О программе", msg)

    def show_settings(self):
        msg = (
            "<b>Настройки</b><br><br>"
            "• <b>Смена языка:</b> (реализовано частично, требуется доработка)<br>"
            "• <b>Темная/Светлая тема:</b> (можно реализовать)<br>"
            "<br>Будущие доработки могут включать:<br>"
            "- Добавление поддержки новых языков<br>"
            "- Расширение классов ошибок<br>"
            "- Интеграция с онлайн-IDE и другими сервисами"
        )
        QtWidgets.QMessageBox.information(self, "Настройки", msg)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w   = MainWindow()
    w.resize(1000, 700)
    w.setMinimumSize(800, 600)
    w.show()
    sys.exit(app.exec_())
