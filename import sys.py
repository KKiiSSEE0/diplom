import sys
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Простое приложение на PyQt6")
        self.setGeometry(100, 100, 300, 200)

        self.button = QPushButton("Нажми меня")
        self.button.clicked.connect(self.say_hello)

        layout = QVBoxLayout()
        layout.addWidget(self.button)
        self.setLayout(layout)

    def say_hello(self):
        print("Привет, PyQt6!")

app = QApplication(sys.argv)
window = MyApp()
window.show()
sys.exit(app.exec())
