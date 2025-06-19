from PyQt5.Qsci import QsciLexerPython, QsciScintilla
from PyQt5 import QtGui, QtCore
from PyQt5 import QtCore, QtGui, QtWidgets

import PySimpleGUI as sg

def create_window():
    layout = [
        [sg.Text("🧠 Система проверки Python-кода на ошибки", font=("Segoe UI", 16))],
        [sg.Frame(layout=[
             [sg.Text("💬 Исходный код")],
             [sg.Multiline(size=(40,20), key='-CODE-')]
         ], title="", pad=(0,0)),

         sg.VerticalSeparator(),

         sg.Frame(layout=[
             [sg.Text("✍ Предполагаемые ошибки")],
             [sg.Multiline(size=(40,20), key='-ERROR-')]
         ], title="", pad=(0,0))
        ],
        [sg.Button("Проверить код"), sg.Button("Загрузить файл"), sg.Button("Очистить"), sg.Button("Выход")],
        [sg.StatusBar(text='', size=(60,1), key='-STATUS-')]
    ]
    return sg.Window("Python Bug Detector", layout, finalize=True)


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1024, 600)
        MainWindow.setMinimumSize(QtCore.QSize(800, 500))
        MainWindow.setStyleSheet("/* Фон всего окна */\n"
"QMainWindow#MainWindow {\n"
"    background-color: #1e1e2f;\n"
"QLabel#labelCode,\n"
"QLabel#labelError {\n"
"    color: #000;                  /* белый текст */\n"
"    background-color: #34405e;       /* чуть светлее тёмно-синего */\n"
"    border-radius: 8px;              /* скруглённые углы */\n"
"    padding: 4px 12px;               /* внутренние отступы */\n"
"    font: bold 12pt \"Segoe UI\";      /* шрифт и жирность */\n"
"    qproperty-alignment: \'AlignCenter | AlignVCenter\';\n"
"}\n"
"\n"
"/* Можно задать разный фон руками, если хотите акцентировать */\n"
"QLabel#labelError {\n"
"    background-color: #3e475f;       /* чуть более «пурпурный» фон */\n"
"}\n"
"\n"
"/* Hover-эффект, если лейблы интерактивные (по желанию) */\n"
"QLabel#labelCode:hover,\n"
"QLabel#labelError:hover {\n"
"    background-color: #41516f;\n"
"}ackground-color: #0d6efd;\n"
"}\n"
"/* Если хочется стилизовать центральный виджет отдельно, можно так */\n"
"QWidget#centralwidget {\n"
"    /* можно оставить пустым или override */\n"
"}\n"
"\n"
"/* Стиль кнопок */\n"
"QPushButton {\n"
"    border: none;\n"
"    border-radius: 15px;\n"
"    padding: 8px 16px;\n"
"    font: 14px \"Segoe UI\";\n"
"    color: #fff;\n"
"    background-color: #0d6efd;\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: #0b5ed7;\n"
"}\n"
"QPushButton:checked {\n"
"    background-color: #fff;\n"
"    color: #0d6efd;\n"
"    border: 2px solid #0d6efd;\n"
"}\n"
"\n"
"/* Пример: отдельный стиль для кнопки button1 */\n"
"QPushButton#button1 {\n"
"    /* тут ваши переопределения */\n"
"}\n"
"\n"
"QPushButton {\n"
"    font: 12pt \"Segoe UI\";\n"
"    padding: 10px 20px;\n"
"}\n"
"QTextEdit {\n"
"    font: 11pt \"Consolas\";\n"
"    color: #000;\n"
"}\n"
"\n"
"/* ----------------------------------\n"
"   1. Глобальный фон\n"
"-----------------------------------*/\n"
"QWidget#centralwidget {\n"
"    background-color: #1e1e2f;\n"
"}\n"
"\n"
"/* ----------------------------------\n"
"   2. Секция (GroupBox) с тенью\n"
"-----------------------------------*/\n"
"QGroupBox {\n"
"    background-color: #27293d;\n"
"    border: 1px solid #3a3c55;\n"
"    border-radius: 8px;\n"
"    margin-top: 16px;\n"
"    padding: 12px;\n"
"    /* внешняя тень */\n"
"    qproperty-flat: false;\n"
"    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.5);\n"
"}\n"
"QGroupBox::title {\n"
"    subcontrol-origin: margin;\n"
"    left: 12px;\n"
"    top: -10px;\n"
"    padding: 0 6px;\n"
"    color: #ffffff;\n"
"    font: bold 14pt \"Segoe UI\";\n"
"}\n"
"\n"
"/* ----------------------------------\n"
"   3. Верхний заголовок (labelHeader)\n"
"-----------------------------------*/\n"
"QLabel#headerLabel {\n"
"    background-color: #34405e;\n"
"    color: #ffffff;\n"
"    font: bold 18pt \"Segoe UI\";\n"
"    padding: 12px 20px;\n"
"    border-radius: 6px;\n"
"    qproperty-alignment: \'AlignCenter\';\n"
"    /* лёгкий внутренний градиент */\n"
"    background-image: qlineargradient(\n"
"        x1:0, y1:0, x2:0, y2:1,\n"
"        stop:0 #3b4d70, stop:1 #2a3652\n"
"    );\n"
"}\n"
"\n"
"/* ----------------------------------\n"
"   4. Статус-бар\n"
"-----------------------------------*/\n"
"QStatusBar {\n"
"    background-color: #272a3a;\n"
"    color: #c0c0c0;\n"
"    font: 10pt \"Segoe UI\";\n"
"    border-top: 1px solid #3a3c55;\n"
"}\n"
"QStatusBar::item {\n"
"    border: none;\n"
"}\n"
"\n"
"/* ----------------------------------\n"
"   5. Метки «Исходный код» и «Предп. ошибки»\n"
"-----------------------------------*/\n"
"QLabel#labelCode,\n"
"QLabel#labelError {\n"
"    background-color: #41516f;\n"
"    color: #ffffff;\n"
"    font: bold 12pt \"Segoe UI\";\n"
"    padding: 4px 12px;\n"
"    border-radius: 6px;\n"
"    qproperty-alignment: \'AlignCenter\';\n"
"}\n"
"QLabel#labelError {\n"
"    background-color: #503c4a;\n"
"}\n"
"QLabel#labelCode:hover,\n"
"QLabel#labelError:hover {\n"
"    background-color: #4c5c8a;\n"
"}\n"
"\n"
"\n"
"QTextEdit:focus {\n"
"  border: 1px solid #0d6efd;\n"
"}\n"
"\n"
"/* шапка */\n"
"QLabel#headerLabel {\n"
"  font: bold 18pt \"Segoe UI\";\n"
"  color: #ffffff;\n"
"}\n"
"\n"
"/* подписи блоков */\n"
"QLabel#labelCode,\n"
"QLabel#labelError {\n"
"  font: bold 12pt \"Segoe UI\";\n"
"  color: #ffffff;\n"
"  background-color: #41516f;\n"
"  padding: 4px 8px;\n"
"  border-radius: 6px;\n"
"}\n"
"\n"
"/* кнопки */\n"
"QPushButton {\n"
"  font: 11pt \"Segoe UI\";\n"
"  padding: 6px 12px;\n"
"  color: #fff;\n"
"  background-color: #0d6efd;\n"
"  border-radius: 8px;\n"
"}\n"
"QPushButton:hover {\n"
"  background-color: #0b5ed7;\n"
"}\n"
"QPushButton:pressed {\n"
"  background-color: #0a58ca;\n"
"}\n"
"")
        MainWindow.setProperty("statusbar", "")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setAutoFillBackground(False)
        self.centralwidget.setStyleSheet("/* == Фон всей рабочей области == */\n"
"QWidget#centralwidget {\n"
"    background-color: #022f59;\n"
"}\n"
"/* фон всего приложения */\n"
"QWidget#centralwidget {\n"
"    background-color: #1e1e2f;  /* тёмно-синий/фиолетовый */\n"
"}\n"
"\n"
"/* стилизуем «шильдики» над текстовыми полями */\n"
"QLabel#labelCode,\n"
"QLabel#labelError {\n"
"    color: #ffffff;                  /* белый текст */\n"
"    background-color: #34405e;       /* чуть светлее фон */\n"
"    border-radius: 8px;              /* скруглённые углы */\n"
"    padding: 4px 12px;               /* внутренние отступы */\n"
"    font: bold 12pt \"Segoe UI\";      /* жирный Segoe UI */\n"
"    qproperty-alignment: \'AlignCenter|AlignVCenter\';\n"
"}\n"
"\n"
"/* чуть другой фон для «ошибок» */\n"
"QLabel#labelError {\n"
"    background-color: #3e475f;\n"
"}\n"
"\n"
"/* hover-эффект (опционально) */\n"
"QLabel#labelCode:hover,\n"
"QLabel#labelError:hover {\n"
"    background-color: #41516f;\n"
"}\n"
"\n"
"/* == Редактор кода == */\n"
"QFrame#frameEditor {\n"
"    background-color: #27293d;\n"
"    border-radius: 8px;\n"
"    padding: 5px;\n"
"}\n"
"\n"
"/* == Область результатов == */\n"
"QFrame#frameResult {\n"
"    background-color: #2e2f42;\n"
"    border: 1px solid #3a3c55;\n"
"    border-radius: 8px;\n"
"    padding: 5px;\n"
"}\n"
"\n"
"/* == Кнопки == */\n"
"QPushButton {\n"
"    border: none;\n"
"    border-radius: 12px;\n"
"    padding: 6px 12px;\n"
"    color: #fff;\n"
"    background-color: #0c064f;\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: #0c064f;\n"
"}\n"
"QPushButton:pressed {\n"
"    background-color: #0a58ca;\n"
"}\n"
"\n"
"QGroupBox {\n"
"    border: 1px solid #3a3c55;\n"
"    border-radius: 8px;\n"
"    margin-top: 10px;\n"
"    padding: 12px;\n"
"    background-color: #27293d;\n"
"}\n"
"QGroupBox::title {\n"
"    subcontrol-origin: margin;\n"
"    left: 10px;\n"
"    top: -7px;\n"
"    padding: 0 3px;\n"
"    color: #ffffff;\n"
"}\n"
"\n"
"QLabel#headerLabel {\n"
"    background-color: #34405e;            /* тёмный градиентный фон */\n"
"    color: #ffffff;                       /* белый текст */\n"
"    font: bold 18pt \"Segoe UI\";           /* шрифт и размер */\n"
"    padding: 10px 16px;                   /* внутренние отступы */\n"
"    border-radius: 6px;                   /* закруглённые углы */\n"
"    qproperty-alignment: \'AlignCenter\';   /* выравнивание текста по центру */\n"
"    /* Дополнительно: вертикальный градиент */\n"
"    background-image: qlineargradient(\n"
"        x1:0, y1:0, x2:0, y2:1,\n"
"        stop:0 #3b4d70, stop:1 #2a3652\n"
"    );\n"
"}")
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setObjectName("textEdit_2")
        self.horizontalLayout_6.addWidget(self.textEdit_2)
        self.gridLayout_3.addLayout(self.horizontalLayout_6, 3, 0, 1, 1)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.gridLayout_3.addLayout(self.horizontalLayout_7, 5, 0, 1, 1)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.button1 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.button1.setFont(font)
        self.button1.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.button1.setStyleSheet("<ul class=\"nav nav-pills nav-fill gap-2 p-1 small bg-primary rounded-5 shadow-sm\" id=\"pillNav2\" role=\"tablist\" style=\"--bs-nav-link-color: var(--bs-white); --bs-nav-pills-link-active-color: var(--bs-primary); --bs-nav-pills-link-active-bg: var(--bs-white);\">\n"
"  <li class=\"nav-item\" role=\"presentation\">\n"
"    <button1 class=\"nav-link active rounded-5\" id=\"home-tab2\" data-bs-toggle=\"tab\" type=\"button\" role=\"tab\" aria-selected=\"true\">Home</button>\n"
"  </li>\n"
"  <li class=\"nav-item\" role=\"presentation\">\n"
"    <butto2n class=\"nav-link rounded-5\" id=\"profile-tab2\" data-bs-toggle=\"tab\" type=\"button\" role=\"tab\" aria-selected=\"false\">Profile</button>\n"
"  </li>\n"
"  <li class=\"nav-item\" role=\"presentation\">\n"
"    <button3 class=\"nav-link rounded-5\" id=\"contact-tab2\" data-bs-toggle=\"tab\" type=\"button\" role=\"tab\" aria-selected=\"false\">Contact</button>\n"
"  </li>\n"
"</ul>")
        self.button1.setCheckable(True)
        self.button1.setChecked(False)
        self.button1.setObjectName("button1")
        self.horizontalLayout_9.addWidget(self.button1)
        self.gridLayout_3.addLayout(self.horizontalLayout_9, 4, 0, 1, 1)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.labelCode = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.labelCode.setFont(font)
        self.labelCode.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.labelCode.setObjectName("labelCode")
        self.horizontalLayout_8.addWidget(self.labelCode)
        self.gridLayout_3.addLayout(self.horizontalLayout_8, 2, 0, 1, 1)
        self.labelError = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.labelError.setFont(font)
        self.labelError.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.labelError.setObjectName("labelError")
        self.gridLayout_3.addWidget(self.labelError, 2, 2, 1, 1)
        self.button3 = QtWidgets.QPushButton(self.centralwidget)
        self.button3.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.button3.setCheckable(True)
        self.button3.setObjectName("button3")
        self.gridLayout_3.addWidget(self.button3, 4, 1, 1, 1)
        self.button2 = QtWidgets.QPushButton(self.centralwidget)
        self.button2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.button2.setCheckable(True)
        self.button2.setObjectName("button2")
        self.gridLayout_3.addWidget(self.button2, 4, 2, 1, 1)
        self.headerLabel = QtWidgets.QLabel(self.centralwidget)
        self.headerLabel.setObjectName("headerLabel")
        self.gridLayout_3.addWidget(self.headerLabel, 1, 0, 1, 3)
        self.textEdit_3 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_3.setObjectName("textEdit_3")
        self.gridLayout_3.addWidget(self.textEdit_3, 3, 2, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1024, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.action_5 = QtWidgets.QAction(MainWindow)
        self.action_5.setObjectName("action_5")
        self.action_9 = QtWidgets.QAction(MainWindow)
        self.action_9.setObjectName("action_9")
        self.action = QtWidgets.QAction(MainWindow)
        self.action.setObjectName("action")
        self.action_2 = QtWidgets.QAction(MainWindow)
        self.action_2.setObjectName("action_2")
        self.action_3 = QtWidgets.QAction(MainWindow)
        self.action_3.setObjectName("action_3")
        self.menu.addAction(self.action)
        self.menu.addAction(self.action_2)
        self.menu.addAction(self.action_3)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.textEdit_2.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Consolas\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'MS Shell Dlg 2\'; font-size:7.8pt;\"><br /></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'MS Shell Dlg 2\'; font-size:7.8pt;\"><br /></p></body></html>"))
        self.button1.setText(_translate("MainWindow", "📝Проверить "))
        self.labelCode.setText(_translate("MainWindow", "💬Исходный код "))
        self.labelError.setText(_translate("MainWindow", "✍Предпологаемые ошибки"))
        self.button3.setText(_translate("MainWindow", "🧹Очистить"))
        self.button2.setText(_translate("MainWindow", "📂Загрузить файл"))
        self.headerLabel.setText(_translate("MainWindow", "⚙️Система проверки Python-кода на ошибки⚙️"))
        self.textEdit_3.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Consolas\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'MS Shell Dlg 2\'; font-size:7.8pt;\"><br /></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'MS Shell Dlg 2\'; font-size:7.8pt;\"><br /></p></body></html>"))
        self.menu.setTitle(_translate("MainWindow", "Информация"))
        self.action_5.setText(_translate("MainWindow", "Возможности ПО"))
        self.action_9.setText(_translate("MainWindow", "О программе"))
        self.action.setText(_translate("MainWindow", "Возможности ПО"))
        self.action_2.setText(_translate("MainWindow", "О программе"))
        self.action_3.setText(_translate("MainWindow", "Настройки"))
