# -*- coding: utf-8 -*-
"""Кнопки «Скопировать CSV» и «Очистить лог» (ui/main_window.py).

Полностью MainWindow здесь не собрать: нужен живой контекст OpenGL.
Поэтому методы вызываются на экземпляре, созданном через __new__, —
исполняется настоящий код из ui/main_window.py, а не его копия.

Запуск (без дисплея):
  QT_QPA_PLATFORM=offscreen python tests/test_results_copy.py
"""
import os
import sys
import shutil
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QTextEdit  # noqa: E402

import ui.main_window as mw  # noqa: E402

FAIL = []
N = [0]


def check(name, cond, extra=""):
    N[0] += 1
    if cond:
        print("  [OK]   %s" % name)
    else:
        print("  [FAIL] %s %s" % (name, extra))
        FAIL.append(name)


app = QApplication.instance() or QApplication([])

# Числа из реального расчёта пользователя: одно крыло, AoA=3°.
CL = 10.56245956
CD = 6.444102703
CM = -65.72787679

w = mw.MainWindow.__new__(mw.MainWindow)
w.all_results = [
    {"aoa": 3.0, "cl": CL, "cd": CD, "cm": CM, "error": False},
    {"aoa": 5.0, "cl": 0.5, "cd": 0.02, "cm": -0.1, "error": False},
    {"aoa": 7.0, "cl": 0.0, "cd": 0.0, "cm": 0.0, "error": True,
     "error_msg": "SU2 has diverged"},
]
w.log_text = QTextEdit()

print("Копирование таблицы результатов")
text = mw.MainWindow.polar_csv_text(w, lineterminator="\n")
lines = text.split("\n")
check("заголовок ровно AoA,Cl,Cd,Cm,L/D", lines[0] == "AoA,Cl,Cd,Cm,L/D",
      repr(lines[0]))
check("строк данных столько, сколько успешных точек (2)",
      len([x for x in lines if x]) == 3, repr(lines))
first = lines[1].split(",")
check("первая строка: AoA=3.0", first[0] == "3.0", first[0])
check("первая строка: Cl без округления", first[1] == repr(CL), first[1])
check("первая строка: Cd без округления", first[2] == repr(CD), first[2])
check("первая строка: Cm отрицательный", first[3] == repr(CM), first[3])
check("первая строка: L/D = Cl/Cd",
      abs(float(first[4]) - CL / CD) < 1e-12, first[4])
check("ошибочная точка не копируется",
      not any(x.startswith("7.0,") for x in lines))

print("Кнопка «Скопировать CSV»")
QApplication.clipboard().setText("")
mw.MainWindow.copy_polar_csv(w)
clip = QApplication.clipboard().text()
check("в буфере тот же текст, что даёт polar_csv_text",
      clip == text, repr(clip[:40]))
check("в буфере переводы строк LF, а не CRLF", "\r" not in clip)
check("в лог написано число скопированных строк",
      "скопировано строк: 2" in w.log_text.toPlainText(),
      w.log_text.toPlainText())

# Негативный контроль: если успешных точек нет, буфер не трогается,
# а в лог уходит понятное сообщение.
print("Негативный контроль копирования")
QApplication.clipboard().setText("НЕ_ТРОНУТО")
w.log_text.clear()
w_err = mw.MainWindow.__new__(mw.MainWindow)
w_err.all_results = [{"aoa": 1.0, "cl": 0.0, "cd": 0.0, "cm": 0.0,
                      "error": True, "error_msg": "diverged"}]
w_err.log_text = QTextEdit()
mw.MainWindow.copy_polar_csv(w_err)
check("буфер не перезаписан, когда копировать нечего",
      QApplication.clipboard().text() == "НЕ_ТРОНУТО")
check("в лог ушло предупреждение",
      "копировать нечего" in w_err.log_text.toPlainText(),
      w_err.log_text.toPlainText())

print("Регрессия экспорта CSV")
tmp = tempfile.mkdtemp(prefix="polar_")
_saved_dir = mw.RESULTS_DIR
_saved_box = mw.QMessageBox
_shown = []


class _FakeBox:
    @staticmethod
    def information(parent, title, msg):
        _shown.append(msg)


try:
    mw.RESULTS_DIR = tmp
    mw.QMessageBox = _FakeBox
    mw.MainWindow.save_polar_csv(w)
finally:
    mw.RESULTS_DIR = _saved_dir
    mw.QMessageBox = _saved_box

files = [f for f in os.listdir(tmp) if f.endswith(".csv")]
check("файл поляры создан (%d)" % len(files), len(files) == 1)
if files:
    with open(os.path.join(tmp, files[0]), "r", encoding="utf-8") as f:
        body = f.read()
    check("содержимое файла совпадает с текстом копирования",
          body.replace("\r\n", "\n") == text, repr(body[:60]))
shutil.rmtree(tmp, ignore_errors=True)

print("Кнопка «Очистить лог»")
w.log_text.setPlainText("строка 1\nстрока 2")
mw.MainWindow.clear_log(w)
check("лог очищен", w.log_text.toPlainText() == "",
      repr(w.log_text.toPlainText()))

# Привязка кнопок проверяется по исходнику: собрать MainWindow целиком
# здесь нельзя — нужен контекст OpenGL.
print("Привязка кнопок (по исходнику)")
src = open(os.path.join(_ROOT, "ui", "main_window.py"),
           encoding="utf-8").read()
check("кнопка «Скопировать CSV» создана и привязана",
      'self.btn_copy_csv = QPushButton("Скопировать CSV")' in src
      and "self.btn_copy_csv.clicked.connect(self.copy_polar_csv)" in src)
check("кнопка «Очистить лог» создана и привязана",
      'self.btn_clear_log = QPushButton("Очистить лог")' in src
      and "self.btn_clear_log.clicked.connect(self.clear_log)" in src)
check("кнопка копирования включается вместе с экспортом",
      "self.btn_copy_csv.setEnabled(True)" in src)

print()
print("Проверок: %d" % N[0])
if FAIL:
    print("ПРОВАЛЕНО: %d -> %s" % (len(FAIL), FAIL))
    sys.exit(1)
print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
