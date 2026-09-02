# -*- coding: utf-8 -*-
"""Справочные данные по реальной геометрии крыла.

Проверяются geometry/reference.py и подключение к
MainWindow.calculate_reference_data (ui/main_window.py).

Полностью MainWindow здесь не собрать — нужен живой контекст OpenGL,
поэтому метод вызывается на экземпляре, созданном через __new__:
исполняется настоящий код, а не его копия.

Запуск:  QT_QPA_PLATFORM=offscreen python tests/test_reference_data.py
"""
import os
import sys
import importlib.util

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference_standalone", os.path.join("geometry", "reference.py"))
gen = _load("generators_standalone", os.path.join("geometry", "generators.py"))

import pyvista as pv  # noqa: E402
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


def faces_of(m):
    f = np.asarray(m.faces)
    return f.reshape(-1, int(f[0]) + 1)[:, 1:]


class Spin:
    def __init__(self, v):
        self._v = float(v)

    def value(self):
        return self._v


# Эталон: сужающееся крыло 9.02 / 1.44 / 0.72
CR, CT, SPAN = 1.44, 0.72, 9.02
S_EXACT = 0.5 * (CR + CT) * SPAN
LAM = CT / CR
MAC_EXACT = (2.0 / 3.0) * CR * ((1 + LAM + LAM ** 2) / (1 + LAM))

wing, _ = gen.generate_wing_mesh(SPAN, CR, CT, 0.0, 0.0, "2412",
                                 -0.64, 0.0, 0.0)

print("geometry/reference.py на эталонном крыле")
g = ref.wing_reference_from_mesh(np.asarray(wing.points), faces_of(wing))
check("размах точный (%.6f)" % g["span"], abs(g["span"] - SPAN) < 1e-6)
check("Sref в пределах 1%% (%.5f против %.5f)" % (g["area"], S_EXACT),
      abs(g["area"] - S_EXACT) / S_EXACT < 0.01)
check("MAC в пределах 1%% (%.5f против %.5f)" % (g["mac"], MAC_EXACT),
      abs(g["mac"] - MAC_EXACT) / MAC_EXACT < 0.01)
check("хорда корня точная (%.4f)" % g["chord_root"],
      abs(g["chord_root"] - CR) < 1e-3)
check("хорда конца точная (%.4f)" % g["chord_tip"],
      abs(g["chord_tip"] - CT) < 1e-3)

print("Независимость от густоты сетки")
# Та же планформа, но 20 сечений вместо 16906 граней.
n = 20
yy = np.linspace(0, SPAN, n)
P = []
F = []
for y in yy:
    c = CR - (CR - CT) * (y / SPAN)
    P += [[-0.64, y, 0.0], [-0.64 + c, y, 0.0]]
for i in range(n - 1):
    a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
    F += [[a, b, d], [a, d, c]]
gc = ref.wing_reference_from_mesh(np.array(P), np.array(F))
check("грубая сетка (%d граней) даёт ту же Sref в пределах 1%%"
      % len(F), abs(gc["area"] - S_EXACT) / S_EXACT < 0.01)
check("грубая сетка даёт тот же MAC в пределах 1%%",
      abs(gc["mac"] - MAC_EXACT) / MAC_EXACT < 0.01)

print("Простые фигуры")
plate = ref.wing_reference_from_mesh(
    np.array([[0, 0, 0.], [0, 6, 0.], [2, 0, 0.], [2, 6, 0.]]),
    np.array([[0, 1, 3], [0, 3, 2]]))
check("пластина из 2 треугольников: S=12 (%.4f)" % plate["area"],
      abs(plate["area"] - 12.0) / 12.0 < 0.01)
check("пластина из 2 треугольников: MAC=2 (%.4f)" % plate["mac"],
      abs(plate["mac"] - 2.0) < 1e-6)
check("вырожденный вход даёт None",
      ref.wing_reference_from_mesh(np.zeros((3, 3)),
                                   np.array([[0, 1, 2]])) is None)
check("несуществующие индексы дают None",
      ref.wing_reference_from_mesh(np.array([[0, 0, 0.], [1, 0, 0.]]),
                                   np.array([[0, 1, 0]])) is None)
check("обёртка pv даёт тот же результат",
      abs(ref.wing_reference_from_pv(wing)["area"] - g["area"]) < 1e-9)

# ------------------------------------------- calculate_reference_data
print("calculate_reference_data")
app = QApplication.instance() or QApplication([])


def make_win(mesh, span, cr, ct):
    w = mw.MainWindow.__new__(mw.MainWindow)
    w.bodies = [{"id": 1, "name": "wing", "role": "wing", "mesh": mesh,
                 "actor": None, "visible": True, "color": None}]
    w.w_span = Spin(span)
    w.w_chord_root = Spin(cr)
    w.w_chord_tip = Spin(ct)
    w.w_sweep = Spin(0.0)
    w.w_pos_x = Spin(-0.64)
    w.w_pos_y = Spin(0.0)
    w.w_pos_z = Spin(0.0)
    w.log_text = QTextEdit()
    return w


# Случай 1: спинбоксы согласованы с моделью — поведение не меняется.
w1 = make_win(wing, SPAN, CR, CT)
lr1, sr1, ox1, oy1, oz1 = mw.MainWindow.calculate_reference_data(w1)
check("согласованные спинбоксы: Lref как по учебной формуле (%.4f)" % lr1,
      abs(lr1 - MAC_EXACT) < 1e-6)
check("согласованные спинбоксы: Sref как по учебной формуле (%.4f)" % sr1,
      abs(sr1 - S_EXACT) < 1e-6)
check("согласованные спинбоксы: предупреждения нет",
      "размах крыла в модели" not in w1.log_text.toPlainText())

# Случай 2: заводские значения спинбоксов при той же модели.
w2 = make_win(wing, 10.0, 1.8, 0.9)
lr2, sr2, ox2, oy2, oz2 = mw.MainWindow.calculate_reference_data(w2)
check("заводские спинбоксы: Sref взята с геометрии (%.4f вместо 13.500)"
      % sr2, abs(sr2 - S_EXACT) / S_EXACT < 0.01)
check("заводские спинбоксы: Lref взят с геометрии (%.4f вместо 1.400)"
      % lr2, abs(lr2 - MAC_EXACT) / MAC_EXACT < 0.01)
check("заводские спинбоксы: в логе объяснена подмена",
      "размах крыла в модели" in w2.log_text.toPlainText(),
      w2.log_text.toPlainText()[:80])

# Случай 3: модель втрое крупнее — площадь должна вырасти в 9 раз.
big_pts = np.asarray(wing.points) * 3.0
big = pv.PolyData(big_pts, np.asarray(wing.faces))
w3 = make_win(big, 10.0, 1.8, 0.9)
lr3, sr3, _, _, _ = mw.MainWindow.calculate_reference_data(w3)
check("модель x3: Sref выросла в 9 раз (%.3f против %.3f)"
      % (sr3, 9.0 * S_EXACT), abs(sr3 - 9.0 * S_EXACT) / S_EXACT < 0.1)
check("модель x3: Lref вырос в 3 раза (%.4f против %.4f)"
      % (lr3, 3.0 * MAC_EXACT),
      abs(lr3 - 3.0 * MAC_EXACT) / MAC_EXACT < 0.01)

# Негативный контроль: если крыла нет, ветка геометрии не вызывается.
w4 = make_win(None, 10.0, 1.8, 0.9)
w4.bodies = []
lr4, sr4, _, _, _ = mw.MainWindow.calculate_reference_data(w4)
check("без крыла RefData по умолчанию (1.0/1.0)",
      lr4 == 1.0 and sr4 == 1.0, (lr4, sr4))

print()
print("Проверок: %d" % N[0])
if FAIL:
    print("ПРОВАЛЕНО: %d -> %s" % (len(FAIL), FAIL))
    sys.exit(1)
print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
