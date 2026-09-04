# -*- coding: utf-8 -*-
"""Единицы модели в Design Rules.

Единицы спрашиваются при импорте CAD, но задать или исправить их можно и
потом — группа «Единицы модели» на странице Design Rules. Проверяется сам
пересчёт координат: метод ``MainWindow.apply_model_units`` вызывается на
подставном объекте, потому что полноценное окно MainWindow в тесте не
собирается.

Тест требует настоящих numpy/pyvista: qt_stubs подменяет pyvista
заглушкой, у которой ``pv.Sphere()`` возвращает ноль точек, и пересчёт
проверить было бы нельзя. При отсутствии зависимостей тест пропускается.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_FAILED = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  СБОЙ ") + name +
          ((" — " + str(extra)) if extra else ""))
    if not cond:
        _FAILED.append(name)


try:
    import numpy as np
    import pyvista as pv
except Exception as _e:                                    # pragma: no cover
    print("ПРОПУЩЕНО: нет numpy/pyvista: %s" % _e)
    sys.exit(0)

from ui.main_window import MainWindow


# ---------------------------------------------------------------------------
# Подстановка: только то, что читают проверяемые методы
# ---------------------------------------------------------------------------
class _Log(list):
    def append(self, m):
        list.append(self, m)


class _Plotter:
    def render(self):
        pass


class _Combo:
    """Мини-копия QComboBox: currentData/currentText."""

    def __init__(self, factor, text):
        self._factor, self._text = factor, text

    def currentData(self):
        return self._factor

    def currentText(self):
        return self._text


class _Label:
    def __init__(self):
        self.text = ""

    def setText(self, t):
        self.text = t


def _fake(n_bodies=2, factor=1.0, units=(0.001, "Миллиметры")):
    f = types.SimpleNamespace()
    f.bodies = [{"name": "body%d" % i,
                 "mesh": pv.Sphere(radius=1.0, center=(0.0, 0.0, 0.0))}
                for i in range(n_bodies)]
    f._model_unit_factor = factor
    f.log_text = _Log()
    f.plotter = _Plotter()
    f.lbl_model_units = _Label()
    f.combo_model_units = _Combo(*units)
    f.invalidate_calls = []
    f.invalidate_mesh = lambda reason: f.invalidate_calls.append(reason)
    f._model_bbox = lambda: MainWindow._model_bbox(f)
    f._refresh_model_units_label = \
        lambda: MainWindow._refresh_model_units_label(f)
    return f


def _extent(mesh):
    p = np.asarray(mesh.points, dtype=float)
    return p.max(0) - p.min(0)


print("== Единицы модели: пересчёт геометрии ==")

f = _fake()
before = _extent(f.bodies[0]["mesh"])
check("геометрия до пересчёта ненулевая", float(before.max()) > 1.9,
      " ".join("%.4f" % v for v in before))

MainWindow.apply_model_units(f)
after = _extent(f.bodies[0]["mesh"])
check("мм: координаты умножены на 0.001",
      np.allclose(after, before * 0.001, rtol=0, atol=1e-12),
      " ".join("%.6f" % v for v in after))
check("мм: множитель запомнен", f._model_unit_factor == 0.001,
      f._model_unit_factor)
check("мм: пересчитаны все тела",
      all(np.allclose(_extent(b["mesh"]), before * 0.001, atol=1e-12)
          for b in f.bodies))
check("мм: сетка помечена устаревшей",
      f.invalidate_calls == ["изменены единицы модели"], f.invalidate_calls)
check("мм: в логе назван множитель",
      any("множитель 0.001" in m for m in f.log_text), list(f.log_text))
check("подпись обновлена и показывает габарит",
      "Габарит модели сейчас" in f.lbl_model_units.text,
      f.lbl_model_units.text.replace("\n", " | "))

print("== Идемпотентность ==")
n_log = len(f.log_text)
MainWindow.apply_model_units(f)
again = _extent(f.bodies[0]["mesh"])
check("повтор с тем же выбором геометрию не меняет",
      np.allclose(again, after, atol=1e-12),
      " ".join("%.6f" % v for v in again))
check("повтор объясняет, что ничего не делал",
      any("не менялась" in m for m in f.log_text[n_log:]),
      list(f.log_text[n_log:]))
check("повтор не инвалидирует сетку второй раз",
      f.invalidate_calls == ["изменены единицы модели"], f.invalidate_calls)

print("== Обратный ход ==")
f.combo_model_units = _Combo(1.0, "Метры")
MainWindow.apply_model_units(f)
back = _extent(f.bodies[0]["mesh"])
check("возврат к метрам восстанавливает координаты",
      np.allclose(back, before, rtol=0, atol=1e-12),
      " ".join("%.6f" % v for v in back))
check("возврат к метрам: множитель 1.0", f._model_unit_factor == 1.0,
      f._model_unit_factor)

print("== Сантиметры и промежуточный переход ==")
g = _fake(n_bodies=1)
b0 = _extent(g.bodies[0]["mesh"])
g.combo_model_units = _Combo(0.001, "Миллиметры")
MainWindow.apply_model_units(g)
g.combo_model_units = _Combo(0.01, "Сантиметры")
MainWindow.apply_model_units(g)
c1 = _extent(g.bodies[0]["mesh"])
# После мм (x0.001) переход к см даёт ещё x10, то есть суммарно x0.01
# от исходных координат.
check("мм -> см даёт суммарный множитель 0.01",
      np.allclose(c1, b0 * 0.01, atol=1e-12),
      " ".join("%.6f" % v for v in c1))
check("мм -> см: _model_unit_factor = 0.01", g._model_unit_factor == 0.01,
      g._model_unit_factor)

print("== Крайние случаи ==")
e = _fake()
e.bodies = []
MainWindow.apply_model_units(e)
check("нет геометрии: предупреждение, без исключения",
      any("нечего" in m for m in e.log_text), list(e.log_text))
check("нет геометрии: сетка не инвалидируется", e.invalidate_calls == [],
      e.invalidate_calls)
check("_model_bbox без тел возвращает пустую строку",
      MainWindow._model_bbox(e) == "", repr(MainWindow._model_bbox(e)))

h = _fake(n_bodies=2)
h.bodies[1]["mesh"] = None
MainWindow.apply_model_units(h)
check("тело без mesh пропускается, остальные считаются",
      np.allclose(_extent(h.bodies[0]["mesh"]),
                  _extent(pv.Sphere(radius=1.0)) * 0.001, atol=1e-12))
check("в логе сказано, сколько тел пересчитано",
      any("Пересчитано тел: 1 из 2" in m for m in h.log_text),
      list(h.log_text))

print()
if _FAILED:
    print("ПРОВАЛЕНО ПРОВЕРОК: %d -> %s" % (len(_FAILED), _FAILED))
    sys.exit(1)
print("Все проверки пройдены.")
