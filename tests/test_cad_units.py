# -*- coding: utf-8 -*-
"""Единицы CAD-модели при импорте.

Координаты STEP читаются как есть, а расчёт идёт в метрах. Файл FreeCAD
по умолчанию объявляет SI_UNIT(.MILLI.,.METRE.), поэтому деталь длиной
65 мм приезжала в приложение как «крыло размахом 65.077 м», и от этого
считались справочные данные, шаг сетки и число Рейнольдса.

Тест проверяет:
  * определение объявленных единиц по тексту STEP, в том числе на точном
    фрагменте из plane_wing.step пользователя;
  * масштабирование геометрии при конвертации;
  * что диалог импорта подставляет определённые единицы по умолчанию.

Запуск:  python tests/test_cad_units.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_passed = 0
_failed = []


def check(name, cond, extra=None):
    global _passed
    if cond:
        _passed += 1
        print("  ✅ %s" % name)
    else:
        _failed.append(name)
        print("  ❌ %s%s" % (name, "" if extra is None else " %s" % (extra,)))


def _write(text):
    fd, path = tempfile.mkstemp(suffix=".step")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def main():
    from geometry.generators import cad_detect_units

    print("== определение единиц по тексту STEP ==")
    # Точный фрагмент из plane_wing.step пользователя.
    frag = _write(
        "ISO-10303-21;\nDATA;\n"
        "#2117 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3) \n"
        "GLOBAL_UNIT_ASSIGNED_CONTEXT((#2118,#2119,#2120)) );\n"
        "#2118 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );\n"
        "#2119 = ( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) );\n"
        "ENDSEC;\n")
    got = cad_detect_units(frag)
    check("plane_wing.step: объявлены миллиметры",
          got == ("мм", 0.001), str(got))
    os.remove(frag)

    for pre, name, factor in (("MICRO", "мкм", 1e-6), ("MILLI", "мм", 1e-3),
                              ("CENTI", "см", 1e-2), ("DECI", "дм", 1e-1),
                              ("$", "м", 1.0), ("KILO", "км", 1e3)):
        p = _write("#1 = ( LENGTH_UNIT() NAMED_UNIT(*) "
                   "SI_UNIT(.%s.,.METRE.) );" % pre)
        got = cad_detect_units(p)
        check("префикс .%s. -> %s" % (pre, name),
              got is not None and got[0] == name
              and abs(got[1] - factor) < 1e-15, str(got))
        os.remove(p)

    p = _write("#1 = ( LENGTH_UNIT() NAMED_UNIT(*) "
               "CONVERSION_BASED_UNIT('INCH',#2) );")
    got = cad_detect_units(p)
    check("CONVERSION_BASED_UNIT('INCH') -> 0.0254",
          got is not None and abs(got[1] - 0.0254) < 1e-15, str(got))
    os.remove(p)

    p = _write("ISO-10303-21;\nDATA;\n#1 = CARTESIAN_POINT('',(0.,0.,0.));\n")
    check("без объявления единиц -> None", cad_detect_units(p) is None)
    os.remove(p)

    check("несуществующий файл -> None",
          cad_detect_units("/нет/такого/файла.step") is None)

    print()
    print("== диалог выбора масштаба ==")
    from PyQt5.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])
    import ui.main_window as MW
    from ui.main_window import MainWindow

    captured = {}

    class _FakeQID:
        @staticmethod
        def getItem(parent, title, text, items, current, editable):
            captured.clear()
            captured.update(title=title, text=text, items=list(items),
                            current=current)
            return items[current], True

    _real = MW.QInputDialog
    MW.QInputDialog = _FakeQID
    try:
        ui = MainWindow.__new__(MainWindow)

        p = _write("#2118 = ( LENGTH_UNIT() NAMED_UNIT(*) "
                   "SI_UNIT(.MILLI.,.METRE.) );")
        k = MainWindow._cad_ask_scale(ui, p)
        check("для файла в мм по умолчанию выбирается мм → м",
              abs(k - 0.001) < 1e-15, str(k))
        check("в вопросе названы определённые единицы",
              "мм" in captured["text"], captured["text"][:60])
        check("вариантов на выбор четыре", len(captured["items"]) == 4,
              str(captured["items"]))
        os.remove(p)

        p = _write("ISO-10303-21;\nDATA;\n#1 = CARTESIAN_POINT('',(0.,0.,0.));\n")
        k = MainWindow._cad_ask_scale(ui, p)
        check("без объявления единиц диалог всё равно спрашивает",
              k is not None and "не удалось" in captured["text"],
              captured["text"][:60])
        os.remove(p)

        class _Cancel:
            @staticmethod
            def getItem(*a, **k):
                return None, False

        MW.QInputDialog = _Cancel
        check("отмена диалога возвращает None",
              MainWindow._cad_ask_scale(ui, "/нет/такого.step") is None)
    finally:
        MW.QInputDialog = _real

    print()
    print("== масштабирование при конвертации ==")
    try:
        import gmsh  # noqa: F401
    except Exception as e:
        print("  — gmsh недоступен (%s), проверка конвертации пропущена" % e)
    else:
        from geometry.generators import cad_to_stl
        import trimesh
        import numpy as np

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        tag = gmsh.model.occ.addBox(0, 0, 0, 65.077, 19.0, 1.5)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        src = _write("")
        gmsh.write(src)
        gmsh.finalize()

        try:
            a = cad_to_stl(src, src + ".a.stl", scale=1.0)
            b = cad_to_stl(src, src + ".b.stl", scale=0.001)
            da = trimesh.load(a, process=False).bounds
            db = trimesh.load(b, process=False).bounds
            la = float((da[1] - da[0]).max())
            lb = float((db[1] - db[0]).max())
            check("scale=1 геометрию не меняет", abs(la - 65.077) < 1e-3,
                  "%.4f" % la)
            check("scale=0.001 уменьшает в 1000 раз",
                  abs(lb - 0.065077) < 1e-5, "%.6f" % lb)
            check("отношение ровно 1000", abs(la / lb - 1000.0) < 1e-3,
                  "%.3f" % (la / lb))
        finally:
            for f in (src, src + ".a.stl", src + ".b.stl"):
                try:
                    os.remove(f)
                except Exception:
                    pass

    print()
    print("Пройдено: %d" % _passed)
    if _failed:
        print("ПРОВАЛЕНО ТЕСТОВ: %d → %s" % (len(_failed), _failed))
        raise SystemExit(1)
    print("Все проверки пройдены.")


if __name__ == "__main__":
    main()
