# -*- coding: utf-8 -*-
"""Проверка разрешающей способности фоновой сетки.

Сетка строится вырезанием ячеек фона: ячейка удаляется, если её ЦЕНТР
попал внутрь тела. Поэтому тело тоньше одного шага фона попадает в сетку
как ступенчатая пластина в одну ячейку — или не попадает вовсе.

На полном самолёте при качестве «Средняя» шаг у тела считается от размаха
(~9 м) и равен ~0.135 м, а толщина ГО/ВО/руля (хорда 0.70 м, профиль 12%)
— 0.084 м, то есть 0.62 шага. Расчёт на такой сетке расходится.

Тест гоняет настоящий generate_mesh_impl на двух телах — тонком и
толстом — и проверяет, что генератор печатает честный отчёт.

Отдельный файл: tests/qt_stubs.py подменяет pyvista заглушкой, а здесь
нужен настоящий. Модуль грузится по пути файла, потому что mesh/__init__.py
тянет PyQt5.

Запуск:  python tests/test_mesh_resolution.py
"""
import io
import os
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pyvista as pv
except ImportError as exc:                      # pragma: no cover
    print("ПРОПУЩЕНО: нет pyvista (%s)" % exc)
    raise SystemExit(0)

import importlib.util                           # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "gmsh_generator_standalone",
    os.path.join(_ROOT, "mesh", "gmsh_generator.py"))
_gm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gm)

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


def main():
    workdir = tempfile.mkdtemp()
    fus = os.path.join(workdir, "fuselage.stl")
    tail = os.path.join(workdir, "h_stab.stl")

    # Фюзеляж: диаметр 1.20 м. ГО: хорда 0.70 м, профиль 12% -> 0.084 м.
    pv.Box(bounds=(-4, 4, -0.6, 0.6, -0.6, 0.6)).triangulate().save(fus)
    pv.Box(bounds=(-0.35, 0.35, -0.5, 0.5, -0.042, 0.042)).triangulate().save(tail)

    buf = io.StringIO()
    with redirect_stdout(buf):
        # Этот тест проверяет предупреждение картезианского фона о том,
        # что тонкое тело не разрешается его шагом. Телооблекающая сетка
        # (TetGen) поверхность облегает и предупреждение не печатает —
        # поэтому путь TetGen здесь отключён, иначе проверялся бы не тот
        # код. Поведение TetGen покрыто в tests/test_bodyfit_tetgen.py.
        import types as _types
        _stub = _types.ModuleType("mesh.bodyfit_tetgen")
        _stub.build_body_fitted_grid = lambda *a, **k: None
        _saved = sys.modules.get("mesh.bodyfit_tetgen")
        sys.modules["mesh.bodyfit_tetgen"] = _stub
        try:
            ok, msg = _gm.generate_mesh_impl(
                [fus, tail], quality_text="Средняя",
                progress_cb=lambda p, t: None)
        finally:
            if _saved is None:
                sys.modules.pop("mesh.bodyfit_tetgen", None)
            else:
                sys.modules["mesh.bodyfit_tetgen"] = _saved
    log = buf.getvalue()

    print("== отчёт о разрешающей способности ==")
    for line in log.splitlines():
        if "шага" in line or "Проверка разрешающей" in line:
            print("   " + line.strip())
    print()

    check("генератор отработал", ok is True, msg)
    check("отчёт о разрешающей способности напечатан",
          "Проверка разрешающей способности" in log)
    check("тонкое тело помечено как неразрешаемое",
          "h_stab.stl" in log and "НЕ РАЗРЕШАЕТСЯ" in log)
    check("толстое тело помечено как разрешаемое",
          "fuselage.stl: мин. габарит 1.2000 м" in log and "разрешается" in log)
    check("число шагов для ГО посчитано верно (0.084/h < 1)",
          "0.70 шага" in log or "0.62 шага" in log,
          [l for l in log.splitlines() if "h_stab" in l])
    check("объяснена причина — сетка не облегает поверхность",
          "не облегает поверхность" in log)
    check("предложено два выхода: мельче шаг или gmsh по STL",
          "облегающая поверхность" in log and "gmsh по STL" in log
          and "шаг у тела не более" in log)

    # Тонкое тело действительно почти не вырезается из фона — это и есть
    # механизм расходимости, а не теория.
    import re as _re
    _removed = dict()
    for _m in _re.finditer(r"Компонент (\d+): удалено (\d+) ячеек", log):
        _removed[int(_m.group(1))] = int(_m.group(2))
    check("удалено ячеек по обоим компонентам", len(_removed) == 2, _removed)
    if len(_removed) == 2:
        _thick = max(_removed.values())
        _thin = min(_removed.values())
        check("тонкое тело вырезается на порядки хуже толстого",
              _thin * 20 < _thick, "%d против %d" % (_thin, _thick))

    print()
    print("Пройдено: %d" % _passed)
    if _failed:
        print("ПРОВАЛЕНО ТЕСТОВ: %d → %s" % (len(_failed), _failed))
        raise SystemExit(1)
    print("Все проверки пройдены.")


if __name__ == "__main__":
    main()
