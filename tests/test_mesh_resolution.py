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
    logged = []          # то, что ушло в лог приложения через log_cb
    with redirect_stdout(buf):
        # Этот тест проверяет предупреждение картезианского фона о том,
        # что тонкое тело не разрешается его шагом. Телооблекающая сетка
        # (TetGen) поверхность облегает и предупреждение не печатает —
        # поэтому путь TetGen здесь отключён, иначе проверялся бы не тот
        # код. Поведение TetGen покрыто в tests/test_bodyfit_tetgen.py.
        import types as _types
        _stub = _types.ModuleType("mesh.bodyfit_tetgen")

        def _stub_build(*a, **k):
            # Настоящий генератор объясняет отказ через тот же колбэк,
            # поэтому и заглушка обязана это делать: проверяется канал,
            # а не только текст.
            k.get("log", lambda *_: None)(
                "   Внимание: TetGen недоступен, картезианская сетка фона")
            return None

        _stub.build_body_fitted_grid = _stub_build
        _saved = sys.modules.get("mesh.bodyfit_tetgen")
        sys.modules["mesh.bodyfit_tetgen"] = _stub
        try:
            ok, msg = _gm.generate_mesh_impl(
                [fus, tail], quality_text="Средняя",
                progress_cb=lambda p, t: None,
                log_cb=logged.append)
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

    # У собранного exe окна консоли нет (CREATE_NO_WINDOW), поэтому stdout
    # не видит никто. Диагностика обязана доходить и до лога приложения.
    joined = "\n".join(logged)
    check("log_cb получил строки диагностики", len(logged) > 0,
          "получено %d" % len(logged))
    check("причина отката на картезианский путь дошла до лога",
          "TetGen недоступен" in joined, joined[:80])
    check("отчёт о разрешении дошёл до лога",
          "Проверка разрешающей способности" in joined)
    check("вердикт по тонкому телу дошёл до лога",
          "НЕ РАЗРЕШАЕТСЯ" in joined)
    check("диагностика не дублируется в логе",
          joined.count("Проверка разрешающей способности") == 1,
          "%d раз" % joined.count("Проверка разрешающей способности"))
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

    # ---------------------------------------------------------------
    # Диагностика обязана доходить до лога приложения, а не только в
    # stdout: у собранного exe окна консоли нет (CREATE_NO_WINDOW), и
    # причину отката на картезианскую сетку пользователь иначе не видит.
    # Проверяется настоящий MeshWorker с настоящим Qt.
    print()
    print("== диагностика доходит до лога приложения ==")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    _qapp = QApplication.instance() or QApplication([])
    import mesh.mesh_worker as _msw

    _seen = []
    _done = []
    _orig = _msw.generate_mesh_impl

    def _fake(paths, quality_text="Средняя", progress_cb=None,
              cancel_cb=None, use_symmetry=False, symmetry_planes=None,
              log_cb=None):
        _seen.append(log_cb)
        if log_cb:
            log_cb("Внимание: откат на картезианскую сетку фона")
        return True, "тестовая сетка"

    _msw.generate_mesh_impl = _fake
    try:
        _w = _msw.MeshWorker(["a.stl"], "Средняя")
        _w.log_signal.connect(_done.append)
        _w.run()      # напрямую, без потока: соединение прямое
    finally:
        _msw.generate_mesh_impl = _orig

    check("MeshWorker передаёт log_cb в генератор",
          len(_seen) == 1 and _seen[0] is not None,
          "вызовов %d" % len(_seen))
    check("log_cb привязан к этому воркеру",
          bool(_seen) and getattr(_seen[0], "__self__", None) is _w)
    check("текст диагностики доходит до сигнала дословно",
          _done == ["Внимание: откат на картезианскую сетку фона"], str(_done))

    # Метод подключения в окне — исполняется настоящий код MainWindow.
    from ui.main_window import MainWindow as _MW
    _ui = _MW.__new__(_MW)
    _lines = []

    class _FakeLog:
        def append(self, s):
            _lines.append(s)

    _ui.log_text = _FakeLog()
    _w2 = _msw.MeshWorker(["a.stl"])
    _MW._connect_mesh_log(_ui, _w2)
    _w2.log_signal.emit("проверка канала")
    check("_connect_mesh_log направляет сигнал в лог окна",
          _lines == ["проверка канала"], str(_lines))

    class _NoSig:
        pass

    _MW._connect_mesh_log(_ui, _NoSig())
    check("_connect_mesh_log терпит воркер без log_signal", True)

    # ---------------------------------------------------------------
    # Абсолютный пол шага у тела (0.05/0.08/0.04 м) на мелкой модели
    # перевешивает пресет, и сетка становится непригодной. Раньше
    # условие `h_near < _h_floor` не срабатывало никогда — h_near уже
    # посчитан как max(пресет, пол), а пресет «Средней» равен
    # body_size*0.015, то есть самому _h_floor. Предупреждение было
    # мёртвым кодом: на модели 0.065 м шаг молча становился 0.0500 м.
    print()
    print("== предупреждение о применённом поле шага ==")
    tiny = os.path.join(workdir, "tiny.stl")
    pv.Box(bounds=(0, 0.065, 0, 0.019, 0, 0.0015)).triangulate().save(tiny)
    _stub2 = _types.ModuleType("mesh.bodyfit_tetgen")
    _stub2.build_body_fitted_grid = lambda *a, **k: None
    _saved2 = sys.modules.get("mesh.bodyfit_tetgen")
    sys.modules["mesh.bodyfit_tetgen"] = _stub2
    try:
        _tiny_log = []
        _buf2 = io.StringIO()
        with redirect_stdout(_buf2):
            _ok2, _msg2 = _gm.generate_mesh_impl(
                [tiny], quality_text="Средняя", log_cb=_tiny_log.append)
    finally:
        if _saved2 is None:
            sys.modules.pop("mesh.bodyfit_tetgen", None)
        else:
            sys.modules["mesh.bodyfit_tetgen"] = _saved2
    _tiny_joined = "\n".join(_tiny_log)
    # На модели 0.065 м генератор закономерно отказывает: при шаге
    # 0.05 м центр ни одной ячейки не попадает в пластину толщиной
    # 0.0015 м. Отказ правильный, а предупреждение объясняет причину
    # до него, а не после.
    check("на модели 0.065 м генератор честно отказывает",
          _ok2 is False and "не вырезана" in str(_msg2), str(_msg2)[:80])
    check("предупреждение о применённом поле шага напечатано",
          "ниже допустимого минимума" in _tiny_joined, _tiny_joined[:90])
    check("в предупреждении названа доля размера модели",
          "%" in _tiny_joined and "габарите модели" in _tiny_joined)
    check("предупреждение подсказывает проверить масштаб",
          "масштаб" in _tiny_joined)

    # Негативный контроль: на модели нормального размера пол не
    # применяется, и предупреждения быть не должно.
    _stub3 = _types.ModuleType("mesh.bodyfit_tetgen")
    _stub3.build_body_fitted_grid = lambda *a, **k: None
    sys.modules["mesh.bodyfit_tetgen"] = _stub3
    try:
        _norm_log = []
        _buf3 = io.StringIO()
        with redirect_stdout(_buf3):
            _gm.generate_mesh_impl([fus, tail], quality_text="Средняя",
                                   log_cb=_norm_log.append)
    finally:
        if _saved2 is None:
            sys.modules.pop("mesh.bodyfit_tetgen", None)
        else:
            sys.modules["mesh.bodyfit_tetgen"] = _saved2
    check("на модели 8 м пол не применяется и не пугает зря",
          "ниже допустимого минимума" not in "\n".join(_norm_log),
          "\n".join(_norm_log)[:90])

    # ---------------------------------------------------------------
    # Причина отката на картезианский фон обязана доходить до лога.
    # Раньше отказ импорта mesh.bodyfit_tetgen проглатывался голым
    # except, и пользователь получал сетку без объяснений.
    print()
    print("== причина отката на картезианский фон видна в логе ==")
    import types as _t2

    class _Boom(_t2.ModuleType):
        def __getattr__(self, name):
            raise ImportError("No module named 'tetgen'")

    _real = sys.modules.get("mesh.bodyfit_tetgen")
    sys.modules["mesh.bodyfit_tetgen"] = _Boom("mesh.bodyfit_tetgen")
    try:
        _imp_log = []
        _buf4 = io.StringIO()
        with redirect_stdout(_buf4):
            _gm.generate_mesh_impl([fus, tail], quality_text="Средняя",
                                   log_cb=_imp_log.append)
    finally:
        if _real is None:
            sys.modules.pop("mesh.bodyfit_tetgen", None)
        else:
            sys.modules["mesh.bodyfit_tetgen"] = _real
    _imp_joined = "\n".join(_imp_log)
    check("отказ импорта модуля попал в лог",
          "телооблекающая сетка недоступна" in _imp_joined, _imp_joined[:90])
    check("в сообщении сказано, чем это грозит",
          "не облегает" in _imp_joined)

    import mesh.bodyfit_tetgen as _BF
    check("tetgen_missing перечисляет tetgen при его отсутствии",
          _BF.tetgen_available() is True or "tetgen" in _BF.tetgen_missing(),
          str(_BF.tetgen_missing()))
    _keep = (_BF.HAS_TETGEN, _BF.HAS_TRIMESH, _BF.HAS_SCIPY, _BF.HAS_PYVISTA)
    _BF.HAS_TETGEN, _BF.HAS_TRIMESH = False, True
    try:
        check("tetgen_missing называет именно отсутствующий пакет",
              _BF.tetgen_missing() == ["tetgen"], str(_BF.tetgen_missing()))
    finally:
        (_BF.HAS_TETGEN, _BF.HAS_TRIMESH,
         _BF.HAS_SCIPY, _BF.HAS_PYVISTA) = _keep
    check("tetgen_missing пуст, когда всё на месте",
          _BF.tetgen_available() is False or _BF.tetgen_missing() == [],
          str(_BF.tetgen_missing()))

    print()
    print("Пройдено: %d" % _passed)
    if _failed:
        print("ПРОВАЛЕНО ТЕСТОВ: %d → %s" % (len(_failed), _failed))
        raise SystemExit(1)
    print("Все проверки пройдены.")


if __name__ == "__main__":
    main()
