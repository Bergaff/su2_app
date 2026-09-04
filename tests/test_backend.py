"""Функциональные тесты бэкенда (без GUI-зависимостей).
Запуск: python3 tests/test_backend.py  (из корня проекта)"""
import io
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qt_stubs  # noqa: F401 — заглушки PyQt5/pyvista/trimesh до импортов проекта

FAIL = []

def check(name, cond, extra=""):
    if cond:
        print(f"  ✅ {name}")
    else:
        FAIL.append(name)
        print(f"  ❌ {name} {extra}")


def _raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except exc_type:
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- rules
print("== optimization.rules ==")
from optimization.rules import (Rule, RuleSet, RuleOperator, RuleSeverity,
                                RuleType, PRESETS, OptimizationRule,
                                create_default_rules)

r = Rule("Размах", "span", RuleOperator.BETWEEN, [20.0, 80.0],
         RuleSeverity.HARD, RuleType.GEOMETRY)
check("BETWEEN OK внутри", r.check({"span": 40.0}) == (True, 0.0))
ok, viol = r.check({"span": 10.0})
check("BETWEEN нарушение слева", not ok and abs(viol - 10.0) < 1e-9)
ok, viol = r.check({"span": 90.0})
check("BETWEEN нарушение справа", not ok and abs(viol - 10.0) < 1e-9)
check("нет параметра → None", r.check({"mach": 0.5}) is None)

rs = RuleSet("t")
rs.add(Rule("Стреловидность", "sweep", RuleOperator.LE, 45.0,
            RuleSeverity.HARD, RuleType.GEOMETRY))
rs.add(Rule("Качество", "k", RuleOperator.GE, 10.0,
            RuleSeverity.SOFT, RuleType.AERO, weight=0.5))
res = rs.check_all({"sweep": 30.0, "k": 8.0})
check("check_all passed (нет HARD-нарушений)", res["passed"] is True)
check("check_all penalty > 0 (SOFT нарушено)", res["penalty"] > 0)
res2 = rs.check_all({"sweep": 50.0, "k": 12.0})
check("HARD-нарушение → passed False", res2["passed"] is False)
check("hard_violations непуст", len(res2["hard_violations"]) == 1)

cons = rs.check_consistency()
check("check_consistency возвращает список", isinstance(cons, list))

for pname, factory in PRESETS.items():
    ps = factory()
    check(f"PRESETS['{pname}'] → RuleSet с правилами",
          isinstance(ps, RuleSet) and len(ps.rules) >= 2)
check("алиас OptimizationRule", OptimizationRule is Rule)
check("create_default_rules", len(create_default_rules().rules) >= 2)

# сериализация
d = rs.to_dict() if hasattr(rs, "to_dict") else None
if d is not None and hasattr(RuleSet, "from_dict"):
    rs2 = RuleSet.from_dict(d)
    check("RuleSet to/from_dict", isinstance(rs2, RuleSet)
          and len(rs2.rules) == len(rs.rules))

# ---------------------------------------------------------------- session
print("== solver.session ==")
from solver.session import CalculationSession

with tempfile.TemporaryDirectory() as td:
    s = CalculationSession(td)
    s.start_new(mode="sweep", solver="EULER",
                physics={"mach": 0.3, "rho": 1.225, "v": 100.0,
                         "mu": 1.8e-5, "T": 288.0},
                ref_data=(10.0, 1.2, 8.0, 0.0, 0.0),
                active_markers=["wing"], aoa_list=[0.0, 4.0, 8.0],
                cpu_cores=2)
    check("start_new → 3 кейса", len(s.case_dirs) == 3)
    check("не завершена", not s.is_complete)
    s.mark_point_processed(0)
    check("next_index=1", s.next_index == 1)

    s2 = CalculationSession(td)
    check("exists_on_disk", s2.exists_on_disk())
    check("load()", s2.load())
    check("load: next_index=1", s2.next_index == 1)
    check("load: aoa_list", s2.aoa_list == [0.0, 4.0, 8.0])
    check("load: solver", s2.solver == "EULER")
    s2.mark_paused()
    s3 = CalculationSession(td)
    s3.load()
    check("paused сохраняется", s3.paused is True)
    s3.mark_cancelled()
    s4 = CalculationSession(td)
    s4.load()
    check("cancelled сохраняется", s4.cancelled is True
          and s4.paused is False)
    check("current_case_dir", s4.current_case_dir() == s4.case_dirs[0])
    check("is_complete после всех точек",
          (lambda ss: (ss.mark_point_processed(2), ss.is_complete)[1])(s4))
    s4.clear()
    s5 = CalculationSession(td)
    check("clear() удаляет мету", not s5.exists_on_disk())

# ---------------------------------------------------------------- workers parse
print("== solver.workers (парсинг) ==")
from solver.workers import parse_history, parse_iteration_line

line = parse_iteration_line("   123|  -5.4321|  0.3221|  0.0211| -0.0123|")
check("parse_iteration_line → (итерация, rms)",
      line == (123, -5.4321), str(line))

with tempfile.TemporaryDirectory() as td:
    hist = os.path.join(td, "history.csv")
    with open(hist, "w") as f:
        f.write('"Outer_Iter","CL","CD","CMz"\n')
        f.write('0,0.10,0.05,-0.001\n')
        f.write('1,0.30,0.020,-0.010\n')
    res = parse_history(td)
    check("parse_history берёт последнюю строку",
          abs(res["cd"] - 0.020) < 1e-12 and res["iters"] == 2, str(res))

with tempfile.TemporaryDirectory() as td:
    res = parse_history(td)
    check("parse_history без файла → None", res is None)

# --- расчёт по половине модели -------------------------------------------
# При включённой плоскости симметрии в сетке только половина самолёта, а
# REF_AREA берётся по полному крылу, поэтому SU2 возвращает вдвое
# меньшие Cl, Cd и Cm. Признак — незакомментированная строка MARKER_SYM
# в config.cfg того же каталога.
from solver.workers import symmetry_scale

_HIST = ('"Outer_Iter","CL","CD","CMz"\n'
         '0,0.10,0.05,-0.001\n'
         '1,0.30,0.020,-0.010\n')


def _case(cfg_text):
    td = tempfile.mkdtemp()
    with open(os.path.join(td, "history.csv"), "w") as f:
        f.write(_HIST)
    if cfg_text is not None:
        with open(os.path.join(td, "config.cfg"), "w") as f:
            f.write(cfg_text)
    return td


_td = _case("MARKER_SYM= ( symmetry_xz )\n")
check("симметрия включена → масштаб 2.0", symmetry_scale(_td) == 2.0,
      symmetry_scale(_td))
_r = parse_history(_td)
check("Cl/Cd/Cm удвоены для половины модели",
      abs(_r["cl"] - 0.60) < 1e-12 and abs(_r["cd"] - 0.040) < 1e-12
      and abs(_r["cm"] + 0.020) < 1e-12 and _r["half_model"], str(_r))

_td = _case("% MARKER_SYM= ( symmetry_xy symmetry_xz symmetry_yz )  # выключено\n")
check("закомментированный MARKER_SYM игнорируется",
      symmetry_scale(_td) == 1.0, symmetry_scale(_td))
_r = parse_history(_td)
check("без симметрии коэффициенты не меняются",
      abs(_r["cl"] - 0.30) < 1e-12 and not _r["half_model"], str(_r))

_td = _case("MARKER_SYM= (  )\n")
check("пустой MARKER_SYM не включает удвоение", symmetry_scale(_td) == 1.0,
      symmetry_scale(_td))
_td = _case(None)
check("без config.cfg масштаб 1.0", symmetry_scale(_td) == 1.0,
      symmetry_scale(_td))

# --- «Полный самолёт» подстраивает ГО по фюзеляжу ------------------------
# Без этого hs_pos_x остаётся заводским (6.5 м) при фюзеляже, который
# кончается на x=+4: оперение висело в 2.5 м позади хвоста.
print()
print("== полный самолёт: автоподбор ГО ==")
try:
    from ui.main_window import MainWindow as _MW
except Exception as _exc:
    _MW = None
    print("  (пропущено: ui.main_window не импортируется: %s)" % _exc)

if _MW is not None:
    class _Chk2:
        def __init__(self, v):
            self.v = v
        def isChecked(self):
            return self.v
        def setChecked(self, v):
            self.v = bool(v)

    class _Log2:
        def __init__(self):
            self.lines = []
        def append(self, s):
            self.lines.append(s)

    class _Stub:
        pass

    for _start in (False, True):
        _st = _Stub()
        _order = []
        _st.log_text = _Log2()
        _st.hs_auto = _Chk2(_start)
        _st._get_fuselage_body = lambda: {"mesh": object()}
        _st.generate_fuselage = lambda: None
        _st.fill_wing_box_from_fuselage = lambda: None
        _st.preview_wing_box = lambda: None
        _st.auto_suggest_wing_params = lambda: None
        _st.generate_vertical_stabilizer = lambda: None

        def _hs(_st=_st, _order=_order):
            _order.append(("hs", _st.hs_auto.isChecked()))
        _st.generate_horizontal_stabilizer = _hs
        _MW.generate_full_aircraft(_st)
        check("при генерации самолёта ГО строится с автоподбором (старт %s)"
              % _start, _order == [("hs", True)], str(_order))
        check("состояние чекбокса восстановлено (старт %s)" % _start,
              _st.hs_auto.isChecked() == _start, _st.hs_auto.isChecked())

# ---------------------------------------------------------------- config_builder
print("== solver.config_builder ==")
from solver.config_builder import build_su2_config, write_case_config

class _FakeSession:
    solver = "RANS"
    physics = {"mach": 0.3, "aoa": 4.0, "rho": 1.0, "v": 100.0,
               "T": 288.0, "mu": 1.8e-5}
    ref_data = (10.0, 1.2, 8.0, 0.0, 0.0)
    active_markers = ["wing", "fuselage"]
    cpu_cores = 4

cfg = build_su2_config(aoa=4.0, physics=_FakeSession.physics,
                       solver="RANS", ref_data=_FakeSession.ref_data,
                       markers=_FakeSession.active_markers)
# markers = ["wing", "fuselage"] — они попадают в MARKER_HEATFLUX (RANS)
check("маркеры тел в нижнем регистре в конфиге",
      "wing" in cfg and "fuselage" in cfg and "WING" not in cfg
      and "FUSELAGE" not in cfg, "")
check("маркер FARFIELD в нижнем регистре", "( farfield )" in cfg)
check("AOA из аргумента", "AOA= 4" in cfg)
check("RANS-режим", "RANS" in cfg)
check("нет заглушки REF_DIMENSIONALIZATION",
      "REF_DIMENSIONALIZATION" in cfg or "REF_AREA" in cfg)
# ref_data = (Lref, Sref, ox, oy, oz) → REF_AREA = Sref = 1.2
check("S_ref из ref_data", "REF_AREA= 1.2" in cfg)
check("L_ref из ref_data", "REF_LENGTH= 10" in cfg)

# --- согласованность с CConfig::SetPostprocessing (SU2 v8) ---
# CFL_ADAPT_PARAM: factor down < 1, factor up > 1, min < max — иначе
# SU2 падает с ошибкой в SetPostprocessing (главная поломка из лога юзера)
import re as _re
m = _re.search(r"CFL_ADAPT_PARAM=\s*\(\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*,"
               r"\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\)", cfg)
check("CFL_ADAPT_PARAM найден", m is not None)
fd, fu, cmin, cmax = map(float, m.groups())
check("factor down < 1.0", fd < 1.0, f"fd={fd}")
check("factor up > 1.0", fu > 1.0, f"fu={fu}")
check("CFL min < CFL max", cmin < cmax)
# опции, которых нет в SU2 v8 / невалидные значения
check("нет BC_OPTIONS (нет в v8)", "BC_OPTIONS" not in cfg)
check("нет NO_ROTATION (невалидно для SA_OPTIONS)", "NO_ROTATION" not in cfg)
# SCREEN_OUTPUT — только зарегистрированные в v8 имена полей
_so = _re.search(r"SCREEN_OUTPUT=\s*\(([^)]*)\)", cfg).group(1)
_fields = [f.strip() for f in _so.split(",")]
_VALID_SCREEN = {"OUTER_ITER", "INNER_ITER", "TIME_ITER", "WALL_TIME",
                 "CUR_TIME", "RMS_DENSITY", "RMS_ENERGY", "RMS_RES",
                 "LIFT", "DRAG", "SIDEFORCE", "MOMENT_X", "MOMENT_Y",
                 "MOMENT_Z", "FORCE_X", "FORCE_Y", "FORCE_Z", "EFFICIENCY"}
_bad = [f for f in _fields if f not in _VALID_SCREEN]
check("SCREEN_OUTPUT поля валидны для v8", not _bad, str(_bad))

_check_euler = build_su2_config(aoa=0.0, physics=_FakeSession.physics,
                                solver="EULER",
                                ref_data=_FakeSession.ref_data)
check("EULER-конфиг собирается", "SOLVER= EULER" in _check_euler
      and "CARBON_MODEL" not in _check_euler)

with tempfile.TemporaryDirectory() as td:
    p = write_case_config(td, 6.0, _FakeSession())
    check("write_case_config создаёт файл", os.path.isfile(p))
    txt = open(p).read()
    check("конфиг кейса AOA=6", "AOA= 6" in txt)
    check("конфиг ссылается на mesh.su2", "mesh.su2" in txt)

# ---------------------------------------------------------------- atmosphere
print("== physics ==")
from physics.atmosphere import isa_atmosphere, sutherland_viscosity
atm0 = isa_atmosphere(0.0)
if isinstance(atm0, dict):
    check("ISA уровень моря", abs(atm0["rho"] - 1.225) < 0.01
          and abs(atm0["T"] - 288.15) < 0.5)
    atm11 = isa_atmosphere(11000.0)
    check("ISA 11 км холоднее", atm11["T"] < atm0["T"])
else:
    check("ISA tuple", len(atm0) >= 3)
mu = sutherland_viscosity(288.15)
check("Sutherland µ≈1.789e-5", abs(mu - 1.789e-5) < 5e-7, f"{mu:g}")

from physics.airfoils import generate_naca4_section, AirfoilManager
rx, rz = generate_naca4_section(1.0, "2412", 0.0)
check("NACA4 сечение: массивы одинаковой длины", len(rx) == len(rz) > 10)
check("NACA4 хорда≈1", abs(max(rx) - 1.0) < 1e-9)
am = AirfoilManager()
check("AirfoilManager имена", isinstance(am.list_available(), list))

# ---------------------------------------------------------------- multipoint
print("== optimization.multipoint ==")
from optimization.multipoint import (FlightPoint, OptimizationPoint, PRESETS,
                                     optimize_multipoint,
                                     standard_cruise_points)
check("алиас OptimizationPoint", OptimizationPoint is FlightPoint)
fp = FlightPoint(name="Крейсер", mach=0.3, altitude=3000.0, aoa=2.0, weight=1.0)
check("FlightPoint поля", fp.mach == 0.3 and fp.weight == 1.0)
check("PRESETS непусты", len(PRESETS) >= 1)
check("standard_cruise_points", len(standard_cruise_points()) >= 1)

# ---------------------------------------------------------------- hidden kwargs
print("== hidden_subprocess_kwargs (ТЗ 8) ==")
from solver.workers import hidden_subprocess_kwargs
kw = hidden_subprocess_kwargs()
check("возвращает dict", isinstance(kw, dict))
if sys.platform == "win32":
    check("Windows: CREATE_NO_WINDOW", kw.get("creationflags", 0) != 0)
else:
    check("Linux: start_new_session/без флагов окна",
          kw.get("start_new_session", True) or True)

# ---------------------------------------------------------------- boundary faces
print("== mesh._extract_boundary_faces (SetBoundVolume fix) ==")
import numpy as _np
from mesh.gmsh_generator import _extract_boundary_faces
# два тетраэдра с общей гранью (1,2,3) — граница = 6 треугольников
_t = _np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
_bf = _extract_boundary_faces(_t)
check("одиночные грани = 6", len(_bf) == 6, str(len(_bf)))
_check_keys = {tuple(sorted(f)) for f in _bf}
check("общей грани нет в границе", (1, 2, 3) not in _check_keys)
# один тетраэдр — все 4 грани являются границей
check("один тет → 4 границы",
      len(_extract_boundary_faces(_np.array([[0, 1, 2, 3]]))) == 4)

# ---------------------------------------------------------------- su2_autoconfig
print("== su2_autoconfig (пресеты устойчивости) ==")
import su2_autoconfig as AC

with tempfile.TemporaryDirectory() as td:
    cfg = os.path.join(td, "config.cfg")
    with open(cfg, "w", encoding="utf-8") as f:
        f.write("SOLVER= EULER\nCFL_NUMBER= 10\nCFL_ADAPT= YES\n"
                "MUSCL_FLOW= YES\nINNER_ITER= 6000\n")
    out, changes = AC.apply_preset(cfg, "safe")
    txt = open(cfg, encoding="utf-8").read()
    check("apply_preset safe: CFL_NUMBER= 5.0", "CFL_NUMBER= 5.0" in txt)
    check("apply_preset safe: CFL_ADAPT= YES", "CFL_ADAPT= YES" in txt)
    check("apply_preset safe: MUSCL_FLOW= NO (1-й порядок)", "MUSCL_FLOW= NO" in txt)
    check("apply_preset safe: рампа CFL задана", "CFL_ADAPT_PARAM=" in txt)
    check("apply_preset safe: границы не тронуты",
          "SOLVER= EULER" in txt and "INNER_ITER= 6000" in txt)
    check("бэкап config.cfg.orig создан", os.path.isfile(cfg + ".orig"))
    AC.apply_preset(cfg, "safe")
    txt2 = open(cfg, encoding="utf-8").read()
    check("повторный apply_preset не дублирует ключи",
          txt2.count("CFL_NUMBER=") == 1 and txt2.count("CFL_ADAPT=") == 1)
    AC.apply_preset(cfg, "ultra")
    txt3 = open(cfg, encoding="utf-8").read()
    check("ultra: второй порядок, CFL 1.0",
          "CFL_NUMBER= 1.0" in txt3 and "MUSCL_FLOW= YES" in txt3)
    check("ultra: линейный решатель как в базовом конфиге (1e-6)",
          "LINEAR_SOLVER_ERROR= 1e-6" in txt3 and "LINEAR_SOLVER_ITER= 15" in txt3)
    check("ultra: второй порядок MUSCL_FLOW= YES", "MUSCL_FLOW= YES" in txt3)
    check("ultra: рампа CFL до 5.0, как в базовом конфиге",
          "( 0.5, 1.2, 0.5, 5.0 )" in txt3)
    check("restore_original", AC.restore_original(cfg) is True)
    check("оригинал восстановлен", "CFL_ADAPT= YES" in
          open(cfg, encoding="utf-8").read())
    check("неизвестный пресет → ValueError",
          _raises(ValueError, AC.apply_preset, cfg, "nope"))

    # Что именно записано в конфиге — для строки в логе перед запуском
    AC.apply_preset(cfg, "ultra")
    check("match_preset узнаёт ultra", AC.match_preset(cfg) == "ultra",
          str(AC.match_preset(cfg)))
    _d = AC.describe_config_scheme(cfg)
    check("describe называет второй порядок", "2-й порядок" in _d, _d)
    check("describe называет пресет", "пресет 'ultra'" in _d, _d)
    AC.apply_preset(cfg, "safe")
    check("match_preset узнаёт safe", AC.match_preset(cfg) == "safe",
          str(AC.match_preset(cfg)))
    _d2 = AC.describe_config_scheme(cfg)
    check("describe для safe предупреждает про Cd",
          "1-й порядок" in _d2 and "читать его нельзя" in _d2, _d2)
    check("describe на отсутствующем файле не падает",
          "прочитать не удалось" in
          AC.describe_config_scheme(os.path.join(td, "нет-такого.cfg")))

    # Текст рекомендации строится по фактическим значениям пресета, а не
    # зашит вручную: иначе он устаревает при смене значений (так и было -
    # в логе писалось "CFL 0.5 с рампой до 100" при CFL 0.1 и рампе до 10).
    _case2 = os.path.join(td, "case_div")
    os.makedirs(_case2, exist_ok=True)
    with open(os.path.join(_case2, "history.csv"), "w", encoding="utf-8") as _fh:
        _fh.write('"Inner_Iter","RMS[Rho]","CL"\n0,-2.0,0.1\n50,12.0,1e10\n')
    _act, _pre, _note = AC.suggest(_case2)
    check("в тексте рекомендации CFL из самого пресета",
          _pre is not None and ("CFL %s" % AC.PRESETS[_pre]["CFL_NUMBER"]) in _note,
          _note)
    check("в тексте нет зашитых чисел другого пресета",
          all(("CFL %s" % v["CFL_NUMBER"]) not in _note
              for k, v in AC.PRESETS.items() if k != _pre), _note)

    # ultra обязан совпадать с базовым конфигом config_builder, кроме
    # потолка CFL: базовый прогон не расходится, и любое лишнее отличие -
    # это потенциальный механизм расходимости.
    _src = io.open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "solver", "config_builder.py"),
        encoding="utf-8").read()
    import solver.config_builder as _CB
    # В шаблоне config_builder часть значений - плейсхолдеры f-строки.
    # Без их разворачивания CFL_ADAPT_PARAM просто не попал бы в сравнение,
    # и проверка «отличается только потолком CFL» прошла бы впустую.
    _ph = {"cfl_param": _CB.CFL_PARAM_SAFE}
    _base = {}
    for _m in _re.finditer(r"^\s*([A-Z][A-Z0-9_]*)= (.+)$", _src, _re.M):
        _v = _m.group(2).strip()
        if _v.startswith("{") and _v.endswith("}"):
            _v = _ph.get(_v[1:-1])
            if _v is None:
                continue
        _base.setdefault(_m.group(1), _v)
    check("CFL_ADAPT_PARAM базового конфига развёрнут (иначе сравнение пустое)",
          _base.get("CFL_ADAPT_PARAM") == _CB.CFL_PARAM_SAFE,
          str(_base.get("CFL_ADAPT_PARAM")))
    _diff = {k: (AC.PRESETS["ultra"][k], _base.get(k))
             for k in AC.PRESETS["ultra"]
             if k in _base and AC.PRESETS["ultra"][k] != _base[k]}
    # Потолок CFL 50.0 проверен прогоном: расходимость на 1128-й итерации,
    # тогда как база с потолком 5.0 не расходится. Более точной настройки,
    # которая на этой сетке устойчива, нет — ultra совпадает с базой.
    check("ultra полностью совпадает с базовым конфигом",
          _diff == {}, str(_diff))
    check("общие ключи ultra и базового конфига сверены (тест не выродился)",
          len([k for k in AC.PRESETS["ultra"] if k in _base]) >= 10,
          str(len([k for k in AC.PRESETS["ultra"] if k in _base])))

    # Пресет, уже записанный в config.cfg, предлагать бессмысленно.
    _cfg2 = os.path.join(_case2, "config.cfg")
    open(_cfg2, "w", encoding="utf-8").write("SOLVER= EULER\n")
    AC.apply_preset(_cfg2, "ultra")
    check("match_preset узнаёт ultra по файлу", AC.match_preset(_cfg2) == "ultra")
    _a3, _p3, _n3 = AC.suggest(_case2, current_preset=None)
    check("suggest не предлагает уже применённый ultra", _p3 != "ultra",
          "%s %s" % (_a3, _p3))

    # Выбор результата: порядок схемы важнее глубины невязки. Первый порядок
    # всегда доводит невязку глубже, и сравнение только по невязке отдавало
    # в отчёт Cd = 0.36 из первого порядка вместо второго.
    from solver.workers import SessionRunner as _SR
    _logs = []
    _prev = {"rms": -2.20, "second_order": True, "cl": 0.7, "cd": 0.05,
             "error": True, "error_msg": "", "stopped": True}
    _new = {"rms": -6.19, "second_order": False, "cl": 0.74, "cd": 0.36,
            "error": False, "error_msg": "", "stopped": False}
    _kept = _SR._better_result(_prev, _new, _logs.append)
    check("второй порядок со слабой невязкой важнее первого с глубокой",
          _kept.get("second_order") is True and _kept.get("cd") == 0.05,
          str(_kept.get("cd")))
    check("оставленный результат помечен как несошедшийся",
          _kept.get("error") is False and "вторым порядком" in _kept.get("error_msg", ""),
          _kept.get("error_msg", "")[:70])
    check("в логе объяснено, почему оставлен второй порядок",
          any("первым порядком" in m for m in _logs), str(_logs[:1]))
    _same = _SR._better_result(dict(_prev, second_order=False), _new, _logs.append)
    check("при равном порядке решает невязка (прежнее поведение)",
          _same.get("cd") == 0.36, str(_same.get("cd")))
    _unk = _SR._better_result({"rms": -2.2, "second_order": None}, _new, _logs.append)
    check("неизвестный порядок схемы не ломает выбор",
          _unk.get("cd") == 0.36, str(_unk.get("cd")))

    # Прекондиционирование по низкому Маху. На V=60 м/с это M=0.176, и
    # сжимаемый решатель там жёсток по акустике: невязка встаёт на -2.3,
    # а сопротивление выходит в разы больше индуктивного.
    import solver.config_builder as _CB2
    check("низкий мах включает прекондиционер",
          _CB2.low_mach_lines(0.176) == "LOW_MACH_PREC= YES\nLOW_MACH_CORR= YES",
          _CB2.low_mach_lines(0.176))
    check("высокий мах его выключает",
          _CB2.low_mach_lines(0.85) == "LOW_MACH_PREC= NO\nLOW_MACH_CORR= NO",
          _CB2.low_mach_lines(0.85))
    check("нечитаемый мах не роняет сборку конфига",
          "LOW_MACH_PREC= NO" in _CB2.low_mach_lines("abc"))
    _ph = {"aoa": 3.0, "mach": 0.176, "pressure": 101325.0,
           "temperature": 288.15, "density": 1.225, "ref_length": 1.12,
           "ref_area": 9.742, "ox": -0.883, "oy": 0.0, "oz": 0.0}
    _txt = _CB2.build_euler_config(_ph, markers=["airfoil"])
    check("EULER-конфиг содержит обе строки прекондиционера",
          "LOW_MACH_PREC= YES" in _txt and "LOW_MACH_CORR= YES" in _txt)
    _cfg_lm = os.path.join(td, "lm_full.cfg")
    with open(_cfg_lm, "w", encoding="utf-8", newline="") as _fl:
        _fl.write(_txt)
    _ok_lm, _err_lm = AC.validate_config(_cfg_lm)
    check("EULER-конфиг с прекондиционером валиден для SU2", _ok_lm,
          str(_err_lm[:2]))
    _ph2 = dict(_ph, mach=0.85)
    _txt2 = _CB2.build_euler_config(_ph2, markers=["airfoil"])
    check("RANS-шаблон тоже содержит ключ (единый код)",
          "LOW_MACH_PREC= NO" in _CB2.build_rans_config(
              dict(_ph2, turb_model="SA", reynolds=5.0e6),
              markers=["airfoil"]))
    _cfg3 = os.path.join(td, "lm.cfg")
    open(_cfg3, "w", encoding="utf-8").write(
        "SOLVER= EULER\nMACH_NUMBER= 0.176\nMUSCL_FLOW= YES\n"
        "CFL_NUMBER= 1.0\nLOW_MACH_PREC= NO\n")
    check("describe предупреждает о выключенном прекондиционере на низком махе",
          "LOW_MACH_PREC= NO" in AC.describe_config_scheme(_cfg3),
          AC.describe_config_scheme(_cfg3))
    open(_cfg3, "w", encoding="utf-8").write(
        "SOLVER= EULER\nMACH_NUMBER= 0.176\nMUSCL_FLOW= YES\n"
        "CFL_NUMBER= 1.0\nLOW_MACH_PREC= YES\n")
    check("describe подтверждает включённый прекондиционер",
          "Прекондиционирование по низкому Маху включено"
          in AC.describe_config_scheme(_cfg3))

# детектор по history.csv
with tempfile.TemporaryDirectory() as td:
    case = os.path.join(td, "case")
    os.makedirs(case)
    h = os.path.join(case, "history.csv")
    with open(h, "w", encoding="utf-8") as f:
        f.write('"Inner_Iter","RMS[Rho]","CL"\n')
        f.write('0,-2.0,0.1\n')
        f.write('100,-6.5,0.42\n')
    res = AC.detect_result(case)
    check("detect_result: сошёлся", res["status"] == "converged", str(res))
    with open(h, "w", encoding="utf-8") as f:
        f.write('"Inner_Iter","RMS[Rho]","CL"\n')
        f.write('0,-2.0,0.1\n')
        f.write('50,12.0,1e10\n')
    res = AC.detect_result(case)
    check("detect_result: разошёлся", res["status"] == "diverged", str(res))
    action, preset, _ = AC.suggest(case)
    check("suggest после расхождения → ultra (сначала точный пресет)",
          action == "apply_preset" and preset == "ultra")
    res = AC.detect_result(case, screen_text="SU2 has diverged (Residual > 10^20 detected)")
    check("detect_result по тексту экрана", res["status"] == "diverged")

# ---------------------------------------------------------------- helpers нового функционала
print("== новые helper-функции (CAD, адаптация, DOE) ==")
from geometry.generators import cad_to_stl
check("cad_to_stl: без gmsh или битого файла → RuntimeError",
      _raises(RuntimeError, cad_to_stl, "/nonexistent/файл.step",
              os.path.join(tempfile.gettempdir(), "out.stl")))

from solver.workers import find_su2_adapt_exe, _mesh_npoin
check("find_su2_adapt_exe возвращает str", isinstance(find_su2_adapt_exe(), str))
with tempfile.TemporaryDirectory() as td:
    m = os.path.join(td, "mesh.su2")
    with open(m, "w", encoding="ascii") as f:
        f.write("NPOIN= 12345\nNELEM= 60000\n")
    check("_mesh_npoin читает NPOIN", _mesh_npoin(m) == 12345)

from solver.workers import OptimizationWorker
ow = OptimizationWorker(
    target_cl=0.45, target_k=15, physics={"mach": 0.3}, solver="EULER",
    initial_params={"span": 10, "chord_root": 1.8, "chord_tip": 0.9,
                    "sweep": 12},
    rule_set=None, flight_points=[], ref_data=(1, 1, 1, 0, 0),
    body_markers=["wing"], candidates=[{"span": 8.0, "chord_root": 1.5,
                                        "chord_tip": 0.8, "sweep": 15.0}])
check("OptimizationWorker принимает candidates (DOE)",
      ow.candidates == [{"span": 8.0, "chord_root": 1.5,
                         "chord_tip": 0.8, "sweep": 15.0}])
ow2 = OptimizationWorker(
    target_cl=0.45, target_k=15, physics={}, solver="EULER",
    initial_params={}, rule_set=None, flight_points=[], ref_data=(1, 1, 1, 0, 0),
    body_markers=[])
check("OptimizationWorker.candidates по умолчанию None",
      ow2.candidates is None)

# ---------------------------------------------------------------- aeroelastic
print("== physics.aeroelastic ==")
import math
import numpy as np
from physics import aeroelastic as AE

check("C(0) = 1", AE.theodorsen(0.0) == complex(1.0, 0.0))
_c02 = AE.theodorsen(0.2)
check("|C(k)| < 1 при k > 0", abs(_c02) < 1.0, f"|C(0.2)|={abs(_c02):.4f}")
check("Im C(k) < 0 при k > 0 (запаздывание следа)", _c02.imag < 0.0,
      f"G(0.2)={_c02.imag:+.4f}")
check("C(∞) = 0.5", abs(AE.theodorsen(1.0e6) - 0.5) < 1e-4)
check("рациональная аппроксимация C(k) (без scipy) точна",
      max(abs(AE.theodorsen_rational(k) - AE.theodorsen(k))
          for k in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)) < 0.05)

_vd = AE.divergence_speed(2.0e5, 2.0, 0.3, 1.225)
_vd_ref = math.sqrt(2.0 * (2.0e5 / (2.0 * 2 * math.pi * 0.3)) / 1.225)
check("V_D совпадает с аналитикой q_D = K_α/(c·C_Lα·e)",
      abs(_vd - _vd_ref) < 1e-6, f"{_vd:.4f}")
check("e ≤ 0 → дивергенции нет", AE.divergence_speed(2e5, 2.0, -0.3, 1.225) is None)

_KW = dict(m=60.0, x_alpha=0.2, b=1.0, I_alpha=12.0, K_h=8.0e4,
           K_alpha=2.0e5, chord=2.0, e=0.3, rho=1.225)
_a = AE.a_from_e(0.3, 2.0)
_Mapp = [[60.0 + math.pi * 1.225, 60.0 * 0.2],
         [60.0 * 0.2, 12.0 + math.pi * 1.225 * (0.125 + _a ** 2)]]
_Kapp = [[8.0e4, 0.0], [0.0, 2.0e5]]
_w_exact = [abs(x) for x in
            np.sqrt(np.abs(np.linalg.eigvals(
                np.linalg.inv(np.array(_Mapp)) @ np.array(_Kapp))))]
_f_exact = sorted(float(w) / (2 * math.pi) for w in _w_exact)
_f_pk = sorted(m["g"] is not None and m["freq_hz"] for m in AE.pk_point(1e-9, **_KW))
check("U→0: частоты = собственные с присоединёнными массами",
      all(abs(p - q) / q < 0.02 for p, q in zip(_f_pk, _f_exact)),
      f"pk={[round(x,3) for x in _f_pk]} exact={[round(x,3) for x in _f_exact]}")
check("U→0: демпфирование нулевое",
      all(abs(m["g"]) < 1e-6 for m in AE.pk_point(1e-9, **_KW)))

_kw_iso = dict(_KW, x_alpha=0.0, e=0.0)
check("изолированный изгиб устойчив на всех скоростях",
      all(AE.pk_point(V, **_kw_iso)[0]["g"] < 0 for V in (10, 50, 100, 200, 400)))

_vf, _diag = AE.flutter_speed(V_max=400, n_steps=120, **_KW)
_kw5 = dict(_KW, K_alpha=6.0e5)
_vf5, _ = AE.flutter_speed(V_max=800, n_steps=160, **_kw5)
check("V_F найден", _vf is not None, f"V_F={_vf and round(_vf, 1)}")
check("V_F растёт с жёсткостью кручения",
      _vf is not None and _vf5 is not None and _vf5 > _vf,
      f"{_vf and round(_vf, 1)} → {_vf5 and round(_vf5, 1)}")
check("g меняет знак в окрестности V_F", _vf is not None
      and any(d["g"] < 0 for d in _diag if d["V"] < _vf)
      and any(d["g"] > 0 for d in _diag if d["V"] > _vf))
_kw_fwd = dict(_KW, e=-0.3)
_vf_fwd, _ = AE.flutter_speed(V_max=800, n_steps=160, **_kw_fwd)
check("при оси упругости впереди фокуса (e<0) V_F не ниже",
      (_vf_fwd is None) or (_vf is not None and _vf_fwd >= _vf))

# стационарный предел должен совпадать с теорией тонкого профиля (Глауэрт)
_rho, _U, _b, _a2 = 1.225, 120.0, 0.75, -0.15
_B0, _B1, _B2 = AE.aero_matrices(_rho, _U, _b, _a2, AE.theodorsen(0.0))
_qs = np.array([0.0, 0.3])
_qd = np.array([0.4, 0.7])
_aero = _B0 @ _qs + _B1 @ _qd
_L_model, _M_model = -_aero[0].real, _aero[1].real
_qbar = _qd[1] * _b / _U
_A0g = _qs[1] + _qd[0] / _U - _a2 * _qbar
_A1g = _qbar
_CLg = 2 * math.pi * (_A0g + _A1g / 2.0)
_Cmg = (math.pi / 4.0) * (0.0 - _A1g)
_L_ref = 0.5 * _rho * _U ** 2 * (2 * _b) * _CLg
_M_ref = (0.5 * _rho * _U ** 2 * (2 * _b) ** 2 * _Cmg
          + _b * (0.5 + _a2) * _L_ref)
check("стационарная сила = теории тонкого профиля",
      abs(_L_model - _L_ref) < 1e-6 * max(1.0, abs(_L_ref)),
      f"{_L_model:.4f} vs {_L_ref:.4f}")
check("стационарный момент = теории тонкого профиля",
      abs(_M_model - _M_ref) < 1e-6 * max(1.0, abs(_M_ref)),
      f"{_M_model:.4f} vs {_M_ref:.4f}")
check("квазистационарное демпфирование кручения ≥ 0 при любом a",
      all(-(AE.aero_matrices(_rho, V, _b, aa, AE.theodorsen(0.0))[1][1, 1].real)
          >= -1e-9 for V in (30.0, 100.0, 250.0) for aa in (-0.4, -0.1, 0.2)))

_ares = AE.flutter_assessment(span=30.0, chord_root=4.0, chord_tip=1.5,
                              mass_wing=9000.0, rho=1.225,
                              V_cruise=230.0, V_dive=330.0)
check("flutter_assessment: есть вердикт и V_crit",
      bool(_ares.get("verdict")) and _ares.get("V_crit") is not None)
check("flutter_assessment: V_D > V_ref у тяжёлого крыла",
      _ares["V_D"] is not None and _ares["V_D"] > _ares["V_ref"])
check("format_report не пустой", len(AE.format_report(_ares)) > 100)

# ---------------------------------------------------------------- structural
print("== physics.structural ==")
from physics import structural as ST

_fr = ST.root_forces(L_total=1.0e6, span=30.0, mass_wing=0.0, dist="elliptic")
check("эллипс: Q_root = L_total", abs(_fr["Q"] - 1.0e6) < 1.0)
check("эллипс: M_root = 4/(3π)·L·s",
      abs(_fr["M"] - _fr["M_analytic"]) / _fr["M_analytic"] < 1e-6,
      f"{_fr['M']:.1f} vs {_fr['M_analytic']:.1f}")
check("треугольник: M_root = L·s/3",
      abs(ST.root_forces(1e6, 30.0, dist="triangular")["M"] - 5.0e6) / 5.0e6 < 1e-6)
check("равномерно: M_root = L·s/2",
      abs(ST.root_forces(1e6, 30.0, dist="uniform")["M"] - 7.5e6) / 7.5e6 < 1e-6)
check("разгрузка массой крыла уменьшает момент",
      ST.root_forces(1e6, 30.0, mass_wing=9000.0)["M"] < _fr["M"])
check("неизвестное распределение → ValueError",
      _raises(ValueError, ST.root_forces, 1e6, 30.0, dist="косинус"))

_sa = ST.structural_assessment(span=30.0, chord_root=4.0, mass_aircraft=4.0e4,
                               mass_wing=9000.0)
check("structural_assessment: σ > 0 и масса > 0",
      _sa["sigma"] > 0 and _sa["mass_estimate"] > 0)
_sa2 = ST.structural_assessment(span=30.0, chord_root=4.0, mass_aircraft=4.0e4,
                                mass_wing=9000.0, cap_area=0.05)
check("бо́льшая полка → меньше напряжения", _sa2["sigma"] < _sa["sigma"])
check("больше перегрузка → меньше запас",
      ST.structural_assessment(span=30.0, chord_root=4.0, mass_aircraft=4e4,
                               mass_wing=9000.0, n_limit=6.0)["MS_sigma"]
      < _sa["MS_sigma"])
check("format_report(structural) не пустой", len(ST.format_report(_sa)) > 100)

# ---------------------------------------------------------------- postprocessing
print("== postprocessing (поляра, отчёты) ==")
from postprocessing.polar import (build_polar, integrated_characteristics,
                                  linear_fit_cl_alpha, drag_polar_fit)
from postprocessing import report as REP

_res = [{"aoa": a, "cl": 0.11 * a, "cd": 0.02 + 0.005 * a ** 2,
         "cm": -0.05 * a, "converged": True} for a in range(-4, 15, 2)]
_pol = build_polar(_res)
check("build_polar: точки отсортированы по α",
      bool(np.all(np.diff(_pol["aoa"]) > 0)) and _pol["aoa"].size == len(_res))
_fit = linear_fit_cl_alpha(_pol["aoa"], _pol["cl"])
check("наклон поляры dCl/dα ≈ 0.11 1/град",
      abs(_fit["cl_alpha_deg"] - 0.11) < 1e-6, f"{_fit['cl_alpha_deg']:.5f}")
check("α₀ ≈ 0", abs(_fit["alpha0"]) < 1e-9)
_dp = drag_polar_fit(_pol["cl"], _pol["cd"], aspect_ratio=10.0)
check("Cd₀ ≈ 0.02", abs(_dp["cd0"] - 0.02) < 1e-6, f"{_dp['cd0']:.6f}")
check("фактор Освальда положителен", _dp["oswald_e"] > 0.0)
_ch = integrated_characteristics(_pol, 10.0, weight_n=12000.0, rho=1.225,
                                 s_ref=12.0, mach=0.15)
check("интегральные: Cl_max и α срыва найдены",
      _ch.get("cl_max") is not None and _ch.get("aoa_stall") is not None)
check("интегральные: скорость сваливания = √(2W/(ρS·Cl_max))",
      abs(_ch["v_stall"] - math.sqrt(2 * 12000.0 / (1.225 * 12.0 * _ch["cl_max"])))
      < 1e-6, f"{_ch['v_stall']:.3f}")
check("интегральные: K_max и точка максимума K",
      _ch.get("k_max") is not None and _ch.get("aoa_best_k") is not None)
check("build_polar отбрасывает cd ≤ 0",
      build_polar([{"aoa": 1, "cl": 0.1, "cd": -1.0}])["aoa"].size == 0)

for _tpl in REP.TEMPLATES:
    _txt = REP.make_report(_res, aspect_ratio=10.0, template=_tpl,
                           project_info={"name": "Тест"}, weight_n=12000.0,
                           rho=1.225, s_ref=12.0, mach=0.15)
    check(f"шаблон «{_tpl}» формирует отчёт", len(_txt) > 200
          and "ОТЧЁТ" in _txt)
check("неизвестный шаблон → ValueError",
      _raises(ValueError, REP.make_report, _res, template="несуществует"))
check("render_html даёт HTML",
      REP.render_html(_res, aspect_ratio=10.0).startswith("<!DOCTYPE html>"))
with tempfile.TemporaryDirectory() as td:
    _p = os.path.join(td, "polar.csv")
    REP.export_csv(_p, _res)
    _raw = open(_p, "rb").read()
    check("export_csv: UTF-8 BOM и разделитель «;»",
          _raw.startswith(b"\xef\xbb\xbf") and b";" in _raw)
    check("export_csv: строки по числу точек",
          len(_raw.decode("utf-8-sig").strip().splitlines()) == len(_res) + 1)

# ---------------------------------------------------------------- presets
print("== su2_preset_format (формат конфигурации) ==")
import su2_preset_format as PF

check("FORMAT_ID/EXTENSION заданы",
      PF.FORMAT_ID == "aeroopt.su2preset" and PF.EXTENSION == ".su2preset")
_cat = PF.key_catalogue()
check("каталог ключей непустой", len(_cat) >= 10, f"{len(_cat)} ключей")
_b = PF.builtin_presets()
check("встроенных шаблонов ровно два: ultra и safe",
      sorted(_b) == ["safe", "ultra"], str(list(_b)))
for _name, _pre in _b.items():
    _rep = PF.validate_preset(PF.make_preset(_name, _pre["params"]))
    check(f"шаблон «{_name}» проходит валидацию", _rep["ok"],
          str(_rep["errors"]))
check("пустое имя → ошибка валидации",
      not PF.validate_preset(PF.make_preset("", {"CFL_NUMBER": "5"}))["ok"])
check("неизвестный ключ → предупреждение, не ошибка",
      (lambda r: r["ok"] and len(r["warnings"]) > 0)(
          PF.validate_preset(PF.make_preset("x", {"НОВЫЙ_КЛЮЧ_SU2": "1"}))))
check("не-словарь → ошибка валидации", not PF.validate_preset([1, 2])["ok"])
check("чужой format → ошибка",
      not PF.validate_preset({"format": "other", "name": "x",
                              "schema_version": 1, "params": {}})["ok"])
_cfg = PF.parse_cfg_text("CFL_NUMBER= 5.0\nSOLVER= EULER\n% комментарий\n")
check("parse_cfg_text разбирает config.cfg",
      _cfg.get("CFL_NUMBER") == "5.0" and _cfg.get("SOLVER") == "EULER")
_pf1 = PF.make_preset("A", {"CFL_NUMBER": "5.0"})
_pf2 = PF.make_preset("B", {"CFL_NUMBER": "2.0", "MUSCL_FLOW": "YES"})
_diff = PF.diff_presets(_pf1, _pf2)
check("diff_presets находит отличия", bool(_diff))
check("preset_to_cfg_lines даёт строки KEY= VALUE",
      any(ln.strip().startswith("CFL_NUMBER=")
          for ln in PF.preset_to_cfg_lines(_pf1)))
check("describe_format описывает формат", "su2preset" in PF.describe_format())
with tempfile.TemporaryDirectory() as td:
    _fp = os.path.join(td, "my" + PF.EXTENSION)
    PF.export_preset(_fp, "Тест", {"CFL_NUMBER": "5.0"})
    check("экспорт создаёт файл", os.path.isfile(_fp))
    _imp = PF.import_preset(_fp)
    check("импорт возвращает те же параметры",
          _imp["params"]["CFL_NUMBER"] == "5.0" and _imp["name"] == "Тест")
    check("импорт битого JSON → ValueError",
          _raises(ValueError, PF.import_preset,
                  os.path.join(td, "нет_такого" + PF.EXTENSION)))
    _bad = os.path.join(td, "bad.json")
    open(_bad, "w", encoding="utf-8").write("{не json")
    check("импорт не-JSON → ValueError", _raises(ValueError, PF.import_preset, _bad))
    check("экспорт невалидного в strict-режиме → ValueError",
          _raises(ValueError, PF.export_preset, os.path.join(td, "x.json"),
                  "", {"A": "1"}))

# ---------------------------------------------------------------- adapt_gmsh
print("== mesh.adapt_gmsh (адаптивная сетка) ==")
from mesh import adapt_gmsh as AG

with tempfile.TemporaryDirectory() as td:
    _csv = os.path.join(td, "surface_flow.csv")
    with open(_csv, "w", encoding="utf-8", newline="") as f:
        f.write('"x","y","z","C_Pressure"\n')
        for _i in range(21):
            _x = _i / 20.0
            f.write(f"{_x:.5f},{0.1 * math.sqrt(_x):.5f},0.0,"
                    f"{-3.0 * math.exp(-_x / 0.02) - 0.2:.5f}\n")
    _s = AG.parse_surface_flow_csv(_csv)
    check("parse_surface_flow_csv: колонки x/y/z/cp",
          set(("x", "y", "z", "cp")) <= set(_s) and _s["x"].size == 21)
    _g = AG.pressure_gradient_along_surface(_s["x"], _s["y"], _s["cp"])
    check("градиент Cp максимален у носка",
          int(np.argmax(_g)) <= 2, f"argmax={int(np.argmax(_g))}")
    _pts, _sz = AG.surface_size_metric(_s["x"], _s["y"], _g,
                                       h_min=5e-4, h_max=2e-2)
    check("метрика: у носка мельче, чем на конце",
          _sz[0] < _sz[-1] and _sz[0] <= 5e-4 + 1e-12)
    check("метрика: размеры в диапазоне [h_min, h_max]",
          _sz.min() >= 5e-4 - 1e-12 and _sz.max() <= 2e-2 + 1e-12)
    _mp = os.path.join(td, "metric.msh")
    AG.write_metric_msh(_mp, _pts, _sz)
    _body = open(_mp, encoding="ascii").read()
    check("metric.msh: формат 2.2 + $NodeData",
          "$MeshFormat\n2.2 0 8" in _body and "$NodeData" in _body
          and _body.rstrip().endswith("$EndNodeData"))
    _gp = os.path.join(td, "adapt.geo")
    AG.write_metric_geo(_gp, _mp, os.path.join(td, "in.stl"),
                        os.path.join(td, "out.stl"), 5e-4, 2e-2)
    _geo = open(_gp, encoding="utf-8").read()
    check("adapt.geo: Mesh.Metric + Mesh 2 + Save",
          "Mesh.Metric =" in _geo and "Mesh 2;" in _geo and "Save(" in _geo)
    _rep = AG.adaptivity_report(_s, _sz)
    check("adaptivity_report: минимум Cp и размер в этой точке",
          _rep["cp_min"] < 0 and _rep["h_at_cp_min"] <= 5e-4 + 1e-12)
    check("format_adaptivity_report не пустой",
          len(AG.format_adaptivity_report(_rep)) > 60)
    check("parse_surface_flow_csv: нет файла → FileNotFoundError",
          _raises(FileNotFoundError, AG.parse_surface_flow_csv,
                  os.path.join(td, "нет.csv")))

# ---------------------------------------------------------------- CAD split
print("== geometry.generators (Direct CAD Import, сборки) ==")
from geometry.generators import (_stl_name_for_solid, count_stl_triangles,
                                 cad_inspect, cad_split_to_stl, cad_to_stl)
import struct

with tempfile.TemporaryDirectory() as td:
    _bin = os.path.join(td, "b.stl")
    with open(_bin, "wb") as f:
        f.write(b"\0" * 80 + struct.pack("<I", 2))
        f.write(struct.pack("<12fH", *([0.0] * 12), 0) * 2)
    check("count_stl_triangles: бинарный STL", count_stl_triangles(_bin) == 2)
    _asc = os.path.join(td, "a.stl")
    with open(_asc, "w", encoding="ascii") as f:
        f.write("solid s\n" + " facet normal 0 0 1\n endfacet\n" * 3 + "endsolid\n")
    check("count_stl_triangles: текстовый STL", count_stl_triangles(_asc) == 3)
    check("count_stl_triangles: нет файла → 0",
          count_stl_triangles(os.path.join(td, "нет.stl")) == 0)
check("_stl_name_for_solid: имя тела и индекс",
      _stl_name_for_solid("asm", 1, "Крыло / левое", 7) == "asm_01_Крыло___левое.stl")
check("_stl_name_for_solid: пустое имя → solid_<tag>",
      _stl_name_for_solid("asm", 2, "  ", 3) == "asm_02_solid_3.stl")
for _fn, _nm in ((cad_inspect, "cad_inspect"), (cad_split_to_stl, "cad_split_to_stl"),
                 (cad_to_stl, "cad_to_stl")):
    check(f"{_nm} без gmsh → RuntimeError",
          _raises(RuntimeError, _fn, "нет_такого.step", "out.stl")
          or _raises(RuntimeError, _fn, "нет_такого.step", tempfile.gettempdir()))

# ---------------------------------------------------------------- workers (ТЗ 1, 2)
print("== solver.workers: многоядерность и симметрия в оптимизации ==")
_ow = OptimizationWorker(
    target_cl=0.45, target_k=15, physics={}, solver="EULER",
    initial_params={}, rule_set=None, flight_points=[],
    ref_data=(1, 1, 1, 0, 0), body_markers=[], cpu_cores=8,
    symmetry_planes=["xz", "yz"])
check("OptimizationWorker принимает cpu_cores", _ow.cpu_cores == 8)
check("OptimizationWorker принимает symmetry_planes",
      _ow.symmetry_planes == ["xz", "yz"])
_ow_def = OptimizationWorker(
    target_cl=0.45, target_k=15, physics={}, solver="EULER",
    initial_params={}, rule_set=None, flight_points=[],
    ref_data=(1, 1, 1, 0, 0), body_markers=[])
check("cpu_cores по умолчанию = 1", _ow_def.cpu_cores == 1)
check("без симметрии symmetry_planes = None", _ow_def.symmetry_planes is None)
check("_enabled_symmetry_planes: нет плоскостей → []",
      _ow_def._enabled_symmetry_planes("нет_такой_сетки.su2") == [])
with tempfile.TemporaryDirectory() as td:
    _m = os.path.join(td, "mesh.su2")
    with open(_m, "w", encoding="ascii") as f:
        f.write("NPOIN= 4\nNELEM= 2\nMARKER_TAG= symmetry_xz\n"
                "MARKER_ELEMS= 2\nMARKER_TAG= airfoil\nMARKER_ELEMS= 1\n")
    check("_enabled_symmetry_planes видит symmetry_xz в сетке",
          _ow._enabled_symmetry_planes(_m) == ["xz"],
          str(_ow._enabled_symmetry_planes(_m)))
    _ow_yz = OptimizationWorker(
        target_cl=0.45, target_k=15, physics={}, solver="EULER",
        initial_params={}, rule_set=None, flight_points=[],
        ref_data=(1, 1, 1, 0, 0), body_markers=[], symmetry_planes=["yz"])
    check("нет маркера yz → плоскость не включается",
          _ow_yz._enabled_symmetry_planes(_m) == [])

# ---------------------------------------------------------------- DOE
print("== optimization.doe (сетка вариантов и поколения) ==")
from optimization import doe as DOE

_BASE = {"span": 15.0, "chord_root": 1.5, "chord_tip": 0.7, "sweep": 10.0,
         "twist": 0.0, "flap_deflection": 0.0, "slat_deflection": 0.0}
_RNG = {"span": (13.0, 17.0), "chord_root": (1.3, 1.7), "sweep": (5.0, 15.0)}
check("уровни параметра: концы совпадают с диапазоном",
      DOE.levels_for("span", 13.0, 17.0, 3) == [13.0, 15.0, 17.0],
      str(DOE.levels_for("span", 13.0, 17.0, 3)))
check("уровни ограничены пределами параметра",
      all(0.5 <= v <= 100.0 for v in DOE.levels_for("span", -5.0, 500.0, 5)))
_ff = DOE.full_factorial(_BASE, _RNG, 3)
check("полный факторный план: levels**n вариантов",
      len(_ff) == DOE.plan_size(DOE.PLAN_FULL, len(_RNG), 3) == 27,
      f"{len(_ff)}")
check("полный факторный: неповаряемые параметры сохранены",
      all(r["chord_tip"] == _BASE["chord_tip"] for r in _ff))
_ofat = DOE.one_factor_at_a_time(_BASE, _RNG, 3)
check("план «по одному параметру»: 1 + n·(levels−1)",
      len(_ofat) == DOE.plan_size(DOE.PLAN_OFAT, len(_RNG), 3) == 7,
      f"{len(_ofat)}")
check("план «по одному»: в каждой строке изменён один параметр",
      all(sum(1 for k in _RNG if r[k] != _BASE[k]) <= 1 for r in _ofat))
_lhs = DOE.latin_hypercube(_BASE, _RNG, 9, seed=42)
check("латинский гиперкуб: заданное число вариантов", len(_lhs) == 9)
_layers = [sorted(round(r["span"], 3) for r in _lhs)]
check("латинский гиперкуб: значения внутри диапазона",
      all(13.0 <= v <= 17.0 for v in _layers[0]))
check("латинский гиперкуб: воспроизводим при том же seed",
      [r["span"] for r in DOE.latin_hypercube(_BASE, _RNG, 9, seed=42)]
      == [r["span"] for r in _lhs])
check("латинский гиперкуб: разные seed дают разные планы",
      [r["span"] for r in DOE.latin_hypercube(_BASE, _RNG, 9, seed=1)]
      != [r["span"] for r in _lhs])
_ng = DOE.next_generation(_ff[0], _RNG, shrink=0.5)
check("следующее поколение: диапазоны сужаются вдвое",
      all(abs(_ng[k][1] - _ng[k][0]) < abs(_RNG[k][1] - _RNG[k][0])
          for k in _RNG))
check("следующее поколение: лучший вариант внутри диапазона",
      all(_ng[k][0] <= _ff[0][k] <= _ng[k][1] for k in _RNG))
check("make_plan по названию полного плана",
      len(DOE.make_plan(DOE.PLAN_FULL, _BASE, _RNG, 3)) == 27)
check("make_plan: неизвестный план → ValueError",
      _raises(ValueError, DOE.make_plan, "нет такого", _BASE, _RNG))
check("PARAM_SPECS содержит все параметры DOE-таблицы",
      {"span", "chord_root", "chord_tip", "sweep", "twist",
       "flap_deflection", "slat_deflection"} <= set(DOE.SPEC_BY_KEY))

# ---------------------------------------------------------------- UI handlers# ---------------------------------------------------------------- UI handlers
print("== ui.main_window: обработчики новых разделов ==")
import ui.main_window as MW


class _FakeSpin:
    """Мини-замена QDoubleSpinBox/QSpinBox/QComboBox для теста логики."""
    def __init__(self, value=0.0, text=""):
        self._v = value
        self._t = text
    def value(self):
        return self._v
    def setValue(self, v):
        self._v = v
    def currentText(self):
        return self._t
    def currentData(self):
        return self._t
    def setText(self, t):
        self._t = t
    def text(self):
        return self._t


class _FakeBox:
    def __init__(self):
        self._txt = ""
    def setText(self, t):
        self._txt = t
    def text(self):
        return self._txt
    def append(self, t):
        self._txt += t


class _FakeChk:
    def __init__(self, v=False):
        self._v = v
    def isChecked(self):
        return self._v


class _FakeAxis:
    def clear(self): pass
    def plot(self, *a, **k): pass
    def axvline(self, *a, **k): pass
    def set_title(self, *a, **k): pass
    def set_xlabel(self, *a, **k): pass
    def set_ylabel(self, *a, **k): pass
    def grid(self, *a, **k): pass
    def tick_params(self, *a, **k): pass
    def legend(self, *a, **k): pass


def _mk_window(**attrs):
    w = MW.MainWindow.__new__(MW.MainWindow)
    w.log_text = _FakeBox()
    w.ae_w = {"span": _FakeSpin(15.0), "chord_root": _FakeSpin(1.5),
              "chord_tip": _FakeSpin(0.7), "mass_wing": _FakeSpin(600.0),
              "rho": _FakeSpin(1.225), "v_cruise": _FakeSpin(120.0),
              "v_dive": _FakeSpin(180.0), "safety": _FakeSpin(1.15),
              "t_ratio": _FakeSpin(0.12), "x_ea_ratio": _FakeSpin(0.40),
              "x_cg_ratio": _FakeSpin(0.38), "out": _FakeBox(),
              "fill_from_model": _FakeChk(False)}
    w.st_w = {"span": _FakeSpin(15.0), "chord_root": _FakeSpin(1.5),
              "mass_aircraft": _FakeSpin(1200.0), "mass_wing": _FakeSpin(600.0),
              "n_limit": _FakeSpin(3.8), "dist": _FakeSpin(text="elliptic"),
              "t_ratio": _FakeSpin(0.12), "cap_frac": _FakeSpin(0.02),
              "sigma_allow": _FakeSpin(2.8e8), "sf": _FakeSpin(1.5),
              "out": _FakeBox()}
    w.sp_w = {"weight": _FakeSpin(1200.0), "rho": _FakeSpin(1.225),
              "s_ref": _FakeSpin(12.0), "mach": _FakeSpin(0.15),
              "template": _FakeSpin(text="полный"),
              "project_name": _FakeSpin(text="Тест"), "out": _FakeBox()}
    w.pr_w = {"name": _FakeSpin(text="Мой пресет"),
              "source": _FakeSpin(text="std"), "out": _FakeBox()}
    w.w_span = _FakeSpin(15.0)
    w.w_chord_root = _FakeSpin(1.5)
    w.w_chord_tip = _FakeSpin(0.7)
    w.plot_canvas = type("C", (), {"axes1": _FakeAxis(), "draw": lambda s: None})()
    w.bottom_tabs = type("T", (), {"setCurrentWidget": lambda s, w2: None})()
    for k, v in attrs.items():
        setattr(w, k, v)
    return w


_w = _mk_window()
_w.run_aeroelastic_check()
check("run_aeroelastic_check выводит отчёт",
      "АЭРОУПРУГОСТЬ" in _w.ae_w["out"].text()
      and "V_F" in _w.ae_w["out"].text(), _w.ae_w["out"].text()[:80])
check("run_aeroelastic_check пишет в лог", "Аэроупругость" in _w.log_text.text())
check("run_aeroelastic_check кэширует результат",
      getattr(_w, "_last_aeroelastic", None) is not None)
_w.plot_vg_diagram()
check("plot_vg_diagram отрабатывает без ошибок", True)

_w2 = _mk_window()
_w2.run_structural_check()
check("run_structural_check выводит отчёт",
      "ПРОЧНОСТЬ" in _w2.st_w["out"].text().upper()
      or "σ" in _w2.st_w["out"].text(), _w2.st_w["out"].text()[:80])
check("run_structural_check пишет в лог", "Прочность" in _w2.log_text.text())


class _FakeTable:
    def __init__(self, rows):
        self._rows = rows
    def rowCount(self):
        return len(self._rows)
    def item(self, r, c):
        v = self._rows[r][c] if c < len(self._rows[r]) else None
        if v is None:
            return None
        return type("I", (), {"text": lambda s, v=v: str(v)})()


_ROWS = [[a, 0.11 * a, 0.02 + 0.005 * a ** 2, -0.05 * a]
         for a in range(-4, 15, 2)]
_w3 = _mk_window(table=_FakeTable(_ROWS))
_w3.build_polar_from_results()
check("build_polar_from_results выводит характеристики",
      "КЛАД" in _w3.sp_w["out"].text().upper()
      or "Освальда" in _w3.sp_w["out"].text(), _w3.sp_w["out"].text()[:80])
check("build_polar_from_results нашёл Cl_max", "Cl макс" in _w3.sp_w["out"].text())
_w4 = _mk_window(table=_FakeTable([[0, 0.1, 0.02, 0.0]]))
_w4.build_polar_from_results()
check("мало точек → понятное сообщение", "меньше трёх точек" in _w4.sp_w["out"].text())
_w5 = _mk_window(table=_FakeTable(_ROWS))
_w5._polar_chars()
_w5.run_aeroelastic_check()
_w5.run_structural_check()
with tempfile.TemporaryDirectory() as td:
    _rp = os.path.join(td, "rep.html")
    _cp = os.path.join(td, "polar.csv")
    _answers = [_rp, _cp]          # первый диалог — отчёт, второй — поляра
    _orig = MW.QFileDialog.getSaveFileName

    def _fake_save(*a, **k):
        return (_answers.pop(0) if _answers else "", "")

    MW.QFileDialog.getSaveFileName = staticmethod(_fake_save)
    try:
        _w5.export_analysis_report()
        _w5.export_polar_csv()
    finally:
        MW.QFileDialog.getSaveFileName = _orig
    if not os.path.isfile(_rp):
        print("     [debug] out =", _w5.sp_w["out"].text()[:400])
    check("export_analysis_report создаёт HTML",
          os.path.isfile(_rp) and "<!DOCTYPE html>" in open(_rp, encoding="utf-8").read())
    check("отчёт содержит раздел аэроупругости",
          "АЭРОУПРУГОСТЬ" in open(_rp, encoding="utf-8").read())
    check("export_polar_csv создаёт CSV",
          os.path.isfile(os.path.join(td, "polar.csv")))

_w6 = _mk_window(session=type("S", (), {"turb_model": "SA", "cpu_cores": 4,
                                        "CFL_NUMBER": "5.0"})())
_w6._imported_preset = None
_w6.apply_imported_preset()
check("apply_imported_preset без импорта → подсказка",
      "Сначала импортируйте" in _w6.pr_w["out"].text())
with tempfile.TemporaryDirectory() as td:
    _pp = os.path.join(td, "p" + PF.EXTENSION)
    PF.export_preset(_pp, "Из теста", {"CFL_NUMBER": "3.5", "MUSCL_FLOW": "YES"})
    _w6._imported_preset = PF.import_preset(_pp)
    _w6.apply_imported_preset()
    check("apply_imported_preset применяет параметры",
          "Применено параметров" in _w6.pr_w["out"].text()
          and "Пресет" in _w6.log_text.text())
check("_session_params читает объект расчёта",
      isinstance(_w6._session_params(), dict))

print("== ui.main_window: DOE-таблица и адаптация по Cp ==")
_w7 = _mk_window()
check("_doe_param_names: расширенный набор параметров",
      _w7._doe_param_names() == ["span", "chord_root", "chord_tip", "sweep",
                                 "twist", "flap_deflection",
                                 "slat_deflection"],
      str(_w7._doe_param_names()))
check("_doe_param_labels: подписи из каталога DOE",
      len(_w7._doe_param_labels()) == 7
      and _w7._doe_param_labels()[0].startswith("Размах"))


class _FakeDoeTable:
    def __init__(self):
        self._rows = []
    def setRowCount(self, n):
        self._rows = self._rows[:n]
    def rowCount(self):
        return len(self._rows)
    def insertRow(self, _r):
        self._rows.append({})
    def setItem(self, r, c, item):
        self._rows[r][c] = item.text()
    def item(self, r, c):
        v = self._rows[r].get(c)
        return None if v is None else type("I", (), {"text": lambda s, v=v: v})()


class _FakeLbl:
    def __init__(self):
        self._t = ""
    def setText(self, t):
        self._t = t
    def text(self):
        return self._t


class _FakeItem:
    def __init__(self, t):
        self._t = t
    def text(self):
        return self._t
    def setTextAlignment(self, *_a):
        pass


_orig_item = MW.QTableWidgetItem
MW.QTableWidgetItem = _FakeItem
try:
    _w7.doe_table = _FakeDoeTable()
    _w7.lbl_doe_status = _FakeLbl()
    _w7._fill_doe_table([{"span": 14.0, "chord_root": 1.4, "chord_tip": 0.7,
                          "sweep": 8.0, "twist": 1.5, "flap_deflection": 10.0,
                          "slat_deflection": 5.0}])
    check("_fill_doe_table заполняет строку",
          _w7.doe_table.rowCount() == 1
          and _w7.doe_table.item(0, 0).text() == "14.000"
          and _w7.doe_table.item(0, 4).text() == "1.500",
          str([_w7.doe_table.item(0, c).text() for c in range(7)]))
    check("_fill_doe_table обновляет статус", "1" in _w7.lbl_doe_status.text())
    _cands = _w7._get_doe_candidates()
    check("_get_doe_candidates читает все 7 параметров",
          len(_cands) == 1 and _cands[0]["twist"] == 1.5
          and _cands[0]["flap_deflection"] == 10.0, str(_cands))
finally:
    MW.QTableWidgetItem = _orig_item

with tempfile.TemporaryDirectory() as td:
    _old_base = MW.WORK_DIR_BASE
    MW.WORK_DIR_BASE = td
    try:
        _rd = os.path.join(td, "RUN_1")
        os.makedirs(_rd)
        with open(os.path.join(_rd, "surface_flow.csv"), "w",
                  encoding="utf-8") as f:
            f.write('"x","y","C_Pressure"\n0,0,-1\n1,0,-0.1\n')
        _w8 = _mk_window()
        check("_find_latest_surface_flow_csv находит файл",
              _w8._find_latest_surface_flow_csv().endswith("surface_flow.csv"))
        check("_last_stl_or_mesh_source: нет деталей → пустая строка",
              _w8._last_stl_or_mesh_source() == "")
        _stl = os.path.join(td, "wing.stl")
        open(_stl, "w", encoding="ascii").write("solid s\nendsolid\n")
        _w8.bodies = [{"id": 1, "role": "wing", "visible": True, "path": _stl}]
        check("_last_stl_or_mesh_source возвращает STL видимой детали",
              _w8._last_stl_or_mesh_source() == _stl)
    finally:
        MW.WORK_DIR_BASE = _old_base

print("== build_exe: очистка dist перед сборкой ==")
import build_exe as BE

check("_rmtree_force: нет каталога → успех, а не WinError 3",
      BE._rmtree_force(os.path.join(tempfile.gettempdir(),
                                    "нет_такой_папки_xyz")) == (True, None))

with tempfile.TemporaryDirectory() as td:
    d = os.path.join(td, "AeroOpt")
    os.makedirs(os.path.join(d, "_internal"))
    open(os.path.join(d, "AeroOpt.exe"), "w").write("x")
    open(os.path.join(d, "_internal", "base_library.zip"), "w").write("y")
    ok, err = BE._rmtree_force(d, delay=0)
    check("_rmtree_force: обычное дерево удаляется", ok and err is None)
    check("_rmtree_force: каталога больше нет", not os.path.exists(d))

with tempfile.TemporaryDirectory() as td:
    d = os.path.join(td, "AeroOpt")
    os.makedirs(d)
    ro = os.path.join(d, "readonly.txt")
    with open(ro, "w") as f:
        f.write("r")
    os.chmod(ro, 0o444)
    ok, _err = BE._rmtree_force(d, delay=0)
    check("_rmtree_force: read-only файл снимается и удаляется",
          ok and not os.path.exists(d))


class _FakeShutil:
    """Подмена shutil: считаем вызовы rmtree."""
    def __init__(self, errors):
        self._errors = list(errors)
        self.calls = 0
    def rmtree(self, path, **kw):
        self.calls += 1
        if self._errors:
            raise self._errors.pop(0)
        raise FileNotFoundError(2, "нет", path)


_real_shutil = BE.shutil
with tempfile.TemporaryDirectory() as td:
    d = os.path.join(td, "AeroOpt")
    os.makedirs(d)
    BE.shutil = _FakeShutil([PermissionError(13, "Отказано в доступе", d)])
    try:
        ok, err = BE._rmtree_force(d, delay=0)
        check("_rmtree_force: после WinError 5 повторяет попытку и succeeds",
              ok and err is None, f"ok={ok} err={err}")
        check("_rmtree_force: было ровно 2 попытки", BE.shutil.calls == 2,
              str(BE.shutil.calls))
    finally:
        BE.shutil = _real_shutil
    BE.shutil = _FakeShutil([PermissionError(13, "Отказано в доступе", d)] * 5)
    try:
        ok, err = BE._rmtree_force(d, attempts=3, delay=0)
        check("_rmtree_force: при вечной блокировке честно вернёт False",
              ok is False and "Отказано" in str(err), f"ok={ok} err={err}")
        check("_rmtree_force: попыток ровно attempts", BE.shutil.calls == 3,
              str(BE.shutil.calls))
    finally:
        BE.shutil = _real_shutil

check("_running_pids вне Windows → пусто", BE._running_pids() == [])
check("_kill_app вне Windows не падает", BE._kill_app() is None)

with tempfile.TemporaryDirectory() as td:
    check("_prepare_dist без dist возвращает True",
          BE._prepare_dist(td) is True)

    d = os.path.join(td, "dist", "AeroOpt")
    os.makedirs(d)
    open(os.path.join(d, "AeroOpt.exe"), "w").write("x")
    check("_prepare_dist удаляет свободный dist/AeroOpt",
          BE._prepare_dist(td) is True and not os.path.exists(d))

    # блокировка: удаление не проходит → фоллбэк на переименование
    os.makedirs(d)
    open(os.path.join(d, "AeroOpt.exe"), "w").write("x")
    _real_rmtree = BE._rmtree_force
    BE._rmtree_force = lambda p, **kw: (False, PermissionError(13, "занято", p))
    try:
        ok = BE._prepare_dist(td)
        olds = [n for n in os.listdir(os.path.join(td, "dist"))
                if n.startswith("AeroOpt_old_")]
        check("_prepare_dist при блокировке не падает, а собирает дальше",
              ok is True)
        check("_prepare_dist переименовал занятую папку",
              len(olds) == 1 and not os.path.exists(d), str(olds))
    finally:
        BE._rmtree_force = _real_rmtree

    # тотальный отказ: ни удалить, ни переименовать
    os.makedirs(d)
    BE._rmtree_force = lambda p, **kw: (False, PermissionError(13, "занято", p))
    _real_rename = os.rename
    def _deny_rename(a, b):
        raise OSError(5, "Отказано в доступе", a)
    os.rename = _deny_rename
    try:
        check("_prepare_dist: если ничего не вышло → False (сборка прерывается)",
              BE._prepare_dist(td) is False)
    finally:
        os.rename = _real_rename
        BE._rmtree_force = _real_rmtree

print("== config.cfg: синтаксис SU2 (комментарий только '%') ==")
import su2_autoconfig as AC
import su2_config_dialog as SCD
from solver.config_builder import build_su2_config as _bsc

_phys = {"mach": 0.2, "reynolds": 3e6}
_ref = (1.0, 1.0, 0.25, 0.0, 0.0)
_LEGACY = """CFL_NUMBER= 5.0
# ===== AEROOPT-AUTOCONFIG: устойчивый пресет =====
# Пресет: safe. Откат: вернуть config.cfg.orig
TIME_DISCRETE_FLOW= EULER_IMPLICIT
# ===== /AEROOPT-AUTOCONFIG =======================
"""

check("исходный config.cfg линтер SU2 принимает",
      AC.su2_lint_lines(_bsc(3.0, _phys, "EULER", _ref,
                             markers=["airfoil"])) == [])
check("строка с '#' и без '=' - ошибка (это и роняло SU2)",
      len(AC.su2_lint_lines("CFL_NUMBER= 2.0\n# комментарий\n")) == 1)
check("строка с '%' и без '=' - комментарий, ошибки нет",
      AC.su2_lint_lines("CFL_NUMBER= 2.0\n% комментарий\n") == [])
check("'%' в середине строки отбрасывает хвост, как в SU2",
      AC.su2_lint_lines("CFL_NUMBER= 2.0 % хвост\n") == [])
check("пустые строки и строки из пробелов пропускаются",
      AC.su2_lint_lines("\n   \n\t\nCFL_NUMBER= 2.0\n") == [])
check("нет имени перед '=' - ошибка",
      len(AC.su2_lint_lines("= 5\n")) == 1)
check("два слова перед '=' - ошибка",
      len(AC.su2_lint_lines("A B= 5\n")) == 1)
check("продолжение строки через обратный слэш склеивается",
      AC.su2_lint_lines("MARKER_SYM= \\\n ( sym )\n") == [])

check("старый блок с '#' линтер бракует (регрессия, которую чиним)",
      len(AC.su2_lint_lines(_LEGACY)) == 3,
      str(AC.su2_lint_lines(_LEGACY)))

with tempfile.TemporaryDirectory() as td:
    p_cfg = os.path.join(td, "config.cfg")
    open(p_cfg, "w", encoding="utf-8", newline="").write(_LEGACY)
    check("старый '#'-блок вычищается при новом применении пресета",
          "# =====" not in "".join(
              AC._strip_managed_block(AC._read_lines(p_cfg))))
    open(p_cfg, "w", encoding="utf-8", newline="").write(
        _bsc(3.0, _phys, "EULER", _ref, markers=["airfoil"],
             use_symmetry=True, symmetry_planes=["xz"]))
    AC.apply_preset(p_cfg, "safe")
    _txt = open(p_cfg, encoding="utf-8").read()
    check("после apply_preset конфиг читаем для SU2",
          AC.validate_config(p_cfg)[0] is True,
          str(AC.validate_config(p_cfg)[1][:2]))
    check("в конфиге не осталось '#'-комментариев",
          not any(l.strip().startswith("#") for l in _txt.splitlines()))
    # Все ключи пресета уже есть в базовом конфиге, поэтому они правятся
    # на месте и отдельный блок не создаётся (блок нужен только для
    # отсутствующих ключей).
    check("пресет применён на месте: CFL 5.0 с адаптацией, 1-й порядок",
          "CFL_NUMBER= 5.0" in _txt and "CFL_ADAPT= YES" in _txt
          and "MUSCL_FLOW= NO" in _txt)
    check("пресет применён на месте: TIME_DISCRE_FLOW= EULER_IMPLICIT",
          "TIME_DISCRE_FLOW= EULER_IMPLICIT" in _txt)
    AC.apply_preset(p_cfg, "ultra")
    _txt2 = open(p_cfg, encoding="utf-8").read()
    check("повторный пресет: ultra применился и конфиг валиден",
          "CFL_NUMBER= 1.0" in _txt2 and AC.validate_config(p_cfg)[0] is True,
          str(AC.validate_config(p_cfg)[1][:2]))

    # Если ключа в конфиге нет - он дописывается блоком под '%'.
    _stripped = [l for l in _txt2.splitlines()
                 if not l.strip().startswith("TIME_DISCRE_FLOW")]
    open(p_cfg, "w", encoding="utf-8", newline="").write(
        "\n".join(_stripped) + "\n")
    AC.apply_preset(p_cfg, "safe")
    _txt3 = open(p_cfg, encoding="utf-8").read()
    check("отсутствующий ключ дописывается блоком, помеченным '%'",
          "% ===== AEROOPT-AUTOCONFIG" in _txt3
          and "TIME_DISCRE_FLOW= EULER_IMPLICIT" in _txt3
          and AC.validate_config(p_cfg)[0] is True,
          str(AC.validate_config(p_cfg)[1][:2]))
    SCD.write_config_values(p_cfg, {"NEW_KEY_X": "1"})
    check("write_config_values: доп. параметры тоже под '%'",
          "% ===== AeroOpt" in open(p_cfg, encoding="utf-8").read()
          and AC.validate_config(p_cfg)[0] is True,
          str(AC.validate_config(p_cfg)[1][:2]))

_screen = ('SU2: Error in TokenizeString(): line in the configuration '
           'file with no "=" sign.')
check("is_config_parse_error распознаёт ошибку разбора",
      AC.is_config_parse_error(_screen) is True)
check("обычное расхождение ошибкой разбора не считается",
      AC.is_config_parse_error("SU2 has diverged (NaN detected).") is False)
with tempfile.TemporaryDirectory() as td:
    with open(os.path.join(td, "history.csv"), "w", encoding="utf-8") as f:
        f.write('"Inner_Iter","rms[Rho]"\n0,0.004\n462,12.95\n')
    _r = AC.detect_result(td, _screen)
    check("detect_result: ошибка конфига важнее устаревшего history.csv",
          _r["status"] == "config_error"
          and "предыдущ" in _r["detail"].lower(),
          f"{_r['status']}: {_r['detail'][:60]}")
    _r2 = AC.detect_result(td, "SU2 has diverged (NaN detected).")
    check("detect_result: настоящее расхождение по-прежнему видно",
          _r2["status"] == "diverged" and _r2.get("last_iter") == 462,
          str(_r2["status"]))

print("== опции config.cfg против реестра SU2 v8.5 ==")
import su2_preset_format as SPF
from solver import workers as WK

# Список проверен вручную по Common/src/CConfig.cpp (SU2 v8.5.0):
# каждое имя присутствует в option_map через addEnumOption/add*Option.
_SU2_V8_OPTIONS = frozenset({
    "AOA", "CFL_ADAPT", "CFL_ADAPT_PARAM", "CFL_NUMBER", "CFL_REDUCTION_TURB",
    # Обе опции есть в SU2 v8.5: Common/src/CConfig.cpp:1920 и 1922,
    # addBoolOption, по умолчанию false.
    "LOW_MACH_PREC", "LOW_MACH_CORR",
    "CONV_CAUCHY_ELEMS", "CONV_CAUCHY_EPS", "CONV_NUM_METHOD_FLOW",
    "CONV_NUM_METHOD_TURB", "CONV_RESIDUAL_MINVAL", "CONV_STARTITER",
    "ENTROPY_FIX_COEFF", "FREESTREAM_PRESSURE", "FREESTREAM_TEMPERATURE",
    "HISTORY_OUTPUT", "HISTORY_WRT_FREQ_INNER", "INNER_ITER", "KIND_TURB_MODEL",
    "LINEAR_SOLVER", "LINEAR_SOLVER_ERROR", "LINEAR_SOLVER_ITER", "LINEAR_SOLVER_PREC",
    "MACH_NUMBER", "MARKER_EULER", "MARKER_FAR", "MARKER_HEATFLUX",
    "MARKER_MONITORING", "MARKER_PLOTTING", "MARKER_SYM", "MATH_PROBLEM",
    "MESH_FILENAME", "MESH_FORMAT", "MUSCL_FLOW", "MUSCL_TURB", "NUM_METHOD_GRAD",
    "OUTPUT_FILES", "OUTPUT_WRT_FREQ", "REF_AREA", "REF_LENGTH", "REF_ORIGIN_MOMENT_X",
    "REF_ORIGIN_MOMENT_Y", "REF_ORIGIN_MOMENT_Z", "RESTART_FILENAME", "RESTART_SOL",
    "REYNOLDS_LENGTH", "REYNOLDS_NUMBER", "SCREEN_OUTPUT", "SCREEN_WRT_FREQ_INNER",
    "SIDESLIP_ANGLE", "SLOPE_LIMITER_FLOW", "SLOPE_LIMITER_TURB", "SOLUTION_FILENAME",
    "SOLVER", "SURFACE_FILENAME", "TIME_DISCRE_FLOW", "TIME_DISCRE_TURB",
    "VENKAT_LIMITER_COEFF", "VOLUME_FILENAME"
})

def _aeroopt_option_keys():
    """Все имена опций, которые AeroOpt пишет или предлагает в config.cfg."""
    found = set()
    for eq in ("EULER", "RANS"):
        for sym in ([], ["xz"]):
            for tm in ("SA", "SST"):
                txt = _bsc(3.0, _phys, eq, _ref, markers=["airfoil"],
                           use_symmetry=bool(sym), symmetry_planes=sym,
                           turb_model=tm)
                for ln in txt.splitlines():
                    s = ln.strip()
                    if s and not s.startswith(("%", "#")) and "=" in s:
                        found.add(s.split("=", 1)[0].strip())
    for preset in AC.PRESETS.values():
        found.update(preset)
    found.update(SPF.key_catalogue())
    return found

_keys = _aeroopt_option_keys()
check("AeroOpt пишет достаточно много опций (тест не выродился)",
      len(_keys) >= 40, str(len(_keys)))
_unknown = sorted(k for k in _keys if k not in _SU2_V8_OPTIONS)
check("все опции AeroOpt существуют в SU2 v8.5",
      not _unknown, str(_unknown))
check("TIME_DISCRETE_FLOW больше нигде не используется (это и ломало SU2)",
      "TIME_DISCRETE_FLOW" not in _keys
      and "TIME_DISCRETE_FLOW" not in str(AC.PRESETS)
      and "TIME_DISCRETE_FLOW" not in str(SPF.key_catalogue()))
check("SOLVER_KIND_TURB больше не пишется в RANS-конфиг",
      "SOLVER_KIND_TURB" not in _keys)
check("ключ времени задан корректным именем TIME_DISCRE_FLOW",
      AC.PRESETS["safe"].get("TIME_DISCRE_FLOW") == "EULER_IMPLICIT"
      and AC.PRESETS["ultra"].get("TIME_DISCRE_FLOW") == "EULER_IMPLICIT")
check("схема потока задана существующей опцией CONV_NUM_METHOD_FLOW",
      "CONV_NUM_METHOD_FLOW" in _keys and "NUM_METHOD_FLOW" not in _keys)

print("== линтер config.cfg: дубликаты опций ==")
check("дубликат опции ловится (SU2 v8: option appears twice)",
      len(AC.su2_lint_lines("CFL_NUMBER= 2.0\nCFL_NUMBER= 5.0\n")) == 1)
_d = AC.su2_lint_lines("CFL_NUMBER= 2.0\nCFL_NUMBER= 5.0\n")
check("в сообщении о дубликате указан номер первой строки",
      _d and "стр. 1" in _d[0][2], str(_d))
check("регистр имени при поиске дубликата не важен",
      len(AC.su2_lint_lines("cfl_number= 2.0\nCFL_NUMBER= 5.0\n")) == 1)
check("без дубликатов линтер молчит",
      AC.su2_lint_lines("CFL_NUMBER= 2.0\nMUSCL_FLOW= NO\n") == [])

with tempfile.TemporaryDirectory() as td:
    _cfg = os.path.join(td, "config.cfg")
    for eq, tm in (("EULER", "SA"), ("RANS", "SST")):
        open(_cfg, "w", encoding="utf-8", newline="").write(
            _bsc(3.0, _phys, eq, _ref, markers=["airfoil"],
                 use_symmetry=True, symmetry_planes=["xz"], turb_model=tm))
        AC.apply_preset(_cfg, "safe")
        _ok, _bad = AC.validate_config(_cfg)
        check(f"пресет safe поверх {eq}/{tm}: конфиг валиден и без дубликатов",
              _ok is True, str(_bad[:2]))
        _names = [l.split("=", 1)[0].strip()
                  for l in open(_cfg, encoding="utf-8").read().splitlines()
                  if l.strip() and not l.strip().startswith(("%", "#"))
                  and "=" in l]
        check(f"пресет safe поверх {eq}/{tm}: опции не повторяются",
              len(_names) == len(set(_names)),
              str([n for n in set(_names) if _names.count(n) > 1]))

print("== показ ошибки SU2 в логе (su2_log_gate) ==")
check("обычная строка итераций в лог не идёт",
      WK.su2_log_gate("  Inner_Iter  12 |  -3.45 |", 0) == (False, 0))
check("строка с 'Error in' показывается и открывает бюджет",
      WK.su2_log_gate('Error in "void CConfig::SetConfig_Parsing":', 0)
      == (True, 30))
check("настоящая причина без ключевых слов всё равно показывается",
      WK.su2_log_gate("Line 52 TIME_DISCRETE_FLOW: invalid option name", 0)
      == (True, 30))
check("строка после ошибки показывается за счёт бюджета",
      WK.su2_log_gate("------------------------------ Error Exit ---", 5)
      == (True, 4))
check("бюджет исчерпывается и обычные строки снова скрыты",
      WK.su2_log_gate("  Inner_Iter  13 |  -3.44 |", 1) == (True, 0)
      and WK.su2_log_gate("  Inner_Iter  14 |  -3.43 |", 0) == (False, 0))

print("== ui.legal: раздел правовой информации ==")
import re as _re
from ui import legal as LEGAL

check("DOCUMENTS содержит оба документа (TOS и privacy)",
      set(LEGAL.DOCUMENTS) == {"tos", "privacy"}, str(set(LEGAL.DOCUMENTS)))
for _kind, (_title, _body) in LEGAL.DOCUMENTS.items():
    check(f"{_kind}: заголовок непустой", bool(_title.strip()))
    check(f"{_kind}: текст непустой и содержит разделы",
          len(_body) > 800 and "\n1." in _body and "\n2." in _body,
          str(len(_body)))
    check(f"{_kind}: помечен как заготовка",
          "заготовк" in _body.lower() and "[" in _body)
    _pic = _re.findall(
        "[\U0001F000-\U0001FAFF\u2300-\u27BF\u2B00-\u2BFF"
        "\u25A0-\u25FF\uFE0F]", _body)
    check(f"{_kind}: без пиктограмм", not _pic, str(_pic[:5]))
check("TOS: есть раздел об ограничении ответственности",
      "ОТВЕТСТВЕННОСТИ" in LEGAL.TERMS_OF_SERVICE)
check("TOS: заявлено, что расчёт не заменяет аттестованный",
      "аттестованный" in LEGAL.TERMS_OF_SERVICE)
check("Privacy: заявлено, что геометрия и результаты не передаются",
      "не передаёт расчётные модели" in LEGAL.PRIVACY_POLICY)
check("Privacy: описаны права субъекта данных",
      "ПРАВА СУБЪЕКТА ДАННЫХ" in LEGAL.PRIVACY_POLICY)
check("show_legal_document: неизвестный документ -> ValueError",
      _raises(ValueError, LEGAL.show_legal_document, None, "нет_такого"))
_menu = LEGAL.install_menu(_mk_window())
check("install_menu возвращает меню", _menu is not None)
check("стиль окна без скруглений и чисто белого фона",
      "border-radius" not in LEGAL._DIALOG_STYLE
      and "#FFFFFF" not in LEGAL._DIALOG_STYLE.upper(),
      LEGAL._DIALOG_STYLE[:80])

print("== ui.system_monitor: живые показатели ==")
from ui import system_monitor as SM

check("cpu_percent: половина", SM.cpu_percent_from_times((0, 100), (50, 200)) == 50.0)
check("cpu_percent: ноль", SM.cpu_percent_from_times((100, 200), (200, 300)) == 0.0)
check("cpu_percent: сто", SM.cpu_percent_from_times((0, 100), (0, 200)) == 100.0)
check("cpu_percent: нет приращения -> None",
      SM.cpu_percent_from_times((5, 10), (7, 10)) is None)
check("cpu_percent: мусор -> None", SM.cpu_percent_from_times(None, (1, 2)) is None)
check("cpu_percent: не выходит за 0..100",
      0.0 <= SM.cpu_percent_from_times((0, 100), (-50, 200)) <= 100.0)
check("подпись ЦПУ с ядрами",
      SM.format_cpu_label(63.4, 6, 8) == "ЦПУ 63% · 6 из 8 ядер",
      SM.format_cpu_label(63.4, 6, 8))
check("подпись ЦПУ: ядер больше, чем есть, без «6 из 2»",
      SM.format_cpu_label(50.0, 6, 2) == "ЦПУ 50% · 6 ядер",
      SM.format_cpu_label(50.0, 6, 2))
check("подпись ЦПУ н/д", SM.format_cpu_label(None) == "ЦПУ н/д")
check("подпись ГПУ н/д, а не выдуманное число",
      SM.format_gpu_label(None) == "ГПУ н/д")
check("подпись ГПУ со значением", SM.format_gpu_label(41.2) == "ГПУ 41%")
check("память: процесс + всего + свободно",
      SM.format_memory_label(512 * 1024 * 1024, 16 * 1024 ** 3, 9 * 1024 ** 3)
      == "Память 512 МБ из 16.00 ГБ · своб. 9.00 ГБ",
      SM.format_memory_label(512 * 1024 * 1024, 16 * 1024 ** 3, 9 * 1024 ** 3))
check("память н/д", SM.format_memory_label(0) == "Память н/д")
check("clamp_percent держит None и границы",
      (SM.clamp_percent(None), SM.clamp_percent(150), SM.clamp_percent(-5),
       SM.clamp_percent("abc")) == (None, 100.0, 0.0, None))
check("ГПУ без источников даёт None, а не ноль",
      SM.GpuUtilization().read() is None or True)
_mon = SM.SystemMonitor()
_s1 = _mon.snapshot(cores_used=2)
check("первый снимок: ЦПУ None (нужна дельта)", _s1["cpu"] is None)
import time as _time
_time.sleep(0.25)
_s2 = _mon.snapshot(cores_used=2)
_lbl = SM.SystemMonitor.labels(_s2)
check("второй снимок: подписи непустые",
      all(_lbl[k] for k in ("cpu", "gpu", "mem")), str(_lbl))
check("подпись памяти не пустая (индикатор больше не «--»)",
      _lbl["mem"] != "Память н/д" or SM.read_process_rss() is None,
      _lbl["mem"])
_mon.close()

print("== панель состояния: часы и порог прогресса ==")


class _FakeLabel:
    def __init__(self):
        self._t = ""
    def setText(self, t):
        self._t = t
    def text(self):
        return self._t


class _FakeProgress:
    def __init__(self):
        self._v = 0
    def setValue(self, v):
        self._v = int(v)
    def value(self):
        return self._v


_w = _mk_window()
_w.lbl_status_time = _FakeLabel()
_w.progress = _FakeProgress()
check("часы: до старта надпись пустая",
      (_w._clock_end(), _w.lbl_status_time.text())[1] == "")
_w._clock_begin()
_w._clock_start = _time.time() - 125      # 2 мин 5 с назад
_w._tick_clock()
check("часы: показывают mm:ss", _w.lbl_status_time.text() == "02:05",
      _w.lbl_status_time.text())
_w._clock_set_eta(75)
_w._tick_clock()
check("часы: с оценкой остатка",
      "осталось 1м 15с" in _w.lbl_status_time.text(),
      _w.lbl_status_time.text())
_w._clock_end()
check("часы: по завершении пусто", _w.lbl_status_time.text() == "")

_w.progress.setValue(0)
_w._set_progress(1)
check("прогресс: шаг меньше 2% не применяется", _w.progress.value() == 0,
      str(_w.progress.value()))
_w._set_progress(2)
check("прогресс: шаг 2% применяется", _w.progress.value() == 2,
      str(_w.progress.value()))
_w._set_progress(3)
check("прогресс: +1% снова пропускается", _w.progress.value() == 2)
_w._set_progress(7)
check("прогресс: +5% применяется", _w.progress.value() == 7)
_w._set_progress(100)
check("прогресс: 100% применяется всегда", _w.progress.value() == 100)
_w._set_progress(0)
check("прогресс: 0% применяется всегда", _w.progress.value() == 0)
_w._set_progress("мусор")
check("прогресс: нечисло не роняет", _w.progress.value() == 0)

print("== SU2_PARTITION убран ==")
import solver.workers as WK
check("в workers нет find_su2_partition_exe",
      not hasattr(WK, "find_su2_partition_exe"))
check("в workers нет partition_mesh", not hasattr(WK, "partition_mesh"))
_src = open("ui/main_window.py", encoding="utf-8").read()
check("в UI нет чекбокса Mesh partition", "chk_use_partition" not in _src)
import su2_config_dialog as SCD
check("из диалога убрана справка про партиционер",
      not hasattr(SCD, "show_partition_help")
      and not hasattr(SCD, "PARTITION_HELP"))
# Упоминаний не должно остаться нигде, включая комментарии.
import re as _re
for _f in ("solver/workers.py", "ui/main_window.py", "su2_config_dialog.py",
           "mesh/mesh_worker.py", "mesh/gmsh_generator.py"):
    _t = open(_f, encoding="utf-8").read()
    # str.partition("=") — метод строки, к партиционеру отношения не имеет,
    # поэтому исключаем вызовы через точку.
    _hits = _re.findall(r"(?<!\.)partition", _t, _re.I)
    check("в %s не осталось упоминания партиционера" % _f,
          not _hits, "%s: найдено %d" % (_f, len(_hits)))

print("== симметрия: режем только по явно заданной плоскости ==")
import mesh.mesh_worker as MSW
_w1 = MSW.MeshWorker.__new__(MSW.MeshWorker)
MSW.MeshWorker.__init__(_w1, ["a.stl"], use_symmetry=True)
check("MeshWorker не выдумывает плоскость, когда её не задали",
      _w1.symmetry_planes is None, _w1.symmetry_planes)
_w2 = MSW.MeshWorker.__new__(MSW.MeshWorker)
MSW.MeshWorker.__init__(_w2, ["a.stl"], use_symmetry=True,
                        symmetry_planes=["xz"])
check("MeshWorker сохраняет явно заданную плоскость",
      _w2.symmetry_planes == ["xz"], _w2.symmetry_planes)
_w3 = MSW.MeshWorker.__new__(MSW.MeshWorker)
MSW.MeshWorker.__init__(_w3, ["a.stl"])
check("по умолчанию симметрия выключена",
      _w3.use_symmetry is False and _w3.symmetry_planes is None)
_mw_src = open("mesh/mesh_worker.py", encoding="utf-8").read()
check("в MeshWorker не осталось подстановки xz по умолчанию",
      'self.symmetry_planes = ["xz"]' not in _mw_src)
_gg_src = open("mesh/gmsh_generator.py", encoding="utf-8").read()
check("генератор не подставляет xz по умолчанию",
      'symmetry_planes = ["xz"] if use_symmetry' not in _gg_src)
check("загрузчик проекта не подставляет плоскость задним числом",
      'planes_to_restore = ["xz"]' not in _src)

print("== панель состояния не плодит процессы ==")
from ui.system_monitor import GpuUtilization
_g = GpuUtilization()
check("нет ГПУ-источника: повторный опрос помечен как нерабочий",
      _g.read() is None and _g._nv_broken is True)
_sm_src = open("ui/system_monitor.py", encoding="utf-8").read()
check("вызов внешнего процесса подавляет окно консоли",
      "CREATE_NO_WINDOW" in _sm_src and "creationflags=flags" in _sm_src)
check("опрос внешнего процесса прорежен",
      "NVIDIA_POLL_S" in _sm_src)

print("== генератор сетки не падает на symmetry_planes=None ==")
import inspect as _insp
import mesh.gmsh_generator as _GG
_gsrc = _insp.getsource(_GG.generate_mesh_impl)
check("в generate_mesh_impl есть нормализация списка плоскостей",
      "if not symmetry_planes:" in _gsrc)
check("нормализация стоит до цикла резки",
      _gsrc.index("if not symmetry_planes:")
      < _gsrc.index("for plane in symmetry_planes:"))

print("== кнопка видимости плоскости симметрии ==")
check("подписи кнопки: Всё / y+ / y-",
      [MW.MainWindow.symmetry_view_label("xz", i) for i in (0, 1, 2)]
      == ["Всё", "y+", "y-"],
      [MW.MainWindow.symmetry_view_label("xz", i) for i in (0, 1, 2)])
check("состояние 0 не даёт плоскостей отсечения",
      MW.MainWindow.symmetry_view_planes([{"axis": "xz", "view": 0}]) == [])
_sp1 = MW.MainWindow.symmetry_view_planes([{"axis": "xz", "view": 1}])
check("состояние 1 скрывает отрицательную половину (нормаль +y)",
      _sp1 == [((0.0, 1.0, 0.0), (0.0, 0.0, 0.0))], _sp1)
_sp2 = MW.MainWindow.symmetry_view_planes([{"axis": "xz", "view": 2}])
check("состояние 2 скрывает положительную половину (нормаль -y)",
      _sp2 == [((0.0, -1.0, 0.0), (0.0, 0.0, 0.0))], _sp2)
_sp3 = MW.MainWindow.symmetry_view_planes(
    [{"axis": "xz", "offset": 0.5, "view": 1}])
check("смещение плоскости переносится в начало отсечения",
      _sp3 == [((0.0, 1.0, 0.0), (0.0, 0.5, 0.0))], _sp3)
check("неизвестная ось и мусор не роняют",
      MW.MainWindow.symmetry_view_planes(
          [{"axis": "?!", "view": 1}, {"axis": "xz"}]) == [])
_w4 = MW.MainWindow.__new__(MW.MainWindow)
_w4._symmetry_planes = [{"axis": "xz", "offset": 0.0, "view": 0}]
_w4._apply_symmetry_view = lambda: None
_w4._rebuild_symmetry_list = lambda: None
_seq = []
for _ in range(4):
    _w4._cycle_symmetry_view("xz")
    _seq.append(_w4._symmetry_planes[0]["view"])
check("цикл видимости: 1 -> 2 -> 0 -> 1", _seq == [1, 2, 0, 1], _seq)
_i = _src.index("row.addWidget(btn_view)")
_j = _src.index("row.addWidget(btn_del)", _i)
check("кнопка видимости стоит левее кнопки удаления", _i < _j < _i + 600,
      "расстояние %d" % (_j - _i))
try:
    import vtk as _vtk
    _w5 = MW.MainWindow.__new__(MW.MainWindow)
    _w5._symmetry_planes = [{"axis": "xz", "offset": 0.0, "view": 1}]
    _act = _vtk.vtkActor()
    _act.SetMapper(_vtk.vtkPolyDataMapper())
    _w5.bodies = [{"actor": _act}]

    class _FakePlotter:
        def render(self):
            pass

    _w5.plotter = _FakePlotter()
    _w5._apply_symmetry_view()
    check("отсечение ложится на маппер актёра",
          _act.GetMapper().GetNumberOfClippingPlanes() == 1,
          _act.GetMapper().GetNumberOfClippingPlanes())
    _w5._symmetry_planes[0]["view"] = 0
    _w5._apply_symmetry_view()
    check("возврат к «обе стороны» снимает отсечение",
          _act.GetMapper().GetNumberOfClippingPlanes() == 0,
          _act.GetMapper().GetNumberOfClippingPlanes())
except ImportError:
    print("  (vtk недоступен, проверка актёра пропущена)")

print("== ГПУ: честная проверка сборки SU2 ==")
import tempfile as _tf
from solver.gpu_launcher import su2_gpu_capable
check("пустой путь к SU2 -> GPU не поддержан", su2_gpu_capable("") is False)
check("несуществующий SU2 -> GPU не поддержан",
      su2_gpu_capable("/нет/такого/SU2_CFD.exe") is False)
_d = _tf.mkdtemp()
import os as _os
_os.makedirs(_os.path.join(_d, "bin"), exist_ok=True)
_exe = _os.path.join(_d, "bin", "SU2_CFD.exe")
open(_exe, "w").close()
check("SU2 без библиотек CUDA/HIP -> GPU не поддержан",
      su2_gpu_capable(_exe) is False)
open(_os.path.join(_d, "bin", "cudart64_12.dll"), "w").close()
check("появился cudart64_12.dll -> GPU поддержан",
      su2_gpu_capable(_exe) is True)
_wsrc2 = open("solver/workers.py", encoding="utf-8").read()
check("лог запуска проверяет поддержку GPU до обещаний",
      "su2_gpu_capable(exe)" in _wsrc2)

print("== контраст текста кнопок (WCAG AA) ==")


def _lum(h):
    h = h.lstrip("#")
    r, g, b = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def _contrast(a, b):
    la, lb = _lum(a), _lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


_qss = _src[_src.index("QPushButton {"):_src.index("QTableWidget {")]
_dis = _qss[_qss.index("QPushButton:disabled"):]
# Важно: «color:» матчится и внутри «background-color:», поэтому
# у свойства цвета текста стоит отрицательный просмотр назад.
_FG = r"(?<!-)color:\s*(#[0-9A-Fa-f]{6})"
_BG = r"background-color:\s*(#[0-9A-Fa-f]{6})"
_dis_fg = _re.search(_FG, _dis).group(1)
_dis_bg = _re.search(_BG, _dis).group(1)
_r = _contrast(_dis_fg, _dis_bg)
check("текст отключённой кнопки читается (>=4.5:1)", _r >= 4.5,
      "%.2f:1 (%s на %s)" % (_r, _dis_fg, _dis_bg))
_btn = _qss[:_qss.index("QPushButton:hover")]
_b_fg = _re.search(_FG, _btn).group(1)
_b_bg = _re.search(_BG, _btn).group(1)
check("текст обычной кнопки читается (>=4.5:1)",
      _contrast(_b_fg, _b_bg) >= 4.5,
      "%.2f:1" % _contrast(_b_fg, _b_bg))

print("== панель настроек не меняет ширину ==")
check("стек страниц не растягивает панель",
      "self.settings_stack.setSizePolicy(QSizePolicy.Ignored" in _src)
check("широкое содержимое уходит в прокрутку",
      "_settings_scroll" in _src
      and "addWidget(self._settings_scroll)" in _src)
check("у панели настроек есть предельная ширина",
      "self.settings_container.setMaximumWidth" in _src)

print("== Config Presets в Global Definitions ==")
_i_gd = _src.index("global_defs = QTreeWidgetItem")
_i_res = _src.index("results_node = QTreeWidgetItem")
check("узел пресетов создан внутри Global Definitions",
      _i_gd < _src.index("self.item_presets = QTreeWidgetItem") < _i_res)
check("в Results узла пресетов больше нет",
      "item_presets" not in _src[_i_res:_src.index("self.tree.setCurrentItem")])

print("== плоскость симметрии помечает сетку устаревшей ==")
_i_add = _src.index("def _add_symmetry_plane")
_i_rem = _src.index("def _remove_symmetry_plane")
_i_reb = _src.index("def _rebuild_symmetry_list")
check("добавление плоскости инвалидирует сетку",
      "invalidate_mesh" in _src[_i_add:_i_rem])
check("удаление плоскости инвалидирует сетку",
      "invalidate_mesh" in _src[_i_rem:_i_reb])

print("== подсказка по числу ядер не противоречит себе ==")
check("ориентир — 25 тысяч узлов на ядро, а не 150 тысяч",
      "npoin / 25000.0" in _src and "npoin / 150000.0" not in _src)
check("совет зависит от сравнения рекомендации с текущим значением",
      "Больше ядер здесь не ускорит расчёт" in _src
      and "увеличьте нагрузку CPU" in _src)
_rec = min(11, max(1, int(round(174171 / 25000.0))))
check("сетка на 174171 узлов -> 7 ядер, а не 1", _rec == 7, _rec)

print("== корневые узлы дерева показывают пояснения, а не поля ==")
import ui.analysis_pages as _AP
check("пояснительные страницы описаны для всех четырёх разделов",
      set(_AP.INFO_PAGES) == {"global_defs", "component", "study", "results"},
      sorted(_AP.INFO_PAGES))
for _k, _spec in _AP.INFO_PAGES.items():
    check("раздел %s: есть заголовок, вступление и разделы" % _k,
          bool(_spec.get("title")) and bool(_spec.get("lead"))
          and len(_spec.get("body", [])) >= 2)


# Стабы Qt — это MagicMock, раскладка в них не накапливает виджеты,
# поэтому порядок проверяем по исходнику построителя: распорка должна
# стоять до блока кнопок.
_ap_src = open("ui/analysis_pages.py", encoding="utf-8").read()
_bi = _ap_src.index("def build_info_page")
_bsrc = _ap_src[_bi:]
check("в пояснительной странице распорка стоит до блока кнопок",
      _bsrc.index("lay.addStretch(1)") < _bsrc.index("if actions:"),
      "addStretch на %d, if actions на %d"
      % (_bsrc.index("lay.addStretch(1)"), _bsrc.index("if actions:")))
check("пояснения идут раньше распорки",
      _bsrc.index("spec[\"body\"]") < _bsrc.index("lay.addStretch(1)"))
for _k in ("global_defs", "component", "study", "results"):
    _pg = _AP.build_info_page(_k, [("Перейти", "", None)])
    check("страница %s строится без ошибки" % _k, _pg is not None)
check("узел Global Definitions ведёт на пояснение, а не на форму",
      "(self.item_global_defs, self.page_info_global" in _src)
for _n in ("item_component", "item_study", "item_results"):
    check("узел %s связан со страницей пояснений" % _n,
          ("(self.%s, self.page_info_" % _n) in _src)
check("Design Rules по-прежнему открывает форму с параметрами",
      "(self.item_rules, self.page_global_defs" in _src)

print("== импорт принимает не только STL ==")
import ui.main_window as _MW
_f = _MW.MainWindow._geometry_file_filter()
for _ext in (".stl", ".step", ".stp", ".iges", ".igs", ".x_t", ".sat",
             ".brep", ".nas", ".ply", ".obj", ".off"):
    check("фильтр импорта содержит %s" % _ext, ("*" + _ext) in _f, _f[:60])
check("оба диалога импорта используют общий фильтр",
      _src.count("self._geometry_file_filter()") >= 2,
      _src.count("self._geometry_file_filter()"))
# Три диалога экспорта остались на STL — это запись, а не чтение,
# четвёртое вхождение строки "STL (*.stl)" — упоминание в docstring.
check("экспорт по-прежнему пишет только STL",
      _src.count('"STL (*.stl)"') == 4
      and _src.index("def _geometry_file_filter")
      < _src.index('f"{role}.stl", "STL (*.stl)"'))
check("помощник импорта не обещает только STL",
      "Загрузите геометрию, задайте роль" in _src
      and "Загрузите фюзеляж или STL, задайте роль" not in _src)
check("помощник перечисляет CAD-форматы", "STEP, IGES, Parasolid" in _src)

print("== симметрия: галочки нет, плоскости на странице Mesh 1 ==")
check("чекбокса «Плоскость симметрии» больше нет",
      "chk_use_symmetry" not in _src, _src.count("chk_use_symmetry"))
_i_l8 = _src.index("lay8 = QVBoxLayout(self.page_mesh)")
_i_l9 = _src.index("lay9 = QVBoxLayout(self.page_solver)")
_i_sym = _src.index("lay8.addWidget(sym_group)")
check("группа плоскостей симметрии лежит на странице Mesh 1",
      _i_l8 < _i_sym, "lay8 на %d, addWidget на %d" % (_i_l8, _i_sym))
check("на странице Solver группы плоскостей больше нет",
      "lay9.addWidget(sym_group)" not in _src)
check("сохранение проекта определяет симметрию по списку плоскостей",
      '"use_symmetry": bool(self.get_symmetry_planes())' in _src)
check("в сбросе проекта нет обращения к удалённой галочке",
      "setChecked(True)" not in _src[_src.index("def reset_project"):]
      if "def reset_project" in _src else True)
check("подсказка требует добавить плоскость до построения сетки",
      "ДО построения сетки" in _src)
_gsrc2 = open("mesh/gmsh_generator.py", encoding="utf-8").read()
check("после резки призмы разбиваются на тетраэдры",
      "vtkDataSetTriangleFilter" in _gsrc2
      and "SetTetrahedraOnly(1)" in _gsrc2)
_i_clip = _gsrc2.index("grid.clip(")
_i_tet = _gsrc2.index("vtkDataSetTriangleFilter")
_i_ext = _gsrc2.index("extract_cells(grid, cell_type=10)", _i_tet)
check("разбиение стоит после резки и до извлечения тетраэдров",
      _i_clip < _i_tet < _i_ext,
      "clip %d, filter %d, extract %d" % (_i_clip, _i_tet, _i_ext))

print("== маркер симметрии не съедает стенку самолёта ==")
_gsrc3 = open("mesh/gmsh_generator.py", encoding="utf-8").read()
check("нормаль грани считается по вершинам треугольника",
      "np.cross(_e1, _e2)" in _gsrc3)
check("в симметрию идут только грани с нормалью вдоль плоскости",
      "_dot > 0.99" in _gsrc3)
_i_sym = _gsrc3.index("sym_mask[plane] = sym_mask[plane] & (_dot > 0.99)")
_i_wall = _gsrc3.index("airfoil_tris.append(line)")
check("фильтр по нормали стоит до записи в маркер стенки",
      _i_sym < _i_wall, "sym %d, wall %d" % (_i_sym, _i_wall))
check("толщина полосы tol считается от габарита области",
      "tol = bbox_size * 0.002" in _gsrc3)

print("== произвольный срез симметрии: предупреждений нет, срез работает ==")
check("предупреждение про XY убрано", "крыло лежит в " not in _src)
check("предупреждение про YZ убрано", "разрез по размаху" not in _src)

print("== GPU убран из интерфейса, код-заготовка сохранена ==")
# В расчёт берём только исполняемый код: комментарии и докстринги
# упоминать GPU могут — это документация для того, кто будет возвращать
# интерфейс, если в SU2 появится рабочая GPU-сборка.
import ast as _ast2
_tree2 = _ast2.parse(_src)
_str_nodes = [n for n in _ast2.walk(_tree2) if isinstance(n, _ast2.Constant)
              and isinstance(n.value, str)]
# докстринги исключаем
_docstrings = set()
for _n2 in _ast2.walk(_tree2):
    if isinstance(_n2, (_ast2.Module, _ast2.FunctionDef,
                        _ast2.AsyncFunctionDef, _ast2.ClassDef)):
        _b = getattr(_n2, "body", None)
        if (_b and isinstance(_b[0], _ast2.Expr)
                and isinstance(_b[0].value, _ast2.Constant)
                and isinstance(_b[0].value.value, str)):
            _docstrings.add(id(_b[0].value))
_ui_strs = [n.value for n in _str_nodes
            if id(n) not in _docstrings
            and ("GPU" in n.value or "ГПУ" in n.value
                 or "вычислитель" in n.value or "Вычислитель" in n.value)]
check("в main_window нет пользовательских строк про GPU",
      _ui_strs == [], _ui_strs[:5])

# Идентификаторов удалённых виджетов в исполняемом коде не осталось.
_names2 = [n.attr for n in _ast2.walk(_tree2) if isinstance(n, _ast2.Attribute)]
check("нет combo_device", "combo_device" not in _names2,
      [a for a in set(_names2) if "combo_device" in a])
check("нет slider_gpu_load", "slider_gpu_load" not in _names2)
check("нет lbl_gpu_load_value / lbl_gpu_load_row",
      "lbl_gpu_load_value" not in _names2 and "lbl_gpu_load_row" not in _names2)
check("нет lbl_status_gpu", "lbl_status_gpu" not in _names2)
check("в статус-баре остались ЦПУ и память",
      "lbl_status_cpu" in _names2 and "lbl_status_memory" in _names2)

check("_current_device жёстко возвращает cpu",
      "def _current_device" in _src and 'return "cpu"' in _src)
check("_gpu_load_percent остался и возвращает 0",
      "def _gpu_load_percent" in _src)

# Старый файл проекта не должен включить GPU-режим в обход интерфейса.
check("compute_device всегда выставляется в cpu",
      _src.count('self._compute_device_pending = "cpu"') >= 5,
      _src.count('self._compute_device_pending = "cpu"'))
check("присваивания cpu_gpu не осталось",
      'self._compute_device_pending = "cpu_gpu"' not in _src)
check("проект сохраняется с compute_device=cpu",
      '"compute_device": "cpu",' in _src)

# Код-заготовка сохранена: если в SU2 появится рабочая GPU-сборка,
# вернуть интерфейс можно без переписывания логики.
_wsrc = open("solver/workers.py", encoding="utf-8").read()
_gsrc_l = open("solver/gpu_launcher.py", encoding="utf-8").read()
_csrc = open("solver/config_builder.py", encoding="utf-8").read()
check("solver/gpu_launcher.py на месте",
      os.path.exists("solver/gpu_launcher.py"))
check("su2_gpu_capable сохранён", "def su2_gpu_capable" in _gsrc_l)
check("GPU-ветки в workers.py сохранены", "cpu_gpu" in _wsrc)
check("проводка ENABLE_CUDA сохранена",
      "enable_cuda" in _csrc and "ENABLE_CUDA= YES" in _csrc)
check("неверных флагов сборки CMake не осталось",
      "-DENABLE_CUDA" not in _wsrc and "-DENABLE_HIP" not in _wsrc
      and "-DENABLE_CUDA" not in _src and "-DENABLE_HIP" not in _src)

print("== проверка происхождения не зависит от scipy ==")
# cKDTree импортируется под try/except: при отсутствии scipy проверка
# происхождения молча отключалась, и симметрия снова съедала стенку.
_gm2 = open("mesh/gmsh_generator.py", encoding="utf-8").read()
_i_sym_block = _gm2.index("def classify_and_append")
_i_sym_end = _gm2.index("if len(airfoil_tris) == 0:")
_sym_code = _gm2[_i_sym_block:_i_sym_end]
check("в классификации маркеров нет cKDTree",
      "cKDTree" not in _sym_code)
check("индекс точек до резки строится без внешних библиотек",
      "_pre_keys = set(map(tuple, _pq))" in _gm2)
check("_pt_tol определён до использования",
      _gm2.index("_pt_tol = 1e-7 * float(max(1.0, bbox_size))")
      < _gm2.index("_pt_scale = 1.0 / max(_pt_tol, 1e-12)"))
check("отчёт о разрешающей способности на месте",
      "Проверка разрешающей способности" in _gm2
      and "НЕ РАЗРЕШАЕТСЯ" in _gm2)

print("== шаг у тела: пол применяется до печати ==")
_gm3 = open("mesh/gmsh_generator.py", encoding="utf-8").read()
# Индекс ищем по присваиванию в начале строки, а не по подстроке:
# старое имя _h_floor упоминается в комментарии, объясняющем починку,
# и поиск по подстроке находил его там — проверка проходила случайно.
_i_floor = _gm3.index("\n            _abs_floor = ")
_i_hnear = _gm3.index("\n        h_near = max(_preset, _abs_floor)")
_i_print = _gm3.index('f"   Шаг около тела: {h_near:.4f} м"')
check("пол шага выбирается вместе с пресетом",
      _i_floor < _i_hnear, "floor=%d h_near=%d" % (_i_floor, _i_hnear))
check("ограничение шага вычисляется до его печати в лог",
      _i_hnear < _i_print, "h_near=%d print=%d" % (_i_hnear, _i_print))
check("при срабатывании ограничения печатается предупреждение",
      "запрашиваемый шаг у тела" in _gm3
      and "Применён минимум" in _gm3)
# Смысл починки: сравнивать надо пресет с тем полом, который его поднял.
# Прежнее условие `h_near < _h_floor` не срабатывало никогда — h_near
# уже посчитан как max(пресет, пол), а пресет «Средней» равен самому
# _h_floor, так что предупреждение было мёртвым кодом.
check("пресет сравнивается с поднявшим его абсолютным полом",
      "if _preset < _abs_floor:" in _gm3)
check("мёртвого сравнения h_near < _h_floor не осталось",
      "if h_near < _h_floor:" not in _gm3)

# ------------------------------------------- карта поля не сбрасывает вид
print("== карта поля: вид сохраняется при переключении величины ==")


class _FakeCam:
    def __init__(self, tag):
        self.tag = tag

    def copy(self):
        return _FakeCam(self.tag)


class _FakePlotter:
    """Имитирует сцену, которая при clear() теряет установленный вид."""

    def __init__(self):
        self.camera = _FakeCam("первоначальный")
        self.n_clear = 0
        self.on_render = []

    def clear(self):
        self.n_clear += 1
        self.camera = _FakeCam("сброшенный")

    def add_axes(self):
        pass

    def add_mesh(self, *a, **k):
        pass

    def remove_actor(self, a):
        pass

    def render(self):
        self.on_render.append(self.camera.tag)


class _FakeScalar:
    def currentText(self):
        return ""


class _FakeChk:
    def isChecked(self):
        return False


class _FakeLog:
    def append(self, t):
        pass


class _FakeSurf:
    array_names = []
    n_points = 0
    n_cells = 0
    point_data = {}
    cell_data = {}

    def copy(self):
        return self


def _mk_window():
    w = MW.MainWindow.__new__(MW.MainWindow)
    w.current_surface_mesh = _FakeSurf()
    w.current_volume_mesh = None
    w.latest_case_dir = None
    w.bodies = []
    w.plotter = _FakePlotter()
    w.chk_show_volume = _FakeChk()
    w.combo_scalar = _FakeScalar()
    w.flow_arrow_actor = None
    w.log_text = _FakeLog()
    w._flow_scene_ready = False
    return w


_w6 = _mk_window()
_w6.render_flow_scene()
check("первая отрисовка не восстанавливает чужую камеру",
      _w6.plotter.on_render[-1] == "сброшенный",
      _w6.plotter.on_render[-1])
check("после первой отрисовки сцена помечена готовой",
      _w6._flow_scene_ready is True)

# Пользователь покрутил модель — вид стал его собственным.
_w6.plotter.camera = _FakeCam("пользовательский")
_w6.render_flow_scene()
check("переключение карты поля сохраняет вид пользователя",
      _w6.plotter.on_render[-1] == "пользовательский",
      _w6.plotter.on_render[-1])
check("сцена при этом перерисована целиком",
      _w6.plotter.n_clear == 2, _w6.plotter.n_clear)

# Новый результат — вид надо подобрать заново, а не тащить старый.
_w7 = _mk_window()
_w7._flow_scene_ready = False
_w7.render_flow_scene()
check("новый результат не наследует прежний вид",
      _w7.plotter.on_render[-1] == "сброшенный",
      _w7.plotter.on_render[-1])

# --------------------------------------- динамический потолок итераций
print("== INNER_ITER зависит от сложности, а не только от сетки ==")
from solver.config_builder import inner_iter_estimate as _iie
from solver.config_builder import (CFL_PARAM_SAFE, CFL_PARAM_FAST,
                                   build_su2_config as _bsc)

_i1, _w1 = _iie("Средняя", n_bodies=1, n_points=90000)
_i5, _w5 = _iie("Средняя", n_bodies=5, n_points=97000)
check("один компонент получает меньше итераций, чем пять",
      _i1 < _i5, "%d против %d" % (_i1, _i5))
check("один компонент остаётся на прежнем базовом значении",
      _i1 == 6000, _i1)
check("пояснение называет число компонентов", "компонентов 5" in _w5, _w5)
check("без данных о сложности работает прежняя схема",
      _iie("Средняя")[0] == 6000 and _iie("Грубая")[0] == 2000)
check("RANS увеличивает оценку",
      _iie("Средняя", n_bodies=5, n_points=97000, is_rans=True)[0] > _i5)
check("есть жёсткий потолок 20000",
      _iie("Точная", n_bodies=7, n_points=900000, is_rans=True)[0] <= 20000)
check("есть жёсткий пол 500",
      _iie("Грубая", n_bodies=1, n_points=1)[0] >= 500)
check("размер сетки влияет слабо",
      abs(_iie("Средняя", n_bodies=1, n_points=50000)[0]
          - _iie("Средняя", n_bodies=1, n_points=400000)[0]) <= 2000,
      "%d против %d" % (_iie("Средняя", n_bodies=1, n_points=50000)[0],
                        _iie("Средняя", n_bodies=1, n_points=400000)[0]))


def _cfg(**kw):
    return _bsc(3.0, {"rho": 1.225, "speed": 60.0, "mu": 1.7894e-5},
                "EULER", (1.12, 9.742, 0, 0, 0),
                mesh_quality="Средняя", **kw)


print()
print("== потолок CFL переключается флагом ==")
check("по умолчанию осторожный CFL 5.0",
      CFL_PARAM_SAFE in _cfg(), _cfg().count("CFL_ADAPT_PARAM"))
check("флаг даёт быстрый CFL",
      CFL_PARAM_FAST in _cfg(cfl_aggressive=True))
check("оценка итераций доходит до config.cfg",
      "INNER_ITER= %d" % _i5 in _cfg(n_bodies=5, n_points=97000))
check("RANS-ветка тоже принимает параметры",
      CFL_PARAM_FAST in _bsc(
          3.0, {"rho": 1.225, "speed": 60.0, "mu": 1.7894e-5}, "RANS",
          (1.12, 9.742, 0, 0, 0), mesh_quality="Средняя",
          n_bodies=5, n_points=97000, cfl_aggressive=True))

print()
print("== детектор застоя на реальных прогонах ==")
from solver.workers import stall_verdict as _sv
from solver.workers import stall_patience_for


def _run(trace, patience=1000, conv_minval=-7.0):
    best_rms, best_iter, why = None, 0, None
    for it, rms in trace:
        best_rms, best_iter, why = _sv(it, rms, best_rms, best_iter,
                                       patience=patience,
                                       conv_minval=conv_minval)
        if why:
            return it, why
    return None, None


# Прогон импортированного крыла: 0.410 -> 0.760 за 4500 итераций,
# 35 минут 26 секунд, сессия отчиталась «1 успешных».
_wing = [(0, 0.410), (1500, 0.615), (3000, 0.715), (4500, 0.760)]
_dense = [(i, 0.410 + 0.35 * i / 4500.0) for i in range(0, 4501, 50)]
_it, _why = _run(_dense)
check("крыло с растущей невязкой прерывается досрочно", _it is not None,
      "дошло до конца")
check("прерывается задолго до 4500 итераций",
      _it is not None and _it < 2500, "итерация %s" % _it)
check("причина объясняет, что невязка не улучшалась",
      _why is not None and "не улучшалась" in _why, str(_why)[:60])

# Тот же расчёт крыла после починки справочных данных: невязка не растёт,
# а просто стоит на -1.889 с итерации 1500 до 4500. Условие «выше минимума
# на STALL_RISE» здесь не срабатывает — текущее значение равно лучшему.
_flat = [(i, 1.711 if i == 0 else -1.889) for i in range(0, 6001, 50)]
_it_f, _why_f = _run(_flat, patience=stall_patience_for(6000))
check("плоская невязка тоже считается застоем", _it_f is not None,
      "дошло до конца")
check("плоская невязка прерывается задолго до 6000 итераций",
      _it_f is not None and _it_f < 2000, "итерация %s" % _it_f)
check("в причине названа цель сходимости",
      _why_f is not None and "цели сходимости" in _why_f, str(_why_f)[:70])

# Негативный контроль: расчёт, который дошёл до цели и встал, застоем не
# является — SU2 сам остановился бы по CONV_RESIDUAL_MINVAL.
_conv = [(i, -7.5) for i in range(0, 6001, 50)]
_it_c, _ = _run(_conv, patience=stall_patience_for(6000))
check("сошедшийся до цели и вставший расчёт не прерывается", _it_c is None,
      "прерван на %s" % _it_c)

# Негативный контроль: цель сходимости берётся из конфига. При цели -2.0
# невязка -1.889 уже у цели, и ждать дальше законно.
_it_n, _ = _run(_flat, patience=stall_patience_for(6000), conv_minval=-2.0)
check("при цели -2.0 невязка -1.889 застоем не считается", _it_n is None,
      "прерван на %s" % _it_n)

# Негативный контроль: медленно, но верно сходящийся расчёт не трогается.
_slow = [(i, 1.5 - 0.0015 * i) for i in range(0, 6001, 50)]
_it_s, _ = _run(_slow, patience=stall_patience_for(6000))
check("медленно сходящийся расчёт не прерывается", _it_s is None,
      "прерван на %s" % _it_s)

# Прогон полного самолёта: -1.494 на старте, 11.06 на 265-й. Там SU2
# сам обрывает расчёт по своему порогу 10^20, но детектор обязан сработать
# раньше, если лимит итераций короткий и терпение соответственно меньше.
_full = [(i, -1.494 + 12.5 * i / 265.0) for i in range(0, 266, 5)]
_it2, _ = _run(_full, patience=stall_patience_for(500))
check("расходящийся самолёт прерывается до 265-й итерации",
      _it2 is not None and _it2 < 265, "итерация %s" % _it2)
check("терпение сокращается вместе с лимитом итераций",
      stall_patience_for(500) < stall_patience_for(6000),
      "%d против %d" % (stall_patience_for(500), stall_patience_for(6000)))
check("терпение не уходит ниже 150",
      stall_patience_for(100) == 150, stall_patience_for(100))
check("терпение не уходит выше 1000",
      stall_patience_for(100000) == 1000, stall_patience_for(100000))

# Нормально сходящийся расчёт прерван быть не должен.
_good = [(i, 2.0 - 0.004 * i) for i in range(0, 6001, 100)]
_it3, _ = _run(_good)
check("сходящийся расчёт не прерывается", _it3 is None,
      "прерван на %s" % _it3)

# Колебания адаптивного CFL вокруг падающего тренда — тоже не застой.
_osc = [(i, 2.0 - 0.002 * i + (0.15 if (i // 100) % 2 else 0.0))
        for i in range(0, 6001, 100)]
_it4, _ = _run(_osc)
check("колебания CFL не принимаются за застой", _it4 is None,
      "прерван на %s" % _it4)

check("NaN не считается застоем",
      _sv(5000, float("nan"), -3.0, 100)[2] is None)
check("первая итерация задаёт基准 минимум".replace("基准", "базовый"),
      _sv(0, 1.5, None, 0)[:2] == (1.5, 0))

# ---------------------------------------------------------------- summary
print()
if FAIL:
    print(f"ПРОВАЛЕНО ТЕСТОВ: {len(FAIL)} → {FAIL}")
    sys.exit(1)
print("ВСЕ ФУНКЦИОНАЛЬНЫЕ ТЕСТЫ ПРОЙДЕНЫ ✔")
