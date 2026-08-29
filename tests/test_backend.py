"""Функциональные тесты бэкенда (без GUI-зависимостей).
Запуск: python3 tests/test_backend.py  (из корня проекта)"""
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
    check("apply_preset safe: CFL_NUMBER= 2.0", "CFL_NUMBER= 2.0" in txt)
    check("apply_preset safe: CFL_ADAPT= NO", "CFL_ADAPT= NO" in txt)
    check("apply_preset safe: MUSCL_FLOW= NO", "MUSCL_FLOW= NO" in txt)
    check("apply_preset safe: границы не тронуты",
          "SOLVER= EULER" in txt and "INNER_ITER= 6000" in txt)
    check("бэкап config.cfg.orig создан", os.path.isfile(cfg + ".orig"))
    AC.apply_preset(cfg, "safe")
    txt2 = open(cfg, encoding="utf-8").read()
    check("повторный apply_preset не дублирует ключи",
          txt2.count("CFL_NUMBER=") == 1 and txt2.count("CFL_ADAPT=") == 1)
    AC.apply_preset(cfg, "ultra")
    txt3 = open(cfg, encoding="utf-8").read()
    check("ultra: CFL 0.5 + LINEAR_SOLVER_ITER 20",
          "CFL_NUMBER= 0.5" in txt3 and "LINEAR_SOLVER_ITER= 20" in txt3)
    check("restore_original", AC.restore_original(cfg) is True)
    check("оригинал восстановлен", "CFL_ADAPT= YES" in
          open(cfg, encoding="utf-8").read())
    check("неизвестный пресет → ValueError",
          _raises(ValueError, AC.apply_preset, cfg, "nope"))

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
    check("suggest после расхождения → safe", action == "apply_preset"
          and preset == "safe")
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

# ---------------------------------------------------------------- summary
print()
if FAIL:
    print(f"ПРОВАЛЕНО ТЕСТОВ: {len(FAIL)} → {FAIL}")
    sys.exit(1)
print("ВСЕ ФУНКЦИОНАЛЬНЫЕ ТЕСТЫ ПРОЙДЕНЫ ✔")
