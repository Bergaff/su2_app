# -*- coding: utf-8 -*-
"""
official_cases/catalog.py — каталог официальных тест-кейсов SU2.

Содержит реестр :data:`OFFICIAL_CASES` — библиотеку эталонных расчётов,
взятых из официальных репозиториев SU2 (su2code/SU2 и su2code/Tutorials,
ветка master):

  * ``TestCases/``  — регрессионные кейсы SU2 с эталонными значениями
    из ``TestCases/serial_regression.py`` (те самые числа «от разработчиков»);
  * ``Tutorials/``  — обучающие кейсы с готовыми mesh-файлами (3D-геометрия),
    которые SU2 выкладывает отдельным репозиторием.

Для пользователя AeroOpt это прежде всего **калибровка**: если расчёт даёт
неправдоподобно большие Cl/Cd, сравните свой ``config.cfg`` с ближайшим
официальным кейсом (``official_cases.compare``) — почти всегда причина в
численной схеме или нормировке, а не в геометрии.

Всё, что здесь есть, — stdlib-only (без numpy/Qt), поэтому модуль можно
импортировать в любом окружении и в тестах.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional

# Репозитории, откуда берутся конфиги и сетки.
REPO_SU2 = "su2code/SU2"
REPO_TUTORIALS = "su2code/Tutorials"

# ---------------------------------------------------------------------------
# Модель данных
# ---------------------------------------------------------------------------


@dataclass
class OfficialCase:
    """Описание одного официального кейса SU2.

    Поля ``mesh_*`` позволяют ``downloader`` скачать официальную 3D-сетку,
    а ``ref_*`` — эталонные аэродинамические коэффициенты из регрессии SU2.
    """

    id: str                      # короткий идентификатор (slug)
    name: str                    # человеческое имя
    description: str             # что за кейс и зачем он нужен
    dimension: int               # 2 или 3 (измерения геометрии)
    solver: str                  # SOLVER= (EULER / RANS / INC_EULER / INC_RANS)
    config_file: str             # файл в official_cases/configs/
    source_path: str             # путь в официальном репозитории
    repo: str = REPO_SU2         # официальный репозиторий-источник
    su2_version: str = ""        # версия SU2 из шапки конфига

    # --- геометрия / режим ---------------------------------------------
    mach: Optional[float] = None
    aoa: Optional[float] = None
    reynolds: Optional[float] = None

    # --- эталонные коэффициенты (из регрессии SU2) ---------------------
    ref_cl: Optional[float] = None
    ref_cd: Optional[float] = None
    ref_cm: Optional[float] = None
    ref_iter: Optional[int] = None   # итерация, на которой сняты ref_*
    ref_source: str = ""             # откуда взяты числа

    # --- официальная сетка (3D-модель) ---------------------------------
    mesh_filename: str = ""          # имя, которое ждёт конфиг (MESH_FILENAME)
    mesh_path: str = ""              # путь к сетке в меш-репозитории
    mesh_repo: str = REPO_TUTORIALS  # репозиторий, где лежит сетка
    mesh_size: int = 0               # размер сетки в байтах (0 — неизвестно)

    notes: str = ""                  # комментарий для пользователя
    tags: list = field(default_factory=list)

    # --- удобные свойства ----------------------------------------------
    @property
    def config_path(self) -> str:
        """Абсолютный путь к встроенному config.cfg."""
        return os.path.join(_configs_dir(), self.config_file)

    @property
    def label(self) -> str:
        return (f"SU2 {self.solver} — {self.name} ({self.dimension}D)")

    def to_dict(self) -> dict:
        return asdict(self)


def _configs_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


# ---------------------------------------------------------------------------
# Реестр официальных кейсов
# ---------------------------------------------------------------------------
#
# ``ref_cl/ref_cd/ref_cm`` — числа из ``TestCases/serial_regression.py`` SU2
# (это и есть официальные ожидаемые значения). ``ref_iter`` указывает, на
# какой итерации регрессия их снимает: у ряда RANS/невязких кейсов это
# далеко от полной сходимости, поэтому читайте их вместе с ``ref_source``.
#
OFFICIAL_CASES: dict = {
    # ---------------- 3D: крыло / самолёт ---------------------------------
    "inv_oneram6": OfficialCase(
        id="inv_oneram6",
        name="ONERA M6 — невязкое обтекание крыла",
        description=(
            "Классическое 3D-крыло ONERA M6 в трансзвуковом потоке "
            "(M=0.8395, AoA=3.06°). Эталонный кейс SU2 для проверки "
            "невязкой внешней аэродинамики; хорошо сходится (JST + "
            "многосеточный метод)."
        ),
        dimension=3,
        solver="EULER",
        config_file="inv_ONERAM6.cfg",
        source_path="compressible_flow/Inviscid_ONERAM6/inv_ONERAM6.cfg",
        repo=REPO_TUTORIALS,
        su2_version="5.0.0",
        mach=0.8395, aoa=3.06,
        ref_cl=0.280800, ref_cd=0.008623, ref_iter=10,
        ref_source="SU2 TestCases/serial_regression.py: oneram6",
        mesh_filename="mesh_ONERAM6_inv_ffd.su2",
        mesh_path="compressible_flow/Inviscid_ONERAM6/mesh_ONERAM6_inv_ffd.su2",
        mesh_repo=REPO_TUTORIALS,
        mesh_size=26636752,
        notes=(
            "REF_AREA= 0 — SU2 сам считает площадь по сетке. Полезно как "
            "репер для нормировки: у ONERA M6 размах b=1.196 м, "
            "Sref≈0.7·b·MAC."
        ),
        tags=["3d", "wing", "euler", "transonic", "inviscid"],
    ),
    "turb_oneram6": OfficialCase(
        id="turb_oneram6",
        name="ONERA M6 — вязкое обтекание (RANS, SA)",
        description=(
            "То же крыло ONERA M6, но с моделью турбулентности Spalart–"
            "Allmaras (Re=11.72e6, M=0.8395, AoA=3.06°). Калибровочный "
            "RANS-кейс для крыла; Cd здесь уже имеет физический смысл."
        ),
        dimension=3,
        solver="RANS",
        config_file="turb_ONERAM6.cfg",
        source_path="compressible_flow/Turbulent_ONERAM6/turb_ONERAM6.cfg",
        repo=REPO_TUTORIALS,
        su2_version="8.x",
        mach=0.8395, aoa=3.06, reynolds=11.72e6,
        ref_cl=0.238581, ref_cd=0.158951, ref_iter=10,
        ref_source=(
            "SU2 TestCases/serial_regression.py: turb_oneram6 "
            "(снято на 10-й итерации — далеко от сходимости)"
        ),
        mesh_filename="mesh_ONERAM6_turb_hexa_43008.su2",
        mesh_path="compressible_flow/Turbulent_ONERAM6/mesh_ONERAM6_turb_hexa_43008.su2",
        mesh_repo=REPO_TUTORIALS,
        mesh_size=5959428,
        notes=(
            "REYNOLDS_LENGTH= REF_LENGTH= 0.64607 м. Сетка 43 тыс. "
            "гексаэдров — быстрый вариант для сравнения с вашим крылом."
        ),
        tags=["3d", "wing", "rans", "sa", "transonic"],
    ),
    "inv_crm": OfficialCase(
        id="inv_crm",
        name="NASA CRM — невязкое обтекание самолёта (JST)",
        description=(
            "NASA Common Research Model — трансзвуковой транспортный "
            "самолёт (фюзеляж + крыло + горизонтальное оперение), "
            "M=0.8395, AoA=3.06°. Полносамолётный невязкий кейс: полезен, "
            "чтобы проверить, что нормировка на площадь крыла и знак "
            "сопротивления заданы правильно, когда в сетке не одно крыло."
        ),
        dimension=3,
        solver="EULER",
        config_file="inv_CRM_JST.cfg",
        source_path="TestCases/euler/CRM/inv_CRM_JST.cfg",
        repo=REPO_SU2,
        su2_version="8.x",
        mach=0.8395, aoa=3.06,
        ref_cl=None, ref_cd=None, ref_iter=None,
        ref_source="SU2 TestCases/euler/CRM — проверьте на вашей сетке",
        mesh_filename="grid_crm_dpw4_MB-structured.su2",
        mesh_path="",
        mesh_repo=REPO_TUTORIALS,
        mesh_size=0,
        notes=(
            "Официальная сетка CRM не выложена в su2code/Tutorials, поэтому "
            "скачать её автоматически нельзя (mesh_path пуст). Конфиг "
            "полезен как эталон численной схемы и нормировки для "
            "полносамолётной сборки."
        ),
        tags=["3d", "aircraft", "euler", "transonic", "full-aircraft"],
    ),

    # ---------------- 2D: профиль (валидация) ----------------------------
    "inv_naca0012": OfficialCase(
        id="inv_naca0012",
        name="NACA0012 — невязкий профиль (JST)",
        description=(
            "Классический профиль NACA0012 в трансзвуковом потоке "
            "(M=0.8, AoA=1.25°). Быстрый 2D-кейс для проверки схемы и "
            "нормировки Cl/Cd без вязкости."
        ),
        dimension=2,
        solver="EULER",
        config_file="inv_NACA0012.cfg",
        source_path="TestCases/euler/naca0012/inv_NACA0012.cfg",
        repo=REPO_SU2,
        su2_version="8.x",
        mach=0.8, aoa=1.25,
        ref_cl=None, ref_cd=None, ref_iter=None,
        ref_source="SU2 TestCases/euler/naca0012 — проверьте на вашей сетке",
        mesh_filename="mesh_NACA0012_inv.su2",
        mesh_path="design/Inviscid_2D_Unconstrained_NACA0012/mesh_NACA0012_inv.su2",
        mesh_repo=REPO_TUTORIALS,
        mesh_size=485272,
        notes=(
            "REF_AREA= 1.0 (2D, хорда=1). Невязкий Cd теоретически "
            "стремится к нулю — если у вас он большой при M<0.3, это "
            "численная вязкость, а не физика."
        ),
        tags=["2d", "airfoil", "euler", "transonic"],
    ),
    "turb_naca0012_sa": OfficialCase(
        id="turb_naca0012_sa",
        name="NACA0012 — вязкий профиль (RANS, SA)",
        description=(
            "NACA0012, модель SA, M=0.15, AoA=10°, Re=6e6. Канонический "
            "низкомаховый RANS-кейс: демонстрирует, что сжимаемый решатель "
            "на M=0.15 работает корректно при хорошей сетке и высоком CFL."
        ),
        dimension=2,
        solver="RANS",
        config_file="turb_NACA0012_sa.cfg",
        source_path="TestCases/rans/naca0012/turb_NACA0012_sa.cfg",
        repo=REPO_SU2,
        su2_version="8.x",
        mach=0.15, aoa=10.0, reynolds=6.0e6,
        ref_cl=1.080346, ref_cd=0.018385, ref_iter=5,
        ref_source=(
            "SU2 TestCases/serial_regression.py: turb_naca0012_sa; "
            "FUN3D (тончайшая сетка) CL=1.0983, Cd=0.01242"
        ),
        mesh_filename="n0012_225-65.su2",
        mesh_path="compressible_flow/UQ_NACA0012/mesh_n0012_225-65.su2",
        mesh_repo=REPO_TUTORIALS,
        mesh_size=1253807,
        notes=(
            "Пример «низкомахового» расчёта в сжимаемой постановке: "
            "CONV_NUM_METHOD_FLOW= ROE, MUSCL_FLOW= YES, CFL_NUMBER= 1000 "
            "без адаптации CFL. Прямо противоположно тем настройкам, "
            "которые в AeroOpt включаются по умолчанию."
        ),
        tags=["2d", "airfoil", "rans", "sa", "low-mach"],
    ),
    "turb_rae2822_sa": OfficialCase(
        id="turb_rae2822_sa",
        name="RAE2822 — трансзвуковой профиль (RANS, SA)",
        description=(
            "Трансзвуковой профиль RAE2822 (case 6): M=0.729, AoA=2.31°, "
            "Re=6.5e6, модель SA. Один из самых известных верификационных "
            "кейсов для вязкого трансзвукового обтекания."
        ),
        dimension=2,
        solver="RANS",
        config_file="turb_SA_RAE2822.cfg",
        source_path="TestCases/rans/rae2822/turb_SA_RAE2822.cfg",
        repo=REPO_SU2,
        su2_version="8.x",
        mach=0.729, aoa=2.31, reynolds=6.5e6,
        ref_cl=0.287676, ref_cd=0.104861, ref_iter=20,
        ref_source=(
            "SU2 TestCases/serial_regression.py: rae2822_sa "
            "(снято на 20-й итерации — не сходимость); "
            "эталон case 6 ≈ CL 0.74, Cd 0.013"
        ),
        mesh_filename="mesh_RAE2822_turb.su2",
        mesh_path="design/Turbulent_2D_Constrained_RAE2822/mesh_RAE2822_turb.su2",
        mesh_repo=REPO_TUTORIALS,
        mesh_size=1101375,
        notes=(
            "LINEAR_SOLVER= BCGSTAB с LINEAR_SOLVER_ERROR= 1E-1 — "
            "AeroOpt использует FGMRES+ILU, это приемлемое отличие."
        ),
        tags=["2d", "airfoil", "rans", "sa", "transonic"],
    ),

    # ---------------- Низкомаховые (incompressible) ----------------------
    "inc_naca0012": OfficialCase(
        id="inc_naca0012",
        name="NACA0012 — невязкий, несжимаемый (INC_EULER)",
        description=(
            "Несжимаемое невязкое обтекание NACA0012. Официальный способ "
            "SU2 считать на малых скоростях (M≈0.1 и ниже) — вместо "
            "сжимаемого EULER, который на таких режимах даёт большую "
            "несжимаемую/численную добавку к Cd."
        ),
        dimension=2,
        solver="INC_EULER",
        config_file="incomp_NACA0012.cfg",
        source_path="TestCases/incomp_euler/naca0012/incomp_NACA0012.cfg",
        repo=REPO_SU2,
        su2_version="8.x",
        mach=None, aoa=None,
        ref_cl=0.519589, ref_cd=0.008977, ref_iter=20,
        ref_source="SU2 TestCases/serial_regression.py: inc_euler_naca0012",
        mesh_filename="mesh_NACA0012_5deg_6814.su2",
        mesh_path="incompressible_flow/Inc_Inviscid_Hydrofoil/mesh_NACA0012_5deg_6814.su2",
        mesh_repo=REPO_TUTORIALS,
        mesh_size=326569,
        notes=(
            "Ключевые отличия от сжимаемого кейса: SOLVER= INC_EULER, "
            "INC_DENSITY_INIT, INC_VELOCITY_INIT, INC_NONDIM= INITIAL_VALUES, "
            "CONV_NUM_METHOD_FLOW= FDS. Именно это сулит правдоподобный Cl/Cd "
            "на малых скоростях."
        ),
        tags=["2d", "airfoil", "incompressible", "low-speed"],
    ),
    "inc_turb_naca0012": OfficialCase(
        id="inc_turb_naca0012",
        name="NACA0012 — вязкий, несжимаемый (INC_RANS, SA)",
        description=(
            "Несжимаемое вязкое обтекание NACA0012 (SA). Это то, что "
            "нужно для типичного режима AeroOpt (V≈60 м/с у земли, "
            "M≈0.18): несжимаемый решатель не даёт искусственной "
            "акустической жёсткости, из-за которой сжимаемый EULER/RANS "
            "завышает Cd."
        ),
        dimension=2,
        solver="INC_RANS",
        config_file="inc_turb_NACA0012.cfg",
        source_path="incompressible_flow/Inc_Turbulent_NACA0012/turb_naca0012.cfg",
        repo=REPO_TUTORIALS,
        su2_version="8.x",
        mach=None, aoa=None, reynolds=None,
        ref_cl=None, ref_cd=None, ref_iter=None,
        ref_source="SU2 Tutorials: Inc_Turbulent_NACA0012 (проверьте на вашей сетке)",
        mesh_filename="n0012_897-257.su2",
        mesh_path="incompressible_flow/Inc_Turbulent_NACA0012/n0012_897-257.su2",
        mesh_repo=REPO_TUTORIALS,
        mesh_size=21594641,
        notes=(
            "INC_VELOCITY_INIT задаёт и модуль, и угол (AoA закодирован в "
            "компонентах скорости). CONV_RESIDUAL_MINVAL= -14 — очень "
            "строгая цель, но несжимаемый решатель её достигает."
        ),
        tags=["2d", "airfoil", "incompressible", "rans", "sa", "low-speed"],
    ),
}


# ---------------------------------------------------------------------------
# Публичные функции доступа
# ---------------------------------------------------------------------------


def list_cases() -> list:
    """Список id всех официальных кейсов (в порядке объявления)."""
    return list(OFFICIAL_CASES.keys())


def get_case(case_id: str) -> OfficialCase:
    """Возвращает кейс по id или кидает KeyError с понятным списком."""
    try:
        return OFFICIAL_CASES[case_id]
    except KeyError:
        raise KeyError(
            f"Неизвестный официальный кейс SU2: {case_id!r}. "
            f"Доступны: {', '.join(list_cases())}"
        ) from None


def find_by_solver(solver: str) -> list:
    """Все кейсы с указанным SOLVER (или вхождением, если 'EULER' → 'INC_EULER')."""
    s = str(solver or "").upper().strip()
    if not s:
        return []
    exact = [c for c in OFFICIAL_CASES.values() if c.solver == s]
    if exact:
        return exact
    return [c for c in OFFICIAL_CASES.values() if s in c.solver]


def find_by_mach(mach: float, tol: float = 0.15) -> list:
    """Кейсы, чей Mach близок к заданному; None (incompressible) не учитывается."""
    try:
        m = float(mach)
    except (TypeError, ValueError):
        return []
    return [c for c in OFFICIAL_CASES.values()
            if c.mach is not None and abs(c.mach - m) <= tol]


def nearest_for(mach: float, solver: str = "") -> list:
    """Ближайшие к режиму кейсы — для сравнения и калибровки.

    Сначала — те же/близкие по SOLVER, затем — по Mach.
    """
    m = float(mach or 0.0)
    by_solver = find_by_solver(solver) if solver else []
    by_mach = find_by_mach(m)
    seen = set()
    out = []
    for c in by_solver + by_mach:
        if c.id in seen:
            continue
        seen.add(c.id)
        out.append(c)
    return out
