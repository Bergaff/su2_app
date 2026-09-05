# solver/config_builder.py
"""Построение и запись конфигураций SU2.

Файл сохраняет существующие шаблоны Euler/RANS и восстанавливает публичные
функции, которые импортируются из solver.__init__ и solver.workers:

    build_su2_config
    write_su2_config
    write_case_config
"""

from __future__ import annotations

import math
import os
from typing import Iterable, Mapping, Optional, Sequence


def format_marker_list(markers: Optional[Iterable[str]]) -> str:
    """Форматирует маркеры как ``( tag1, tag2 )`` для SU2."""
    clean = [str(tag).strip() for tag in (markers or []) if str(tag).strip()]
    if not clean:
        clean = ["airfoil"]
    return f"( {', '.join(clean)} )"


def _format_marker_value_pairs(markers: Optional[Iterable[str]],
                               value: float = 0.0) -> str:
    """Форматирует пары ``маркер, значение`` для MARKER_HEATFLUX."""
    clean = [str(tag).strip() for tag in (markers or []) if str(tag).strip()]
    if not clean:
        clean = ["airfoil"]
    pairs = []
    for tag in clean:
        pairs.extend((tag, f"{float(value):g}"))
    return f"( {', '.join(pairs)} )"


# Потолок CFL. Формат CFL_ADAPT_PARAM в SU2 —
# (коэффициент вниз, коэффициент вверх, CFL минимум, CFL максимум).
#
# Старое значение ( 0.5, 1.2, 0.5, 5.0 ) не давало CFL подняться выше 5.0.
# Для неявной схемы EULER_IMPLICIT это крайне робкий шаг: официальный
# туториал SU2 по невязкому обтеканию ONERA M6 — тот же класс задач —
# использует ( 0.1, 2.0, 100.0, 1e10 ), а туториал по соплу
# ( 0.1, 2.0, 10.0, 1000.0 ). То есть прежний потолок был в 20 раз ниже
# минимума из туториала.
#
# Агрессивный вариант включается отдельным флагом, а не подменяет
# осторожный по умолчанию: на сетке, которая не разрешает геометрию,
# высокий CFL не ускоряет расчёт, а роняет его быстрее. Пока маркер
# airfoil берётся из лестницы объёмной сетки, безопаснее остаться на 5.0.
# Порог, ниже которого сжимаемый решатель становится жёстким по акустике:
# звуковые волны идут в 1/M раз быстрее потока, поэтому невязка плотности
# почти не падает, как бы долго ни шёл расчёт. На V=60 м/с у земли это
# M=0.176 — то есть типичный режим AeroOpt, а не редкий случай.
LOW_MACH_THRESHOLD = 0.3


# Автоматическое включение прекондиционера ВЫКЛЮЧЕНО. Проверено прогоном:
# базовый конфиг без него на M=0.176 не расходился, а стоял на
# log10(rms[Rho]) = -2.34; тот же конфиг с LOW_MACH_PREC/CORR= YES
# разошёлся на 1111-й итерации (Residual > 10^20). Официальный кейс
# TestCases/euler/turbofan_MFR_coupling использует этот прекондиционер на
# M=0.167, но со схемой JST и MUSCL_FLOW= NO, а у нас ROE с MUSCL_FLOW=
# YES — сочетание, которое ни один кейс не подтверждает. Опция оставлена
# в диалоге настроек SU2: включать её нужно осознанно, одним прогоном, а
# не по умолчанию.
LOW_MACH_AUTO = False


def low_mach_lines(mach, auto=LOW_MACH_AUTO) -> str:
    """Строки прекондиционирования для малых чисел Маха.

    ``LOW_MACH_PREC`` — прекондиционер Roe-Turkel, снимает жёсткость
    системы по акустике; ``LOW_MACH_CORR`` — поправка после реконструкции
    MUSCL на избыточную диссипацию схемы Роу. Обе в SU2 по умолчанию
    выключены (Common/src/CConfig.cpp:1920 и 1922), а CONV_NUM_METHOD_FLOW=
    ROE задан у нас явно — то есть это ровно тот случай, для которого они
    сделаны.

    Без них на M=0.176 невязка встаёт на log10(res) ~ -2.3 и не идёт ниже
    даже за 6000 итераций, а сопротивление выходит в разы больше
    индуктивного: схема Роу на низких махах вносит лишнюю вязкость.
    """
    if not auto:
        return "LOW_MACH_PREC= NO\nLOW_MACH_CORR= NO"
    try:
        m = float(mach)
    except (TypeError, ValueError):
        m = 0.0
    on = "YES" if 0.0 < m < LOW_MACH_THRESHOLD else "NO"
    return "LOW_MACH_PREC= %s\nLOW_MACH_CORR= %s" % (on, on)


# Итераций линейного решателя. В официальных кейсах SU2 это 2..5, но там
# препроцессор LU_SGS, а у нас ILU: при меньшем числе итераций ньютоновский
# шаг решается грубее. Снижение до 5 было проверено прогоном вместе с
# прекондиционером и дало расходимость, поэтому возвращено проверенное 15.
LINEAR_SOLVER_ITER = 15

CFL_PARAM_SAFE = "( 0.5, 1.2, 0.5, 5.0 )"
CFL_PARAM_FAST = "( 0.1, 2.0, 10.0, 1000.0 )"


def inner_iter_estimate(mesh_quality=None, n_bodies=1, n_points=0,
                        is_rans=False):
    """Оценка потолка итераций и пояснение, из чего она получилась.

    Прежняя привязка была только к качеству сетки: Грубая 2000, Средняя
    6000, Точная 12000 — и не зависела от того, что именно считается.
    Одиночное крыло получало те же 6000 итераций, что и самолёт из пяти
    компонентов с механизацией, хотя сошёлось бы за пятую часть.

    Здесь число компонентов — главный множитель, а размер сетки влияет
    слабо (в четвёртой степени), потому что сам по себе он плохой
    прогнозатор: грубая сетка сложной формы требует больше итераций, чем
    мелкая сетка простого тела. RANS добавляет полтора раза — уравнения
    турбулентности сходятся медленнее.

    Важно, что это ПОТОЛОК, а не цель. Сходящийся расчёт останавливается
    сам на CONV_RESIDUAL_MINVAL, а расходящийся теперь прерывается
    детектором застоя в solver/workers.py. Поэтому значение можно брать с
    запасом: оно почти не влияет на время успешного расчёта.

    Возвращает (число, пояснение).
    """
    base = _inner_iter_for_quality(mesh_quality)

    nb = max(1, int(n_bodies or 1))
    # Один компонент — множитель 1.0, дальше +0.15 за каждый, потолок 1.8.
    complexity = min(1.8, 1.0 + 0.15 * (nb - 1))

    np_ = max(0, int(n_points or 0))
    if np_ > 0:
        # Зависимость намеренно слабая: размер сетки — плохой прогнозатор,
        # грубая сетка сложной формы требует больше итераций, чем мелкая
        # сетка простого тела. Степень 0.12 и узкий коридор 0.90..1.25
        # дают поправку, а не вторую привязку к сетке.
        mesh_factor = float(min(1.25, max(0.90, (np_ / 100000.0) ** 0.12)))
    else:
        mesh_factor = 1.0

    turb = 1.5 if is_rans else 1.0

    value = int(round(base * complexity * mesh_factor * turb / 500.0) * 500)
    # Жёсткий потолок: при «Точной» на семи компонентах с RANS оценка
    # уходит за 48 тысяч, а столько не нужен ни один осмысленный прогон.
    value = max(500, min(20000, value))

    parts = ["база %d" % base, "компонентов %d (x%.2f)" % (nb, complexity)]
    if np_ > 0:
        parts.append("узлов %d (x%.2f)" % (np_, mesh_factor))
    if is_rans:
        parts.append("RANS (x1.5)")
    return value, ", ".join(parts)


def _inner_iter_for_quality(mesh_quality) -> int:
    """Подбор INNER_ITER в зависимости от качества сетки.

    Грубая  — быстрая оценка, хватает 2000 итераций.
    Средняя — рабочий режим, 6000.
    Точная  — прецизионный расчёт, до 12000.

    Можно передать None или неизвестную строку — будет 6000 (default).
    """
    if not mesh_quality:
        return 6000
    q = str(mesh_quality).strip().lower()
    if "груб" in q or "coarse" in q or "rough" in q:
        return 2000
    if "точн" in q or "fine" in q or "high" in q:
        return 12000
    # По умолчанию — «средняя»
    return 6000


def _sym_marker_names(planes):
    """Возвращает список имён маркеров симметрии для config.cfg.

    Обратная совместимость: старая плоскость XZ имеет два имени —
    symmetry_plane (старое) и symmetry_xz (новое). Берём оба, чтобы
    config.cfg работал с любой mesh.su2.
    """
    names = []
    for p in (planes or []):
        p = str(p).lower()
        if p == "xz" and "symmetry_xz" not in names:
            # Раньше сюда добавлялось и старое имя symmetry_plane. В
            # mesh.su2 те же грани больше не дублируются под двумя
            # маркерами, поэтому в MARKER_SYM остаётся одно имя.
            names.append("symmetry_xz")
        elif p == "xy":
            names.append("symmetry_xy")
        elif p == "yz":
            names.append("symmetry_yz")
    return names


def build_euler_config(p: Mapping, markers=None, restart: bool = False,
                      mesh_quality=None,
                      use_symmetry: bool = False,
                      symmetry_planes: list = None,
                      enable_cuda: bool = False,
                      n_bodies: int = 0, n_points: int = 0,
                      cfl_aggressive: bool = False) -> str:
    restart_str = "YES" if restart else "NO"
    body_markers = format_marker_list(markers)
    if n_bodies or n_points:
        inner_iter, inner_why = inner_iter_estimate(
            mesh_quality, n_bodies=n_bodies, n_points=n_points, is_rans=False)
    else:
        inner_iter, inner_why = _inner_iter_for_quality(mesh_quality), ""
    cfl_param = CFL_PARAM_FAST if cfl_aggressive else CFL_PARAM_SAFE
    # Адаптация CFL остаётся включённой по умолчанию: именно с ней базовый
    # конфиг дошёл до log10(rms[Rho]) = -2.34 без расходимости, а
    # выключение было проверено прогоном и в расходимость не спасло.
    # В официальных кейсах SU2 адаптация выключена везде (8 из 8), но там
    # другие сетки; менять проверенное на рекомендованное сразу нельзя.
    cfl_adapt = "YES"
    lin_iter = LINEAR_SOLVER_ITER
    lm_lines = low_mach_lines(p['mach'])
    # === T1: MARKER_SYM — плоскости симметрии ========================
    # Если включены плоскости (XY/XZ/YZ) и в сетке есть
    # соответствующие маркеры, SU2 посчитает только симметричную
    # часть модели. Ускорение: 1 плоскость ~1.8x, 2 плоскости ~3.5x,
    # 3 плоскости (полная симметрия) ~7x.
    if symmetry_planes is None and use_symmetry:
        symmetry_planes = ["xz"]  # обратная совместимость
    sym_names = _sym_marker_names(symmetry_planes)
    if sym_names:
        sym_line = "MARKER_SYM= ( " + " ".join(sym_names) + " )"
    else:
        sym_line = "% MARKER_SYM= ( symmetry_xy symmetry_xz symmetry_yz )  # выключено"
    # ==================================================================

    # ENABLE_CUDA включает в SU2 GPU-ветку произведения матрицы на вектор
    # (CMatrixVectorProduct.hpp). Без неё видеокарта не используется,
    # даже если SU2_CFD собран с -Denable-cuda=true.
    cuda_line = ("ENABLE_CUDA= YES" if enable_cuda
                 else "% ENABLE_CUDA= NO   # выключено")

    return f"""SOLVER= EULER
MATH_PROBLEM= DIRECT
RESTART_SOL= {restart_str}
SOLUTION_FILENAME= restart.dat
RESTART_FILENAME= restart.dat
MESH_FILENAME= {os.path.basename(str(p.get('MESH_FILENAME', 'mesh.su2')))}
MESH_FORMAT= SU2
MACH_NUMBER= {float(p['mach']):.6f}
AOA= {float(p['aoa']):.8g}
SIDESLIP_ANGLE= 0.0
FREESTREAM_PRESSURE= {float(p['pressure']):.3f}
FREESTREAM_TEMPERATURE= {float(p['temperature']):.3f}
REF_LENGTH= {float(p['ref_length']):.12g}
REF_AREA= {float(p['ref_area']):.12g}
REF_ORIGIN_MOMENT_X= {float(p['ox']):.12g}
REF_ORIGIN_MOMENT_Y= {float(p['oy']):.12g}
REF_ORIGIN_MOMENT_Z= {float(p['oz']):.12g}
MARKER_EULER= {body_markers}
{cuda_line}
MARKER_FAR= ( farfield )
{sym_line}
MARKER_MONITORING= {body_markers}
MARKER_PLOTTING= {body_markers}
CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.05
ENTROPY_FIX_COEFF= 0.005
NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
{lm_lines}
CFL_NUMBER= 1.0
CFL_ADAPT= {cfl_adapt}
CFL_ADAPT_PARAM= {cfl_param}
TIME_DISCRE_FLOW= EULER_IMPLICIT
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1e-6
LINEAR_SOLVER_ITER= {lin_iter}
INNER_ITER= {inner_iter}
CONV_RESIDUAL_MINVAL= -7
CONV_STARTITER= 50
CONV_CAUCHY_ELEMS= 100
CONV_CAUCHY_EPS= 1e-5
SCREEN_OUTPUT= (INNER_ITER, RMS_DENSITY, LIFT, DRAG)
HISTORY_OUTPUT= (INNER_ITER, RMS_RES, AERO_COEFF)
OUTPUT_FILES= (RESTART, PARAVIEW, SURFACE_PARAVIEW)
% SU2 7.x appends .vtu to these names itself — write name without extension.
VOLUME_FILENAME= flow
SURFACE_FILENAME= surface_flow
SCREEN_WRT_FREQ_INNER= 50
HISTORY_WRT_FREQ_INNER= 1
OUTPUT_WRT_FREQ= 100
"""


def build_rans_config(p: Mapping, markers=None, restart: bool = False,
                      mesh_quality=None,
                      turb_model: str = "SA",
                      use_symmetry: bool = False,
                      symmetry_planes: list = None,
                      use_ramp_aoa: bool = False,
                      enable_cuda: bool = False,
                      n_bodies: int = 0, n_points: int = 0,
                      cfl_aggressive: bool = False) -> str:
    """Шаблон config.cfg для RANS.

    Параметры:
        turb_model      — "SA" (Spalart-Allmaras, по умолчанию) или "SST" (Menter SST).
        use_symmetry    — обратная совместимость: True → плоскость XZ.
        symmetry_planes — список плоскостей ["xy", "xz", "yz"].
        use_ramp_aoa    — добавить RAMP_AOA_*: плавный разгон от 0° до AOA
                          за первые 100 итераций (улучшает сходимость на
                          жёстких моделях и высоких углах атаки).
    """
    restart_str = "YES" if restart else "NO"
    body_markers = format_marker_list(markers)
    heatflux_markers = _format_marker_value_pairs(markers, 0.0)
    if n_bodies or n_points:
        inner_iter, inner_why = inner_iter_estimate(
            mesh_quality, n_bodies=n_bodies, n_points=n_points, is_rans=True)
    else:
        inner_iter, inner_why = _inner_iter_for_quality(mesh_quality), ""
    cfl_param = CFL_PARAM_FAST if cfl_aggressive else CFL_PARAM_SAFE
    # Адаптация CFL остаётся включённой по умолчанию: именно с ней базовый
    # конфиг дошёл до log10(rms[Rho]) = -2.34 без расходимости, а
    # выключение было проверено прогоном и в расходимость не спасло.
    # В официальных кейсах SU2 адаптация выключена везде (8 из 8), но там
    # другие сетки; менять проверенное на рекомендованное сразу нельзя.
    cfl_adapt = "YES"
    lin_iter = LINEAR_SOLVER_ITER
    lm_lines = low_mach_lines(p['mach'])

    # === ENABLE_CUDA: GPU-ветка произведения матрицы на вектор =========
    cuda_line = "ENABLE_CUDA= YES" if enable_cuda else "% ENABLE_CUDA= NO   # выключено"
    # ====================================================================

    # === T1: MARKER_SYM — плоскости симметрии ========================
    if symmetry_planes is None and use_symmetry:
        symmetry_planes = ["xz"]
    sym_names = _sym_marker_names(symmetry_planes)
    if sym_names:
        sym_line = "MARKER_SYM= ( " + " ".join(sym_names) + " )"
    else:
        sym_line = "% MARKER_SYM= ( symmetry_xy symmetry_xz symmetry_yz )  # выключено"
    # ====================================================================

    # === T4: RAMP-функции для AoA =======================================
    # Плавно наращиваем угол атаки от 0 до AOA за первые 100 итераций.
    # Это даёт устойчивую сходимость на жёстких конфигурациях
    # (закрылки, высокие AoA, плохо обусловленные сетки).
    if use_ramp_aoa:
        ramp_block = (
            "RAMP_AOA_1= 0\n"
            "RAMP_AOA_2= 50\n"
            "RAMP_AOA_3= 100"
        )
    else:
        ramp_block = "% RAMP_AOA_1= 0   RAMP_AOA_2= 50   RAMP_AOA_3= 100   # выключено"
    # ====================================================================

    # === T4: SST vs SA ==================================================
    turb_upper = str(turb_model or "SA").strip().upper()
    if turb_upper in ("SST", "MENTER", "KW", "K-OMEGA", "K_OMEGA"):
        turb_block = (
            "KIND_TURB_MODEL= SST\n"
                        "CONV_NUM_METHOD_TURB= ROE\n"
            "MUSCL_TURB= YES\n"
            "SLOPE_LIMITER_TURB= VENKATAKRISHNAN\n"
        )
    else:
        turb_block = (
            "KIND_TURB_MODEL= SA\n"
            "CONV_NUM_METHOD_TURB= SCALAR_UPWIND\n"
            "MUSCL_TURB= NO\n"
            "CFL_REDUCTION_TURB= 0.5\n"
        )
    # ====================================================================

    return f"""SOLVER= RANS
{turb_block}MATH_PROBLEM= DIRECT
RESTART_SOL= {restart_str}
SOLUTION_FILENAME= restart.dat
RESTART_FILENAME= restart.dat
MESH_FILENAME= {os.path.basename(str(p.get('MESH_FILENAME', 'mesh.su2')))}
MESH_FORMAT= SU2
MACH_NUMBER= {float(p['mach']):.6f}
AOA= {float(p['aoa']):.8g}
SIDESLIP_ANGLE= 0.0
FREESTREAM_PRESSURE= {float(p['pressure']):.3f}
FREESTREAM_TEMPERATURE= {float(p['temperature']):.3f}
REYNOLDS_NUMBER= {float(p['reynolds']):.6f}
REYNOLDS_LENGTH= {float(p['ref_length']):.12g}
REF_LENGTH= {float(p['ref_length']):.12g}
REF_AREA= {float(p['ref_area']):.12g}
REF_ORIGIN_MOMENT_X= {float(p['ox']):.12g}
REF_ORIGIN_MOMENT_Y= {float(p['oy']):.12g}
REF_ORIGIN_MOMENT_Z= {float(p['oz']):.12g}
{ramp_block}
{cuda_line}
MARKER_HEATFLUX= {heatflux_markers}
MARKER_FAR= ( farfield )
{sym_line}
MARKER_MONITORING= {body_markers}
MARKER_PLOTTING= {body_markers}
CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.05
ENTROPY_FIX_COEFF= 0.005
NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
{lm_lines}
CFL_NUMBER= 1.0
CFL_ADAPT= {cfl_adapt}
CFL_ADAPT_PARAM= {cfl_param}
TIME_DISCRE_FLOW= EULER_IMPLICIT
TIME_DISCRE_TURB= EULER_IMPLICIT
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1e-6
LINEAR_SOLVER_ITER= {lin_iter}
INNER_ITER= {inner_iter}
CONV_RESIDUAL_MINVAL= -7
CONV_STARTITER= 50
CONV_CAUCHY_ELEMS= 100
CONV_CAUCHY_EPS= 1e-5
SCREEN_OUTPUT= (INNER_ITER, RMS_DENSITY, LIFT, DRAG)
HISTORY_OUTPUT= (INNER_ITER, RMS_RES, AERO_COEFF)
OUTPUT_FILES= (RESTART, PARAVIEW, SURFACE_PARAVIEW)
% SU2 7.x appends .vtu to these names itself — write name without extension.
VOLUME_FILENAME= flow
SURFACE_FILENAME= surface_flow
SCREEN_WRT_FREQ_INNER= 50
HISTORY_WRT_FREQ_INNER= 1
OUTPUT_WRT_FREQ= 100
"""


def _inc_velocity(speed: float, aoa_deg: float) -> tuple:
    """Скорость набегающего потока для несжимаемого решателя INC_*.

    SU2 кодирует угол атаки в компонентах ``INC_VELOCITY_INIT``. Ось
    размаха крыла у нас — Y, вертикаль — Z, поэтому AoA поворачивает поток
    в плоскости X-Z: u = V·cos(a), v = 0, w = V·sin(a).
    """
    a = math.radians(float(aoa_deg or 0.0))
    v = float(speed or 0.0)
    return v * math.cos(a), 0.0, v * math.sin(a)


def build_inc_config(p: Mapping, markers=None, restart: bool = False,
                     mesh_quality=None,
                     use_symmetry: bool = False,
                     symmetry_planes: list = None,
                     enable_cuda: bool = False,
                     n_bodies: int = 0, n_points: int = 0,
                     cfl_aggressive: bool = False,
                     solver_name: str = "INC_EULER",
                     turb_model: str = "SA") -> str:
    """Шаблон config.cfg для несжимаемого решателя SU2 (INC_*).

    Это официальный способ SU2 считать внешнее обтекание на малых скоростях
    (M<0.3): сжимаемый EULER/RANS на таких режимах становится жёстким по
    акустике и завышает сопротивление. Ключевые параметры:
        SOLVER= INC_EULER / INC_NAVIER_STOKES / INC_RANS
        INC_DENSITY_INIT, INC_VELOCITY_INIT, INC_NONDIM= INITIAL_VALUES
        CONV_NUM_METHOD_FLOW= FDS   (официальная схема для incompressible)

    ``solver_name`` — одно из ``INC_EULER``, ``INC_NAVIER_STOKES``,
    ``INC_RANS``. Для ``INC_RANS`` используется ``turb_model`` ("SA"/"SST").
    """
    sol = str(solver_name or "INC_EULER").strip().upper()
    viscos = sol in ("INC_NAVIER_STOKES", "INC_RANS")
    turbulent = sol == "INC_RANS"

    restart_str = "YES" if restart else "NO"
    body_markers = format_marker_list(markers)
    if n_bodies or n_points:
        inner_iter, _why = inner_iter_estimate(
            mesh_quality, n_bodies=n_bodies, n_points=n_points,
            is_rans=viscos)
    else:
        inner_iter, _why = _inner_iter_for_quality(mesh_quality), ""
    cfl_param = CFL_PARAM_FAST if cfl_aggressive else CFL_PARAM_SAFE
    cfl_adapt = "YES"
    lin_iter = LINEAR_SOLVER_ITER

    u, v, w = _inc_velocity(p.get("speed", 0.0), p.get("aoa", 0.0))
    density = float(p.get("rho", 1.225) or 1.225)

    # === ENABLE_CUDA: GPU-ветка произведения матрицы на вектор =========
    cuda_line = "ENABLE_CUDA= YES" if enable_cuda else "% ENABLE_CUDA= NO   # выключено"
    # ====================================================================

    # === T1: MARKER_SYM — плоскости симметрии ========================
    if symmetry_planes is None and use_symmetry:
        symmetry_planes = ["xz"]
    sym_names = _sym_marker_names(symmetry_planes)
    if sym_names:
        sym_line = "MARKER_SYM= ( " + " ".join(sym_names) + " )"
    else:
        sym_line = "% MARKER_SYM= ( symmetry_xy symmetry_xz symmetry_yz )  # выключено"
    # ====================================================================

    solver = "INC_RANS" if viscos else "INC_EULER"

    # === T4: SST vs SA (только для турбулентного INC_RANS) ============
    if turbulent and str(turb_model or "SA").strip().upper() in (
            "SST", "MENTER", "KW", "K-OMEGA", "K_OMEGA"):
        turb_block = (
            "KIND_TURB_MODEL= SST\n"
            "CONV_NUM_METHOD_TURB= ROE\n"
            "MUSCL_TURB= YES\n"
            "SLOPE_LIMITER_TURB= VENKATAKRISHNAN\n"
        )
    else:
        turb_block = (
            "KIND_TURB_MODEL= SA\n"
            "CONV_NUM_METHOD_TURB= SCALAR_UPWIND\n"
            "MUSCL_TURB= NO\n"
            "CFL_REDUCTION_TURB= 0.5\n"
        )
    heatflux_markers = _format_marker_value_pairs(markers, 0.0)

    if viscos:
        # Вязкий INC (INC_NAVIER_STOKES / INC_RANS): стенка как heatflux.
        wall_line = f"MARKER_HEATFLUX= {heatflux_markers}"
    else:
        wall_line = f"MARKER_EULER= {body_markers}"

    screen = ("(INNER_ITER, WALL_TIME, RMS_PRESSURE, RMS_VELOCITY-X, "
              "LIFT, DRAG)")
    if turbulent:
        screen = ("(INNER_ITER, WALL_TIME, RMS_PRESSURE, RMS_NU_TILDE, "
                  "LIFT, DRAG)")

    # Вязкость для INC_RANS / INC_NAVIER_STOKES. Без этих строк SU2
    # падает при чтении конфига: по умолчанию VISCOSITY_MODEL= SUTHERLAND,
    # а он для несжимаемого решателя запрещён («Sutherland's law only
    # valid for ideal gases in incompressible flows. Must use
    # VISCOSITY_MODEL=CONSTANT_VISCOSITY and set viscosity with
    # MU_CONSTANT» — замер на SU2 8.5). Динамическую вязкость берём по
    # Сазерленду от заданной температуры потока: при 288.15 K это
    # 1.789e-5 Па·с — табличное значение для воздуха.
    if viscos:
        _t = float(p.get("temperature", 288.15) or 288.15)
        _mu = 1.458e-6 * _t ** 1.5 / (_t + 110.4)
        visc_lines = ("VISCOSITY_MODEL= CONSTANT_VISCOSITY\n"
                      "MU_CONSTANT= %.6E" % _mu)
    else:
        visc_lines = "% INC_EULER вязкость не читает"

    return f"""SOLVER= {sol}
{turb_block if turbulent else ''}MATH_PROBLEM= DIRECT
RESTART_SOL= {restart_str}
SOLUTION_FILENAME= restart.dat
RESTART_FILENAME= restart.dat
MESH_FILENAME= {os.path.basename(str(p.get('MESH_FILENAME', 'mesh.su2')))}
MESH_FORMAT= SU2
INC_DENSITY_INIT= {density:.6f}
INC_VELOCITY_INIT= ( {u:.6f}, {v:.6f}, {w:.6f} )
INC_NONDIM= INITIAL_VALUES
INC_DENSITY_REF= 1.0
INC_VELOCITY_REF= 1.0
INC_TEMPERATURE_REF= 1.0
{visc_lines}
REF_LENGTH= {float(p['ref_length']):.12g}
REF_AREA= {float(p['ref_area']):.12g}
REF_ORIGIN_MOMENT_X= {float(p['ox']):.12g}
REF_ORIGIN_MOMENT_Y= {float(p['oy']):.12g}
REF_ORIGIN_MOMENT_Z= {float(p['oz']):.12g}
{wall_line}
{cuda_line}
MARKER_FAR= ( farfield )
{sym_line}
MARKER_MONITORING= {body_markers}
MARKER_PLOTTING= {body_markers}
CONV_NUM_METHOD_FLOW= FDS
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.05
NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
CFL_NUMBER= 25.0
CFL_ADAPT= {cfl_adapt}
CFL_ADAPT_PARAM= {cfl_param}
TIME_DISCRE_FLOW= EULER_IMPLICIT
TIME_DISCRE_TURB= EULER_IMPLICIT
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1e-6
LINEAR_SOLVER_ITER= {lin_iter}
INNER_ITER= {inner_iter}
CONV_RESIDUAL_MINVAL= -10
CONV_STARTITER= 50
CONV_CAUCHY_ELEMS= 100
CONV_CAUCHY_EPS= 1e-5
SCREEN_OUTPUT= {screen}
HISTORY_OUTPUT= (INNER_ITER, RMS_RES, AERO_COEFF)
OUTPUT_FILES= (RESTART, PARAVIEW, SURFACE_PARAVIEW)
% SU2 7.x appends .vtu to these names itself — write name without extension.
VOLUME_FILENAME= flow
SURFACE_FILENAME= surface_flow
SCREEN_WRT_FREQ_INNER= 50
HISTORY_WRT_FREQ_INNER= 1
OUTPUT_WRT_FREQ= 100
"""


def _unpack_ref_data(ref_data: Sequence[float]):
    if ref_data is None or len(ref_data) != 5:
        raise ValueError(
            "ref_data должен содержать 5 значений: "
            "(ref_length, ref_area, ox, oy, oz)"
        )

    ref_length, ref_area, ox, oy, oz = map(float, ref_data)
    if ref_length <= 0.0:
        raise ValueError("REF_LENGTH должен быть больше нуля")
    if ref_area <= 0.0:
        raise ValueError("REF_AREA должен быть больше нуля")
    return ref_length, ref_area, ox, oy, oz


def _calculate_reynolds(physics: Mapping, ref_length: float) -> float:
    """Возвращает Re из physics либо вычисляет его по rho, speed и mu."""
    explicit = physics.get("reynolds")
    if explicit is not None:
        try:
            value = float(explicit)
            if value > 0.0:
                return value
        except (TypeError, ValueError):
            pass

    rho = float(physics.get("rho", 1.225))
    speed = float(physics.get("speed", 0.0))
    mu = float(physics.get("mu", 1.7894e-5))
    if mu <= 0.0:
        raise ValueError("Динамическая вязкость mu должна быть больше нуля")
    return max(rho * speed * ref_length / mu, 1.0)


def build_su2_config(aoa: float, physics: Mapping, solver: str,
                     ref_data: Sequence[float], markers=None,
                     restart: bool = False,
                     mesh_filename: str = "mesh.su2",
                     mesh_quality=None,
                     use_symmetry: bool = False,
                     symmetry_planes: list = None,
                     turb_model: str = "SA",
                     use_ramp_aoa: bool = False,
                     enable_cuda: bool = False,
                     n_bodies: int = 0, n_points: int = 0,
                     cfl_aggressive: bool = False) -> str:
    """Строит полный текст ``config.cfg`` для одной расчётной точки.

    Параметры T1+T4:
        use_symmetry    — обратная совместимость: True → плоскость XZ.
        symmetry_planes — список плоскостей ["xy", "xz", "yz"].
        turb_model      — "SA" (по умолчанию) или "SST" (Menter SST).
        use_ramp_aoa    — плавный разгон AoA от 0° до нужного за 100 итераций.
        enable_cuda     — писать в config.cfg ``ENABLE_CUDA= YES``.

    Про ``ENABLE_CUDA``: это реальная опция SU2 (CConfig.cpp,
    ``addBoolOption("ENABLE_CUDA", Enable_Cuda, false)``). Она включает
    GPU-ветку в ``CMatrixVectorProduct.hpp`` — вызов
    ``matrix.GPUMatrixVectorProduct`` вместо ``matrix.MatrixVectorProduct``.
    Без неё SU2 не трогает видеокарту, даже если собран с CUDA.

    Если написать ``ENABLE_CUDA= YES``, а SU2_CFD собран без
    ``-Denable-cuda=true``, SU2 завершится с ошибкой «ENABLE_CUDA is set
    to YES / Please compile with CUDA options enabled in Meson». Поэтому
    опция включается только после проверки сборки решателя.
    """
    if physics is None:
        raise ValueError("physics не задан")

    ref_length, ref_area, ox, oy, oz = _unpack_ref_data(ref_data)

    p = {
        "MESH_FILENAME": mesh_filename,
        "mach": float(physics.get("mach", 0.0)),
        "aoa": float(aoa),
        "pressure": float(physics.get("pressure", 101325.0)),
        "temperature": float(physics.get("temperature", 288.15)),
        "reynolds": _calculate_reynolds(physics, ref_length),
        "ref_length": ref_length,
        "ref_area": ref_area,
        "ox": ox,
        "oy": oy,
        "oz": oz,
        # Для несжимаемого решателя INC_*: плотность и скорость потока.
        "rho": float(physics.get("rho", 1.225) or 1.225),
        "speed": float(physics.get("speed", 0.0) or 0.0),
    }

    solver_name = str(solver or "EULER").strip().upper()
    # === T-low-mach: несжимаемый решатель (INC_EULER / INC_RANS) =========
    # Проверяем ДО RANS/EULER, потому что "INC_RANS" содержит "RANS", а
    # "INC_EULER" содержит "EULER" — иначе они ушли бы в сжимаемые ветки.
    if solver_name.startswith("INC_"):
        return build_inc_config(
            p, markers=markers, restart=restart,
            mesh_quality=mesh_quality,
            use_symmetry=use_symmetry,
            symmetry_planes=symmetry_planes,
            enable_cuda=enable_cuda,
            n_bodies=n_bodies, n_points=n_points,
            cfl_aggressive=cfl_aggressive,
            solver_name=solver_name,
            turb_model=turb_model,
        )
    if solver_name.startswith("RANS") or "RANS" in solver_name:
        return build_rans_config(
            p, markers=markers, restart=restart,
            mesh_quality=mesh_quality,
            turb_model=turb_model,
            use_symmetry=use_symmetry,
            symmetry_planes=symmetry_planes,
            use_ramp_aoa=use_ramp_aoa,
            enable_cuda=enable_cuda,
            n_bodies=n_bodies, n_points=n_points,
            cfl_aggressive=cfl_aggressive,
        )
    if solver_name.startswith("EULER") or "EULER" in solver_name:
        return build_euler_config(
            p, markers=markers, restart=restart, enable_cuda=enable_cuda,
            mesh_quality=mesh_quality,
            use_symmetry=use_symmetry,
            symmetry_planes=symmetry_planes,
            n_bodies=n_bodies, n_points=n_points,
            cfl_aggressive=cfl_aggressive,
        )

    raise ValueError(f"Неподдерживаемый решатель SU2: {solver!r}")


def write_su2_config(path: str, aoa: float, physics: Mapping, solver: str,
                     ref_data: Sequence[float], markers=None,
                     restart: bool = False,
                     mesh_filename: str = "mesh.su2",
                     mesh_quality=None,
                     n_bodies: int = 0, n_points: int = 0,
                     cfl_aggressive: bool = False,
                     use_symmetry: bool = False,
                     symmetry_planes: list = None,
                     turb_model: str = "SA",
                     use_ramp_aoa: bool = False,
                     enable_cuda: bool = False) -> str:
    """Записывает конфигурацию и возвращает полный путь к ``config.cfg``.

    Параметры T1+T4 пробрасываются в build_su2_config.
    """
    path = os.fspath(path)
    if path.lower().endswith(".cfg"):
        cfg_path = os.path.abspath(path)
    else:
        cfg_path = os.path.abspath(os.path.join(path, "config.cfg"))

    parent = os.path.dirname(cfg_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    text = build_su2_config(
        aoa=aoa,
        physics=physics,
        solver=solver,
        ref_data=ref_data,
        markers=markers,
        restart=restart,
        mesh_filename=mesh_filename,
        mesh_quality=mesh_quality,
        n_bodies=n_bodies, n_points=n_points,
        cfl_aggressive=cfl_aggressive,
        use_symmetry=use_symmetry,
        symmetry_planes=symmetry_planes,
        turb_model=turb_model,
        use_ramp_aoa=use_ramp_aoa,
        enable_cuda=enable_cuda,
    )

    with open(cfg_path, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)

    return cfg_path


def low_mach_incompressible_solver(solver: str, mach) -> str:
    """Сопоставляет сжимаемый решатель с несжимаемым на низком Mach.

    Используется в write_case_config для включения INC_* по умолчанию на
    малых скоростях (M < LOW_MACH_THRESHOLD) — это официальный способ SU2
    получать правдоподобные Cl/Cd вместо завышенного Cd сжимаемого
    решателя на таких режимах.

    Возвращает исходный решатель, если Mach не мал или решатель уже
    несжимаемый/неподдерживаемый.
    """
    s = str(solver or "").strip().upper()
    try:
        m = float(mach)
    except (TypeError, ValueError):
        return str(solver or "EULER")
    if m >= LOW_MACH_THRESHOLD:
        return s
    mapping = {
        "EULER": "INC_EULER",
        "NAVIER_STOKES": "INC_NAVIER_STOKES",
        "RANS": "INC_RANS",
        "INC_EULER": "INC_EULER",
        "INC_NAVIER_STOKES": "INC_NAVIER_STOKES",
        "INC_RANS": "INC_RANS",
    }
    return mapping.get(s, s)


def write_case_config(case_dir: str, aoa: float, session) -> str:
    """Создаёт ``config.cfg`` по данным ``CalculationSession``.

    Эта сигнатура соответствует вызову из ``SessionRunner._prepare_case``.
    При наличии ``restart.dat`` расчётная точка продолжается с рестарта.

    Важно: ``session.active_markers`` содержит роли деталей GUI (wing,
    fuselage и т. п.), а файл сетки сейчас объединяет их в один SU2-маркер
    ``airfoil``. Поэтому роли нельзя напрямую писать в MARKER_EULER/
    MARKER_HEATFLUX — используется фактический маркер сетки ``airfoil``.

    Параметры T1+T4 берутся из session:
        session.use_symmetry (bool, default False)
        session.turb_model   ("SA" | "SST", default "SA")
        session.use_ramp_aoa (bool, default False)
        session.mesh_quality (для INNER_ITER)
    """
    if session is None:
        raise ValueError("session не задан")

    os.makedirs(case_dir, exist_ok=True)
    restart = os.path.exists(os.path.join(case_dir, "restart.dat"))
    mesh_quality = getattr(session, "mesh_quality", None)
    turb_model = str(getattr(session, "turb_model", "SA") or "SA")
    use_ramp_aoa = bool(getattr(session, "use_ramp_aoa", False))
    # Оценка потолка итераций и агрессивный CFL. getattr с нулями и False —
    # старые сессии без этих полей работают ровно как раньше.
    n_bodies = int(getattr(session, "n_bodies", 0) or 0)
    n_points = int(getattr(session, "n_points", 0) or 0)
    cfl_aggressive = bool(getattr(session, "cfl_aggressive", False))

    # T1: собираем список плоскостей симметрии. Берём из session.symmetry_planes,
    # если нет — обратная совместимость: session.use_symmetry=True → ["xz"].
    # Затем проверяем mesh.su2 — если нужного маркера нет, плоскость
    # исключается (SU2 упадёт с "MARKER_SYM not found in mesh" иначе).
    planes_attr = getattr(session, "symmetry_planes", None)
    use_symmetry_legacy = bool(getattr(session, "use_symmetry", False))
    if planes_attr is None:
        planes_attr = ["xz"] if use_symmetry_legacy else []

    mesh_path = os.path.join(case_dir, "mesh.su2")
    enabled_planes = []
    if planes_attr:
        for p in planes_attr:
            p_lower = str(p).lower()
            # Имя маркера, которое пишется в mesh.su2:
            #   XZ → symmetry_plane (обратная совместимость) И symmetry_xz
            #   XY → symmetry_xy
            #   YZ → symmetry_yz
            possible_tags = []
            if p_lower == "xz":
                possible_tags = ["symmetry_plane", "symmetry_xz"]
            elif p_lower == "xy":
                possible_tags = ["symmetry_xy"]
            elif p_lower == "yz":
                possible_tags = ["symmetry_yz"]
            # Проверяем хотя бы одно имя в mesh.su2
            for tag in possible_tags:
                if _mesh_has_marker(mesh_path, tag):
                    enabled_planes.append(p_lower)
                    break

    # Старое поле use_symmetry — для обратной совместимости
    use_symmetry_final = bool(enabled_planes)

    # === T-low-mach: по умолчанию переключаем на несжимаемый решатель ======
    # На малых скоростях (M<0.3) сжимаемый EULER/RANS даёт завышенный Cd.
    # Включение INC_* по умолчанию управляется флагом session, а не
    # подменяет выбор пользователя: если он сам выбрал INC_* — остаётся
    # INC_*, а если выбрал сжимаемый и режим малый — уходит в INC_*.
    solver_final = session.solver
    if getattr(session, "use_incompressible_at_low_mach", True):
        # Mach предпочтительно берём из physics; если его там нет (старые
        # сессии), восстанавливаем из скорости потока и скорости звука,
        # чтобы низкомаховые прогоны всегда уходили в INC_*, а не падали
        # в расходимость сжимаемого EULER/RANS.
        _physics = getattr(session, "physics", {}) or {}
        _mach = _physics.get("mach")
        if _mach is None:
            _sp = _physics.get("speed")
            _a = _physics.get("a")
            try:
                _mach = float(_sp) / max(float(_a), 1e-9)
            except (TypeError, ValueError):
                _mach = None
        solver_final = low_mach_incompressible_solver(
            session.solver, _mach)

    return write_su2_config(
        path=os.path.join(case_dir, "config.cfg"),
        aoa=aoa,
        physics=session.physics,
        solver=solver_final,
        ref_data=session.ref_data,
        markers=["airfoil"],
        restart=restart,
        mesh_filename="mesh.su2",
        mesh_quality=mesh_quality,
        use_symmetry=use_symmetry_final,
        symmetry_planes=enabled_planes,
        turb_model=turb_model,
        use_ramp_aoa=use_ramp_aoa,
        enable_cuda=bool(getattr(session, "enable_cuda", False)),
    )


def _mesh_has_marker(mesh_path: str, marker_tag: str) -> bool:
    """Возвращает True, если в mesh.su2 есть маркер с указанным тегом
    и у него ненулевое MARKER_ELEMS.

    Используется в write_case_config для безопасного включения
    MARKER_SYM (T1): если в сетке нет маркера symmetry_plane,
    SU2 упадёт — лучше отключить опцию, чем падать в рантайме.
    """
    try:
        if not os.path.exists(mesh_path):
            return False
        with open(mesh_path, "r", encoding="ascii", errors="ignore") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return False
    for i, ln in enumerate(lines):
        if ln.startswith("MARKER_TAG=") and marker_tag in ln:
            # Следующая строка должна быть MARKER_ELEMS= N
            if i + 1 < len(lines) and lines[i + 1].startswith("MARKER_ELEMS="):
                try:
                    n = int(lines[i + 1].split("=", 1)[1].strip())
                except (ValueError, IndexError):
                    n = 0
                return n > 0
            return False
    return False
