# -*- coding: utf-8 -*-
"""
physics/aeroelastic.py — аэроупругость: флатер и дивергенция крыла.

ТЗ, пункт «Средний приоритет — учёт аэроупругости (Flutter)».

Что реализовано
---------------
1. **Метод типичного сечения (2 DOF)** — изгиб ``h`` + кручение ``α``.
   Аэродинамика — полная теория Теодорсена для несжимаемого потока:
   циркуляционная часть с функцией ``C(k)`` (точная, через функции
   Бесселя) плюс нециркуляционная (присоединённые массы).

2. **p-k метод** — для каждой скорости решается квадратичная задача на
   собственные значения, находится ``λ = g/2 + iω``. Скорость флатера
   ``V_F`` — точка, где структурное демпфирование ``g`` пересекает ноль.

3. **Дивергенция** — статическая потеря устойчивости по кручению,
   аналитическая формула (совпадение проверяется тестами)::

       q_D = K_α / (c · C_Lα · e),   e = (x_ea − 0.25) · c

   возможна только при ``e > 0`` (ось упругости за аэродинамическим
   фокусом).

4. **Оценка параметров сечения** по параметрам крыла — консольная балка
   с эквивалентным лонжероном-коробкой.

Соглашения по знакам
--------------------
``h`` — перемещение ВНИЗ (м); ``α`` — угол атаки, нос вверх (рад);
момент положителен, если поднимает нос. ``b = c/2`` — полу-хорда,
``a = 2·x_ea − 1`` — положение оси упругости от средней хорды в
полу-хордах (положительно к хвосту), ``e = b(1/2 + a)`` — смещение оси
упругости от аэродинамического фокуса (0.25c).

Аэродинамика (на единицу размаха), ``C(k)`` — функция Теодорсена::

    Q   = U·α + ḣ + b(1/2 − a)·α̇                    эффективный скос в 3/4 хорды
    L_c = 2πρUb·C(k)·Q                              циркуляционная сила (в фокусе)
    L   = πρb²·(ḧ − a·b·α̈) + L_c                    полная сила (вверх +)
    M   = a·b·πρb²(ḧ − a·b·α̈) − (πρb⁴/8)·α̈
          − (π/2)ρUb³·α̇ + b(1/2 + a)·L_c            момент отн. оси упругости

Происхождение слагаемых: ``πρb²`` — присоединённая масса профиля
(приложена в средней хорде), ``πρb⁴/8`` — присоединённый момент инерции
при вращении относительно средней хорды, ``−(π/2)ρUb³α̇`` — стационарный
момент тонкого профиля относительно фокуса при угловой скорости
(``C_m,ac = −π/4·q̄``), ``b(1/2+a)·L_c`` — плечо от фокуса до оси упругости.

Структурная часть (``S_α = m·x_α·b``); обобщённые аэродинамические силы
``Q_aero = [−L, +M]`` (работа силы вверх на перемещении вниз
отрицательна)::

    m·ḧ + S_α·α̈ + K_h·h   = −L
    S_α·ḧ + I_α·α̈ + K_α·α =  M

В коде это записано как квадратичная задача на собственные значения
``λ²(M−B2) − λB1 + (K−B0) = 0``, где ``Q_aero = (B0 + λB1 + λ²B2)·[h, α]``.

Контрольные пределы, на которых модель проверена
(``tests/check_physics_new.py``, ``tests/test_backend.py``):
  * ``C(0) = 1``, ``C(∞) = 0.5``, ``|C| < 1`` и ``Im C < 0`` при ``k > 0``;
  * стационарные сила и момент совпадают с теорией тонкого профиля
    (Глауэрт) с точностью до машинной;
  * статическая дивергенция совпадает с аналитикой;
  * при ``U → 0`` частоты совпадают с собственными частотами с
    присоединёнными массами, демпфирование нулевое;
  * изолированный изгиб устойчив на всех скоростях;
  * квазистационарное демпфирование кручения неотрицательно при любом ``a``;
  * ``V_F`` растёт с жёсткостью кручения и отсутствует при оси упругости
    впереди фокуса (``e ≤ 0``).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # функции Бесселя — для точной функции Теодорсена
    from scipy.special import j0 as _j0, j1 as _j1, y0 as _y0, y1 as _y1
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

C_L_ALPHA_DEFAULT = 2.0 * math.pi   # тонкий профиль, 1/рад


# ---------------------------------------------------------------------------
# Функция Теодорсена C(k) = F(k) + i·G(k)
# ---------------------------------------------------------------------------

def theodorsen(k: float) -> complex:
    """Точная функция Теодорсена ``C(k) = F(k) + i·G(k)``.

    ``k = ω·b/U`` — безразмерная частота. Определяется через функции
    Ханкеля второго рода (``H_n = J_n − i·Y_n``):

        C(k) = H_1 / (H_1 + i·H_0)

    что в вещественной записи даёт

        D = (J1+Y0)² + (J0−Y1)²
        F = [J1·(J1+Y0) − Y1·(J0−Y1)] / D
        G = −[J1·J0 + Y1·Y0] / D

    Пределы: ``C(0) = 1`` (квазистационарный), ``C(∞) = 0.5``;
    ``G(k) < 0`` при ``k > 0`` — запаздывание вихревого следа, именно оно
    даёт физическое (положительное) демпфирование.
    """
    k = abs(float(k))
    if k < 1e-9:
        return complex(1.0, 0.0)
    if not _HAVE_SCIPY:
        return theodorsen_rational(k)
    J0, J1 = float(_j0(k)), float(_j1(k))
    Y0, Y1 = float(_y0(k)), float(_y1(k))
    den = (J1 + Y0) ** 2 + (J0 - Y1) ** 2
    if den == 0.0:  # pragma: no cover
        return theodorsen_rational(k)
    F = (J1 * (J1 + Y0) - Y1 * (J0 - Y1)) / den
    G = -(J1 * J0 + Y1 * Y0) / den
    return complex(F, G)


# Коэффициенты рациональной (Паде [2/2]) аппроксимации C(k), подобраны
# методом наименьших квадратов по точной функции Теодорсена на
# k ∈ [0.01, 20]; ``s = i·k``:
#     C(k) ≈ (1 + a1·s + 0.5·b2·s²) / (1 + b1·s + b2·s²)
# Автоматически выполнены C(0) = 1 и C(∞) = 0.5. Максимальное отклонение
# |ΔC| < 0.05; полюса знаменателя s = −0.1373 и −0.7647 — вещественные и
# отрицательные, т.е. аппроксимация годится для реализации в пространстве
# состояний без внесения неустойчивости.
_RFA_A1, _RFA_B1, _RFA_B2 = 5.485905, 8.592119, 9.525996


def theodorsen_rational(k: float) -> complex:
    """Рациональная аппроксимация ``C(k)`` (запасной вариант без scipy)."""
    s = 1j * abs(float(k))
    if abs(s) < 1e-12:
        return complex(1.0, 0.0)
    num = 1.0 + _RFA_A1 * s + 0.5 * _RFA_B2 * s * s
    den = 1.0 + _RFA_B1 * s + _RFA_B2 * s * s
    return complex(num / den)


def theodorsen_jones(k: float) -> complex:
    """Синоним :func:`theodorsen_rational` (старое название)."""
    return theodorsen_rational(k)


# ---------------------------------------------------------------------------
# Геометрические соотношения
# ---------------------------------------------------------------------------

def a_from_e(e: float, chord: float) -> float:
    """Параметр Теодорсена ``a`` по смещению оси упругости ``e``.

    ``e`` — расстояние от аэродинамического фокуса (0.25c) до оси
    упругости, м; положительно, если ось упругости ЗА фокусом.
    """
    return 2.0 * float(e) / max(float(chord), 1e-9) - 0.5


def e_from_a(a: float, chord: float) -> float:
    """Обратное преобразование: ``e`` по ``a``."""
    return 0.5 * float(chord) * (float(a) + 0.5)


# ---------------------------------------------------------------------------
# Матрицы типичного сечения
# ---------------------------------------------------------------------------

def section_matrices(m: float, x_alpha: float, b: float, I_alpha: float,
                     K_h: float, K_alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Массовая и жёсткостная матрицы типичного сечения."""
    S_alpha = m * x_alpha * b
    M = np.array([[m, S_alpha], [S_alpha, I_alpha]], dtype=float)
    K = np.array([[K_h, 0.0], [0.0, K_alpha]], dtype=float)
    return M, K


def aero_matrices(rho: float, U: float, b: float, a: float, Ck: complex
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Обобщённые аэродинамические силы ``B0, B1, B2`` такие, что

        Q_aero = [-L, +M] = (B0 + λ·B1 + λ²·B2) · [h, α]

    где ``λ`` — комплексная частота (``q = q₀·e^{λt}``), ``h`` — прогиб
    ВНИЗ, ``α`` — угол атаки (нос вверх), ``a`` — положение оси упругости
    от средней хорды в полу-хордах (положительно к хвосту).

    Подъёмная сила и момент относительно оси упругости:

        Q   = Uα + ḣ + b(1/2 − a)α̇
        L   = πρb²[ḧ − abα̈ + Uα̇] + 2πρUb·C(k)·Q
        M   = ab·L_nc − (πρb⁴/8)α̈ + b(1/2 + a)·L_c

    где ``L_nc`` — нециркуляционная часть (присоединённые массы, точка
    приложения — средняя хорда, плюс чистая пара от углового ускорения),
    ``L_c`` — циркуляционная (приложена в аэродинамическом фокусе).
    """
    pi = math.pi
    C = complex(Ck)

    # Сила L: коэффициенты при ḧ, α̈, ḣ, α, α̇
    L_h2 = pi * rho * b * b                       # ḧ  (присоединённая масса)
    L_a2 = -pi * rho * b ** 3 * a                 # α̈
    L_h1 = 2.0 * pi * rho * U * b * C             # ḣ
    L_a0 = 2.0 * pi * rho * U * U * b * C         # α
    L_a1 = 2.0 * pi * rho * U * b * b * (0.5 - a) * C   # α̇

    # Момент M (относительно оси упругости)
    M_h2 = pi * rho * b ** 3 * a
    M_a2 = -pi * rho * b ** 4 * (a * a + 0.125)
    M_h1 = 2.0 * pi * rho * U * b * b * (0.5 + a) * C
    M_a0 = 2.0 * pi * rho * U * U * b * b * (0.5 + a) * C
    # −(π/2)ρUb³α̇ — стационарный момент относительно фокуса при угловой
    # скорости (тонкий профиль: C_m,ac = −π/4·q̄), плюс плечо e от L_c.
    M_a1 = -0.5 * pi * rho * U * b ** 3 \
        + 2.0 * pi * rho * U * b ** 3 * (0.5 + a) * (0.5 - a) * C

    # Обобщённые силы: для координаты h (вниз) работа силы L вверх = −L,
    # для координаты α (нос вверх) работа момента M = +M.
    B2 = np.array([[-L_h2, -L_a2], [M_h2, M_a2]], dtype=complex)
    B1 = np.array([[-L_h1, -L_a1], [M_h1, M_a1]], dtype=complex)
    B0 = np.array([[0.0, -L_a0], [0.0, M_a0]], dtype=complex)
    return B0, B1, B2


def _quadratic_eig(M: np.ndarray, A1: np.ndarray, Keff: np.ndarray
                   ) -> np.ndarray:
    """Собственные значения ``λ`` уравнения ``λ²M q + λA1 q + Keff q = 0``."""
    n = M.shape[0]
    Minv = np.linalg.inv(M)
    top = np.hstack([-Minv @ A1, -Minv @ Keff])
    bot = np.hstack([np.eye(n, dtype=complex), np.zeros((n, n), dtype=complex)])
    return np.linalg.eigvals(np.vstack([top, bot]))


def _select_root(eigs: np.ndarray, omega: float,
                 imag_tol: float = 1e-7) -> Optional[int]:
    """Выбор ветви собственных значений в итерации p-k.

    Корни квадратичной задачи идут парами ``λ, λ̄``; физическая мода —
    та, у которой ``Im λ > 0``. Из положительных ветвей берём ближайшую
    к текущей частоте ``omega``. Возвращает индекс или ``None``, если
    положительных ветвей нет (чисто вещественный спектр — дивергенция).
    """
    im = np.imag(eigs)
    pos = np.flatnonzero(im > imag_tol)
    if pos.size == 0:
        # вещественные корни: берём ближайший к нулю по модулю
        return int(np.argmin(np.abs(eigs)))
    return int(pos[int(np.argmin(np.abs(im[pos] - omega)))])


def pk_point(U: float, m: float, x_alpha: float, b: float, I_alpha: float,
             K_h: float, K_alpha: float, chord: float, e: float, rho: float,
             n_iter: int = 60, tol: float = 1e-11) -> List[Dict[str, float]]:
    """Одна точка p-k диаграммы: моды при скорости ``U``.

    Возвращает список ``{"omega", "g", "freq_hz"}``; ``g > 0`` —
    нарастание колебаний (неустойчивость).
    """
    M, K = section_matrices(m, x_alpha, b, I_alpha, K_h, K_alpha)
    a = a_from_e(e, chord)
    out: List[Dict[str, float]] = []

    for omega in (math.sqrt(max(K_h, 1e-9) / max(m, 1e-12)),
                  math.sqrt(max(K_alpha, 1e-9) / max(I_alpha, 1e-12))):
        omega = max(abs(omega), 1e-6)
        g = 0.0
        for _ in range(n_iter):
            k = omega * b / max(U, 1e-9)
            Ck = theodorsen(k)
            B0, B1, B2 = aero_matrices(rho, U, b, a, Ck)
            Meff = M.astype(complex) - B2
            Keff = K.astype(complex) - B0
            # Структурное уравнение M q̈ + K q = (B0 + λB1 + λ²B2) q,
            # откуда λ²(M−B2) − λB1 + (K−B0) = 0.
            try:
                eigs = _quadratic_eig(Meff, -B1, Keff)
            except np.linalg.LinAlgError:  # pragma: no cover
                break
            idx = _select_root(eigs, omega)
            if idx is None:
                break
            lam = complex(eigs[idx])
            im = abs(lam.imag)
            if im < 1e-9:
                # апериодическая ветвь (в т.ч. дивергенция) — считаем
                # демпфирование «бесконечно большим» по модулю
                g_new = 1e3 if lam.real > 0 else -1e3
                if abs(g_new - g) < tol:
                    g = g_new
                    break
                g = g_new
                omega = max(omega * 0.9, 1e-6)
                continue
            omega_new = im
            g_new = 2.0 * lam.real / im
            if (abs(omega_new - omega) < tol * max(1.0, omega)
                    and abs(g_new - g) < 1e-9):
                omega, g = omega_new, g_new
                break
            omega, g = omega_new, g_new
        out.append({"omega": float(omega), "g": float(g),
                    "freq_hz": float(omega / (2.0 * math.pi))})
    return out


def vg_diagram(V_range: Sequence[float], **kw) -> List[Dict[str, float]]:
    """V-g диаграмма: для каждой скорости — демпфирование худшей моды."""
    rows = []
    for V in V_range:
        modes = pk_point(float(V), **kw)
        if not modes:
            continue
        worst = max(modes, key=lambda d: d["g"])
        rows.append({"V": float(V), "g": worst["g"],
                     "freq_hz": worst["freq_hz"], "modes": modes})
    return rows


def flutter_speed(V_max: float = 400.0, n_steps: int = 80, **kw
                  ) -> Tuple[Optional[float], List[Dict[str, float]]]:
    """Скорость флатера ``V_F`` — первое пересечение ``g`` нуля снизу."""
    Vs = np.linspace(1.0, float(V_max), int(n_steps))
    diag = vg_diagram(Vs, **kw)
    v_f = None
    for prev, cur in zip(diag, diag[1:]):
        if prev["g"] < 0.0 <= cur["g"]:
            span = cur["g"] - prev["g"]
            frac = 0.0 if span == 0 else (-prev["g"] / span)
            v_f = prev["V"] + frac * (cur["V"] - prev["V"])
            break
    return v_f, diag


def divergence_speed(K_alpha: float, chord: float, e: float, rho: float,
                     cl_alpha: float = C_L_ALPHA_DEFAULT) -> Optional[float]:
    """Скорость дивергенции (статическая аэроупругая неустойчивость).

    Возможна только при ``e > 0`` — ось упругости за аэродинамическим
    фокусом. Иначе возвращает ``None``.
    """
    if e <= 0.0:
        return None
    q_d = K_alpha / (chord * cl_alpha * e)
    if q_d <= 0.0 or rho <= 0.0:
        return None
    return math.sqrt(2.0 * q_d / rho)


# ---------------------------------------------------------------------------
# Оценка параметров крыла (консольная балка с лонжероном-коробкой)
# ---------------------------------------------------------------------------

def wing_section_properties(span: float, chord_root: float, chord_tip: float,
                            mass_wing: float, t_ratio: float = 0.12,
                            x_ea_ratio: float = 0.45,
                            x_cg_ratio: float = 0.42,
                            E: float = 71.0e9, G: float = 27.0e9,
                            spar_area_frac: float = 0.02) -> Dict[str, float]:
    """Оценка погонных характеристик сечения в корне крыла.

    Модель: консоль длиной ``L = span/2`` с эквивалентным
    лонжероном-коробкой высотой ``h = t_ratio·c_root``.
    """
    span = max(float(span), 1e-6)
    c_root = max(float(chord_root), 1e-6)
    L = span / 2.0
    b = c_root / 2.0

    m = max(float(mass_wing), 1e-3) / span          # кг/м

    h_box = max(t_ratio * c_root, 1e-3)
    A_box = max(spar_area_frac * c_root * h_box, 1e-6)
    I_beam = 2.0 * A_box * (h_box / 2.0) ** 2
    J_tors = 2.0 * (c_root * h_box) ** 2 * (A_box / 4.0) \
        / max(c_root + h_box, 1e-6)

    K_h = 3.0 * E * I_beam / max(L ** 3, 1e-9)      # Н/м
    K_alpha = G * J_tors / max(L, 1e-6)             # Н·м/рад

    x_alpha = 2.0 * (float(x_cg_ratio) - float(x_ea_ratio))
    r_gyr = 0.25 * c_root
    I_alpha = m * (r_gyr ** 2 + (x_alpha * b) ** 2)

    e = (float(x_ea_ratio) - 0.25) * c_root

    omega_h = math.sqrt(max(K_h, 0.0) / max(m, 1e-12))
    omega_alpha = math.sqrt(max(K_alpha, 0.0) / max(I_alpha, 1e-12))

    return {
        "b": b, "chord": c_root, "e": e,
        "m": m, "I_alpha": I_alpha, "x_alpha": x_alpha,
        "K_h": K_h, "K_alpha": K_alpha,
        "omega_h": omega_h, "omega_alpha": omega_alpha,
        "span": span, "half_span": L,
        "a": a_from_e(e, c_root),
    }


def flutter_assessment(span: float, chord_root: float, chord_tip: float,
                       mass_wing: float, rho: float, V_cruise: float,
                       V_dive: Optional[float] = None,
                       t_ratio: float = 0.12, x_ea_ratio: float = 0.45,
                       x_cg_ratio: float = 0.42,
                       safety_factor: float = 1.15,
                       V_max_scan: Optional[float] = None,
                       **prop_kw) -> Dict[str, object]:
    """Полная оценка аэроупругости крыла (V_F, V_D, запасы, вердикт)."""
    props = wing_section_properties(span, chord_root, chord_tip, mass_wing,
                                    t_ratio=t_ratio, x_ea_ratio=x_ea_ratio,
                                    x_cg_ratio=x_cg_ratio, **prop_kw)
    v_ref = float(V_dive) if V_dive else float(V_cruise)
    scan_max = float(V_max_scan) if V_max_scan else max(4.0 * v_ref, 200.0)

    v_f, diag = flutter_speed(
        V_max=scan_max,
        m=props["m"], x_alpha=props["x_alpha"], b=props["b"],
        I_alpha=props["I_alpha"], K_h=props["K_h"], K_alpha=props["K_alpha"],
        chord=props["chord"], e=props["e"], rho=rho,
    )
    v_d = divergence_speed(props["K_alpha"], props["chord"], props["e"], rho)

    v_crit = min([x for x in (v_f, v_d) if x is not None], default=None)
    margin = (v_crit / v_ref) if (v_crit and v_ref > 0) else None
    ok = bool(margin is not None and margin >= safety_factor)

    return {
        "props": props, "V_F": v_f, "V_D": v_d, "V_crit": v_crit,
        "V_ref": v_ref, "margin": margin, "safety_factor": safety_factor,
        "ok": ok,
        "omega_h_hz": props["omega_h"] / (2 * math.pi),
        "omega_alpha_hz": props["omega_alpha"] / (2 * math.pi),
        "vg_diagram": diag,
        "verdict": ("Аэроупругая устойчивость обеспечена" if ok else
                    "ТРЕБУЕТСЯ ПРОВЕРКА: критическая скорость ниже допуска"),
    }


def format_report(res: Dict[str, object]) -> str:
    """Человекочитаемый отчёт по результатам оценки."""
    def _fmt(v, unit="", nd=2):
        return "—" if v is None else f"{float(v):.{nd}f} {unit}".strip()

    lines = [
        "АЭРОУПРУГОСТЬ (флатер / дивергенция)",
        "=" * 44,
        f"Частота изгиба      : {_fmt(res.get('omega_h_hz'), 'Гц')}",
        f"Частота кручения    : {_fmt(res.get('omega_alpha_hz'), 'Гц')}",
        f"Скорость флатера V_F: {_fmt(res.get('V_F'), 'м/с')}",
        f"Скорость дивергенции: {_fmt(res.get('V_D'), 'м/с')}",
        f"Критическая скорость: {_fmt(res.get('V_crit'), 'м/с')}",
        f"Опорная скорость    : {_fmt(res.get('V_ref'), 'м/с')}",
        f"Запас               : {_fmt(res.get('margin'), '', 2)}×"
        f" (требуется ≥ {res.get('safety_factor')}×)",
        f"Вердикт             : {res.get('verdict')}",
    ]
    if res.get("V_F") is None:
        lines.append("Флатер в просканированном диапазоне скоростей не найден.")
    return "\n".join(lines)
