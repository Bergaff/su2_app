# -*- coding: utf-8 -*-
"""
physics/structural.py — оценочная прочность крыла.

ТЗ, пункт «Низкий приоритет — прочность».

Модель
------
Крыло рассматривается как консольная балка с эквивалентным
лонжероном-коробкой. Расчёт идёт в три шага:

1. **Нагрузка по размаху.** Подъёмная сила распределяется по размаху
   эллиптически (по умолчанию) или треугольно; масса крыла вычитается
   как разгрузка (inertia relief).

2. **Усилия в корне.** Перерезывающая сила ``Q`` и изгибающий момент
   ``M`` считаются интегрированием по полуразмаху. Для эллиптического
   распределения результат совпадает с аналитикой (проверяется
   тестами)::

       Q_root = L_total
       M_root = (4 / 3π) · L_total · (span/2) ≈ 0.4244 · L · s

3. **Напряжения и запас.** Нормальные напряжения в полках лонжерона
   ``σ = M / (h · A_cap)``, касательные в стенке, масса конструкции из
   требуемой площади полок. Запас прочности — относительно
   допустимых напряжений с учётом эксплуатационной перегрузки и
   коэффициента безопасности 1.5.

Модуль не заменяет прочностной расчёт КД — это оценка на ранней стадии
проектирования, которая показывает порядок величин и «красные флаги».
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

G0 = 9.80665                      # м/с², стандартное ускорение
SAFETY_FACTOR_DEFAULT = 1.5       # авиационный коэффициент безопасности
RHO_ALUMINUM = 2700.0             # кг/м³
SIGMA_ALLOW_ALUMINUM = 2.8e8      # Па (допустимое, Д16Т с запасом по ресурсу)
TAU_ALLOW_ALUMINUM = 1.6e8        # Па


# ---------------------------------------------------------------------------
# 1. Распределение нагрузки по размаху
# ---------------------------------------------------------------------------

def spanwise_shape(y: np.ndarray, half_span: float, dist: str = "elliptic"
                   ) -> np.ndarray:
    """Безразмерная форма распределения (максимум = 1 в корне).

    ``dist``: ``"elliptic"`` — эллипс, ``"triangular"`` — линейное спадание
    к концу крыла, ``"uniform"`` — равномерно.
    """
    y = np.asarray(y, dtype=float)
    eta = np.clip(y / max(half_span, 1e-9), 0.0, 1.0)
    d = str(dist).lower()
    if d.startswith("ell"):
        return np.sqrt(np.clip(1.0 - eta ** 2, 0.0, None))
    if d.startswith("tri"):
        return 1.0 - eta
    if d.startswith("uni"):
        return np.ones_like(eta)
    raise ValueError(f"Неизвестное распределение: {dist!r}")


def _shape_integral(half_span: float, dist: str) -> Tuple[float, float]:
    """Аналитические ∫shape·dη и ∫shape·η·dη по η∈[0,1]."""
    d = str(dist).lower()
    if d.startswith("ell"):
        return math.pi / 4.0, 1.0 / 3.0
    if d.startswith("tri"):
        return 0.5, 1.0 / 6.0
    if d.startswith("uni"):
        return 1.0, 0.5
    raise ValueError(f"Неизвестное распределение: {dist!r}")


def root_forces(L_total: float, span: float, mass_wing: float = 0.0,
                dist: str = "elliptic", n_load: float = 1.0,
                n_points: int = 2001) -> Dict[str, float]:
    """Перерезывающая сила и изгибающий момент в корне крыла.

    ``L_total``   — полная подъёмная сила (на обе половины), Н;
    ``mass_wing`` — масса крыла, кг (разгружает корень);
    ``n_load``    — перегрузка, с которой считается вес конструкции.

    Возвращает ``{"Q": Н, "M": Н·м, "M_analytic": Н·м, "q_root": Н/м}``.
    """
    s = max(float(span), 1e-9) / 2.0
    shape_i, shape_m = _shape_integral(s, dist)

    # Погонная нагрузка в корне (аэродинамика)
    q_aero_root = L_total / (2.0 * s * shape_i)
    # Погонная масса крыла в корне (та же форма распределения)
    q_mass_root = (mass_wing * G0 * n_load) / (2.0 * s * shape_i)
    q_root = q_aero_root - q_mass_root

    Q = q_root * 2.0 * s * shape_i
    M = q_root * 2.0 * s * s * shape_m

    # Аналитическая проверка для эллипса без разгрузки
    M_analytic = (4.0 / (3.0 * math.pi)) * L_total * s if \
        str(dist).lower().startswith("ell") else float("nan")

    return {"Q": float(Q), "M": float(M), "M_analytic": float(M_analytic),
            "q_root": float(q_root), "half_span": s}


# ---------------------------------------------------------------------------
# 2. Напряжения в эквивалентном лонжероне
# ---------------------------------------------------------------------------

def spar_stresses(M_root: float, Q_root: float, chord_root: float,
                  t_ratio: float = 0.12, web_thickness: float = 0.005,
                  cap_area: Optional[float] = None) -> Dict[str, float]:
    """Напряжения в лонжероне-коробке.

    ``cap_area`` — площадь одной полки, м². Если не задана, берётся
    требуемая по моменту (см. :func:`cap_area_required`).
    """
    c = max(float(chord_root), 1e-6)
    h = max(t_ratio * c, 1e-4)
    if cap_area is None:
        cap_area = cap_area_required(M_root, h)
    cap_area = max(float(cap_area), 1e-9)

    sigma = abs(M_root) / (h * cap_area)          # Н/м²
    web_area = h * max(web_thickness, 1e-6)
    tau = abs(Q_root) / max(web_area, 1e-9)
    return {"sigma": float(sigma), "tau": float(tau),
            "cap_area": float(cap_area), "spar_height": float(h)}


def cap_area_required(M_root: float, spar_height: float,
                      sigma_allow: float = SIGMA_ALLOW_ALUMINUM) -> float:
    """Требуемая площадь одной полки лонжерона (пара полок = силовая пара)."""
    h = max(float(spar_height), 1e-6)
    return abs(float(M_root)) / (h * max(sigma_allow, 1.0))


def wing_structural_mass(cap_area: float, span: float, chord_root: float,
                         t_ratio: float = 0.12,
                         web_thickness: float = 0.005,
                         rib_factor: float = 1.35) -> float:
    """Оценка массы силовой конструкции крыла, кг.

    Полки лонжерона (2 шт., с учётом схода на нет — 0.6 от корневой
    площади в среднем) + стенка + обшивка/нервюры через ``rib_factor``.
    """
    s = max(float(span), 1e-6) / 2.0
    h = max(t_ratio * max(chord_root, 1e-6), 1e-4)
    m_caps = 2.0 * cap_area * 0.6 * s * RHO_ALUMINUM
    m_web = h * max(web_thickness, 1e-6) * s * RHO_ALUMINUM
    m_skin = 0.02 * chord_root * 0.0015 * s * 2.0 * RHO_ALUMINUM
    return float((m_caps + m_web + m_skin) * rib_factor)


# ---------------------------------------------------------------------------
# 3. Оценка целиком
# ---------------------------------------------------------------------------

def structural_assessment(span: float, chord_root: float,
                          mass_aircraft: float, mass_wing: float,
                          n_limit: float = 2.5,
                          dist: str = "elliptic",
                          t_ratio: float = 0.12,
                          web_thickness: float = 0.005,
                          sigma_allow: float = SIGMA_ALLOW_ALUMINUM,
                          tau_allow: float = TAU_ALLOW_ALUMINUM,
                          safety_factor: float = SAFETY_FACTOR_DEFAULT,
                          cap_area: Optional[float] = None) -> Dict[str, object]:
    """Полная оценочная проверка прочности крыла.

    Возвращает словарь с усилиями, напряжениями, массой, запасами и
    вердиктом. Нагрузка — симметричный доворот с перегрузкой ``n_limit``.
    """
    L_total = float(n_limit) * float(mass_aircraft) * G0
    forces = root_forces(L_total, span, mass_wing=mass_wing, dist=dist,
                         n_load=float(n_limit))
    st = spar_stresses(forces["M"], forces["Q"], chord_root,
                       t_ratio=t_ratio, web_thickness=web_thickness,
                       cap_area=cap_area)
    mass = wing_structural_mass(st["cap_area"], span, chord_root,
                                t_ratio=t_ratio, web_thickness=web_thickness)

    ms_sigma = (sigma_allow / safety_factor) / max(st["sigma"], 1e-9) - 1.0
    ms_tau = (tau_allow / safety_factor) / max(st["tau"], 1e-9) - 1.0
    ok = bool(ms_sigma >= 0.0 and ms_tau >= 0.0)

    return {
        "L_total": L_total,
        "Q_root": forces["Q"],
        "M_root": forces["M"],
        "M_analytic": forces["M_analytic"],
        "sigma": st["sigma"],
        "tau": st["tau"],
        "cap_area": st["cap_area"],
        "spar_height": st["spar_height"],
        "mass_estimate": mass,
        "MS_sigma": ms_sigma,
        "MS_tau": ms_tau,
        "safety_factor": safety_factor,
        "n_limit": n_limit,
        "ok": ok,
        "verdict": ("Прочность обеспечена (оценочно)" if ok else
                    "НЕ ОБЕСПЕЧЕНА: увеличьте полки лонжерона "
                    "или высоту сечения"),
    }


def format_report(res: Dict[str, object]) -> str:
    """Человекочитаемый отчёт по прочности."""
    def _fmt(v, unit="", nd=2):
        return "—" if v is None else f"{float(v):.{nd}f} {unit}".strip()

    return "\n".join([
        "ПРОЧНОСТЬ КРЫЛА (оценка)",
        "=" * 44,
        f"Перегрузка расчётная : {_fmt(res.get('n_limit'), 'g', 2)}",
        f"Подъёмная сила       : {_fmt(res.get('L_total'), 'Н', 0)}",
        f"Перерезывающая Q     : {_fmt(res.get('Q_root'), 'Н', 0)}",
        f"Момент в корне M     : {_fmt(res.get('M_root'), 'Н·м', 0)}",
        f"  (аналитика, эллипс): {_fmt(res.get('M_analytic'), 'Н·м', 0)}",
        f"Высота лонжерона     : {_fmt(res.get('spar_height'), 'м', 4)}",
        f"Площадь полки        : {_fmt(res.get('cap_area') * 1e4, 'см²', 2)}",
        f"σ изгиба             : {_fmt(res.get('sigma') / 1e6, 'МПа', 1)}",
        f"τ сдвига             : {_fmt(res.get('tau') / 1e6, 'МПа', 1)}",
        f"Масса конструкции    : {_fmt(res.get('mass_estimate'), 'кг', 1)}",
        f"Запас по σ           : {_fmt(res.get('MS_sigma'), '', 2)}",
        f"Запас по τ           : {_fmt(res.get('MS_tau'), '', 2)}",
        f"Вердикт              : {res.get('verdict')}",
    ])
