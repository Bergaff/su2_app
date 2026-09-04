# -*- coding: utf-8 -*-
"""
postprocessing/polar.py — поляра и интегральные характеристики.

Вход — список результатов расчёта (такие же словари, какие кладёт в
``self.all_results`` главное окно): минимум ``aoa``, ``cl``, ``cd``;
опционально ``cm``, ``mach``, ``converged``.

Все функции — чистый numpy, без Qt, поэтому легко проверяются тестами.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

POLAR_KEYS = ("aoa", "cl", "cd", "k", "cm")
G0 = 9.80665


# ---------------------------------------------------------------------------
# Построение поляры
# ---------------------------------------------------------------------------

def build_polar(results: Sequence[dict],
                include_non_converged: bool = False) -> Dict[str, np.ndarray]:
    """Собирает поляру из результатов, сортируя по углу атаки.

    Точки с неположительным ``cd`` отбрасываются (нерасчётные).
    Возвращает словарь массивов по ключам ``POLAR_KEYS`` (``cm`` — NaN,
    если данных нет).
    """
    rows = []
    for r in results or []:
        try:
            aoa = float(r.get("aoa", r.get("alpha", float("nan"))))
            cl = float(r.get("cl", float("nan")))
            cd = float(r.get("cd", float("nan")))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(aoa) and math.isfinite(cl) and math.isfinite(cd)):
            continue
        if cd <= 0.0:
            continue
        if not include_non_converged and r.get("converged") is False:
            continue
        try:
            cm = float(r.get("cm", float("nan")))
        except (TypeError, ValueError):
            cm = float("nan")
        rows.append((aoa, cl, cd, cl / cd, cm))

    if not rows:
        empty = np.array([])
        return {k: empty.copy() for k in POLAR_KEYS}

    rows.sort(key=lambda t: t[0])
    arr = np.array(rows, dtype=float)
    return {"aoa": arr[:, 0], "cl": arr[:, 1], "cd": arr[:, 2],
            "k": arr[:, 3], "cm": arr[:, 4]}


# ---------------------------------------------------------------------------
# Интегральные характеристики
# ---------------------------------------------------------------------------

def _linear_region_mask(cl: np.ndarray, fraction: float = 0.85) -> np.ndarray:
    """Точки линейной части поляры: |Cl| ≤ fraction·max|Cl|."""
    if cl.size == 0:
        return np.array([], dtype=bool)
    lim = fraction * float(np.max(np.abs(cl)))
    return np.abs(cl) <= max(lim, 1e-9)


def linear_fit_cl_alpha(aoa: np.ndarray, cl: np.ndarray,
                        fraction: float = 0.85) -> Dict[str, float]:
    """Наклон ``dCl/dα`` и угол нулевой подъёмной силы.

    Аппроксимация только по линейной части поляры. Возвращает
    ``{"cl_alpha_deg": 1/град, "cl_alpha_rad": 1/рад, "alpha0": град,
    "r2": …, "n_points": …}``.
    """
    aoa = np.asarray(aoa, dtype=float)
    cl = np.asarray(cl, dtype=float)
    if aoa.size < 2:
        return {"cl_alpha_deg": float("nan"), "cl_alpha_rad": float("nan"),
                "alpha0": float("nan"), "r2": float("nan"), "n_points": int(aoa.size)}

    mask = _linear_region_mask(cl, fraction)
    if int(mask.sum()) < 2:
        mask = np.ones_like(cl, dtype=bool)
    x, y = aoa[mask], cl[mask]

    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    alpha0 = -intercept / slope if abs(slope) > 1e-12 else float("nan")

    return {"cl_alpha_deg": float(slope),
            "cl_alpha_rad": float(slope * 180.0 / math.pi),
            "alpha0": float(alpha0),
            "r2": float(r2),
            "n_points": int(mask.sum())}


def drag_polar_fit(cl: np.ndarray, cd: np.ndarray, aspect_ratio: float,
                   fraction: float = 0.85) -> Dict[str, float]:
    """Аппроксимация ``Cd = Cd0 + Cl² / (π·e·AR)``.

    Возвращает ``{"cd0": …, "oswald_e": …, "r2": …}``. ``oswald_e`` —
    фактор Освальда (эффективность крыла в плане).
    """
    cl = np.asarray(cl, dtype=float)
    cd = np.asarray(cd, dtype=float)
    ar = float(aspect_ratio)
    if cl.size < 2 or ar <= 0:
        return {"cd0": float("nan"), "oswald_e": float("nan"), "r2": float("nan")}

    mask = _linear_region_mask(cl, fraction)
    if int(mask.sum()) < 2:
        mask = np.ones_like(cl, dtype=bool)
    x = cl[mask] ** 2
    y = cd[mask]

    # Cd = Cd0 + x·(1/(π e AR))  →  линейная регрессия по x
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    e = 1.0 / (math.pi * ar * slope) if slope > 1e-12 else float("nan")

    return {"cd0": float(intercept), "oswald_e": float(e), "r2": float(r2)}


def best_ld_point(polar: Dict[str, np.ndarray]) -> Optional[Dict[str, float]]:
    """Точка максимального аэродинамического качества."""
    k = polar.get("k")
    if k is None or k.size == 0:
        return None
    i = int(np.argmax(k))
    return {"aoa": float(polar["aoa"][i]), "cl": float(polar["cl"][i]),
            "cd": float(polar["cd"][i]), "k": float(k[i])}


def stall_point(polar: Dict[str, np.ndarray]) -> Optional[Dict[str, float]]:
    """Точка максимального Cl (условный срыв по расчётной поляре)."""
    cl = polar.get("cl")
    if cl is None or cl.size == 0:
        return None
    i = int(np.argmax(cl))
    return {"aoa": float(polar["aoa"][i]), "cl": float(cl[i]),
            "cd": float(polar["cd"][i]), "k": float(polar["k"][i])}


def cl_at_aoa(polar: Dict[str, np.ndarray], aoa: float) -> float:
    """Интерполяция Cl по углу атаки (вне диапазона — экстраполяция)."""
    x = polar.get("aoa")
    y = polar.get("cl")
    if x is None or x.size == 0:
        return float("nan")
    if x.size == 1:
        return float(y[0])
    return float(np.interp(float(aoa), x, y))


def integrated_characteristics(polar: Dict[str, np.ndarray],
                               aspect_ratio: float,
                               weight_n: Optional[float] = None,
                               rho: float = 1.225,
                               s_ref: float = 1.0,
                               mach: Optional[float] = None
                               ) -> Dict[str, float]:
    """Сводные характеристики по поляре.

    Возвращает: наклон поляры, α₀, Cd₀, фактор Освальда, Cl_max, α срыва,
    K_max, скорость сваливания (если задан вес) и число точек.
    """
    fit = linear_fit_cl_alpha(polar.get("aoa", np.array([])),
                              polar.get("cl", np.array([])))
    dp = drag_polar_fit(polar.get("cl", np.array([])),
                        polar.get("cd", np.array([])), aspect_ratio)
    out: Dict[str, float] = {
        "n_points": int(polar.get("aoa").size) if polar.get("aoa") is not None else 0,
        "cl_alpha_deg": fit["cl_alpha_deg"],
        "cl_alpha_rad": fit["cl_alpha_rad"],
        "alpha0": fit["alpha0"],
        "cd0": dp["cd0"],
        "oswald_e": dp["oswald_e"],
        "aspect_ratio": float(aspect_ratio),
    }
    st = stall_point(polar)
    bl = best_ld_point(polar)
    if st:
        out.update({"cl_max": st["cl"], "aoa_stall": st["aoa"]})
    if bl:
        out.update({"k_max": bl["k"], "aoa_best_k": bl["aoa"]})
    if mach is not None:
        out["mach"] = float(mach)

    if weight_n and st and rho > 0 and s_ref > 0:
        out["v_stall"] = math.sqrt(
            2.0 * float(weight_n) / (rho * s_ref * max(st["cl"], 1e-6)))
    return out
