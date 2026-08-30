# -*- coding: utf-8 -*-
"""
optimization/doe.py — планирование эксперимента (DOE) для перебора вариантов.

ТЗ, п. 5 «Параметрическая оптимизация по таблице параметров»: помимо
ручного заполнения таблицы нужны (а) генерация сетки вариантов по
диапазонам параметров и (б) несколько поколений — каждое следующее
поколение строится вокруг лучшего варианта предыдущего с сужающимся
диапазоном.

Модуль не зависит от Qt, поэтому покрыт тестами напрямую.

Три плана:
  * :func:`full_factorial` — полный факторный план (все комбинации
    уровней). Точен, но растёт как ``levels ** n``;
  * :func:`one_factor_at_a_time` — варьирование по одному параметру
    (``1 + n·(levels−1)`` вариантов) — дёшево и наглядно;
  * :func:`latin_hypercube` — латинский гиперкуб: ``n`` вариантов,
    покрывающих диапазоны равномерно, без полного перебора.
"""

from __future__ import annotations

import itertools
import random
from typing import Dict, List, Optional, Sequence, Tuple

# (ключ, подпись, минимум, максимум, число знаков)
PARAM_SPECS: Tuple[Tuple[str, str, float, float, int], ...] = (
    ("span", "Размах, м", 0.5, 100.0, 3),
    ("chord_root", "Хорда корня, м", 0.05, 30.0, 3),
    ("chord_tip", "Хорда конца, м", 0.02, 30.0, 3),
    ("sweep", "Стреловидность, °", -30.0, 70.0, 2),
    ("twist", "Крутка, °", -15.0, 15.0, 2),
    ("flap_deflection", "Отклонение закрылка, °", 0.0, 45.0, 2),
    ("slat_deflection", "Отклонение предкрылка, °", 0.0, 30.0, 2),
)

SPEC_BY_KEY: Dict[str, Tuple[str, float, float, int]] = {
    k: (label, lo, hi, nd) for k, label, lo, hi, nd in PARAM_SPECS
}

PLAN_FULL = "Полный факторный"
PLAN_OFAT = "По одному параметру"
PLAN_LHS = "Латинский гиперкуб"
PLANS = (PLAN_FULL, PLAN_OFAT, PLAN_LHS)


def _round(v: float, key: str) -> float:
    nd = SPEC_BY_KEY.get(key, ("", 0.0, 1.0, 3))[3]
    return round(float(v), int(nd))


def _clamp(v: float, key: str) -> float:
    _label, lo, hi, _nd = SPEC_BY_KEY.get(key, ("", -1e9, 1e9, 3))
    return min(max(float(v), float(lo)), float(hi))


def levels_for(key: str, low: float, high: float, n_levels: int
               ) -> List[float]:
    """``n_levels`` значений параметра от ``low`` до ``high`` включительно."""
    n = max(1, int(n_levels))
    lo, hi = float(low), float(high)
    if n == 1:
        return [_round(_clamp(lo, key), key)]
    step = (hi - lo) / (n - 1)
    return [_round(_clamp(lo + step * i, key), key) for i in range(n)]


def full_factorial(base: Dict[str, float],
                   ranges: Dict[str, Tuple[float, float]],
                   n_levels: int = 3) -> List[Dict[str, float]]:
    """Полный факторный план: все комбинации уровней варьируемых параметров.

    ``ranges`` — ``{ключ: (мин, макс)}``; параметры, которых нет в
    ``ranges``, берутся из ``base`` без изменения. Первая строка плана —
    базовый вариант (удобно для сравнения).
    """
    keys = [k for k in ranges if k in SPEC_BY_KEY]
    if not keys:
        return [dict(base)]
    grids = [levels_for(k, ranges[k][0], ranges[k][1], n_levels) for k in keys]
    out: List[Dict[str, float]] = []
    for combo in itertools.product(*grids):
        row = dict(base)
        row.update({k: v for k, v in zip(keys, combo)})
        if row not in out:
            out.append(row)
    return out


def one_factor_at_a_time(base: Dict[str, float],
                         ranges: Dict[str, Tuple[float, float]],
                         n_levels: int = 3) -> List[Dict[str, float]]:
    """План «по одному параметру»: базовый вариант + вариации каждого."""
    keys = [k for k in ranges if k in SPEC_BY_KEY]
    out: List[Dict[str, float]] = [dict(base)]
    for k in keys:
        for v in levels_for(k, ranges[k][0], ranges[k][1], n_levels):
            row = dict(base)
            row[k] = v
            if row not in out:
                out.append(row)
    return out


def latin_hypercube(base: Dict[str, float],
                    ranges: Dict[str, Tuple[float, float]],
                    n_samples: int = 9, seed: Optional[int] = None
                    ) -> List[Dict[str, float]]:
    """Латинский гиперкуб: ``n_samples`` вариантов, равномерно по диапазонам.

    Каждый диапазон делится на ``n_samples`` равных слоёв; в каждом слое
    берётся одна точка со случайным сдвигом, причём по каждому параметру
    каждый слой используется ровно один раз (перестановки).
    """
    keys = [k for k in ranges if k in SPEC_BY_KEY]
    n = max(1, int(n_samples))
    if not keys:
        return [dict(base)]
    rng = random.Random(seed)
    perms = {}
    for k in keys:
        order = list(range(n))
        rng.shuffle(order)
        perms[k] = order
    out: List[Dict[str, float]] = []
    for i in range(n):
        row = dict(base)
        for k in keys:
            lo, hi = float(ranges[k][0]), float(ranges[k][1])
            layer = perms[k][i]
            jitter = rng.random()
            v = lo + (hi - lo) * (layer + jitter) / n
            row[k] = _round(_clamp(v, k), k)
        if row not in out:
            out.append(row)
    return out


def make_plan(plan: str, base: Dict[str, float],
              ranges: Dict[str, Tuple[float, float]],
              n_levels: int = 3, n_samples: int = 9,
              seed: Optional[int] = None) -> List[Dict[str, float]]:
    """Построение плана по названию (:data:`PLANS`)."""
    if plan == PLAN_FULL:
        return full_factorial(base, ranges, n_levels)
    if plan == PLAN_OFAT:
        return one_factor_at_a_time(base, ranges, n_levels)
    if plan == PLAN_LHS:
        return latin_hypercube(base, ranges, n_samples, seed=seed)
    raise ValueError(f"Неизвестный план DOE: {plan!r}. Доступны: {PLANS}")


def next_generation(best: Dict[str, float],
                    ranges: Dict[str, Tuple[float, float]],
                    shrink: float = 0.5,
                    keep_center: bool = True) -> Dict[str, Tuple[float, float]]:
    """Диапазоны следующего поколения вокруг лучшего варианта.

    Полуширина каждого диапазона умножается на ``shrink`` и недавится на
    ``best``; при ``keep_center`` лучший вариант остаётся внутри
    диапазона (он и есть центр).
    """
    out: Dict[str, Tuple[float, float]] = {}
    s = min(max(float(shrink), 0.01), 1.0)
    for k, (lo, hi) in ranges.items():
        if k not in SPEC_BY_KEY:
            continue
        center = float(best.get(k, (lo + hi) / 2.0))
        half = abs(hi - lo) / 2.0 * s
        new_lo, new_hi = center - half, center + half
        if keep_center:
            # не выкидываем исходные границы — диапазон только сужается,
            # но лучший вариант обязан остаться внутри
            new_lo = min(new_lo, center)
            new_hi = max(new_hi, center)
        spec_lo, spec_hi = SPEC_BY_KEY[k][1], SPEC_BY_KEY[k][2]
        out[k] = (round(max(new_lo, spec_lo), 6),
                  round(min(new_hi, spec_hi), 6))
    return out


def plan_size(plan: str, n_params: int, n_levels: int = 3,
              n_samples: int = 9) -> int:
    """Число расчётов в плане — для предупреждения пользователя."""
    n_params = max(0, int(n_params))
    n_levels = max(1, int(n_levels))
    if plan == PLAN_FULL:
        return n_levels ** n_params
    if plan == PLAN_OFAT:
        return 1 + n_params * max(0, n_levels - 1)
    if plan == PLAN_LHS:
        return max(1, int(n_samples))
    return 0
