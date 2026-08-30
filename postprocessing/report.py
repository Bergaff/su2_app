# -*- coding: utf-8 -*-
"""
postprocessing/report.py — отчёты по шаблону и экспорт результатов.

ТЗ, пункт 6 «Спецфункции»: формирование отчётов по шаблону.

Шаблоны:
    ``"краткий"``  — сводка: интегральные характеристики + вердикт;
    ``"полный"``   — сводка + таблица точек поляры + данные проекта;
    ``"поляра"``   — только таблица точек.

Форматы вывода: текст (``make_report``), HTML (``render_html``),
CSV (``export_csv``).
"""

from __future__ import annotations

import csv
import datetime
import html
import io
import math
import os
from typing import Dict, Optional, Sequence

import numpy as np

from .polar import build_polar, integrated_characteristics

TEMPLATES = ("краткий", "полный", "поляра")

# Подстановки шаблона: имя поля → (заголовок, единицы, число знаков)
_FIELDS = {
    "n_points": ("Число расчётных точек", "", 0),
    "cl_alpha_deg": ("Наклон поляры dCl/dα", "1/град", 4),
    "alpha0": ("Угол нулевой подъёмной силы", "град", 2),
    "cd0": ("Профильное сопротивление Cd0", "", 5),
    "oswald_e": ("Фактор Освальда e", "", 3),
    "cl_max": ("Cl максимальный", "", 4),
    "aoa_stall": ("Угол атаки при Cl max", "град", 2),
    "k_max": ("Максимальное качество K", "", 2),
    "aoa_best_k": ("Угол атаки при K max", "град", 2),
    "v_stall": ("Скорость сваливания", "м/с", 1),
    "mach": ("Число Маха", "", 3),
    "aspect_ratio": ("Удлинение крыла", "", 2),
}


# ---------------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------------

def _fmt(value, nd: int = 3) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return "—"
    return f"{v:.{nd}f}"


def _polar_table(polar: Dict[str, np.ndarray], sep: str = " | ") -> str:
    aoa = polar.get("aoa")
    if aoa is None or aoa.size == 0:
        return "  (нет точек)"
    lines = [f"{'α, град':>8}{sep}{'Cl':>9}{sep}{'Cd':>9}{sep}"
             f"{'K':>7}{sep}{'Cm':>9}"]
    lines.append("-" * 52)
    for i in range(aoa.size):
        lines.append(
            f"{polar['aoa'][i]:8.2f}{sep}{polar['cl'][i]:9.4f}{sep}"
            f"{polar['cd'][i]:9.5f}{sep}{polar['k'][i]:7.2f}{sep}"
            f"{_fmt(polar['cm'][i], 4):>9}")
    return "\n".join(lines)


def _project_lines(info: Optional[Dict[str, object]]) -> list:
    if not info:
        return []
    out = []
    for k, v in info.items():
        out.append(f"  {k}: {v}")
    return out


# ---------------------------------------------------------------------------
# Текстовый отчёт
# ---------------------------------------------------------------------------

def make_report(results: Sequence[dict],
                aspect_ratio: float = 10.0,
                template: str = "полный",
                project_info: Optional[Dict[str, object]] = None,
                weight_n: Optional[float] = None,
                rho: float = 1.225,
                s_ref: float = 1.0,
                mach: Optional[float] = None) -> str:
    """Формирует отчёт по шаблону.

    ``template`` — один из :data:`TEMPLATES`.
    """
    tpl = str(template or "полный").lower()
    if tpl not in TEMPLATES:
        raise ValueError(f"Неизвестный шаблон: {template!r}. "
                         f"Доступны: {', '.join(TEMPLATES)}")

    polar = build_polar(results)
    chars = integrated_characteristics(polar, aspect_ratio,
                                       weight_n=weight_n, rho=rho,
                                       s_ref=s_ref, mach=mach)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    out = [f"ОТЧЁТ ПО РЕЗУЛЬТАТАМ РАСЧЁТА — AeroOpt",
           f"Шаблон: {tpl}",
           f"Дата:   {stamp}",
           "=" * 52]

    if project_info:
        out.append("ДАННЫЕ ЗАДАЧИ")
        out.extend(_project_lines(project_info))
        out.append("-" * 52)

    if tpl in ("краткий", "полный"):
        out.append("ИНТЕГРАЛЬНЫЕ ХАРАКТЕРИСТИКИ")
        for key, (title, unit, nd) in _FIELDS.items():
            if key not in chars:
                continue
            suffix = f", {unit}" if unit else ""
            out.append(f"  {title}{suffix}: {_fmt(chars[key], nd)}")
        out.append("-" * 52)

    if tpl in ("полный", "поляра"):
        out.append("ПОЛЯРА")
        out.append(_polar_table(polar))

    if tpl == "краткий":
        k = chars.get("k_max")
        cl = chars.get("cl_max")
        out.append("ВЫВОД")
        out.append(f"  K_max = {_fmt(k, 2)} при α = "
                   f"{_fmt(chars.get('aoa_best_k'), 1)} град; "
                   f"Cl_max = {_fmt(cl, 3)}.")

    out.append("=" * 52)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def render_html(results: Sequence[dict], **kw) -> str:
    """Тот же отчёт в виде простой HTML-страницы."""
    text = make_report(results, **kw)
    body = html.escape(text)
    return ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>Отчёт AeroOpt</title><style>"
            "body{font-family:Consolas,monospace;white-space:pre;"
            "margin:24px;background:#fff;color:#222}</style></head><body>"
            f"{body}</body></html>")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def export_csv(path: str, results: Sequence[dict]) -> str:
    """Экспорт поляры в CSV (α, Cl, Cd, K, Cm). Возвращает путь."""
    polar = build_polar(results)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";", lineterminator="\n")
    w.writerow(["AoA_deg", "Cl", "Cd", "K", "Cm"])
    aoa = polar.get("aoa")
    if aoa is not None:
        for i in range(aoa.size):
            w.writerow([f"{polar['aoa'][i]:.4f}", f"{polar['cl'][i]:.6f}",
                        f"{polar['cd'][i]:.6f}", f"{polar['k'][i]:.4f}",
                        f"{polar['cm'][i]:.6f}"])
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(buf.getvalue())
    return path
