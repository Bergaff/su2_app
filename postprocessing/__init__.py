# -*- coding: utf-8 -*-
"""
postprocessing — спецфункции постобработки результатов расчёта.

ТЗ, пункт 6 «Спецфункции» (уточнено заказчиком: спец. функции
постобработки — поляра, диаграммы, интегральные характеристики,
отчёты по шаблону).

Состав:
    postprocessing.polar  — построение поляры и интегральных характеристик
    postprocessing.report — отчёты по шаблону (текст/HTML) и экспорт CSV
"""

from .polar import (build_polar, linear_fit_cl_alpha, drag_polar_fit,
                    best_ld_point, stall_point, cl_at_aoa,
                    integrated_characteristics, POLAR_KEYS)
from .report import make_report, export_csv, render_html, TEMPLATES

__all__ = [
    "build_polar",
    "linear_fit_cl_alpha",
    "drag_polar_fit",
    "best_ld_point",
    "stall_point",
    "cl_at_aoa",
    "integrated_characteristics",
    "POLAR_KEYS",
    "make_report",
    "export_csv",
    "render_html",
    "TEMPLATES",
]
