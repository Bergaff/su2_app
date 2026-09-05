# -*- coding: utf-8 -*-
"""
official_cases — библиотека официальных конфигов и 3D-моделей SU2.

Что это
-------
Вспомогательный пакет, который добавляет в AeroOpt **эталонные тест-кейсы
SU2 от разработчиков** (репозитории su2code/SU2 и su2code/Tutorials):
встроенные официальные ``config.cfg``, возможность скачать соответствующую
3D-сетку и инструмент диагностики, который объясняет, почему ваш расчёт
даёт неправдоподобно большие Cl/Cd.

Публичный API
-------------
    from official_cases import (
        list_cases, get_case, find_by_mach, find_by_solver, nearest_for,
        diagnose, compare_file, render_diagnosis,
        download_mesh, is_downloaded, prepare_case_dir, meshes_report,
    )

CLI (без Qt и numpy, работает в любом окружении):
    python -m official_cases list
    python -m official_cases show inv_oneram6
    python -m official_cases download inv_oneram6
    python -m official_cases prepare inv_oneram6 <out_dir>
    python -m official_cases compare <config.cfg> [--case inv_oneram6]
    python -m official_cases meshes

Назначение (для пользователя)
-----------------------------
1. **Дополнить конфиги официальными.** В ``official_cases/configs/`` лежат
   официальные ``config.cfg`` — их можно открыть и сравнить со своим.
2. **3D-модели.** Официальные сетки (ONERA M6 и др.) скачиваются командой
   ``download`` и кладутся в ``official_cases/meshes/`` (в Git не попадают).
3. **Понять «большие значения».** ``compare`` находит, чем ваш конфиг
   отличается от официального, и указывает наиболее вероятную причину
   (обычно — сжимаемый решатель на низком Mach без несжимаемой постановки
   или без low-Mach-прекондиционера).
"""

from __future__ import annotations

from .catalog import (
    OfficialCase,
    OFFICIAL_CASES,
    REPO_SU2,
    REPO_TUTORIALS,
    list_cases,
    get_case,
    find_by_solver,
    find_by_mach,
    nearest_for,
)
from .compare import (
    diagnose,
    compare_file,
    render_diagnosis,
)
from .downloader import (
    meshes_dir,
    mesh_local_path,
    is_downloaded,
    download_mesh,
    prepare_case_dir,
    prepare_case_run_dir,
    meshes_report,
)
from .loader import (
    load_text,
    parse_keys,
    bundled_config_path,
    bundled_config_text,
    body_markers_from_config,
)
from .surface import (
    parse_su2_text,
    read_su2_boundary,
    is_manifold_closed,
)

__all__ = [
    "OfficialCase",
    "OFFICIAL_CASES",
    "REPO_SU2",
    "REPO_TUTORIALS",
    "list_cases",
    "get_case",
    "find_by_solver",
    "find_by_mach",
    "nearest_for",
    "diagnose",
    "compare_file",
    "render_diagnosis",
    "meshes_dir",
    "mesh_local_path",
    "is_downloaded",
    "download_mesh",
    "prepare_case_dir",
    "prepare_case_run_dir",
    "meshes_report",
    "load_text",
    "parse_keys",
    "bundled_config_path",
    "bundled_config_text",
    "body_markers_from_config",
    "parse_su2_text",
    "read_su2_boundary",
    "is_manifold_closed",
]
