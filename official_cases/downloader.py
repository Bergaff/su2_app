# -*- coding: utf-8 -*-
"""
official_cases/downloader.py — скачивание официальных сеток SU2 (3D-моделей).

Официальные mesh-файлы SU2 живут в отдельном репозитории ``su2code/Tutorials``
и не коммитятся в ``su2code/SU2`` (кроме пары исключений). Здесь мы тянем их
через GitHub Contents API (``Accept: application/vnd.github.raw``) в локальный
кэш ``official_cases/meshes/<case_id>/<mesh_filename>`` и, если нужно, правим
``MESH_FILENAME`` в локальной копии конфига, чтобы сетка подхватилась как есть.

Папка ``meshes/`` добавлена в ``.gitignore``: крупные сетки (десятки МБ) в
Git не кладём.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import urllib.request
from typing import Optional

from .catalog import OfficialCase, get_case, list_cases
from .loader import (
    bundled_config_text, rewrite_mesh_filename,
)

# Кэш сеток рядом с пакетом (не коммитится).
_MESHES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes")

_API = "https://api.github.com/repos/{repo}/contents/{path}"
_RAW = "https://raw.githubusercontent.com/{repo}/{branch}/{path}"
_RAW_ACCEPT = "application/vnd.github.raw"


def meshes_dir() -> str:
    os.makedirs(_MESHES_DIR, exist_ok=True)
    return _MESHES_DIR


def mesh_local_path(case_id: str) -> str:
    """Путь, куда будет скачана сетка кейса (по имени из конфига)."""
    case = get_case(case_id)
    if not case.mesh_filename:
        raise ValueError(f"Кейс {case_id!r} не имеет сетки (mesh_filename пуст).")
    return os.path.join(meshes_dir(), case_id, case.mesh_filename)


def is_downloaded(case_id: str) -> bool:
    case = get_case(case_id)
    return bool(case.mesh_filename) and os.path.exists(mesh_local_path(case_id))


def fetch_remote(repo: str, path: str, branch: str = "master") -> bytes:
    """Скачивает бинарный файл по URL.

    GitHub Contents API без токена ограничен **60 запросами в час** — на
    повторном скачивании (например, второй сетки подряд) сервер отдаёт
    ``HTTP 403: rate limit exceeded``. Поэтому сначала пробуем CDN
    ``raw.githubusercontent.com`` (лимита на публичные репозитории нет), а
    при неудаче (TLS/прокси/CDN не отдаёт файл) — GitHub Contents API с
    ``Accept: application/vnd.github.raw``.
    """
    errors = []
    candidates = [
        _RAW.format(repo=repo, branch=branch, path=path),
        _API.format(repo=repo, path=path),
    ]
    for url in candidates:
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "AeroOpt-official-cases",
                "Accept": _RAW_ACCEPT,
            })
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read()
        except Exception as e:                                 # pragma: no cover
            errors.append(f"{url} → {type(e).__name__}: {e}")
    raise RuntimeError(
        "Не удалось скачать файл ни по одному из адресов:\n"
        + "\n".join(errors))


def _contents_url(repo: str, path: str) -> str:
    return _API.format(repo=repo, path=path)


def download_mesh(case_id: str, overwrite: bool = False) -> str:
    """Скачивает официальную сетку кейса и возвращает путь к файлу.

    Если сетка уже есть и ``overwrite=False`` — возвращает существующий путь.
    """
    case = get_case(case_id)
    if not case.mesh_path or not case.mesh_filename:
        raise ValueError(
            f"Кейс {case_id!r} не имеет источники сетки (mesh_path/mesh_filename). "
            "Сетку для него нужно взять вручную (см. notes)."
        )
    dest = mesh_local_path(case_id)
    if os.path.exists(dest) and not overwrite:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    data = fetch_remote(case.mesh_repo, case.mesh_path)
    with open(dest, "wb") as f:
        f.write(data)
    return dest


def prepare_case_dir(case_id: str, out_dir: str,
                     download: bool = True) -> dict:
    """Готовит каталог ``out_dir`` с конфигом и (опционально) сеткой кейса.

    Возвращает dict: ``{"case_dir": ..., "config": ..., "mesh": ...}``.
    Если сетка не скачана (файл отсутствует), в ``mesh`` будет None, а в
    ``mesh_error`` — пояснение. Конфиг копируется из встроенного, и его
    ``MESH_FILENAME`` приводится к скачанному файлу сетки.
    """
    case = get_case(case_id)
    os.makedirs(out_dir, exist_ok=True)
    text = bundled_config_text(case.config_file)

    mesh_path = None
    mesh_error = None
    mesh_in_dir = None
    if download:
        try:
            target = download_mesh(case_id)
            text = rewrite_mesh_filename(text, os.path.basename(target))
            mesh_path = target
            # Кладём сетку рядом с config.cfg, чтобы SU2 нашёл её по
            # MESH_FILENAME — иначе каталог «конфиг+модель» не запускался.
            mesh_in_dir = os.path.join(out_dir, os.path.basename(target))
            shutil.copy2(target, mesh_in_dir)
        except Exception as e:            # pragma: no cover — сетевой путь
            mesh_error = f"Не удалось скачать сетку: {type(e).__name__}: {e}"

    cfg_out = os.path.join(out_dir, "config.cfg")
    with open(cfg_out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return {
        "case_id": case_id,
        "case_dir": out_dir,
        "config": cfg_out,
        "mesh": mesh_path,
        "mesh_in_dir": mesh_in_dir,
        "mesh_error": mesh_error,
        "mesh_filename": case.mesh_filename,
        "mesh_expected_in_dir": os.path.basename(case.mesh_filename),
    }


# ---------------------------------------------------------------------------
# Информационная справка (без выхода в сеть)
# ---------------------------------------------------------------------------

def _fmt_size(n: int) -> str:
    if not n:
        return "?"
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} ТБ"


def meshes_report() -> list:
    """Таблица «какие официальные 3D-сетки доступны и их состояние»."""
    rows = []
    for cid in list_cases():
        case = get_case(cid)
        if not case.mesh_path:
            rows.append({"id": cid, "name": case.name, "available": False,
                         "size": 0, "downloaded": False})
            continue
        rows.append({
            "id": cid,
            "name": case.name,
            "available": True,
            "size": _fmt_size(case.mesh_size),
            "downloaded": is_downloaded(cid),
            "mesh": case.mesh_filename,
            "repo": case.mesh_repo,
        })
    return rows
