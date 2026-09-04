# -*- coding: utf-8 -*-
"""
official_cases/loader.py — чтение и разбор config.cfg (stdlib-only).

Нужен, чтобы сравнивать конфиг AeroOpt с официальным кейсом SU2 и чтобы
``downloader`` мог править ``MESH_FILENAME`` под скачанную сетку.
"""

from __future__ import annotations

import os
from typing import Dict, Optional


def load_text(path: str) -> str:
    """Читает файл (utf-8, допуская cp1251/повреждённые байты)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def strip_comments(line: str) -> str:
    """Убирает SU2-комментарий '%' (единственный в SU2) и всё после него."""
    if "%" in line:
        return line.split("%", 1)[0]
    return line


def parse_keys(text: str) -> Dict[str, str]:
    """Разбирает текст config.cfg → словарь активных ``KEY= value``.

    Повторы ключей (SU2 на них ругается) схлопываются в последнее значение —
    для диагностики этого достаточно.
    """
    cfg: Dict[str, str] = {}
    for raw in (text or "").splitlines():
        line = strip_comments(raw).strip()
        if not line or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().upper()
        if not key:
            continue
        cfg[key] = val.strip()
    return cfg


def parse_float(value: Optional[str]) -> Optional[float]:
    """Пытается распарсить число из строки SU2 (с учётом списка)."""
    if value is None:
        return None
    v = str(value).strip()
    for token in v.replace("(", " ").replace(")", " ").replace(",", " ").split():
        t = token.strip()
        if not t:
            continue
        try:
            return float(t)
        except ValueError:
            continue
    return None


def get_float(cfg: Dict[str, str], key: str) -> Optional[float]:
    return parse_float(cfg.get(key.upper()))


def mesh_filename_from(text: str) -> str:
    """Возвращает значение ``MESH_FILENAME`` из текста конфига."""
    return parse_keys(text).get("MESH_FILENAME", "")


def rewrite_mesh_filename(text: str, new_name: str) -> str:
    """Заменяет MESH_FILENAME в тексте конфига и оставляет остальное как есть."""
    if not new_name:
        return text
    out = []
    changed = False
    for raw in (text or "").splitlines(keepends=True):
        stripped = strip_comments(raw).strip()
        if stripped.upper().startswith("MESH_FILENAME=") and not changed:
            indent = raw[:len(raw) - len(raw.lstrip())]
            out.append(f"{indent}MESH_FILENAME= {new_name}\n")
            changed = True
        else:
            out.append(raw)
    if not changed:
        out.append(f"MESH_FILENAME= {new_name}\n")
    return "".join(out)


def _configs_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


def bundled_config_path(config_file: str) -> str:
    """Абсолютный путь к встроенному файлу конфига."""
    return os.path.join(_configs_dir(), os.path.basename(config_file))


def bundled_config_text(config_file: str) -> str:
    """Читает встроенный официальный конфиг."""
    return load_text(bundled_config_path(config_file))
