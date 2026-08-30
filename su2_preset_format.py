# -*- coding: utf-8 -*-
"""
su2_preset_format.py — формат пресета конфигурации расчёта SU2.

ТЗ, пункт 6 «Формат Configure» (уточнено заказчиком: формат config.cfg
и именованных пресетов — импорт/экспорт, валидация, шаблоны).

Формат (JSON, UTF-8, расширение ``.su2preset``)
-----------------------------------------------
::

    {
      "format":         "aeroopt.su2preset",
      "schema_version": 1,
      "name":           "Устойчивый крейсер",
      "description":    "CFL 2.0, первый порядок",
      "based_on":       "safe",
      "created":        "2026-08-30T12:00:00",
      "params": {
        "CFL_NUMBER": "2.0",
        "CFL_ADAPT":  "NO",
        "MUSCL_FLOW": "NO"
      }
    }

Правила:
  * ``format`` — обязательное поле, строка ``aeroopt.su2preset``;
  * ``schema_version`` — целое, текущая версия :data:`SCHEMA_VERSION`;
  * ``name`` — непустая строка (имя пресета в меню);
  * ``params`` — словарь «ключ SU2 → значение строкой»; значения в
    config.cfg пишутся как ``KEY= value``;
  * ключи сверяются с каталогом параметров (``su2_config_dialog.PARAMS``,
    если доступен), значения — с типом и диапазоном из каталога;
  * неизвестные ключи не отвергаются (SU2 развивается), но попадают в
    предупреждения — это осознанное решение, чтобы пресеты не ломались
    при обновлении SU2.

Модуль не зависит от Qt, поэтому пригоден и для CLI, и для тестов.
"""

from __future__ import annotations

import datetime
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

FORMAT_ID = "aeroopt.su2preset"
SCHEMA_VERSION = 1
EXTENSION = ".su2preset"


# ---------------------------------------------------------------------------
# Каталог ключей (берётся из su2_config_dialog, если он доступен)
# ---------------------------------------------------------------------------

# Запасной каталог: ключ → (тип, опции/диапазон). Используется, когда
# su2_config_dialog недоступен (например, вне GUI).
_FALLBACK_CATALOGUE: Dict[str, Tuple[str, object]] = {
    "SOLVER": ("combo", ["EULER", "NAVIER_STOKES", "RANS", "INC_EULER",
                         "INC_NAVIER_STOKES", "INC_RANS"]),
    "MATH_PROBLEM": ("combo", ["DIRECT", "CONTINUOUS_ADJOINT"]),
    "MACH_NUMBER": ("double", (0.0, 5.0)),
    "AOA": ("double", (-90.0, 90.0)),
    "FREESTREAM_PRESSURE": ("double", (1000.0, 1e7)),
    "FREESTREAM_TEMPERATURE": ("double", (100.0, 1000.0)),
    "REF_LENGTH": ("double", (0.001, 1000.0)),
    "REF_AREA": ("double", (0.001, 1e6)),
    "MARKER_EULER": ("str", None),
    "MARKER_FAR": ("str", None),
    "CONV_NUM_METHOD_FLOW": ("combo", ["ROE", "JST", "AUSM", "HLLC",
                                       "LAX-FRIEDRICH", "ROE_SW"]),
    "MUSCL_FLOW": ("yesno", None),
    "ENTROPY_FIX_COEFF": ("double", (0.0, 1.0)),
    "NUM_METHOD_GRAD": ("combo", ["WEIGHTED_LEAST_SQUARES", "GREEN_GAUSS"]),
    "TIME_DISCRETE_FLOW": ("combo", ["EULER_IMPLICIT", "EULER_EXPLICIT",
                                     "RUNGE-KUTTA_EXPLICIT"]),
    "CFL_NUMBER": ("double", (0.01, 1000.0)),
    "CFL_ADAPT": ("yesno", None),
    "LINEAR_SOLVER": ("combo", ["FGMRES", "GMRES", "BCGSTAB"]),
    "LINEAR_SOLVER_PREC": ("combo", ["ILU", "JACOBI", "LINELET"]),
    "LINEAR_SOLVER_ITER": ("int", (1, 200)),
    "INNER_ITER": ("int", (10, 200000)),
    "CONV_RESIDUAL_MINVAL": ("double", (-12.0, -1.0)),
    "CONV_STARTITER": ("int", (0, 1000)),
}


def key_catalogue() -> Dict[str, Tuple[str, object]]:
    """Каталог «ключ SU2 → (тип, опции/диапазон)».

    Использует ``su2_config_dialog.PARAMS`` (единый источник правды для
    GUI), при недоступности Qt — запасной каталог.
    """
    try:
        from su2_config_dialog import PARAMS  # type: ignore
    except Exception:
        return dict(_FALLBACK_CATALOGUE)
    cat: Dict[str, Tuple[str, object]] = {}
    for row in PARAMS:
        # (группа, ключ, подпись, тип, опции, пределы, подсказка)
        _grp, key, _label, kind, options, limits, _tip = row
        if kind == "combo":
            cat[key] = ("combo", list(options or []))
        elif kind == "double":
            cat[key] = ("double", (limits[0], limits[1]))
        elif kind == "int":
            cat[key] = ("int", (limits[0], limits[1]))
        else:
            cat[key] = (kind, None)
    return cat


# ---------------------------------------------------------------------------
# Встроенные шаблоны
# ---------------------------------------------------------------------------

def builtin_presets() -> Dict[str, Dict[str, object]]:
    """Шаблоны пресетов, поставляемые с приложением."""
    return {
        "Стандартный": {
            "description": "Настройки по умолчанию: CFL с адаптацией, 2-й порядок.",
            "params": {
                "TIME_DISCRETE_FLOW": "EULER_IMPLICIT",
                "CFL_NUMBER": "5.0",
                "CFL_ADAPT": "YES",
                "MUSCL_FLOW": "YES",
                "ENTROPY_FIX_COEFF": "0.05",
                "NUM_METHOD_GRAD": "WEIGHTED_LEAST_SQUARES",
            },
        },
        "Устойчивый (safe)": {
            "description": "CFL 2.0 без адаптации, первый порядок. "
                           "Для расходящихся задач.",
            "params": {
                "TIME_DISCRETE_FLOW": "EULER_IMPLICIT",
                "CFL_NUMBER": "2.0",
                "CFL_ADAPT": "NO",
                "MUSCL_FLOW": "NO",
                "ENTROPY_FIX_COEFF": "0.1",
                "NUM_METHOD_GRAD": "WEIGHTED_LEAST_SQUARES",
            },
        },
        "Ультра-устойчивый (ultra)": {
            "description": "CFL 0.5, усиленная энтропийная поправка, "
                           "20 итераций линейного решателя.",
            "params": {
                "TIME_DISCRETE_FLOW": "EULER_IMPLICIT",
                "CFL_NUMBER": "0.5",
                "CFL_ADAPT": "NO",
                "MUSCL_FLOW": "NO",
                "ENTROPY_FIX_COEFF": "0.2",
                "NUM_METHOD_GRAD": "WEIGHTED_LEAST_SQUARES",
                "LINEAR_SOLVER_ITER": "20",
            },
        },
    }


# ---------------------------------------------------------------------------
# Валидация
# ---------------------------------------------------------------------------

def validate_preset(data: object,
                    catalogue: Optional[Dict[str, Tuple[str, object]]] = None
                    ) -> Dict[str, object]:
    """Проверяет структуру и значения пресета.

    Возвращает ``{"ok": bool, "errors": [...], "warnings": [...]}``.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(data, dict):
        return {"ok": False,
                "errors": ["Пресет должен быть JSON-объектом."],
                "warnings": []}

    if data.get("format") != FORMAT_ID:
        errors.append(f"Поле 'format' должно быть '{FORMAT_ID}' "
                      f"(получено: {data.get('format')!r}).")

    ver = data.get("schema_version")
    if not isinstance(ver, int) or isinstance(ver, bool):
        errors.append("Поле 'schema_version' должно быть целым числом.")
    elif ver > SCHEMA_VERSION:
        errors.append(f"Версия формата {ver} новее поддерживаемой "
                      f"{SCHEMA_VERSION} — обновите AeroOpt.")

    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append("Поле 'name' должно быть непустой строкой.")

    params = data.get("params")
    if not isinstance(params, dict):
        errors.append("Поле 'params' должно быть словарем ключ→значение.")
        params = {}

    cat = catalogue if catalogue is not None else key_catalogue()
    for key, value in (params or {}).items():
        if not isinstance(key, str) or not key.strip():
            errors.append(f"Ключ параметра должен быть строкой: {key!r}.")
            continue
        if not isinstance(value, (str, int, float)):
            errors.append(f"{key}: значение должно быть строкой или числом.")
            continue
        sval = str(value).strip()
        if key not in cat:
            warnings.append(f"{key}: ключ не из каталога AeroOpt "
                            f"(проверьте написание).")
            continue
        kind, spec = cat[key]
        if kind == "yesno":
            if sval.upper() not in ("YES", "NO"):
                errors.append(f"{key}: ожидается YES или NO, получено {sval!r}.")
        elif kind == "combo":
            allowed = [str(x).upper() for x in (spec or [])]
            if allowed and sval.upper() not in allowed:
                errors.append(f"{key}: {sval!r} не входит в список "
                              f"{', '.join(spec)}.")
        elif kind in ("int", "double"):
            try:
                num = float(str(sval).split()[0])
            except (ValueError, IndexError):
                errors.append(f"{key}: {sval!r} не число.")
                continue
            if spec:
                lo, hi = float(spec[0]), float(spec[1])
                if not (lo <= num <= hi):
                    errors.append(f"{key}: {num} вне диапазона {lo}…{hi}.")
            if kind == "int" and abs(num - round(num)) > 1e-9:
                errors.append(f"{key}: ожидается целое, получено {sval!r}.")

    return {"ok": not errors, "errors": errors, "warnings": warnings}


# ---------------------------------------------------------------------------
# Импорт / экспорт
# ---------------------------------------------------------------------------

def make_preset(name: str, params: Dict[str, object],
                description: str = "", based_on: Optional[str] = None
                ) -> Dict[str, object]:
    """Собирает словарь пресета текущей версии формата."""
    return {
        "format": FORMAT_ID,
        "schema_version": SCHEMA_VERSION,
        "name": str(name).strip(),
        "description": str(description or ""),
        "based_on": based_on,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "params": {str(k).strip(): str(v).strip()
                   for k, v in (params or {}).items()},
    }


def export_preset(path: str, name: str, params: Dict[str, object],
                  description: str = "", based_on: Optional[str] = None,
                  strict: bool = True) -> str:
    """Пишет пресет в файл. Возвращает путь.

    ``strict=True`` — при ошибках валидации файл не пишется
    (поднимается ``ValueError``).
    """
    preset = make_preset(name, params, description, based_on)
    report = validate_preset(preset)
    if strict and not report["ok"]:
        raise ValueError("Пресет не прошёл валидацию:\n  "
                         + "\n  ".join(report["errors"]))
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(preset, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path


def import_preset(path: str) -> Dict[str, object]:
    """Читает пресет из файла и валидирует его.

    При ошибке поднимает ``ValueError`` с человеком читаемым описанием.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        raise ValueError(f"Не удалось прочитать файл пресета: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Файл не является корректным JSON: {e}") from e
    report = validate_preset(data)
    if not report["ok"]:
        raise ValueError("Пресет не прошёл валидацию:\n  "
                         + "\n  ".join(report["errors"]))
    data["_warnings"] = report["warnings"]
    return data


# ---------------------------------------------------------------------------
# Работа с config.cfg
# ---------------------------------------------------------------------------

def parse_cfg_text(text: str) -> Dict[str, str]:
    """Разбирает текст config.cfg → словарь активных ключей."""
    cfg: Dict[str, str] = {}
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or s.startswith("%") or s.startswith("#") or "=" not in s:
            continue
        key, _, val = s.partition("=")
        cfg[key.strip()] = val.strip()
    return cfg


def preset_to_cfg_lines(preset: Dict[str, object]) -> List[str]:
    """Строки ``KEY= value`` для вставки в config.cfg."""
    params = preset.get("params") or {}
    return [f"{k}= {v}" for k, v in params.items()]


def diff_presets(a: Dict[str, object], b: Dict[str, object]
                 ) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """Отличия двух пресетов: [(ключ, значение A, значение B), …]."""
    pa = {str(k): str(v) for k, v in (a.get("params") or {}).items()}
    pb = {str(k): str(v) for k, v in (b.get("params") or {}).items()}
    out = []
    for k in sorted(set(pa) | set(pb)):
        if pa.get(k) != pb.get(k):
            out.append((k, pa.get(k), pb.get(k)))
    return out


def describe_format() -> str:
    """Краткое описание формата — для справки в меню «SU2»."""
    return (
        f"Формат пресета AeroOpt ({FORMAT_ID}, версия {SCHEMA_VERSION}).\n"
        "JSON-файл с расширением .su2preset:\n"
        "  format         — всегда '" + FORMAT_ID + "'\n"
        "  schema_version — целое, сейчас " + str(SCHEMA_VERSION) + "\n"
        "  name           — имя пресета (обязательно, непустое)\n"
        "  description    — свободный текст\n"
        "  based_on       — имя шаблона, от которого произошёл\n"
        "  created        — дата создания (ISO 8601)\n"
        "  params         — словарь «ключ SU2 → значение строкой»\n\n"
        "Значения пишутся в config.cfg как «KEY= value».\n"
        "Ключи сверяются с каталогом параметров; неизвестные ключи\n"
        "дают предупреждение, но не ошибку (совместимость с новыми\n"
        "версиями SU2)."
    )
