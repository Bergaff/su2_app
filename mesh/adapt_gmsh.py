# -*- coding: utf-8 -*-
"""
mesh/adapt_gmsh.py — адаптивное сгущение сетки по решению (ТЗ, п. 3).

Два независимых механизма адаптации в приложении:

1. **SU2_ADAPT** (``solver.workers.run_su2_adapt``) — перестройка сетки
   формата ``.su2`` по градиентам решения. Работает после расчёта, требует
   ``restart.dat``.

2. **Этот модуль** — адаптация на стороне gmsh: по распределению ``Cp``
   с поверхности (``surface_flow.csv`` от SU2) строится поле целевых
   размеров элемента, и сетка перестраивается уже с этим полем. Нужно,
   когда сетка строится через gmsh (в т.ч. из CAD) и хочется получить
   сгущение там, где большие градиенты давления, ещё до дорогого расчёта.

Модуль разделён на две части: чистый Python (разбор CSV, построение
метрики, запись файла метрики) — он покрыт тестами и работает без gmsh, —
и тонкая обёртка над gmsh (:func:`rebuild_with_metric`), которая
импортирует gmsh лениво.

Формат файла метрики — Gmsh ``.msh`` версии 2.2 с узловыми данными
``$NodeData``: gmsh читает его при ``Mesh.Metric = <файл>`` и использует
значение как целевой размер элемента в узле.
"""

from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "parse_surface_flow_csv",
    "pressure_gradient_along_surface",
    "surface_size_metric",
    "write_metric_msh",
    "write_metric_geo",
    "rebuild_with_metric",
    "adaptivity_report",
]


# ---------------------------------------------------------------------------
# Разбор surface_flow.csv (чистый Python)
# ---------------------------------------------------------------------------

_COORD_NAMES = {"x", "x_coord", "x-coordinate"}
_COL_ALIASES = {
    "cp": ("cp", "c_pressure", "c_press", "pressure_coefficient", "cp_avg"),
    "cf": ("cf", "c_friction", "c_fric", "skin_friction_coefficient"),
    "p": ("pressure", "static_pressure", "p_static"),
    "mach": ("mach", "mach_number"),
}


def _clean(name: str) -> str:
    return (name or "").strip().strip('"').strip("'").strip().lower()


def _find_column(header: Sequence[str], wanted: Sequence[str]) -> Optional[int]:
    cleaned = [_clean(h) for h in header]
    for w in wanted:
        if w in cleaned:
            return cleaned.index(w)
    return None


def parse_surface_flow_csv(path: str) -> Dict[str, np.ndarray]:
    """Читает ``surface_flow.csv`` из каталога расчёта SU2.

    Возвращает словарь массивов: ``x``, ``y``, ``z`` (координаты узлов
    поверхности) и, если они есть в файле, ``cp``, ``cf``, ``p``, ``mach``.

    Формат SU2: первая строка — имена колонок в двойных кавычках,
    разделитель — запятая. Парсер устойчив к кавычкам, лишним пробелам,
    другому регистру имён и отсутствию ``z`` (2D-расчёты).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except Exception:
            dialect = csv.excel
        rows = [r for r in csv.reader(f, dialect) if r and any(c.strip() for c in r)]

    if len(rows) < 2:
        raise ValueError(f"В {os.path.basename(path)} нет данных")

    header = rows[0]
    ix = _find_column(header, _COORD_NAMES)
    iy = _find_column(header, ("y", "y_coord", "y-coordinate"))
    iz = _find_column(header, ("z", "z_coord", "z-coordinate"))
    icp = _find_column(header, _COL_ALIASES["cp"])
    icf = _find_column(header, _COL_ALIASES["cf"])
    ip = _find_column(header, _COL_ALIASES["p"])
    im = _find_column(header, _COL_ALIASES["mach"])

    if ix is None or iy is None:
        raise ValueError(
            "Не найдены колонки координат (x, y) в "
            f"{os.path.basename(path)}: {header}")

    data: Dict[str, List[float]] = {"x": [], "y": [], "z": []}
    for key, idx in (("cp", icp), ("cf", icf), ("p", ip), ("mach", im)):
        if idx is not None:
            data[key] = []

    for r in rows[1:]:
        try:
            data["x"].append(float(r[ix]))
            data["y"].append(float(r[iy]))
            data["z"].append(float(r[iz]) if iz is not None else 0.0)
        except (ValueError, IndexError):
            continue
        for key, idx in (("cp", icp), ("cf", icf), ("p", ip), ("mach", im)):
            if idx is None:
                continue
            try:
                data[key].append(float(r[idx]))
            except (ValueError, IndexError):
                data[key].append(float("nan"))

    n = len(data["x"])
    if n == 0:
        raise ValueError(f"Не удалось разобрать ни одной строки в {path}")
    out = {k: np.asarray(v, dtype=float)[:n] for k, v in data.items()}
    return out


# ---------------------------------------------------------------------------
# Поле целевых размеров (чистый Python)
# ---------------------------------------------------------------------------

def pressure_gradient_along_surface(x: np.ndarray, y: np.ndarray,
                                    cp: np.ndarray,
                                    z: Optional[np.ndarray] = None
                                    ) -> np.ndarray:
    """|dCp/ds| вдоль поверхности (s — длина дуги), методом конечных разностей.

    Точки сортируются по длине дуги, производная считается центральными
    разностями (``numpy.gradient``) по фактическому шагу — он неравномерный,
    поэтому передаётся массив координат.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cp = np.asarray(cp, dtype=float)
    if x.size < 2:
        return np.zeros_like(cp)
    pts = np.vstack([x, y] + ([np.asarray(z, dtype=float)] if z is not None else []))
    seg = np.sqrt(((np.diff(pts, axis=1)) ** 2).sum(axis=0))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    # повторяющиеся точки (стык кромок) ломают gradient — убираем дубли
    keep = np.concatenate([[True], np.diff(s) > 1e-12])
    if keep.sum() < 2:
        return np.zeros_like(cp)
    g = np.gradient(cp[keep], s[keep])
    out = np.zeros_like(cp)
    out[keep] = np.abs(g)
    return out


def surface_size_metric(x: np.ndarray, y: np.ndarray, grad: np.ndarray,
                        h_min: float = 0.001, h_max: float = 0.05,
                        power: float = 1.0,
                        z: Optional[np.ndarray] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Целевой размер элемента в каждой точке поверхности.

    Чем больше градиент ``Cp``, тем мельче элемент::

        h = h_max − (h_max − h_min) · (g/g_max) ** power

    Возвращает ``(points, sizes)``: ``points`` — массив (N, 3).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    grad = np.asarray(grad, dtype=float)
    z = (np.asarray(z, dtype=float) if z is not None
         else np.zeros_like(x))
    h_min = max(float(h_min), 1e-9)
    h_max = max(float(h_max), h_min * 1.000001)
    g_max = float(np.nanmax(grad)) if grad.size else 0.0
    if not np.isfinite(g_max) or g_max <= 0.0:
        return np.column_stack([x, y, z]), np.full(x.shape, h_max)
    norm = np.clip(np.nan_to_num(grad / g_max, nan=0.0), 0.0, 1.0)
    sizes = h_max - (h_max - h_min) * (norm ** max(float(power), 1e-6))
    return np.column_stack([x, y, z]), np.asarray(sizes, dtype=float)


def write_metric_msh(path: str, points: np.ndarray, sizes: np.ndarray,
                     view_name: str = "TargetSize") -> str:
    """Пишет поле размеров как Gmsh-файл ``.msh`` (версия 2.2, ``$NodeData``).

    gmsh использует такой файл как фоновую метрику при
    ``Mesh.Metric = <path>`` — целевой размер элемента интерполируется
    по узлам.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts))])
    sizes = np.asarray(sizes, dtype=float).reshape(-1)
    if len(sizes) != len(pts):
        raise ValueError("Число точек и размеров не совпадает")

    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write(f"$Nodes\n{len(pts)}\n")
        for i, p in enumerate(pts, start=1):
            f.write(f"{i} {p[0]:.10g} {p[1]:.10g} {p[2]:.10g}\n")
        f.write("$EndNodes\n")
        f.write("$NodeData\n1\n")
        f.write(f'"{view_name}"\n')
        f.write("1\n")            # 1 вещественный тег:
        f.write("0.0\n")          #   значение времени
        f.write("3\n")            # 3 целочисленных тега:
        f.write("0\n")            #   номер временного шага
        f.write("1\n")            #   компонент в узле (скаляр = 1)
        f.write(f"{len(pts)}\n")  #   число узлов
        for i, s in enumerate(sizes, start=1):
            f.write(f"{i} {float(s):.10g}\n")
        f.write("$EndNodeData\n")
    return path


def write_metric_geo(path: str, metric_msh: str, mesh_in: str, mesh_out: str,
                     h_min: Optional[float] = None,
                     h_max: Optional[float] = None) -> str:
    """Пишет ``.geo``-скрипт, перестраивающий сетку по метрике.

    Скрипт можно выполнить командой ``gmsh script.geo -2 -o out.stl``
    (``rebuild_with_metric`` делает то же самое через API).
    """
    lines = [
        '// Автогенерация: адаптивная сетка по полю целевых размеров',
        f'Mesh.Metric = "{os.path.abspath(metric_msh)}";',
        'Mesh.MetricIsAbsolute = 1;',
    ]
    if h_min is not None:
        lines.append(f"Mesh.CharacteristicLengthMin = {float(h_min):g};")
    if h_max is not None:
        lines.append(f"Mesh.CharacteristicLengthMax = {float(h_max):g};")
    lines += [
        f'Open("{os.path.abspath(mesh_in)}");',
        "Mesh.Algorithm = 6;             // Frontal-Delaunay",
        "Mesh 2;",
        f'Save("{os.path.abspath(mesh_out)}");',
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return path


def rebuild_with_metric(geometry_path: str, metric_msh: str, out_stl: str,
                        h_min: Optional[float] = None,
                        h_max: Optional[float] = None, log=None) -> str:
    """Перестраивает поверхностную сетку с учётом поля размеров (нужен gmsh).

    ``geometry_path`` — исходная геометрия (STL/STEP/BREP/GEO). Возвращает
    путь к новому STL.
    """
    try:
        import gmsh
    except Exception as e:  # pragma: no cover - зависит от окружения
        raise RuntimeError(
            f"Модуль gmsh недоступен — адаптивная сетка требует gmsh. ({e})"
        ) from e

    os.makedirs(os.path.dirname(os.path.abspath(out_stl)) or ".", exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.open(geometry_path)
        if geometry_path.lower().endswith((".step", ".stp", ".iges", ".igs",
                                           ".brep")):
            gmsh.model.occ.synchronize()
        gmsh.option.setString("Mesh.Metric", os.path.abspath(metric_msh))
        gmsh.option.setNumber("Mesh.MetricIsAbsolute", 1)
        if h_min is not None:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(h_min))
        if h_max is not None:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(h_max))
        gmsh.option.setNumber("Mesh.Algorithm", 6)   # Frontal-Delaunay
        gmsh.model.mesh.generate(2)
        gmsh.write(out_stl)
        if log:
            log(f"  Готово: Адаптивная сетка: {os.path.basename(out_stl)}")
    except Exception as e:
        raise RuntimeError(f"Не удалось построить адаптивную сетку: {e}") from e
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass

    if not os.path.isfile(out_stl) or os.path.getsize(out_stl) == 0:
        raise RuntimeError("gmsh не создал адаптивную сетку (пустой файл).")
    return out_stl


def adaptivity_report(samples: Dict[str, np.ndarray], sizes: np.ndarray
                      ) -> Dict[str, float]:
    """Сводка по построенному полю размеров — для отчёта пользователю."""
    sizes = np.asarray(sizes, dtype=float)
    cp = samples.get("cp")
    out = {
        "n_points": int(sizes.size),
        "h_min": float(sizes.min()) if sizes.size else 0.0,
        "h_max": float(sizes.max()) if sizes.size else 0.0,
        "h_mean": float(sizes.mean()) if sizes.size else 0.0,
    }
    if cp is not None and np.size(cp):
        finite = cp[np.isfinite(cp)]
        if finite.size:
            out["cp_min"] = float(finite.min())
            out["cp_max"] = float(finite.max())
            i_min = int(np.nanargmin(np.where(np.isfinite(cp), cp, np.inf)))
            out["x_cp_min"] = float(samples["x"][i_min])
            out["y_cp_min"] = float(samples["y"][i_min])
            out["h_at_cp_min"] = float(sizes[i_min])
    return out


def format_adaptivity_report(rep: Dict[str, float]) -> str:
    """Человекочитаемая сводка (для журнала и отчёта)."""
    lines = ["АДАПТИВНОСТЬ СЕТКИ", "=" * 34,
             f"Точек на поверхности : {rep.get('n_points', 0)}",
             f"Размер элемента      : {rep.get('h_min', 0):.5g} … "
             f"{rep.get('h_max', 0):.5g} (сред. {rep.get('h_mean', 0):.5g})"]
    if "cp_min" in rep:
        lines += [
            f"Cp минимум           : {rep['cp_min']:.4f} "
            f"в ({rep['x_cp_min']:.4f}; {rep['y_cp_min']:.4f})",
            f"Размер в этой точке  : {rep['h_at_cp_min']:.5g}",
        ]
    return "\n".join(lines)
