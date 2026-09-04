# -*- coding: utf-8 -*-
"""
official_cases/surface.py — извлечение поверхности тела из официальной сетки SU2.

Официальные 3D-модели SU2 раздают в виде **объёмного** `.su2`-меша. В нём
поверхность тела (крыло, фюзеляж и т.д.) зашита в границы ``MARKER_TAG= ...``.
Здесь мы вытаскиваем эти границы как плоский треугольный меш — то есть
геометрию, которую AeroOpt умеет загружать в список компонентов.

Модуль stdlib-only (без numpy/Qt), поэтому его можно тестировать и в CI без
тяжёлых зависимостей. Конвертация в pyvista/numpy делается в UI-слое.

Формат SU2 (фрагменты), которые здесь разбираются:
    NDIME= 3
    NELEM= N
    <N строк объёмных элементов>
    NPOIN= M
    <M строк координат точек>
    NMARK= K
    MARKER_TAG= <имя>
    MARKER_ELEMS= L
    <L строк граничных элементов>
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

# Число узлов граничного элемента по его типу (SU2):
#   тип 3 — линия (2 узла), тип 5 — треугольник (3), тип 9 — четырёхугольник (4),
#   тип 10 — пятиугольник (5), тип 11 — шестиугольник (6).
_BOUNDARY_NODES = {
    3: 2,   # line
    5: 3,   # triangle
    9: 4,   # quad
    10: 5,  # pentagon
    11: 6,  # hexagon
}

# Имена маркеров, которые точно НЕ тело (дальнее поле, симметрия, периодика).
_NON_BODY_HINTS = (
    "FAR", "XNORMAL", "YNORMAL", "ZNORMAL", "SYMMETRY", "SYMMETRIC",
    "PERIODIC", "INTERFACE", "OUTLET", "INLET",
)


def _is_body_marker(name: str) -> bool:
    up = str(name).upper()
    return not any(h in up for h in _NON_BODY_HINTS)


def parse_su2_text(text: str) -> dict:
    """Разбирает текст `.su2` в структуру точек и границ.

    Возвращает dict::

        {
          "ndime": int,
          "points": [(x, y, z), ...],
          "markers": {"TAG": [(elem_type, nodes_list), ...], ...},
        }

    Толерантно к комментариям '%' и пустым строкам.
    """
    lines = []
    for raw in (text or "").splitlines():
        s = raw.split("%", 1)[0].strip()
        if not s:
            continue
        if "=" in s and not s.split("=", 1)[0].strip().isdigit():
            lines.append(s)
        else:
            lines.append(s)

    ndime = 3
    # Поиск NDIME.
    for ln in lines:
        if ln.upper().startswith("NDIME"):
            try:
                ndime = int(ln.split("=", 1)[1].split()[0])
            except (ValueError, IndexError):
                ndime = 3
            break

    # --- NPOIN: точки --------------------------------------------------
    points: List[Tuple[float, ...]] = []
    npoin = 0
    idx = 0
    for i, ln in enumerate(lines):
        if ln.upper().startswith("NPOIN"):
            try:
                npoin = int(ln.split("=", 1)[1].split()[0])
            except (ValueError, IndexError):
                npoin = 0
            idx = i + 1
            break
    # Считываем ровно npoin строк координат.
    for j in range(idx, min(idx + npoin, len(lines))):
        toks = lines[j].split()
        if len(toks) < ndime:
            continue
        try:
            points.append(tuple(float(t) for t in toks[:ndime]))
        except ValueError:
            continue
    # Если точек не набрали (npoin могло быть слишком большим/некорректным),
    # добор по оставшимся числовым строкам не делаем — просто вёрнём что есть.

    # --- NMARK: границы -------------------------------------------------
    markers: dict = {}
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        if ln.upper().startswith("NMARK"):
            try:
                nmark = int(ln.split("=", 1)[1].split()[0])
            except (ValueError, IndexError):
                nmark = 0
            i += 1
            for _ in range(nmark):
                tag = None
                count = 0
                # Пропускаем строки до MARKER_TAG / MARKER_ELEMS.
                tag_line = None
                elems_line = None
                while i < n:
                    cu = lines[i].upper()
                    if cu.startswith("MARKER_TAG"):
                        tag_line = lines[i]
                        i += 1
                    elif cu.startswith("MARKER_ELEMS"):
                        elems_line = lines[i]
                        i += 1
                    else:
                        break
                if tag_line is not None:
                    tag = tag_line.split("=", 1)[1].strip()
                if elems_line is not None:
                    try:
                        count = int(elems_line.split("=", 1)[1].split()[0])
                    except (ValueError, IndexError):
                        count = 0
                elems = []
                for _ in range(count):
                    if i >= n:
                        break
                    toks = lines[i].split()
                    i += 1
                    if not toks:
                        continue
                    try:
                        etype = int(toks[0])
                    except ValueError:
                        continue
                    nnodes = _BOUNDARY_NODES.get(etype, 0)
                    nodes = []
                    for t in toks[1:1 + nnodes]:
                        try:
                            nodes.append(int(t))
                        except ValueError:
                            break
                    if len(nodes) >= 3:
                        elems.append((etype, tuple(nodes)))
                if tag is not None:
                    markers[tag] = elems
            break
        i += 1

    return {
        "ndime": ndime,
        "points": points,
        "markers": markers,
    }


def _triangulate(elem_type: int, nodes: Sequence[int]) -> List[Tuple[int, int, int]]:
    """Преобразует граничный элемент в треугольники.

    Треугольник (5) — как есть; четырёхугольник (9) — веером на два
    треугольника с сохранением обхода. Прочие многоугольники разбиваются
    веером от первой вершины.
    """
    n = list(nodes)
    if elem_type == 5 and len(n) >= 3:
        return [(n[0], n[1], n[2])]
    out = []
    if len(n) >= 4:
        for k in range(1, len(n) - 1):
            out.append((n[0], n[k], n[k + 1]))
    return out


def read_su2_boundary(mesh_path: str,
                      markers: Optional[Sequence[str]] = None,
                      exclude_non_body: bool = True,
                      compact: bool = True) -> dict:
    """Читает `.su2` и возвращает поверхность тела.

    ``markers`` — имена маркеров тела (например, ``["WING"]`` или
    ``["UPPER_SIDE", "LOWER_SIDE", "TIP"]``). Если ``None`` — берутся все
    маркеры, не похожие на дальнее поле/симметрию.

    ``compact`` — оставить только точки, реально используемые треугольниками
    (иначе для объёмной сетки возвращаются все узлы, включая поле).

    Возвращает dict::

        {
          "points": [(x, y, z), ...],
          "triangles": [(i, j, k), ...],
          "markers": {"TAG": n_elems},
        }

    Если поверхность не найдена — ``triangles`` пуст.
    """
    with open(mesh_path, "r", encoding="ascii", errors="ignore") as f:
        text = f.read()
    parsed = parse_su2_text(text)
    pts = parsed["points"]
    pts_adjusted = [_pad_point(p, parsed["ndime"]) for p in pts]

    wanted = None
    if markers is not None and any(markers):
        wanted = {str(m).strip() for m in markers if str(m).strip()}

    tris: List[Tuple[int, int, int]] = []
    used: dict = {}
    for tag, elems in parsed["markers"].items():
        if wanted is not None and str(tag) not in wanted:
            continue
        if wanted is None and exclude_non_body and not _is_body_marker(tag):
            continue
        count = 0
        for etype, nodes in elems:
            tris.extend(_triangulate(etype, nodes))
            count += 1
        used[tag] = count

    if compact and tris:
        remap: dict = {}
        compact_tris = []
        for (a, b, c) in tris:
            new_a = remap.setdefault(a, len(remap))
            new_b = remap.setdefault(b, len(remap))
            new_c = remap.setdefault(c, len(remap))
            compact_tris.append((new_a, new_b, new_c))
        compact_pts = [pts_adjusted[old] for old in sorted(remap, key=remap.get)]
        return {
            "points": compact_pts,
            "triangles": compact_tris,
            "markers": used,
        }

    return {
        "points": pts_adjusted,
        "triangles": tris,
        "markers": used,
    }


def _pad_point(p: Sequence[float], ndime: int) -> Tuple[float, float, float]:
    """Дополняет точку до 3 компонент (для 2D → z=0)."""
    vals = [float(x) for x in p]
    while len(vals) < 3:
        vals.append(0.0)
    return (vals[0], vals[1], vals[2])
