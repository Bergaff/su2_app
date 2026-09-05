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

import numpy as np
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
                    # Граничные элементы бывают и линиями (тип 3, 2 узла) в
                    # 2D-сетках — их тоже сохраняем (раньше отбрасывали,
                    # и 2D-профили давали пустую поверхность).
                    if len(nodes) >= 2:
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


def is_manifold_closed(triangles: Sequence[Sequence[int]]) -> bool:
    """Проверяет, является ли треугольная поверхность замкнутой манifold-поверхностью.

    Для замкнутой ориентированной манifold-поверхности каждое ребро входит
    ровно в два треугольника. Если хоть одно ребро имеет другую кратность
    (1 — открытый край, >2 — «бабочка»), тело незамкнуто: его нельзя
    облечь телtoоблекающей сеткой, и re-mesh даст ступенчатую сетку.

    Возвращает True только если треугольников > 0 и все рёбра кратны 2.
    """
    from collections import Counter
    edges = Counter()
    for t in triangles:
        if len(t) < 3:
            continue
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        for (u, v) in ((a, b), (b, c), (c, a)):
            e = (u, v) if u < v else (v, u)
            edges[e] += 1
    if not edges:
        return False
    return all(count == 2 for count in edges.values())


# ---------------------------------------------------------------------------
# Починка геометрии официальных кейсов до замкнутого тела
# ---------------------------------------------------------------------------
# Официальные поверхности SU2 — часто НЕ тела: у 2D-профилей (NACA0012,
# RAE2822) маркер стенки — это ЛИНИИ контура (NDIME=2), у полу-моделей
# (ONERA M6) — открытая оболочка с краем в плоскости симметрии. Вокруг
# такой поверхности телооблегающую сетку не построить. Здесь поверхность
# доводится до замкнутого тела: профиль вытягивается по размаху, полу-
# модель зеркалится. После этого кейс считается как обычная геометрия
# приложения: пользователь сам строит сетку и сам запускает расчёт.


def read_profile_segments(mesh_path: str,
                          markers: Optional[Sequence[str]] = None) -> list:
    """Линейные сегменты маркеров тела из 2D-сетки .su2.

    Возвращает список пар 2D-точек ``[((x1, y1), (x2, y2)), ...]``.
    Для NDIME=3 сегментов не бывает — вернётся пустой список.
    """
    with open(mesh_path, "r", encoding="ascii", errors="ignore") as f:
        text = f.read()
    parsed = parse_su2_text(text)
    pts = parsed["points"]
    wanted = None
    if markers is not None and any(markers):
        wanted = {str(m).strip() for m in markers if str(m).strip()}
    segs = []
    for tag, elems in parsed["markers"].items():
        if wanted is not None and str(tag) not in wanted:
            continue
        if wanted is None and not _is_body_marker(tag):
            continue
        for etype, nodes in elems:
            if int(etype) != 3 or len(nodes) < 2:
                continue
            try:
                a = pts[int(nodes[0])]
                b = pts[int(nodes[1])]
            except (IndexError, ValueError):
                continue
            segs.append(((float(a[0]), float(a[1])),
                         (float(b[0]), float(b[1]))))
    return segs


def chain_loops(segments: Sequence, tol_rel: float = 1e-6) -> list:
    """Склеить сегменты в замкнутые петли.

    ``segments`` — пары 2D-точек. Концы свариваются с допуском
    ``tol_rel`` от диаграммы габаритов. Возвращает список петель
    (списков точек (x, y)); последняя точка НЕ повторяет первую.
    """
    if not segments:
        return []
    xs = [p[0] for s in segments for p in s]
    ys = [p[1] for s in segments for p in s]
    diag = max(max(xs) - min(xs), max(ys) - min(ys), 1e-12)
    tol = tol_rel * diag

    def key(p):
        return (int(round(p[0] / tol)), int(round(p[1] / tol)))

    adj: dict = {}
    for si, (a, b) in enumerate(segments):
        adj.setdefault(key(a), []).append((si, key(b)))
        adj.setdefault(key(b), []).append((si, key(a)))
    used = [False] * len(segments)
    loops = []
    for si0 in range(len(segments)):
        if used[si0]:
            continue
        used[si0] = True
        a, b = segments[si0]
        loop_keys = [key(a), key(b)]
        loop_pts = {key(a): a, key(b): b}
        # расширяем в обе стороны
        for direction in (0, 1):
            while True:
                cur = loop_keys[-1] if direction == 0 else loop_keys[0]
                nxt = None
                for sj, other in adj.get(cur, ()):
                    if not used[sj]:
                        nxt = (sj, other)
                        break
                if nxt is None:
                    break
                sj, other = nxt
                used[sj] = True
                if direction == 0:
                    loop_keys.append(other)
                else:
                    loop_keys.insert(0, other)
                # координаты другого конца сегмента
                sa, sb = segments[sj]
                for pp in (sa, sb):
                    if key(pp) == other:
                        loop_pts[other] = pp
                        break
                if other == loop_keys[0 if direction == 0 else 1] and \
                        len(loop_keys) > 2:
                    break
        # замкнутая петля: первый и последний ключ совпадают
        if len(loop_keys) >= 4 and loop_keys[0] == loop_keys[-1]:
            loop_keys = loop_keys[:-1]
            loops.append([loop_pts[k] for k in loop_keys])
    return loops


def _triangulate_polygon(pts: Sequence) -> list:
    """Триангуляция простого полигона (ухластая вырезка).

    ``pts`` — вершины в порядке обхода против часовой стрелки в плоскости
    (x, z). Возвращает тройки индексов вершин. Вырожденные/невыпуклые
    места терпит; совсем испорченный полигон отдаёт веером.
    """
    n = len(pts)
    if n < 3:
        return []
    idx = list(range(n))
    tris = []

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1])
                - (a[1] - o[1]) * (b[0] - o[0]))

    guard = 0
    while len(idx) > 3 and guard < 20 * n * n:
        guard += 1
        m = len(idx)
        cut = False
        for k in range(m):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if cross(a, b, c) <= 1e-14:
                continue                       # рефлексный/вырожденный угол
            ok = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = pts[j]
                if (cross(a, b, p) >= -1e-14
                        and cross(b, c, p) >= -1e-14
                        and cross(c, a, p) >= -1e-14):
                    ok = False
                    break
            if ok:
                tris.append((i0, i1, i2))
                idx.pop(k)
                cut = True
                break
        if not cut:
            break
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    elif len(idx) > 3:                          # fallback: веер
        for k in range(1, len(idx) - 1):
            tris.append((idx[0], idx[k], idx[k + 1]))
    return tris


def extrude_loop_to_solid(loop: Sequence, span: float) -> tuple:
    """Вытянуть 2D-контур (x, y) в замкнутое 3D-тело по оси Y.

    Контур профиля лежит в плоскости XZ (2D-сетка SU2: x — хорда,
    y — вертикаль -> у приложения вертикаль это Z). Тело занимает
    Y из ``[-span/2, +span/2]``, торцы — плоские крышки. Ориентация
    граней — наружу. Возвращает (points, faces).
    """
    if len(loop) < 3 or span <= 0:
        return [], []
    h = 0.5 * float(span)
    # Контур против часовой стрелки в (x, z): знаковая площадь > 0.
    area2 = 0.0
    for k in range(len(loop)):
        x1, y1 = loop[k]
        x2, y2 = loop[(k + 1) % len(loop)]
        area2 += x1 * y2 - x2 * y1
    pts2d = list(loop) if area2 > 0 else list(loop)[::-1]
    # Убрать дубли вершин (замыкание петли могло остаться в списке).
    clean = []
    seen = set()
    for p in pts2d:
        k = (round(p[0], 12), round(p[1], 12))
        if k in seen:
            continue
        seen.add(k)
        clean.append(p)
    if len(clean) < 3:
        return [], []
    verts = []
    bot_of = []
    top_of = []
    for (x, y) in clean:
        bot_of.append(len(verts)); verts.append((x, -h, y))
        top_of.append(len(verts)); verts.append((x, +h, y))
    faces = []
    n = len(clean)
    # Кожа: для ребра (i -> i+1) CCW наружная нормаль (dz, 0, -dx).
    for i in range(n):
        i2 = (i + 1) % n
        b1, t1 = bot_of[i], top_of[i]
        b2, t2 = bot_of[i2], top_of[i2]
        faces.append((b1, t1, t2))
        faces.append((b1, t2, b2))
    # Крышки: низ наружу -Y, верх наружу +Y.
    cap = _triangulate_polygon(clean)
    for (a, b, c) in cap:
        faces.append((bot_of[a], bot_of[c], bot_of[b]))   # низ: -Y
        faces.append((top_of[a], top_of[b], top_of[c]))   # верх: +Y
    return verts, faces


def open_boundary_plane(points: Sequence, faces: Sequence):
    """Ось, на плоскости по которой лежат ВСЕ открытые рёбра, или None.

    Открытое ребро — ребро с кратностью 1. У полу-моделей SU2 открытый
    край лежит в плоскости симметрии (y=0 и т.п.).
    """
    from collections import Counter
    cnt = Counter()
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        for (u, v) in ((a, b), (b, c), (c, a)):
            cnt[(u, v) if u < v else (v, u)] += 1
    open_edges = [e for e, k in cnt.items() if k == 1]
    if not open_edges:
        return None
    verts = sorted({v for e in open_edges for v in e})
    p = np.asarray([points[v] for v in verts], dtype=float)
    diag = float(np.ptp(np.asarray(points, dtype=float), axis=0).max()) or 1.0
    for axis in range(3):
        vals = p[:, axis]
        if float(np.abs(vals - vals[0]).max()) <= 1e-5 * diag:
            return axis, float(vals[0])
    return None


def mirror_close_solid(points: Sequence, faces: Sequence):
    """Зеркалировать открытую полу-модель по плоскости открытого края.

    Возвращает (points, faces) замкнутого тела или None, если открытый
    край не лежит ни в одной координатной плоскости.
    """
    plane = open_boundary_plane(points, faces)
    if plane is None:
        return None
    axis, coord = plane
    pts = [tuple(float(c) for c in p) for p in points]
    m = pts + [tuple(2.0 * coord - p[axis] if k == axis else p[k]
                     for k in range(3)) for p in pts]
    faces = [tuple(int(x) for x in f) for f in faces]
    mfaces = faces + [tuple(reversed(f)) for f in
                      [(a + len(pts), b + len(pts), c + len(pts))
                       for (a, b, c) in faces]]
    # Сварка совпавших вершин (квантование от габарита).
    arr = np.asarray(m, dtype=float)
    diag = float(np.ptp(arr, axis=0).max()) or 1.0
    q = np.round(arr / (1e-7 * diag)).astype(np.int64)
    uniq: dict = {}
    remap = []
    for row in q:
        t = tuple(row)
        if t not in uniq:
            uniq[t] = len(uniq)
        remap.append(uniq[t])
    out_pts = [None] * len(uniq)
    for old_i, new_i in enumerate(remap):
        if out_pts[new_i] is None:
            out_pts[new_i] = m[old_i]
    # ВАЖНО: не сортируем узлы внутри грани — сортировка уничтожает
    # ориентацию, и знаковый объём замкнутой поверхности схлопывается
    # в нуль (замерено на полу-боксе). Дубликаты снимаем точным
    # совпадением троек в исходном порядке обхода.
    out_faces = [t for t in dict.fromkeys(
        (remap[a], remap[b], remap[c]) for (a, b, c) in mfaces)]
    # Ориентация наружу: знаковый объём > 0, иначе всё вывернуть.
    vol6 = 0.0
    for (a, b, c) in out_faces:
        p0, p1, p2 = np.asarray(out_pts[a]), np.asarray(out_pts[b]), \
            np.asarray(out_pts[c])
        vol6 += float(np.dot(p0, np.cross(p1, p2)))
    if vol6 < 0:
        out_faces = [tuple(reversed(f)) for f in out_faces]
    return out_pts, out_faces


def fix_body_surface(mesh_path: str, markers: Optional[Sequence[str]] = None,
                     span_factor: float = 2.0) -> Optional[dict]:
    """Довести поверхность официального кейса до замкнутого тела.

    Возвращает dict с ключами points/triangles/note или None, если
    починить не удалось. Порядок попыток:

    1. Уже замкнута — вернуть как есть.
    2. 2D-профиль (NDIME=2, контур из линий) — вытянуть по размаху
       ``span_factor`` × хорда (по умолчанию 2 хорды): получается
       конечное крыло; вычислительно это 3D, у торцов теряется часть
       подъёмной силы — честное приближение, а не 2D-постановка.
    3. Открытая полу-модель с краем в координатной плоскости —
       зеркалирование с сваркой.
    """
    surf = read_su2_boundary(mesh_path, markers=markers)
    tris = surf.get("triangles") or []
    pts = surf.get("points") or []
    if tris and is_manifold_closed(tris):
        return {"points": pts, "triangles": tris,
                "note": "поверхность уже замкнута"}
    # Попытка 2: 2D-профиль.
    try:
        segs = read_profile_segments(mesh_path, markers=markers)
        loops = chain_loops(segs)
    except Exception:
        loops = []
    if loops:
        loop = max(loops, key=len)
        xs = [p[0] for p in loop]
        chord = max(xs) - min(xs)
        if chord > 0:
            span = float(span_factor) * chord
            v3, f3 = extrude_loop_to_solid(loop, span)
            if v3 and is_manifold_closed(f3):
                return {"points": v3, "triangles": f3,
                        "note": ("2D-профиль (%d точек контура) вытянут в "
                                 "тело: размах %.4g = %.1f хорд. Это "
                                 "конечное крыло, а не 2D-постановка: у "
                                 "торцов часть подъёмной силы теряется."
                                 % (len(loop), span, span_factor))}
    # Попытка 3: открытая полу-модель.
    if tris and pts:
        try:
            m = mirror_close_solid(pts, tris)
        except Exception:
            m = None
        if m is not None:
            mp, mf = m
            if is_manifold_closed(mf):
                return {"points": mp, "triangles": mf,
                        "note": ("полу-модель отражена по плоскости "
                                 "открытого края: %d -> %d граней"
                                 % (len(tris), len(mf)))}
    return None
