# -*- coding: utf-8 -*-
"""
mesh/surface_remesh.py — изотропная перестройка триангуляции компонента STL.

Зачем
-----
Генераторы геометрии приложения строят тонкие поверхности лофтом из
НЕСКОЛЬКИХ сечений: ГО и руль высоты — 3 сечения по полуразмаху 1.4 м,
киль — 2 сечения по высоте. По хорде сечение даёт ~0.014 м ребра, вдоль
размаха — 1.4 м. Треугольники поверхности получаются с соотношением
сторон до нескольких ТЫСЯЧ (замер на самолёте по умолчанию: руль —
медиана 208, максимум 7127; ГО — медиана 100, максимум 3061; 410 граней
из 474 — вырожденно вытянутые). Объёмный сеточник обязан воспроизвести
такую поверхность как ограничение, и рядом с ней неизбежно рожаются
тетраэдры-бритвы. Именно они валят второй порядок SU2 (MUSCL) с первой
итерации: градиентная реконструкция на ячейке с ребром 2e-4 м даёт
 slopes порядка 1e4, лимитер не спасает, решение NaN.

`refine_to_edge_length` (бисекция длинных рёбер) тут не помогает: он
уменьшает рёбра, но НЕ меняет вытянутость, а частичное уплотнение ещё и
ломает замкнутость (T-стыки). Нужно именно перестроение: равносторонние
треугольники заданного шага на той же геометрии.

Как
---
gmsh умеет перестраивать поверхность STL (tutorial t13):
``classifySurfaces`` разбивает STL на гладкие участки по двугранному
углу, ``createGeometry`` строит по ним параметрическую геометрию, и
обычная 2D-сетка gmsh даёт равносторонние треугольники с сохранением
острых кромок. Результат проверяется на замкнутость (каждое ребро ровно
у двух граней) и на сохранение габарита; не прошло — возвращается None,
и вызывающий код работает с исходной триангуляцией (как раньше).

``curve_angle`` = 25° обязателен: при 40° и выше два участка по разные
стороны острой задней кромки (зазор ~1e-3 хорды) склеиваются в один, и
ребро шва получает 4 соседних грани — незамкнутая поверхность. 25° режет
кромку на отдельные участки, замыкание сохраняется (проверено на всех
компонентах самолёта по умолчанию).

Функции без gmsh (``edge_manifold_bad``, ``aspect_stats``,
``component_target_edge``) чистые и покрыты тестами.
"""

from __future__ import annotations

import math
import os
import signal
import tempfile

import numpy as np

try:
    import gmsh
    HAS_GMSH = True
except Exception:                       # pragma: no cover - нет gmsh
    gmsh = None
    HAS_GMSH = False

# Угол двугранного угла для разбиения STL на участки. 40° — дефолт gmsh,
# но на острой задней кромке (зазор ~1e-3 хорды) он склеивает верх и низ
# в один участок и даёт шов с 4 гранями. 25° — участок на каждую грань
# кромки, замыкание сохраняется. Замерено на самолёте по умолчанию.
CLASSIFY_ANGLE_DEG = 25.0
# Порог «резкой» кривой: угол между соседними участками, ниже которого
# они считаются одной поверхностью. Меньше — больше участков (надёжнее
# замыкание, чуть больше граней).
CURVE_ANGLE_DEG = 25.0

# Границы целевого шага ремешки компонента (м): ниже 0.012 сетка
# бессмысленно мельчает даже для миллиметровых деталей, выше 0.6 —
# перестройка ничего не меняет к лучшему.
TARGET_EDGE_MIN = 0.012
TARGET_EDGE_MAX = 0.6


def edge_manifold_bad(faces):
    """Число рёбер, у которых НЕ ровно две соседние грани.

    0 — поверхность замкнута и многообразна (каждое ребро разделяет
    ровно две грани). Чистый numpy, без gmsh — тестируемо.
    """
    f = np.asarray(faces, dtype=np.int64)
    if len(f) == 0:
        return 0
    e = np.sort(np.stack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]],
                         axis=1).reshape(-1, 2), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return int((cnt != 2).sum())


def aspect_stats(points, faces):
    """(медиана, p95, максимум, доля AR>10) соотношения сторон граней."""
    p = np.asarray(points, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) == 0:
        return 0.0, 0.0, 0.0, 0.0
    tri = p[f]
    e = np.stack([np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
                  np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
                  np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)], axis=1)
    ar = e.max(axis=1) / np.maximum(e.min(axis=1), 1e-12)
    return (float(np.median(ar)), float(np.percentile(ar, 95)),
            float(ar.max()), float((ar > 10).mean()))


def component_target_edge(min_dim, h_surf,
                          lo=TARGET_EDGE_MIN, hi=TARGET_EDGE_MAX):
    """Целевой шаг ремешки компонента по его минимальному габариту.

    Правило: шаг не крупнее «среднего» шага поверхности ``h_surf`` и не
    крупнее 0.75 минимального габарита (тонкую пластину толщиной t нельзя
    честно обклеить треугольниками крупнее ~t — бока пластины шириной t
    потребуют вытянутых граней). Ограничен разумными границами.

    Примеры (h_surf = 0.079, «Средняя»):
        фюзеляж  D=1.2   -> 0.079 (h_surf)
        ГО       t=0.067 -> 0.050 (0.75*t)
        руль     t=0.029 -> 0.022
    """
    if h_surf is None or h_surf <= 0:
        return None
    h = min(float(h_surf), 0.75 * float(min_dim))
    return float(np.clip(h, min(lo, h), max(hi, h)))


def surface_needs_remesh(points, faces, max_ar_p95=10.0, max_ar_median=5.0):
    """True, если триангуляция вытянута настолько, что её стоит перестроить.

    Критерии по p95 и медиане AR: у нормальной поверхности p95 < 10 и
    медиана < 5. Лофт из трёх сечений даёт медиану за сотню.
    """
    med, p95, mx, frac = aspect_stats(points, faces)
    return bool(p95 > max_ar_p95 or med > max_ar_median)


def surface_volume(points, faces):
    """Знаковый объём замкнутой поверхности (формула дивергенции)."""
    p = np.asarray(points, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) < 4:
        return 0.0
    tri = p[f]
    return float(np.einsum('ij,ij->i', tri[:, 0],
                           np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def remesh_component(points, faces, target_edge, log=print,
                     max_faces=600_000, keep_tol_rel=1e-6):
    """Перестроить поверхность компонента равносторонними треугольниками.

    ``points``/``faces`` — замкнутая триангулированная поверхность одного
    компонента. ``target_edge`` — целевое ребро (см.
    ``component_target_edge``).

    Адаптивность: результат ремешки зависит от того, как parametrize
    сложные места (острая задняя кромка, стык крышек). Поэтому делается
    до трёх попыток с шагом 1.0/0.7/0.5 от целевого; принимается первая
    замкнутая, у которой p95 вытянутости упал хотя бы вдвое против
    исходного И объём сохранился в пределах 5%. Если ни одна попытка не
    дала приемлемого качества, но есть замкнутые — возвращается лучшая
    (с минимальным p95), если она всё же лучше исходной; иначе None —
    вызывающий код работает с исходной триангуляцией (прежнее
    поведение). Плохую поверхность (перехлёсты, потеря объёма) ремешка
    вернуть не может по определению.
    """
    if not HAS_GMSH:
        return None
    pts = np.asarray(points, dtype=float)
    fcs = np.asarray(faces, dtype=np.int64)
    if len(pts) < 4 or len(fcs) < 4 or target_edge is None or target_edge <= 0:
        return None
    if not np.isfinite(pts).all():
        return None

    # Квантирование записи STL — float32; шаг мельче эпсилон координат
    # ремешкой не удержать, поэтому режем снизу.
    scale = float(np.abs(pts).max()) or 1.0
    target_edge = max(float(target_edge), keep_tol_rel * scale)

    bbox_before = (pts.min(axis=0), pts.max(axis=0))
    vol_before = abs(surface_volume(pts, fcs))
    ar_med0, ar_p950, _mx0, _f0 = aspect_stats(pts, fcs)

    best = None  # (ar_p95, pts, faces)
    factor = 1.0
    for attempt in range(3):
        tgt = target_edge * factor
        _tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
        _tmp.close()
        try:
            from mesh.bodyfit_gmsh import _write_binary_stl
            _write_binary_stl(pts, fcs, _tmp.name)
            out = _remesh_stl_file(_tmp.name, tgt, log=log,
                                   max_faces=max_faces)
        except Exception as e:                          # pragma: no cover
            log("   Внимание: попытка ремешки %.4f м не удалась (%s: %s)"
                % (tgt, type(e).__name__, e))
            out = None
        finally:
            try:
                os.remove(_tmp.name)
            except OSError:
                pass
        factor *= 0.7
        if out is None:
            continue
        pts2, fcs2 = out

        # Проверка 1: замкнутость (каждое ребро ровно у двух граней).
        bad = edge_manifold_bad(fcs2)
        if bad:
            log("   Внимание: ремешка %.4f м дала незамкнутую поверхность "
                "(%d рёбер с числом граней != 2), пробую шаг мельче."
                % (tgt, bad))
            continue

        # Проверка 2: габарит не уплыл (ремешка двигает узлы по
        # поверхности, но не меняет форму).
        bb2 = (pts2.min(axis=0), pts2.max(axis=0))
        grow = float(np.max(np.abs(np.asarray(bb2[1]) - np.asarray(bb2[0]))
                            - (np.asarray(bbox_before[1]) - bbox_before[0])))
        if grow > 0.02 * tgt + 1e-9:
            log("   Внимание: ремешка %.4f м изменила габарит на %.2e м "
                "(порог %.2e), пробую шаг мельче."
                % (tgt, grow, 0.02 * tgt))
            continue

        # Проверка 3: объём сохранился (перехлёсты/складки его меняют).
        vol_after = abs(surface_volume(pts2, fcs2))
        if vol_before > 1e-12 and abs(vol_after - vol_before) > 0.05 * vol_before:
            log("   Внимание: ремешка %.4f м изменила объём компонента на "
                "%.1f%% (допустимо 5%%) — поверхность с дефектами, пробую "
                "шаг мельче." % (tgt, 100.0 * abs(vol_after - vol_before)
                                 / vol_before))
            continue

        # Проверка 4: бюджет граней.
        if max_faces and len(fcs2) > max_faces:
            log("   Внимание: ремешка %.4f м дала %d граней (лимит %d)."
                % (tgt, len(fcs2), max_faces))
            continue

        # Проверка 5: пробный embed (см. _embed_ok) — ловит складки.
        if not _embed_ok(pts2, fcs2, tgt):
            log("   Внимание: ремешка %.4f м даёт перекрывающиеся грани "
                "(embed не проходит), пробую шаг мельче." % tgt)
            continue

        _ar_med, ar_p95, _mx, _f = aspect_stats(pts2, fcs2)
        if best is None or ar_p95 < best[0]:
            best = (ar_p95, pts2, fcs2)
        # Приемлемо: замкнуто и p95 вытянутости упал хотя бы вдвое.
        if ar_p950 <= 1e-9 or ar_p95 <= max(10.0, 0.5 * ar_p950):
            _orient_consistent(pts2, fcs2)
            return pts2, fcs2

    if best is not None and best[0] < ar_p950:
        log("   Готово: ремешка принята по лучшей из попыток "
            "(AR p95 %.1f -> %.1f)." % (ar_p950, best[0]))
        _orient_consistent(best[1], best[2])
        return best[1], best[2]
    log("   Внимание: ни одна попытка ремешки не дала замкнутой "
        "поверхности лучше исходной — компонент оставлен на исходной "
        "триангуляции.")
    return None



def _embed_ok(points, faces, target_edge):
    """Пробный объёмный прогон: embed поверхности в плотный короб.

    Это ровно та операция, которую выполняет телооблекающий путь, поэтому
    она — честный гейт качества: parametric-ремешка gmsh на острых
    кромках (задняя кромка крыла — зазор ~1e-3 хорды) может дать
    СКЛАДКИ — треугольники, перекрывающиеся геометрически при
    формально замкнутой топологии. На складках embed падает с
    «Invalid boundary mesh (overlapping facets)» / «PLC Error», и вся
    сетка уезжает в картезианский фолбэк. Ловим это на компоненте,
    пока не поздно. Возвращает True/False.
    """
    if not HAS_GMSH:
        return False
    pts = np.asarray(points, dtype=float)
    lo = pts.min(axis=0) - 2.0 * target_edge
    hi = pts.max(axis=0) + 2.0 * target_edge
    ext = hi - lo
    _tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    _tmp.close()
    ok = False
    try:
        from mesh.bodyfit_gmsh import _write_binary_stl
        _write_binary_stl(pts, faces, _tmp.name)
        _orig_signal = signal.signal

        def _noop(sig, handler):
            return handler

        signal.signal = _noop
        try:
            gmsh.initialize()
            signal.signal = _noop
            gmsh.option.setNumber("General.Terminal", 0)
            x0, y0, z0 = [float(v) for v in lo]
            gmsh.model.occ.addBox(x0, y0, z0, float(ext[0]), float(ext[1]),
                                  float(ext[2]))
            gmsh.model.occ.synchronize()
            gmsh.merge(_tmp.name)
            gmsh.model.occ.synchronize()
            box = gmsh.model.getEntities(3)[0][1]
            box_tags = {t for (d, t) in
                        gmsh.model.getBoundary([(3, box)], oriented=False)
                        if d == 2}
            body_tags = [t for (d, t) in gmsh.model.getEntities(2)
                         if t not in box_tags]
            gmsh.model.mesh.embed(2, body_tags, 3, box)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.3 * target_edge)
            gmsh.option.setNumber("Mesh.MeshSizeMax", 3.0 * target_edge)
            gmsh.model.mesh.generate(3)
            ok = True
        except Exception:
            ok = False
        finally:
            signal.signal = _orig_signal
            try:
                gmsh.finalize()
            except Exception:
                pass
    except Exception:
        ok = False
    finally:
        try:
            os.remove(_tmp.name)
        except OSError:
            pass
    return ok


def _remesh_stl_file(stl_path, target_edge, log=print, max_faces=600_000):
    """Сам вызов gmsh: classifySurfaces -> createGeometry -> generate(2).

    Возвращает ``(points, faces)`` или None. gmsh регистрирует обработчики
    сигналов и из не-главного потока падает — на время сессии подменяем
    ``signal.signal`` заглушкой (тот же приём, что в bodyfit_gmsh).
    """
    _orig_signal = signal.signal

    def _noop_signal(sig, handler):
        return handler

    signal.signal = _noop_signal
    try:
        gmsh.initialize()
        signal.signal = _noop_signal
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.merge(stl_path)
        gmsh.model.mesh.classifySurfaces(
            math.radians(CLASSIFY_ANGLE_DEG), False, True,
            math.radians(CURVE_ANGLE_DEG))
        # gmsh >= 4.15: createGeometry переехал в gmsh.model.mesh.
        _cg = getattr(gmsh.model.mesh, "createGeometry", None)
        if _cg is None:                                 # pragma: no cover
            _cg = gmsh.model.createGeometry
        _cg()

        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5 * target_edge)
        gmsh.option.setNumber("Mesh.MeshSizeMax", target_edge)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)      # Frontal-Delaunay 2D
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(node_coords, dtype=float).reshape(-1, 3)
        tag_to_local = {int(t): i for i, t in enumerate(node_tags)}
        fcs = []
        elem_types, _, node_tags_list = gmsh.model.mesh.getElements(2)
        for et, ntl in zip(elem_types, node_tags_list):
            if int(et) != 2:            # 2 = треугольник
                continue
            arr = np.asarray(ntl, dtype=np.int64).reshape(-1, 3)
            for row in arr:
                fcs.append([tag_to_local[int(x)] for x in row])
        if not fcs:
            return None
        return coords, np.asarray(fcs, dtype=np.int64)
    finally:
        signal.signal = _orig_signal
        try:
            gmsh.finalize()
        except Exception:                               # pragma: no cover
            pass


def _orient_consistent(points, faces):
    """Развернуть грани согласованно (общие рёбра — в разные стороны) и
    суммарно наружу. BFS по неориентированным рёбрам: у каждой грани
    запоминается, в каком направлении она проходит общее ребро; сосед,
    проходящий ребро В ту же сторону, относительно неё вывернут и
    переворачивается. Немногообразные рёбра (число граней != 2) в
    связывании не участвуют. Финально — глобальный знак объёма.
    """
    p = np.asarray(points, dtype=float)
    f = np.asarray(faces, dtype=np.int64).copy()
    n = len(f)
    if n == 0:
        return
    from collections import defaultdict, deque
    edge_faces = defaultdict(list)   # неориентированное ребро -> [(грань, forward)]
    for i, (a, b, c) in enumerate(f):
        for (u, v) in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_faces[key].append((i, u < v))
    adj = [[] for _ in range(n)]
    for lst in edge_faces.values():
        if len(lst) != 2:
            continue
        (i, fi), (j, fj) = lst
        same = (fi == fj)          # одно направление = сосед вывернут
        adj[i].append((j, same))
        adj[j].append((i, same))
    visited = np.zeros(n, dtype=bool)
    flip = np.zeros(n, dtype=bool)
    for seed in range(n):
        if visited[seed]:
            continue
        visited[seed] = True
        q = deque([seed])
        while q:
            i = q.popleft()
            for j, same in adj[i]:
                if visited[j]:
                    continue
                flip[j] = flip[i] ^ same
                visited[j] = True
                q.append(j)
    for j in np.nonzero(flip)[0]:
        f[j] = f[j][::-1]
    tri = p[f]
    vol = float(np.sum(tri[:, 0] * np.cross(tri[:, 1], tri[:, 2])) / 6.0)
    if vol < 0:
        f[:] = f[:, [0, 2, 1]]
    faces[:] = f


