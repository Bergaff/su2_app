"""
geometry/solid_union.py — объединение пересекающихся тел в одну поверхность.

Зачем это нужно
---------------
Крыло входит в фюзеляж, ГО и элеватор пересекаются, киль стоит на
фюзеляже. Если тела оставить как есть, то внутри модели остаются
внутренние стенки: в 3D они просвечивают сквозь обшивку, в экспортированном
STL детали пересекаются насквозь, а решатель получает лишние граничные
граней там, где их быть не должно.

Булево объединение (union) разрезает поверхности по линиям пересечения и
отбрасывает то, что оказалось внутри другого тела. Из пяти тел самолёта
получается одна замкнутая оболочка.

Замерено на встроенном самолёте (фюзеляж L=8/D=1.2, крыло 9.02/1.44/0.72,
ГО, элеватор, ВО):

    тело          граней   watertight
    фюзеляж         2592       True
    крыло          15958       True
    ГО             12798       True
    элеватор       12798       True
    ВО              4424       True
    объединение    43156       True, V=7.0704

Сумма объёмов отдельных тел больше объёма объединения ровно на объём
пересечений — это и есть то, что обрезается.

Движок булевых операций выбирается по доступности: manifold3d, затем
blender, затем что найдёт trimesh. Если ни один не сработал или
результат не замкнут, функция возвращает None, и вызывающий код
остаётся на прежнем поведении.
"""
from __future__ import annotations

import numpy as np

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:          # pragma: no cover - pyvista есть в зависимостях
    pv = None
    HAS_PYVISTA = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    trimesh = None
    HAS_TRIMESH = False


def to_triangles(mesh):
    """Привести поверхность к (points[N,3] float64, faces[M,3] int64).

    pyvista хранит грани плоским массивом [3, i, j, k, 3, ...] либо
    массивом одинаковых строк — обрабатываются оба варианта.
    Возвращает (None, None), если поверхность пустая или не треугольная.
    """
    if not HAS_PYVISTA:
        return None, None
    m = mesh if isinstance(mesh, pv.PolyData) else pv.PolyData(mesh)
    try:
        m = m.triangulate()
    except Exception:
        pass
    faces_raw = np.asarray(m.faces)
    if faces_raw.size == 0:
        return None, None
    if faces_raw.ndim == 1:
        stride = int(faces_raw[0]) + 1
        if stride < 4 or len(faces_raw) % stride != 0:
            return None, None
        faces = faces_raw.reshape(-1, stride)[:, 1:]
    else:
        faces = faces_raw
    if faces.shape[1] != 3:
        return None, None
    return np.asarray(m.points, dtype=float), np.ascontiguousarray(
        faces, dtype=np.int64)


def _as_trimesh(mesh):
    pts, fcs = to_triangles(mesh)
    if pts is None or len(fcs) == 0:
        return None
    return trimesh.Trimesh(vertices=pts, faces=fcs, process=True)


def union_meshes(meshes, log=print):
    """Объединить поверхности тел в одну замкнутую.

    Возвращает (points, faces) или None. Замкнутость обязательна: иначе
    непонятно, что считать внутренностью, и ни сетка, ни экспорт не
    будут корректными.
    """
    if not HAS_TRIMESH:
        return None, None
    parts = []
    for m in meshes:
        t = _as_trimesh(m)
        if t is not None and len(t.faces) > 0:
            parts.append(t)
    if not parts:
        return None, None

    if len(parts) == 1:
        merged = parts[0]
    else:
        merged = None
        for engine in ("manifold", "blender", None):
            try:
                if engine is None:
                    merged = trimesh.boolean.union(parts)
                else:
                    merged = trimesh.boolean.union(parts, engine=engine)
                if merged is not None and len(merged.faces) > 0:
                    log("   Готово: тела объединены (движок %s)"
                        % (engine or "по умолчанию"))
                    break
            except Exception as e:
                log("   Внимание: объединение движком %s не удалось (%s)"
                    % (engine or "по умолчанию", e))
                merged = None
        if merged is None:
            log("   Внимание: объединить тела не удалось")
            return None, None

    if not getattr(merged, "is_watertight", False):
        try:
            merged.fill_holes()
        except Exception:
            pass
    if not getattr(merged, "is_watertight", False):
        log("   Внимание: объединённая поверхность не замкнута "
            "(%d граней)" % len(merged.faces))
        return None, None

    return (np.asarray(merged.vertices, dtype=float),
            np.ascontiguousarray(merged.faces, dtype=np.int64))


def union_to_polydata(meshes, log=print):
    """То же, что union_meshes, но возвращает pyvista.PolyData или None."""
    if not HAS_PYVISTA:
        return None
    pts, fcs = union_meshes(meshes, log=log)
    if pts is None:
        return None
    flat = np.hstack([np.full((len(fcs), 1), 3, dtype=np.int64),
                      fcs]).ravel()
    return pv.PolyData(pts, flat)


def union_stats(meshes, log=print):
    """Что именно обрезало объединение.

    Возвращает dict:
        n_bodies     число тел на входе
        facets_in    граней во всех телах до объединения
        facets_out   граней в объединённой поверхности
        volume_in    сумма объёмов отдельных тел
        volume_out   объём объединения
        overlap      volume_in - volume_out, то есть объём пересечений
        mesh         pyvista.PolyData объединённой поверхности или None
    Объёмы считаются только для замкнутых тел; если тело не замкнуто,
    его объём в сумму не попадает.
    """
    facets_in = 0
    volume_in = 0.0
    n_closed = 0
    for m in meshes:
        pts, fcs = to_triangles(m)
        if pts is None:
            continue
        facets_in += int(len(fcs))
        if not HAS_TRIMESH:
            continue
        t = trimesh.Trimesh(vertices=pts, faces=fcs, process=True)
        if getattr(t, "is_watertight", False):
            try:
                volume_in += float(t.volume)
                n_closed += 1
            except Exception:
                pass

    merged = union_to_polydata(meshes, log=log)
    out = {"n_bodies": len(meshes),
           "facets_in": facets_in,
           "facets_out": 0,
           "volume_in": volume_in,
           "volume_out": 0.0,
           "overlap": 0.0,
           "n_closed_in": n_closed,
           "mesh": merged}
    if merged is None:
        return out
    pts, fcs = to_triangles(merged)
    out["facets_out"] = int(len(fcs))
    if HAS_TRIMESH:
        t = trimesh.Trimesh(vertices=pts, faces=fcs, process=True)
        try:
            out["volume_out"] = float(t.volume)
        except Exception:
            out["volume_out"] = 0.0
    out["overlap"] = out["volume_in"] - out["volume_out"]
    return out
