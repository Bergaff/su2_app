"""
geometry/reference.py — справочные данные по реальной геометрии крыла.

Зачем это нужно
---------------
Sref и Lref в config.cfg нормируют все аэродинамические коэффициенты:
Cl = L / (q·Sref), Cm = M / (q·Sref·Lref). Если геометрия одна, а
справочные данные взяты от другой модели, коэффициенты ошибочны ровно
во столько раз, во сколько отличаются площади и длины. Для EULER сами
уравнения масштабно-инвариантны, поэтому ошибка целиком приходит из
нормировки.

Раньше Sref и Lref считались из спинбоксов «Размах / Хорда корня /
Хорда конца». Для крыла, построенного встроенным генератором, это
верно: спинбоксы и есть источник геометрии. Но при импорте детали
спинбоксы не обновляются, и расчёт нормировался на заводские значения
(размах 10 м, хорды 1.8/0.9 м) независимо от того, что реально загружено.

Здесь те же величины берутся прямо из поверхностной сетки крыла.

Как считается
-------------
Планформа рассекается набором горизонтальных линий y = const. Для
каждой линии находятся все рёбра сетки, которые она пересекает, и по
ним восстанавливаются передняя и задняя кромки: x_le(y) и x_te(y).
Хорда c(y) = x_te(y) − x_le(y). Дальше обычное интегрирование:

    Sref = ∫ c(y) dy                     площадь планформы
    MAC  = (1/Sref) · ∫ c(y)² dy         средняя аэродинамическая хорда

Вторая формула — стандартное определение средней аэродинамической
хорды через интеграл по полному размаху. Для сужающегося крыла она
совпадает с учебной (2/3)·c_root·(1+λ+λ²)/(1+λ): при c_root=1.44,
c_tip=0.72, b=9.02 обе дают 1.1200.

Сечение берётся по рёбрам, а не по треугольникам, поэтому результат не
зависит от густоты сетки: прямоугольная пластина из двух треугольников
даёт ровно 12 м² и хорду 2 м. Это важно, потому что импортированный
CAD часто приходит очень грубым (крыло из 382 граней), а разнесение
треугольников по полосам на такой сетке занижает площадь в разы.
"""
from __future__ import annotations

import numpy as np


def wing_reference_from_mesh(points, faces, n_sections=193):
    """Размах, площадь и средняя аэродинамическая хорда по сетке крыла.

    points — массив (N, 3), faces — массив (M, 3) индексов треугольников.
    Ось размаха — Y, хорда откладывается по X.

    Возвращает dict:
        span        размах по Y
        area        площадь планформы (Sref)
        mac         средняя аэродинамическая хорда (Lref)
        chord_root  наибольшая хорда (у корня)
        chord_tip   хорда на самом дальнем от корня сечении
        x_le_mean   средневзвешенная по площади координата передней кромки
    или None, если по сетке ничего нельзя посчитать.
    """
    pts = np.asarray(points, dtype=float)
    fcs = np.asarray(faces, dtype=np.int64)
    if pts.ndim != 2 or pts.shape[0] < 3 or fcs.ndim != 2 or len(fcs) == 0:
        return None
    if fcs.shape[1] != 3 or fcs.max() >= len(pts):
        return None

    tri = pts[fcs]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if not (np.linalg.norm(cross, axis=1) > 0).any():
        return None

    y_min = float(tri[:, :, 1].min())
    y_max = float(tri[:, :, 1].max())
    span = y_max - y_min
    if span <= 1e-9:
        return None

    # Все рёбра сетки: три на треугольник.
    e = np.stack([tri[:, 0], tri[:, 1],
                  tri[:, 1], tri[:, 2],
                  tri[:, 2], tri[:, 0]], axis=1).reshape(-1, 2, 3)
    ya, yb = e[:, 0, 1], e[:, 1, 1]
    xa, xb = e[:, 0, 0], e[:, 1, 0]
    dy_edge = yb - ya
    live = np.abs(dy_edge) > 1e-15

    ys = np.linspace(y_min, y_max, max(8, int(n_sections)))
    chord = np.zeros(len(ys))
    x_le = np.zeros(len(ys))
    for k, yk in enumerate(ys):
        hit = live & ((ya - yk) * (yb - yk) <= 0.0)
        if not hit.any():
            continue
        x = xa[hit] + (yk - ya[hit]) * (xb[hit] - xa[hit]) / dy_edge[hit]
        x_le[k] = float(x.min())
        chord[k] = float(x.max()) - float(x.min())

    if not (chord > 0).any():
        return None

    # Интегрирование трапецией по реально заполненному диапазону, чтобы
    # пустые сечения у концов не растягивали площадь.
    filled = chord > 0
    y_use = ys[filled]
    c_use = chord[filled]
    le_use = x_le[filled]
    if len(y_use) < 2:
        return None
    w = np.gradient(y_use)
    area = float((c_use * w).sum())
    if area <= 1e-12:
        return None
    mac = float((c_use ** 2 * w).sum() / area)
    x_le_mean = float((c_use * le_use * w).sum() / area)

    i_root = int(np.argmax(c_use))
    i_tip = int(max(range(len(c_use)), key=lambda i: abs(i - i_root)))

    return {"span": span,
            "area": area,
            "mac": mac,
            "chord_root": float(c_use[i_root]),
            "chord_tip": float(c_use[i_tip]),
            "x_le_mean": x_le_mean,
            "y_min": y_min,
            "y_max": y_max}


def wing_reference_from_pv(mesh, n_sections=193):
    """То же, что wing_reference_from_mesh, но принимает pyvista.PolyData."""
    import numpy as _np
    faces_raw = _np.asarray(mesh.faces)
    if faces_raw.ndim == 1:
        if faces_raw.size == 0:
            return None
        stride = int(faces_raw[0]) + 1
        if stride < 4 or len(faces_raw) % stride != 0:
            return None
        faces = faces_raw.reshape(-1, stride)[:, 1:]
    else:
        faces = faces_raw
    return wing_reference_from_mesh(_np.asarray(mesh.points), faces,
                                    n_sections)
