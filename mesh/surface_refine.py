# -*- coding: utf-8 -*-
"""mesh/surface_refine.py — уплотнение триангуляции поверхности.

Зачем
-----
Генераторы и импорт STL дают поверхность с сильно вытянутыми
треугольниками. У горизонтального оперения (размах 2.4 м, хорда 0.70 м,
профиль 12%, толщина 0.084 м) измерено: медианное ребро 0.327 м,
максимальное 1.200 м, 44% рёбер длиннее 0.5 м. Причина в лофте тремя
сечениями — по размаху идут два пролёта по 1.2 м.

Сеточник наследует эту триангуляцию. gmsh импортированную STL-поверхность
не переразбивает вовсе: при Mesh.CharacteristicLengthMax 0.20 и 0.05
результат одинаковый, 12 треугольников на боксе; setCompound(2, ...) тоже
ничего не меняет.

Почему не trimesh.remesh.subdivide_to_size
------------------------------------------
Он делит выбранный треугольник 1-к-4, то есть вместе с длинным ребром
halves и короткое. У оперения короткое ребро 0.0011 м (задняя кромка),
длинное 1.200 м; чтобы довести длинное до 0.084 м, нужно 4 прохода, и
короткое становится 0.00007 м. Один компонент даёт 474 * 4**4 ≈ 121 тыс.
вытянутых треугольников, весь самолёт — около 400 тыс.

Что делает этот модуль
----------------------
Бисекция длинного ребра: за проход делятся только рёбра длиннее целевой
длины, треугольник разрезается по числу помеченных рёбер — 1-к-2, 1-к-3
или 1-к-4. Помеченное ребро делится сразу во всех треугольниках, где
встречается, поэтому сетка остаётся конформной, а короткие рёбра не
трогаются. Рост числа треугольников пропорционален отношению длинного
ребра к целевому, а не его четвёртой степени.

Модуль — чистый numpy, без gmsh и без trimesh.
"""

from __future__ import annotations

import numpy as np

__all__ = ["refine_to_edge_length", "edge_length_stats"]


def edge_length_stats(vertices, faces) -> dict:
    """Длины рёбер: минимум, медиана, максимум, число треугольников."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
        return {"n_faces": int(len(f) if f.ndim == 2 else 0), "min": 0.0,
                "median": 0.0, "max": 0.0, "edges": np.zeros(0)}
    e = np.concatenate([
        np.linalg.norm(v[f[:, 0]] - v[f[:, 1]], axis=1),
        np.linalg.norm(v[f[:, 1]] - v[f[:, 2]], axis=1),
        np.linalg.norm(v[f[:, 2]] - v[f[:, 0]], axis=1),
    ])
    return {"n_faces": int(len(f)), "min": float(e.min()),
            "median": float(np.median(e)), "max": float(e.max()), "edges": e}


def _edge_index(a, b, uniq_keys, stride, base):
    """Индекс средней точки для рёбер (a, b).

    ``uniq_keys`` — отсортированные ключи уникальных рёбер текущего
    прохода, ``stride`` — множитель упаковки, ``base`` — индекс первой
    новой точки. Всё векторно, без питоновского цикла по граням.
    """
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    key = lo.astype(np.int64) * stride + hi.astype(np.int64)
    # Ищем все рёбра подряд, включая непомеченные: np.where всё равно
    # вычисляет обе ветви, поэтому для отсутствующих пар позиция уходит
    # за конец массива. Их значения нигде не выбираются, но индекс должен
    # оставаться в диапазоне, иначе упадёт уже на np.stack.
    pos = np.searchsorted(uniq_keys, key)
    np.clip(pos, 0, len(uniq_keys) - 1, out=pos)
    return base + pos


def refine_to_edge_length(vertices, faces, max_edge,
                          max_faces=4_000_000, max_passes=60):
    """Делит рёбра длиннее цели, пока такие есть.

    ``max_edge`` — число либо массив длиной ``len(faces)`` с целевой
    длиной для каждого треугольника. Массив нужен потому, что
    равномерный шаг по всему самолёту не проходит: 0.084 м даёт 385894
    грани и всё равно не разрешает оперение, а 0.028 м — 3705492.
    Дозировать сгущение можно только по месту.

    При разрезании потомки наследуют цель родителя.

    Возвращает ``(vertices, faces, info)``. В ``info``:

    ``passes``      число проходов;
    ``faces_before``/``faces_after`` число треугольников;
    ``capped``      сработал ли предел ``max_faces`` — тогда цель по
                    длине ребра НЕ достигнута, и это надо показать
                    пользователю, а не проглатывать;
    ``reached``     все ли рёбра теперь не длиннее ``max_edge``.
    """
    v = np.asarray(vertices, dtype=float).copy()
    f = np.asarray(faces, dtype=np.int64).copy()
    n0 = len(f)
    if len(f) == 0:
        return v, f, {"passes": 0, "faces_before": n0, "faces_after": len(f),
                      "capped": False, "reached": True, "max_edge": 0.0}
    tgt = np.asarray(max_edge, dtype=float)
    if tgt.ndim == 0:
        tgt = np.full(len(f), float(tgt))
    if len(tgt) != len(f):
        raise ValueError("max_edge: нужно число или массив длиной %d, "
                         "дано %d" % (len(f), len(tgt)))
    if float(tgt.max()) <= 0:
        return v, f, {"passes": 0, "faces_before": n0, "faces_after": len(f),
                      "capped": False, "reached": True, "max_edge": 0.0}

    n_pass = 0
    capped = False
    while n_pass < max_passes:
        l01 = np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1)
        l12 = np.linalg.norm(v[f[:, 2]] - v[f[:, 1]], axis=1)
        l20 = np.linalg.norm(v[f[:, 0]] - v[f[:, 2]], axis=1)
        m01, m12, m20 = l01 > tgt, l12 > tgt, l20 > tgt
        if not (m01.any() or m12.any() or m20.any()):
            break

        # Все рёбра, которые нужно поделить, одним массивом.
        parts = []
        if m01.any():
            parts.append(f[m01][:, [0, 1]])
        if m12.any():
            parts.append(f[m12][:, [1, 2]])
        if m20.any():
            parts.append(f[m20][:, [2, 0]])
        pairs = np.sort(np.vstack(parts), axis=1)
        uniq = np.unique(pairs, axis=0)

        # Решение «делить ребро» принимается по любому из смежных
        # треугольников, а не по своему.
        #
        # При разной цели у соседей (а именно для этого и нужен массив
        # tgt) ребро на границе зон получало пометку в одном
        # треугольнике и не получало в другом: один делился, второй
        # нет, и возникал T-стык. На замкнутой поверхности это сразу
        # is_watertight=False — так и вышло в первом прогоне с полем
        # шагов: 119327 граней, объём сошёлся, а watertight нет.
        n_v0 = len(v)
        stride0 = np.int64(2 * n_v0 + 2)
        marked_keys = np.unique(uniq[:, 0].astype(np.int64) * stride0
                                + uniq[:, 1].astype(np.int64))

        def _marked(a, b):
            lo = np.minimum(a, b).astype(np.int64)
            hi = np.maximum(a, b).astype(np.int64)
            k = lo * stride0 + hi
            pos = np.searchsorted(marked_keys, k)
            np.clip(pos, 0, len(marked_keys) - 1, out=pos)
            return marked_keys[pos] == k

        m01 = _marked(f[:, 0], f[:, 1])
        m12 = _marked(f[:, 1], f[:, 2])
        m20 = _marked(f[:, 2], f[:, 0])
        n_pass += 1

        n_v = len(v)
        m = len(uniq)
        stride = np.int64(n_v + m + 1)
        uniq_keys = uniq[:, 0].astype(np.int64) * stride + uniq[:, 1].astype(np.int64)
        v = np.vstack([v, (v[uniq[:, 0]] + v[uniq[:, 1]]) * 0.5])

        A, B, C = f[:, 0], f[:, 1], f[:, 2]
        pAB = _edge_index(A, B, uniq_keys, stride, n_v)
        pBC = _edge_index(B, C, uniq_keys, stride, n_v)
        pCA = _edge_index(C, A, uniq_keys, stride, n_v)
        cnt = m01.astype(np.int8) + m12.astype(np.int8) + m20.astype(np.int8)

        # Шаблоны разрезания выверены на единичном треугольнике
        # A(0,0) B(1,0) C(0,1): каждый даёт положительную площадь и в
        # сумме ровно 0.5. Прежняя версия писала 1-к-2 как (C,p,A) и
        # (C,B,p) — это та же площадка, но с обратной обходкой, площадь
        # -0.25; на замкнутой поверхности это выворачивало нормали,
        # объём становился отрицательным, а поверхность переставала
        # быть watertight.
        out = []
        s0 = cnt == 0
        if s0.any():
            out.append(np.stack([A[s0], B[s0], C[s0]], axis=1))

        # 1 помеченное ребро — 1-к-2: разрез от середины ребра к
        # противоположной вершине. Для AB это (A,p,C) и (p,B,C);
        # для BC — (B,p,A) и (p,C,A); для CA — (C,p,B) и (p,A,B).
        s1 = cnt == 1
        if s1.any():
            aa, bb, cc = A[s1], B[s1], C[s1]
            q01, q12 = m01[s1], m12[s1]
            apex = np.where(q01, cc, np.where(q12, aa, bb))
            end1 = np.where(q01, aa, np.where(q12, bb, cc))
            end2 = np.where(q01, bb, np.where(q12, cc, aa))
            pnt = np.where(q01, pAB[s1], np.where(q12, pBC[s1], pCA[s1]))
            out.append(np.stack([end1, pnt, apex], axis=1))
            out.append(np.stack([pnt, end2, apex], axis=1))

        # 2 помеченных — 1-к-3. Свободная вершина — та, у которой ребро
        # не помечено; помеченные рёбра из неё выходят.
        s2 = cnt == 2
        if s2.any():
            aa, bb, cc = A[s2], B[s2], C[s2]
            q01, q12, q20 = m01[s2], m12[s2], m20[s2]
            pab, pbc, pca = pAB[s2], pBC[s2], pCA[s2]
            # не помечено CA -> свободна B: (pAB,B,pBC) (A,pAB,C) (pAB,pBC,C)
            # не помечено BC -> свободна A: (pCA,A,pAB) (C,pCA,B) (pCA,pAB,B)
            # не помечено AB -> свободна C: (pBC,C,pCA) (B,pBC,A) (pBC,pCA,A)
            freeB = q01 & q12
            freeA = q01 & q20
            free = np.where(freeB, bb, np.where(freeA, aa, cc))
            m1 = np.where(freeB, pab, np.where(freeA, pca, pbc))
            m2 = np.where(freeB, pbc, np.where(freeA, pab, pca))
            o1 = np.where(freeB, aa, np.where(freeA, cc, bb))
            o2 = np.where(freeB, cc, np.where(freeA, bb, aa))
            out.append(np.stack([m1, free, m2], axis=1))
            out.append(np.stack([o1, m1, o2], axis=1))
            out.append(np.stack([m1, m2, o2], axis=1))

        # 3 помеченных — 1-к-4.
        s3 = cnt == 3
        if s3.any():
            aa, bb, cc = A[s3], B[s3], C[s3]
            pab, pbc, pca = pAB[s3], pBC[s3], pCA[s3]
            out.append(np.stack([aa, pab, pca], axis=1))
            out.append(np.stack([pab, bb, pbc], axis=1))
            out.append(np.stack([pca, pbc, cc], axis=1))
            out.append(np.stack([pab, pbc, pca], axis=1))

        f = np.vstack(out).astype(np.int64)
        # цели потомков: каждый блок out[] соответствует маске s0..s3,
        # поэтому цель родителя просто повторяется нужное число раз
        tg = []
        if s0.any():
            tg.append(tgt[s0])
        if s1.any():
            tg += [tgt[s1], tgt[s1]]
        if s2.any():
            tg += [tgt[s2], tgt[s2], tgt[s2]]
        if s3.any():
            tg += [tgt[s3]] * 4
        tgt = np.concatenate(tg)
        if len(f) > max_faces:
            capped = True
            break

    st = edge_length_stats(v, f)
    lv = np.concatenate([
        np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1),
        np.linalg.norm(v[f[:, 2]] - v[f[:, 1]], axis=1),
        np.linalg.norm(v[f[:, 0]] - v[f[:, 2]], axis=1),
    ]) if len(f) else np.zeros(1)
    reached = bool((lv <= tgt.max() * (1.0 + 1e-9)).all()) if len(f) else True
    return v, f, {"passes": n_pass, "faces_before": n0, "faces_after": len(f),
                  "capped": capped, "reached": reached,
                  "max_edge": st["max"]}


def refine_within_budget(bodies, total_budget=600_000,
                         min_edge_fraction=0.01):
    """Уплотняет набор поверхностей в рамках общего бюджета граней.

    Равномерный шаг по всему самолёту не проходит по памяти: доведение
    каждой грани до толщины оперения (0.084 м) даёт 3.24 млн треугольников
    только на четырёх компонентах, из них 2.97 млн — на фюзеляже, у
    которого вырожденные рёбра по 7.8 м на носке и срезе.

    Поэтому здесь компромисс, который деградирует явно, а не взрывается:

    1. Каждому телу назначается желаемый шаг — треть от его собственного
       минимального габарита, не мельче ``min_edge_fraction`` от
       максимального. Фюзеляжу диаметром 1.20 м не нужно 0.028 м, а
       оперению толщиной 0.084 м нужно.
    2. Тела сортируются по возрастанию минимального габарита: самые
       тонкие получают приоритет.
    3. Бюджет раздаётся по очереди. Если на тело не хватает, оно
       уплотняется настолько, насколько хватает, и это фиксируется в
       ``reached=False`` — вызывающий код обязан это показать.

    ``bodies`` — список ``(vertices, faces)``. Возвращает список
    ``(vertices, faces, info)`` в том же порядке.
    """
    prepared = []
    for v, f in bodies:
        v = np.asarray(v, dtype=float)
        f = np.asarray(f, dtype=np.int64)
        ext = (v.max(axis=0) - v.min(axis=0)) if len(v) else np.zeros(3)
        max_dim = float(ext.max()) if len(v) else 0.0
        # Минимальный габарит — но только по непустым осям.
        #
        # Вертикальное оперение сейчас строится как лист нулевой толщины:
        # generate_vertical_stabilizer_geometry ставит и корневое, и
        # концевое сечение в y=0, поэтому размах по y ровно 0. Если брать
        # ext.min() буквально, такое тело получает желаемый шаг 0,
        # уходит на пол max_dim*0.01 и съедает весь бюджет: в прогоне на
        # пяти компонентах ВО забрал 501932 грани из 600000, а фюзеляж
        # не получил ничего.
        _sig = ext[ext > max(1e-9, 1e-6 * max_dim)]
        min_dim = float(_sig.min()) if len(_sig) else max_dim
        prepared.append({
            "v": v, "f": f,
            "min_dim": min_dim,
            "max_dim": max_dim,
            "flat": bool(len(_sig) < 3),
            "n": int(len(f)),
        })

    order = sorted(range(len(prepared)), key=lambda i: prepared[i]["min_dim"])
    results = [None] * len(prepared)
    left = int(total_budget)

    for idx in order:
        b = prepared[idx]
        if b["n"] == 0:
            results[idx] = (b["v"], b["f"],
                            {"passes": 0, "faces_before": 0, "faces_after": 0,
                             "capped": False, "reached": True,
                             "target": 0.0, "skipped": True})
            continue

        want = b["min_dim"] / 3.0
        floor = b["max_dim"] * min_edge_fraction
        if want < floor:
            want = floor

        avail = max(left - sum(prepared[j]["n"]
                               for j in order[order.index(idx) + 1:]), b["n"])
        target = want
        v2 = f2 = None
        info = None
        while True:
            v2, f2, info = refine_to_edge_length(
                b["v"], b["f"], target, max_faces=max(avail, b["n"]) + 1)
            if len(f2) <= avail or target >= b["max_dim"]:
                break
            target *= 1.5
            if target > b["max_dim"]:
                target = b["max_dim"]
                v2, f2, info = refine_to_edge_length(
                    b["v"], b["f"], target, max_faces=10 ** 9)
                break

        info = dict(info)
        info["target"] = float(target)
        info["skipped"] = False
        info["flat"] = bool(b.get("flat"))
        info["reached"] = bool(info.get("reached")
                               and not info.get("capped"))
        results[idx] = (v2, f2, info)
        left -= len(f2)
        if left <= 0:
            for j in order[order.index(idx) + 1:]:
                bb = prepared[j]
                results[j] = (bb["v"], bb["f"],
                              {"passes": 0, "faces_before": bb["n"],
                               "faces_after": bb["n"], "capped": False,
                               "reached": False, "target": 0.0,
                               "skipped": True})
            break

    return results


def min_nonzero_extent(vertices) -> float:
    """Минимальный габарит тела по непустым осям.

    Оси с нулевым размахом игнорируются: плоский лист дал бы 0 и увёл бы
    желаемый шаг в ноль. Если пустых осей нет — это обычный минимум.
    """
    v = np.asarray(vertices, dtype=float)
    if len(v) == 0:
        return 0.0
    ext = v.max(axis=0) - v.min(axis=0)
    mx = float(ext.max())
    sig = ext[ext > max(1e-9, 1e-6 * mx)]
    return float(sig.min()) if len(sig) else mx


def target_field_from_bodies(vertices, faces, bodies, divisor=3.0,
                             growth=0.30, cap_fraction=0.05, default=None,
                             body_faces=None):
    """Целевая длина ребра для каждого треугольника объединённой поверхности.

    Равномерный шаг не проходит ни в одну сторону: 0.084 м даёт 385894
    грани и всё равно не разрешает оперение, 0.028 м — 3705492. Поэтому
    шаг назначается по месту.

    Но одного «ближайшего тела» мало. При жёстком скачке цели (0.028 м у
    оперения против 0.40 м у фюзеляжа) конформное разрезание растаскивает
    мелкий шаг по всей модели: ребро на границе зон делится, если этого
    хочет хотя бы один из соседей, сосед делится тоже, и так по цепочке.
    В первом прогоне так вышло 2700594 грани, из них 2699346 на
    фюзеляже при медианном ребре 0.0153 м, а оперению досталось 624
    грани — ровно наоборот.

    Поэтому цель растёт с расстоянием:

        candidate_b = dim_b / divisor + growth * dist_b
        target      = min_b candidate_b,  затем в [floor, cap]

    У поверхности тонкого тела шаг dim/divisor, в ``growth`` метрах на
    метр он дорастает до крупного. При growth=0.30 переход от 0.028 до
    0.40 занимает 1.24 м — реальная переходная зона, а не скачок.

    ``bodies``      — вершины исходных (не объединённых) тел.
    ``cap_fraction`` — верхняя граница шага как доля от габарита модели.
    ``default``     — шаг, если тел нет вовсе.
    """
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) == 0:
        return np.zeros(0)

    cents = v[f].mean(axis=1)
    all_ext = (v.max(axis=0) - v.min(axis=0)) if len(v) else np.zeros(3)
    model_max = float(all_ext.max()) if len(v) else 1.0
    cap = model_max * cap_fraction

    thin = []
    for bi, b in enumerate(bodies):
        b = np.asarray(b, dtype=float)
        if len(b) == 0:
            continue
        sample = b
        # Расстояние нужно до поверхности тела, а не до его вершин.
        #
        # На грубом теле центроид крупной грани лежит на поверхности, но
        # далеко от любой вершины: у оперения из двух пролётов по 1.4 м
        # это ~0.39 м, и цель вместо 0.0223 м получалась 0.14 м — тело,
        # которое нужно резать мельче всех, резалось в шесть раз крупнее.
        # Поэтому в выборку добавляются центроиды граней.
        if body_faces is not None and bi < len(body_faces)                 and body_faces[bi] is not None:
            bf = np.asarray(body_faces[bi], dtype=np.int64)
            if bf.ndim == 2 and bf.shape[1] >= 3 and len(bf):
                sample = np.vstack([sample, b[bf[:, :3]].mean(axis=1)])
        if len(sample) > 20_000:
            sample = sample[::max(1, len(sample) // 20_000)]
        thin.append((sample, min_nonzero_extent(b) / divisor))
    if not thin:
        d = cap if default is None else float(default)
        return np.full(len(f), min(max(d, 1e-6), cap))

    tgt = np.full(len(cents), np.inf)
    chunk = 200_000
    for start in range(0, len(cents), chunk):
        sl = slice(start, min(start + chunk, len(cents)))
        c = cents[sl]
        best = np.full(len(c), np.inf)
        for sample, base in thin:
            d = np.sqrt(((c[:, None, :] - sample[None, :, :]) ** 2
                         ).sum(axis=2)).min(axis=1)
            cand = base + growth * d
            np.minimum(best, cand, out=best)
        tgt[sl] = best

    out = np.where(np.isfinite(tgt), tgt, cap if default is None else default)
    return np.clip(out, 1e-6, cap)
