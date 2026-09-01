# -*- coding: utf-8 -*-
"""Уплотнение триангуляции поверхности (mesh/surface_refine.py).

Что проверяется и почему именно это
-----------------------------------
1. Бисекция длинного ребра даёт конформную замкнутую поверхность с тем
   же объёмом. Прежняя версия шаблонов 1-к-2 и 1-к-3 писала обход
   вершин в обратную сторону: на единичном треугольнике A(0,0) B(1,0)
   C(0,1) разрез (C,pAB,A) давал площадь -0.25 вместо +0.25. На
   замкнутой поверхности это выворачивало нормали, объём становился
   отрицательным, а is_watertight — False. Поэтому объём и watertight
   проверяются явно, а не «на глаз» по числу граней.

2. Раздачу бюджета. Равномерный шаг по всему самолёту не проходит:
   доведение каждой грани до толщины оперения 0.084 м даёт 3.24 млн
   треугольников на четырёх компонентах.

3. Плоские тела. ВО сейчас строится листом нулевой толщины (оба сечения
   в y=0). Если брать минимум габарита буквально, такое тело получает
   желаемый шаг 0 и съедает весь бюджет — в прогоне ВО забрал 501932
   грани из 600000, а фюзеляж не получил ничего.

Запуск:  python tests/test_surface_refine.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util                            # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "surface_refine_standalone",
    os.path.join(_ROOT, "mesh", "surface_refine.py"))
sr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sr)

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

_passed = 0
_failed = []


def check(name, cond, extra=None):
    global _passed
    if cond:
        _passed += 1
        print("  ✅ %s" % name)
    else:
        _failed.append(name)
        print("  ❌ %s%s" % (name, "" if extra is None else " %s" % (extra,)))


def box(extents, offset=(0.0, 0.0, 0.0)):
    """Закрытый бокс треугольниками — заведомо watertight, объём известен."""
    dx, dy, dz = extents
    ox, oy, oz = offset
    v = np.array([[ox, oy, oz], [ox + dx, oy, oz], [ox + dx, oy + dy, oz],
                  [ox, oy + dy, oz], [ox, oy, oz + dz], [ox + dx, oy, oz + dz],
                  [ox + dx, oy + dy, oz + dz], [ox, oy + dy, oz + dz]],
                 dtype=float)
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
             (2, 3, 7, 6), (3, 0, 4, 7)]
    f = []
    for a, b, c, d in quads:
        f.append([a, b, c])
        f.append([a, c, d])
    return v, np.array(f, dtype=np.int64)


def plate(span, chord, thick, n_span=2):
    """Пластина, лофченная n_span+1 сечениями, — как generate_tail_surface.

    Даёт те же вытянутые треугольники: при span=2.4 и n_span=2 пролёт
    по размаху 1.2 м при толщине 0.084 м.
    """
    ys = np.linspace(-span / 2, span / 2, n_span + 1)
    pts = []
    for y in ys:
        pts.append([-chord / 2, y, -thick / 2])
        pts.append([+chord / 2, y, -thick / 2])
        pts.append([+chord / 2, y, +thick / 2])
        pts.append([-chord / 2, y, +thick / 2])
    v = np.array(pts, dtype=float)
    f = []
    for k in range(n_span):
        b = 4 * k
        n = 4 * (k + 1)
        for i in range(4):
            j = (i + 1) % 4
            f.append([b + i, b + j, n + j])
            f.append([b + i, n + j, n + i])
    for i in range(4):
        j = (i + 1) % 4
        f.append([i, j, len(v)])
    v = np.vstack([v, [[0, -span / 2, 0]]])
    base = 4 * n_span
    for i in range(4):
        j = (i + 1) % 4
        f.append([base + i, base + j, len(v)])
    v = np.vstack([v, [[0, +span / 2, 0]]])
    return v, np.array(f, dtype=np.int64)


print("== бисекция длинного ребра ==")
v, f = box((1.0, 1.0, 1.0))
st0 = sr.edge_length_stats(v, f)
check("исходный бокс: 12 граней, макс. ребро = диагональ",
      st0["n_faces"] == 12 and abs(st0["max"] - np.sqrt(2.0)) < 1e-9,
      "%.4f" % st0["max"])

v2, f2, info = sr.refine_to_edge_length(v, f, 0.25)
st2 = sr.edge_length_stats(v2, f2)
check("целевая длина ребра достигнута", st2["max"] <= 0.25 + 1e-12,
      "max=%.4f" % st2["max"])
check("info.reached согласован с фактом", info["reached"] is True)
check("число граней выросло", len(f2) > len(f), "%d -> %d" % (len(f), len(f2)))

if HAS_TRIMESH:
    m0 = trimesh.Trimesh(v, f, process=False)
    m2 = trimesh.Trimesh(v2, f2, process=False)
    check("поверхность осталась замкнутой", m2.is_watertight)
    check("ориентация согласована", m2.is_winding_consistent)
    check("объём сохранён", abs(m2.volume - m0.volume) < 1e-9 * max(1.0, abs(m0.volume)),
          "%.9f -> %.9f" % (m0.volume, m2.volume))
    check("объём положительный (нормали не вывернуты)", m2.volume > 0,
          "%.6f" % m2.volume)
    # рёбра не должны разъехаться: каждое ребро ровно у двух граней
    e = np.sort(np.concatenate([f2[:, [0, 1]], f2[:, [1, 2]],
                                f2[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    check("нет T-стыков и дыр: у каждого ребра ровно 2 соседа",
          bool((cnt == 2).all()),
          "распределение %s" % dict(zip(*np.unique(cnt, return_counts=True))))
else:
    print("  (trimesh недоступен — проверки замкнутости пропущены)")

# все три шаблона разрезания exercised: нужно смешать длинные и короткие
print()
print("== все три шаблона разрезания ==")
v, f = plate(2.4, 0.70, 0.084, n_span=2)
st = sr.edge_length_stats(v, f)
check("пластина повторяет проблему оперения: ребро 1.2 м при толщине 0.084",
      st["max"] > 1.1, "max=%.4f" % st["max"])
v2, f2, info = sr.refine_to_edge_length(v, f, 0.084)
st2 = sr.edge_length_stats(v2, f2)
check("длинное ребро доведено до цели", st2["max"] <= 0.084 + 1e-12,
      "max=%.4f" % st2["max"])
if HAS_TRIMESH:
    m0 = trimesh.Trimesh(v, f, process=False)
    m2 = trimesh.Trimesh(v2, f2, process=False)
    check("замкнута после разрезания", m2.is_watertight)
    check("объём пластины сохранён",
          abs(m2.volume - m0.volume) < 1e-9,
          "%.9f -> %.9f" % (m0.volume, m2.volume))

print()
print("== предел по числу граней не проглатывается ==")
v, f = plate(2.4, 0.70, 0.084, n_span=2)
v3, f3, info3 = sr.refine_to_edge_length(v, f, 0.001, max_faces=2000)
check("capped выставлен, когда бюджет исчерпан", info3["capped"] is True)
check("reached при этом False", info3["reached"] is False)
check("граней не больше предела", len(f3) <= 4 * 2000 + 100,
      len(f3))

print()
print("== раздача бюджета ==")
# тонкая пластина + крупный бокс: бюджет должен уйти тонкой,
# а не быть съеденным крупным телом
thin_v, thin_f = plate(2.4, 0.70, 0.084, n_span=2)
big_v, big_f = box((8.0, 1.2, 1.2))
res = sr.refine_within_budget([(thin_v, thin_f), (big_v, big_f)],
                              total_budget=60_000)
(vt, ft, it), (vb, fb, ib) = res
check("бюджет не превышен", len(ft) + len(fb) <= 60_000,
      "%d + %d" % (len(ft), len(fb)))
check("тонкое тело уплотнено сильнее крупного",
      it["target"] < ib["target"],
      "%.4f против %.4f" % (it["target"], ib["target"]))

print()
print("== плоское тело не съедает бюджет ==")
# нулевая толщина по одной оси — как у вертикального оперения сейчас
flat_v, flat_f = plate(1.2, 0.70, 0.0, n_span=2)
res = sr.refine_within_budget([(thin_v, thin_f), (flat_v, flat_f),
                               (big_v, big_f)], total_budget=60_000)
info_flat = res[1][2]
check("плоское тело помечено", info_flat.get("flat") is True)
check("плоское тело не забрало весь бюджет",
      res[1][1].shape[0] < 0.5 * 60_000,
      "плоское %d из 60000" % res[1][1].shape[0])
check("тонкое тело при этом уплотнено", res[0][1].shape[0] > len(thin_f),
      "%d -> %d" % (len(thin_f), res[0][1].shape[0]))

print()
print("== вырожденный вход не роняет ==")
v, f, i = sr.refine_to_edge_length(np.zeros((0, 3)), np.zeros((0, 3), int), 0.1)
check("пустая поверхность", len(f) == 0 and i["faces_after"] == 0)
v, f = box((1, 1, 1))
v2, f2, i2 = sr.refine_to_edge_length(v, f, 0.0)
check("нулевая цель — без изменений", len(f2) == len(f) and i2["passes"] == 0)
v2, f2, i2 = sr.refine_to_edge_length(v, f, 100.0)
check("цель крупнее всех рёбер — без изменений",
      len(f2) == len(f) and i2["passes"] == 0)

print()
print("== целевой шаг на треугольник ==")
v, f = plate(2.4, 0.70, 0.084, n_span=2)
n = len(f)
tgt = np.full(n, 0.5)
tgt[: n // 2] = 0.05          # половина поверхности мелко, половина крупно
v2, f2, i2 = sr.refine_to_edge_length(v, f, tgt)
check("массив целей принят", i2["faces_after"] > n, "%d -> %d" % (n, len(f2)))
if HAS_TRIMESH:
    m2 = trimesh.Trimesh(v2, f2, process=False)
    # Ключевое: при разной цели у соседей ребро на границе зон должно
    # делиться согласованно. Первая версия помечала ребро по своему
    # треугольнику, и на границе зон возникал T-стык — watertight False
    # при сошедшемся объёме.
    check("конформность при смешанных целях (watertight)", m2.is_watertight)
    check("объём сохранён при смешанных целях",
          abs(m2.volume - trimesh.Trimesh(v, f, process=False).volume) < 1e-9,
          "%.6f" % m2.volume)
    e = np.sort(np.concatenate([f2[:, [0, 1]], f2[:, [1, 2]],
                                f2[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    check("у каждого ребра ровно два соседа (нет T-стыков)",
          bool((cnt == 2).all()),
          "распределение %s" % dict(zip(*np.unique(cnt, return_counts=True))))

try:
    sr.refine_to_edge_length(v, f, np.full(n + 1, 0.1))
    check("несовпадение длины массива целей ловится", False)
except ValueError:
    check("несовпадение длины массива целей ловится", True)

print()
print("== минимальный габарит по непустым осям ==")
flat = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
check("плоский лист: минимум по непустым осям, а не 0",
      abs(sr.min_nonzero_extent(flat) - 1.0) < 1e-12,
      "%.6f" % sr.min_nonzero_extent(flat))
solid = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 0.5]], dtype=float)
check("обычное тело: минимум из трёх осей",
      abs(sr.min_nonzero_extent(solid) - 0.5) < 1e-12,
      "%.6f" % sr.min_nonzero_extent(solid))
check("пустой вход не роняет", sr.min_nonzero_extent(np.zeros((0, 3))) == 0.0)

print()
print("== поле целевых шагов ==")
# крупное тело рядом с тонким: у тонкого шаг мелкий, вдали дорастает
big = np.array([[-4, -0.6, -0.6], [4, 0.6, 0.6]], dtype=float)
thin_v, thin_f = plate(2.8, 0.56, 0.067, n_span=2)
thin_v = thin_v + np.array([3.0, 0.0, 0.0])
sur_v = np.vstack([big, thin_v])
sur_f = np.array([[0, 1, 4], [0, 4, 5], [1, 3, 6], [1, 6, 7]], dtype=np.int64)
tgt = sr.target_field_from_bodies(sur_v, sur_f,
                                  bodies=[big[:1] * 0 + np.array(
                                      [[-4, -0.6, -0.6], [4, 0.6, 0.6],
                                       [4, -0.6, 0.6], [-4, 0.6, 0.6],
                                       [-4, -0.6, 0.6], [4, 0.6, -0.6]]),
                                      thin_v])
check("поле построено по числу треугольников", len(tgt) == len(sur_f))
check("у тонкого тела шаг мельче, чем у крупного",
      tgt.min() < tgt.max(), "min=%.4f max=%.4f" % (tgt.min(), tgt.max()))
# Цель считается как толщина/divisor + growth*расстояние, поэтому
# проверять надо формулу, а не число: у треугольников выше их центроиды
# стоят в ~0.25 м от тонкого тела и член роста даёт 0.0223 + 0.30*0.25
# = 0.097. Берём треугольник, лежащий прямо на тонком теле, — там
# расстояние ноль и цель обязана быть ровно толщина/3.
_on = thin_f[0]
sur_f2 = np.array([list(_on)], dtype=np.int64)
sur_v2 = thin_v
t_on = sr.target_field_from_bodies(sur_v2, sur_f2,
                                   bodies=[np.array([[-4., -0.6, -0.6],
                                                     [4., 0.6, 0.6]]),
                                           thin_v],
                                   body_faces=[None, thin_f])
check("на поверхности тонкого тела цель равна толщина/3",
      abs(float(t_on[0]) - 0.067 / 3.0) < 1e-9,
      "%.6f против %.6f" % (t_on[0], 0.067 / 3.0))
check("вдали от тонкого тела цель дорастает",
      tgt.max() > 0.067 / 3.0 * 2,
      "max=%.4f" % tgt.max())
check("шаг ограничен сверху", tgt.max() <= 8.0 * 0.05 + 1e-9,
      "%.4f" % tgt.max())
check("пустой список тел не роняет",
      len(sr.target_field_from_bodies(sur_v, sur_f, bodies=[])) == len(sur_f))
check("пустая поверхность возвращает пустой массив",
      len(sr.target_field_from_bodies(np.zeros((0, 3)),
                                      np.zeros((0, 3), int), [big])) == 0)

print()
print("Пройдено: %d" % _passed)
if _failed:
    print("ПРОВАЛЕНО ТЕСТОВ: %d → %s" % (len(_failed), _failed))
    raise SystemExit(1)
print("Все проверки пройдены.")
