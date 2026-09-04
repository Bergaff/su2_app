"""
mesh/bodyfit_tetgen.py — объёмная сетка, ОБЛЕГАЮЩАЮЩАЯ поверхность тела.

Зачем это нужно
---------------
Основной генератор (mesh/gmsh_generator.py) строит картезианский фон и
удаляет ячейки, чей центр попал внутрь тела. Поверхность тела в такую
сетку не попадает вовсе: границей расчётной области оказывается
«ступенька» из граней фоновых тетраэдров, и маркер airfoil в mesh.su2
собирается именно с этой ступеньки. Для тонких элементов (ГО, ВО, руль:
толщина 0.084 м при шаге фона 0.135 м — 0.62 шага) ступенька профиль не
описывает вообще, и SU2 на такой сетке расходится независимо от настроек
решателя.

Здесь сетка строится иначе. Поверхность тел и прямоугольная расчётная
область объединяются в один PLC и триангулируются TetGen как constrained
Delaunay. Ключевой параметр — ключ Y (nobisect): без него TetGen режет
входные грани ради соблюдения условия Делоне, и часть граней тела в
сетке отсутствует. Измерено на полном самолёте (43752 грани):

    без Y            38068 из 43752   87.0%
    с Y (pqY)        43750 из 43752   100.0%

Дальше тетраэдры, попавшие внутрь тела, удаляются, и границей сетки
оказываются настоящие грани тела. Проверено на том же самолёте:

    тетраэдров всего              317075
    удалено (центр внутри тела)   118447
    осталось                      198628
    объём наружной области        19676.0218  (короб 19683.0 - тело 6.9782)
    граней тела на границе сетки  43743 из 43752  = 99.98%

Как определить, какие тетраэдры внутри тела
------------------------------------------
Основной путь — **топологическая заливка снаружи** (`classify_exterior_tets`):
семена берутся из тетраэдров с центроидом вне габарита тела и заливка идёт
по смежным граням, НЕ пересекая граней тела (`body_face_keys`). Если
поверхность тела замкнута, заливка останавливается на ней: внутренность
тела остаётся непомеченной, и почти все грани тела выходят на границу
сетки (маркер airfoil ~100%). Замерено на полном самолёте: восстановлено
22894 из 22898 граней, а доля на границе (после перенумерации узлов через
vmap) почти 100%. Если же поверхность дырявая, заливка просачивается
внутрь — тогда наружными оказываются почти все тетраэдры, и вышестоящий
код отвергает такой результат.

Прежний геометрический `select_enclosed_points` (точка-в-объёме) сохранён
как **фолбэк**: он ошибается на стыках крыло/фюзеляж (19% граней «утонули»
внутрь, SU2 расходился), поэтому используется, только если топологическая
заливка дала сбой (нет семян / почти всё снаружи) или выбросила исключение.
В обоих случаях итоговая сетка проверяется по доле граней тела, вышедших
на границу: меньше `min_recovery` — возвращается None и честное сообщение
о дырявой поверхности.

Поле размера (mesh sizing function) — ОТКЛЮЧЕНО по умолчанию
----------------------------------------------------------
Поле размера делало сетку конформной: у тела ячейки мелкие (~`target_edge`),
к коробу — плавный рост до `extent/12`. Изотропное поле задавалось через
TetGen `-m` и фоновую сетку с `target_size` в узлах
(`build_size_field_background`). Однако проверено (tetgen 0.8.4, авиационная
геометрия короба + тела): `-m` с фоновой сеткой на **плоскостном PLC**
(расчётный короб из плоских граней) приводит к детерминированному
**SIGSEGV** (exit 139) в «Interpolating mesh size» — независимо от того, как
построена фоновая сетка (структурированная, со сдвигом узлов, случайная
Делоне), независимо от ключей (`pq2Y`, `pq2`, `pqY`, `p`) и даже на одном
выпуклом коробе без тела. Сегфолт — это сбой C++ в самом TetGen, его
**невозможно перехватить** из Python (`try/except` бесполезен). Поэтому
`build_body_fitted_grid` по умолчанию строит сетку **без поля** (`size_field
= False`), а любой явный `size_field=True` также игнорируется с предупреждением
— чтобы собранное приложение не падало. Помощники поля размера
(`size_field_for_points`, `build_size_field_background`, `_bg_axes`,
`_jitter_bg_points`, `_structured_tet_cells`) оставлены для тестов и на случай
будущего исправления TetGen, в рабочий путь они больше не попадают.

Без поля у тела (~0.1 м) и у короба (~6 м) остаётся резкий скачок размера;
это минус (может ухудшить сходимость 2-го порядка), но сетка строится
стабильно. Запасной путь к более мелкому разрешению у тела — уплотнение
самой поверхности (`surface_needs_refinement` / `_sr.refine_to_edge_length`),
которое нигде не роняет TetGen.

Если TetGen или движок булевых операций недоступны, функция возвращает
None, и вызывающий код остаётся на прежнем картезианском пути.
"""
from __future__ import annotations

import os
import numpy as np

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:          # pragma: no cover - pyvista есть в зависимостях
    pv = None
    HAS_PYVISTA = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    cKDTree = None
    HAS_SCIPY = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    trimesh = None
    HAS_TRIMESH = False

try:
    from tetgen import TetGen
    HAS_TETGEN = True
except Exception:
    TetGen = None
    HAS_TETGEN = False


# Доля граней тела, которая обязана оказаться в сетке. Ниже — сетка не
# облегает поверхность, и смысла в этом пути нет: возвращаем None.
DEFAULT_MIN_RECOVERY = 0.90


def tetgen_available():
    """True, если телообтекающий путь вообще может быть построен."""
    return bool(HAS_TETGEN and HAS_TRIMESH and HAS_SCIPY and HAS_PYVISTA)


def tetgen_missing():
    """Чего не хватает для телооблекающей сетки. Для честного сообщения."""
    miss = []
    if not HAS_TETGEN:
        miss.append("tetgen")
    if not HAS_TRIMESH:
        miss.append("trimesh")
    if not HAS_SCIPY:
        miss.append("scipy")
    if not HAS_PYVISTA:
        miss.append("pyvista")
    return miss


def _load_surface_refine():
    """Загрузить mesh/surface_refine.py по пути файла (без mesh/__init__)."""
    try:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "surface_refine.py")
        spec = importlib.util.spec_from_file_location(
            "surface_refine_standalone", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def surface_needs_refinement(points, faces, target_edge):
    """Грубее ли поверхность целевого шага.

    Сравнивается средняя площадь треугольника с площадью равностороннего
    треугольника с ребром ``target_edge`` (0.433 * h^2). Запас взят
    двукратный: уплотнять уже достаточно мелкую поверхность незачем, а
    на самолёте из пяти компонентов это стоило бы миллионов граней.

    Точность формы здесь ни при чём — поверхность и так точная. Речь о
    плотности: TetGen сохраняет входные грани как есть, поэтому размер
    ячеек у тела повторяет размер треугольников поверхности. Замерено на
    пластине plane_wing.step (65.077 x 19 x 1.5 с четырьмя пазами 0.3):
    412 граней дают 5301 тетраэдр, те же грани, уплотнённые до шага у
    тела (0.9762 м) — 17094 грани и 52315 тетраэдров, сохранение 100%
    в обоих случаях, 2.0 с.
    """
    try:
        pts = np.asarray(points, dtype=float)
        fac = np.asarray(faces, dtype=np.int64)
        if len(fac) == 0 or target_edge is None or target_edge <= 0:
            return False
        p0 = pts[fac[:, 0]]; p1 = pts[fac[:, 1]]; p2 = pts[fac[:, 2]]
        area = 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0),
                                          axis=1).sum())
        if area <= 0:
            return False
        mean_area = area / len(fac)
        return mean_area > 2.0 * 0.433 * float(target_edge) ** 2
    except Exception:
        return False


def _load_solid_union():
    """Загрузить geometry/solid_union.py по пути файла.

    Обычный `from geometry.solid_union import ...` исполняет
    geometry/__init__.py, а тот тянет PyQt5 через stl_healer. В фоновом
    процессе генерации сетки и в тестах это лишняя зависимость.
    """
    try:
        import importlib.util
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "geometry", "solid_union.py")
        spec = importlib.util.spec_from_file_location(
            "solid_union_standalone", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


_su = _load_solid_union()

if _su is not None:
    to_triangles = _su.to_triangles
else:
    def to_triangles(mesh):
        """Привести поверхность к (points[N,3], faces[M,3]) — запасной вариант."""
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


def union_surfaces(body_meshes, log=print):
    """Слить поверхности тел в одну замкнутую (общая реализация).

    Пересечения разрешаются булевым объединением: части тел, оказавшиеся
    внутри других тел, вырезаются. Подробное обоснование и замеры — в
    geometry/solid_union.py. Возвращает (points, faces) или None.
    """
    if _su is not None:
        return _su.union_meshes(body_meshes, log=log)
    return None, None


def farfield_bounds(body_min, body_max, margin):
    """Габариты расчётной области — те же, что у картезианского пути."""
    body_min = np.asarray(body_min, dtype=float)
    body_max = np.asarray(body_max, dtype=float)
    return (float(body_min[0] - margin * 0.8),
            float(body_max[0] + margin * 1.2),
            float(body_min[1] - margin),
            float(body_max[1] + margin),
            float(body_min[2] - margin),
            float(body_max[2] + margin))


def box_surface(bounds, h_box):
    """Поверхность расчётной области, разбитая на треугольники.

    Ключ Y (nobisect) запрещает TetGen резать входные грани, поэтому
    короб из 12 треугольников остался бы в mesh.su2 маркером farfield из
    12 гигантских элементов. Разбиваем грани короба заранее: размер
    элементов тогда задаём мы, а не TetGen.

    Все шесть граней строятся на ОДНОЙ решётке узлов (ax, ay, az), и
    узел хранится в единственном экземпляре по своему индексу (i, j, k).
    Иначе на общих рёбрах появляются совпадающие дубли вершин, TetGen
    трактует их как самопересечение и отбрасывает грани (проверено:
    «256 input triangles are skipped due to self-intersections»).

    Возвращает (points, faces), нормали направлены наружу.
    """
    x0, x1, y0, y1, z0, z1 = [float(b) for b in bounds]
    nx = max(1, int(round((x1 - x0) / h_box)))
    ny = max(1, int(round((y1 - y0) / h_box)))
    nz = max(1, int(round((z1 - z0) / h_box)))
    ax = np.linspace(x0, x1, nx + 1)
    ay = np.linspace(y0, y1, ny + 1)
    az = np.linspace(z0, z1, nz + 1)
    center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0])

    pts = []
    index = {}

    def vid(i, j, k):
        key = (int(i), int(j), int(k))
        n = index.get(key)
        if n is None:
            n = len(pts)
            index[key] = n
            pts.append([ax[key[0]], ay[key[1]], az[key[2]]])
        return n

    faces = []
    # Число делений по каждой оси. Пары «свободная ось — её число делений»
    # строятся явно: раньше порядок брался из списка [a for a in (0,1,2)
    # if a != fa], и для граней y=const первой свободной оказывалась ось X,
    # а число делений ей доставалось от Z. На кубическом коробе это
    # незаметно, на некубическом — IndexError.
    counts = {0: nx, 1: ny, 2: nz}
    faces_spec = [(0, 0), (0, nx), (1, 0), (1, ny), (2, 0), (2, nz)]
    for fa, fidx in faces_spec:
        free = [a for a in (0, 1, 2) if a != fa]
        ua, va = free[0], free[1]
        nu, nv = counts[ua], counts[va]
        for i in range(nu):
            for j in range(nv):
                corners = []
                for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    tri_idx = [0, 0, 0]
                    tri_idx[fa] = fidx
                    tri_idx[ua] = i + di
                    tri_idx[va] = j + dj
                    corners.append(vid(*tri_idx))
                a_, b_, c_, d_ = corners
                for t in ((a_, b_, c_), (a_, c_, d_)):
                    p = np.asarray(pts, dtype=float)[list(t)]
                    nrm = np.cross(p[1] - p[0], p[2] - p[0])
                    if np.dot(nrm, p.mean(axis=0) - center) < 0:
                        t = (t[0], t[2], t[1])
                    faces.append(list(t))
    return (np.asarray(pts, dtype=float),
            np.ascontiguousarray(faces, dtype=np.int64))


# ---------------------------------------------------------------------------
# Поле размера (mesh sizing function) для TetGen (-m + .mtr / bgmesh)
# ---------------------------------------------------------------------------
# Максимум узлов фоновой сетки поля размера. Поле лог-линейное (гладкое),
# поэтому фон нужен только как решётка для линейной интерполяции TetGen —
# сгущать до размера ячеек у тела не требуется: их размер задаёт входная
# поверхность (ключ Y её не дробит), а поле лишь сглаживает переход к
# дальнему полю. Скачок «0.1 м у тела -> 6 м у короба» без промежуточных
# ячеек и вызывал расходимость 2-го порядка. Умеренный фон ускоряет сборку
# и не съедает память.
MAX_BG_NODES = 60_000
# Доля шага у тела, на которую случайно сдвигаются узлы фоновой сетки.
# Регулярная решётка фона имеет плоскости узлов (напр. z=0 у симметричной
# модели); входные точки короба/тела, лежащие ровно на такой плоскости,
# оказываются на грани фонового тета, и TetGen в «Interpolating mesh size»
# получает вырожденную барицентрику -> сегфолт. Малый случайный сдвиг убирает
# это совпадение, не искажая поле (поле лог-линейное и гладкое).
# Фиксированный seed даёт воспроизводимую сетку.
BG_JITTER_FRAC = 0.15
BG_JITTER_SEED = 712367


def make_graded_axis(lo, hi, inner_lo, inner_hi, h_near, h_far):
    """Ось с мелким шагом ``h_near`` в зоне [inner_lo, inner_hi] и крупным
    ``h_far`` за её пределами. Возвращает монотонный набор координат.

    Это та же кластеризация, что и у картезианского пути в gmsh_generator:
    вокруг тела густо, к границе области — реже и плавно (без одного
    гигантского скачка). Нужна, чтобы фоновая сетка поля размера имела узлы
    там, где размер ячеек меняется быстрее всего, — у самой поверхности.
    """
    lo, hi = float(lo), float(hi)
    inner_lo = max(lo, float(inner_lo))
    inner_hi = min(hi, float(inner_hi))
    if inner_hi <= inner_lo:
        inner_lo = inner_hi = 0.5 * (lo + hi)
    h_near = max(float(h_near), 1e-9)
    h_far = max(float(h_far), h_near)

    n_in = max(1, int(round((inner_hi - inner_lo) / h_near)))
    inner = np.linspace(inner_lo, inner_hi, n_in + 1)

    left = np.array([inner_lo])
    if inner_lo > lo + 1e-9:
        n_left = max(1, int(np.ceil((inner_lo - lo) / h_far)))
        left = np.linspace(lo, inner_lo, n_left + 1)

    right = np.array([inner_hi])
    if hi > inner_hi + 1e-9:
        n_right = max(1, int(np.ceil((hi - inner_hi) / h_far)))
        right = np.linspace(inner_hi, hi, n_right + 1)

    return np.unique(np.concatenate([left, inner[1:], right[1:]]))


def _nearest_distances(pts, body_pts):
    """Расстояние от каждой точки ``pts`` до ближайшей вершины поверхности тела."""
    pts = np.asarray(pts, dtype=float)
    body_pts = np.asarray(body_pts, dtype=float)
    if HAS_SCIPY:
        return cKDTree(body_pts).query(pts, k=1)[0].ravel()
    # Запасной вариант для малых выборок (тесты без scipy).
    d = np.sqrt(((pts[:, None, :] - body_pts[None, :, :]) ** 2).sum(-1))
    return d.min(axis=1)


def size_field_for_points(points, body_pts, h_near, h_far, L):
    """Целевая длина ребра в точках ``points`` по расстоянию до тела.

    У поверхности ``h_near``, при расстоянии >= ``L`` — ``h_far``, а между
    ними — гладкая геометрическая интерполяция (постоянный логарифмический
    рост). Важно, что рост монотонный и без скачка: именно скачок «мелко у
    тела -> крупно у короба» давал растянутые ячейки, которые взрывались на
    2-м порядке.

    ``L`` — расстояние, на котором размер достигает ``h_far`` (обычно 2-3
    габарита модели). Возвращает массив размера ``len(points)``.
    """
    d = _nearest_distances(points, body_pts)
    h_near = max(float(h_near), 1e-9)
    h_far = max(float(h_far), h_near)
    L = max(float(L), 1e-9)
    t = np.clip(d / L, 0.0, 1.0)
    h = h_near * (h_far / h_near) ** t
    return np.clip(h, h_near, h_far)


def build_size_field_background(bounds, body_min, body_max, body_pts,
                                h_near, h_far, log=print):
    """Тетраэдральная фоновая сетка ``target_size`` для ``-m`` TetGen.

    Сетка покрывает весь расчётный короб (``bounds``), узлы сгущены к телу
    (шаг ``h_near``) и разрежены вдали (``h_far``); в каждом узле лежит
    целевая длина ребра из :func:`size_field_for_points`. TetGen линейно
    интерполирует это поле и строит объёмные ячейки, конформные ему.

    Возвращает ``pyvista.UnstructuredGrid`` (тетраэдры) или ``None``, если
    построить нельзя (нет pyvista/scipy либо слишком многo узлов).
    """
    if not HAS_PYVISTA or not HAS_SCIPY:
        log("   Внимание: для поля размера нужны pyvista и scipy — пропускаю.")
        return None
    x0, x1, y0, y1, z0, z1 = [float(b) for b in bounds]
    h_near = max(float(h_near), 1e-9)
    h_far = max(float(h_far), h_near)
    if h_far <= h_near * 1.05:
        # Поле не даст выигрыша: весь домен одного масштаба.
        return None

    # КРИТИЧНО: фоновая область должна СТРОГО вмещать расчётный короб, а не
    # совпадать с ним. Узлы основного PLC (box_surface от bounds) лежат ровно
    # на границе короба; если фон повторяет те же границы, эти узлы попадают
    # на грани фоновых тетов, и TetGen в «Interpolating mesh size» интерполирует
    # в вырожденной барицентрике -> сегфолт. Отступаем фон наружу на 3% размера
    # области (как это делает штатный пример tetgen с осевым запасом eps).
    extent = max(x1 - x0, y1 - y0, z1 - z0)

    axes, h_near, bmin, bmax = _bg_axes(
        bounds, body_min, body_max, h_near, h_far, max_nodes=MAX_BG_NODES)
    if axes is None:
        log("   Внимание: фоновая сетка поля размера слишком велика — "
            "строю без неё.")
        return None

    pts = _structured_grid_pts(axes)
    # Сбрасываем узлы на малый случайный сдвиг — иначе входные точки
    # симметричной модели (z=0 и т. п.) ложатся точно на фоновую плоскость
    # и TetGen падает в "Interpolating mesh size" (см. _jitter_bg_points).
    pts = _jitter_bg_points(pts, h_near)

    span = float(np.max(bmax - bmin))
    # Расстояние, на котором размер достигает h_far: ~2 габарита тела, но не
    # больше 0.55 габарита области, чтобы у короба поле уже было крупным и
    # согласовалось с его триангуляцией.
    L = max(2.0 * span, 0.05 * extent)
    L = min(L, 0.55 * extent)

    sizes = size_field_for_points(pts, body_pts, h_near, h_far, L)
    grid = _hex_to_tets(axes, pts, sizes)

    log("   Фоновая сетка поля размера: %d узлов (со сдвигом против "
        "совпадений), шаг у тела %.4f м, вдали %.4f м (переход ~%.1f м)"
        % (int(grid.n_points), h_near, h_far, L))
    return grid


def _jitter_bg_points(pts, h_near):
    """Случайный сдвиг узлов фоновой сетки, чтобы входные точки не лежали
    на её плоскостях/гранях (иначе TetGen падает в 'Interpolating mesh size').

    Регулярный фон (meshgrid) имеет узлы строго на осях, а входной PLC
    симметричной модели (короб, фюзеляж) содержит точки ровно на этих осях
    (напр. z=0). Точка на грани фонового тета даёт вырожденную барицентрику
    в интерполяции размера -> сегфолт. Сдвиг на долю локального шага уводит
    узлы фона с этих плоскостей; величина мала, поэтому теты не выворачиваются
    и поле остаётся гладким. RNG с фиксированным seed для воспроизводимости.
    """
    pts = np.asarray(pts, dtype=float).copy()
    rng = np.random.default_rng(BG_JITTER_SEED)
    amp = max(float(h_near), 1e-9) * BG_JITTER_FRAC
    pts += rng.uniform(-amp, amp, size=pts.shape)
    return pts


def _bg_axes(bounds, body_min, body_max, h_near, h_far, max_nodes=MAX_BG_NODES):
    """Оси фоновой сетки поля размера (чистая numpy, без pyvista).

    Возвращает ``(axes, h_near, bmin, bmax)``. Оси **строго шире** ``bounds``
    (запас ``0.03*extent``), чтобы узлы основного короба лежали строго внутри
    фоновой области — иначе TetGen падает в «Interpolating mesh size» на
    грани фонового тета (см. build_size_field_background). Если даже после
    огрубления шага узлов больше ``max_nodes`` — возвращает ``(None, ...)``.
    """
    x0, x1, y0, y1, z0, z1 = [float(b) for b in bounds]
    h_near = max(float(h_near), 1e-9)
    h_far = max(float(h_far), h_near)
    if h_far <= h_near * 1.05:
        h_far = h_near * 1.05

    extent = max(x1 - x0, y1 - y0, z1 - z0)
    pad = float(0.03 * extent)

    bmin = np.asarray(body_min, dtype=float) - 2.0 * h_near
    bmax = np.asarray(body_max, dtype=float) + 2.0 * h_near
    orders = [(x0 - pad, x1 + pad, bmin[0], bmax[0]),
              (y0 - pad, y1 + pad, bmin[1], bmax[1]),
              (z0 - pad, z1 + pad, bmin[2], bmax[2])]

    for _ in range(4):
        axes = [make_graded_axis(lo, hi, ilo, ihi, h_near, h_far)
                for (lo, hi, ilo, ihi) in orders]
        n_nodes = int(len(axes[0]) * len(axes[1]) * len(axes[2]))
        if n_nodes <= max_nodes:
            return axes, h_near, bmin, bmax
        h_near = h_near * 1.6
    return None, h_near, bmin, bmax


def _structured_grid_pts(axes):
    """Точки решётки (axes) в C-порядке meshgrid(..., indexing='ij')."""
    xg, yg, zg = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return np.column_stack([xg.ravel(), yg.ravel(), zg.ravel()])


def _structured_tet_cells(axes):
    """Тетраэдры структурированной решётки (axes) — по 6 на гекс.

    Чистая numpy-часть (без pyvista), чтобы её можно было проверить в тестах.
    Тот же шаблон, что у картезианского пути в gmsh_generator.
    """
    nx = max(1, len(axes[0]) - 1)
    ny = max(1, len(axes[1]) - 1)
    nz = max(1, len(axes[2]) - 1)

    def idx(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                             indexing="ij")
    corners = np.stack([
        idx(ii, jj, kk).ravel(),
        idx(ii + 1, jj, kk).ravel(),
        idx(ii + 1, jj + 1, kk).ravel(),
        idx(ii, jj + 1, kk).ravel(),
        idx(ii, jj, kk + 1).ravel(),
        idx(ii + 1, jj, kk + 1).ravel(),
        idx(ii + 1, jj + 1, kk + 1).ravel(),
        idx(ii, jj + 1, kk + 1).ravel(),
    ], axis=1)

    pattern = np.array([
        [0, 1, 2, 6], [0, 2, 3, 6], [0, 5, 1, 6],
        [0, 4, 5, 6], [0, 3, 7, 6], [0, 7, 4, 6],
    ], dtype=np.int64)
    return np.ascontiguousarray(corners[:, pattern].reshape(-1, 4),
                                dtype=np.int64)


def _hex_to_tets(axes, pts, sizes):
    """Тетраэдральная фоновая сетка из решётки (axes) с point_data['target_size'].

    Строим UnstructuredGrid ВРУЧНУЮ (6 тетов на каждый гекс), а не через
    ``StructuredGrid.triangulate()``: так порядок точек ровно тот, в котором
    посчитан ``sizes``, и поле не разъезжается. ``cell_connectivity`` у
    UnstructuredGrid — ровно 4 числа на тет, что и ожидает TetGen.
    """
    tets = _structured_tet_cells(axes)
    cells = np.hstack([np.full((len(tets), 1), 4, dtype=np.int64),
                       tets]).ravel()
    ctypes = np.full(len(tets), pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, ctypes, np.asarray(pts, dtype=float))
    grid.point_data["target_size"] = np.asarray(sizes, dtype=float)
    return grid


def _tetgen_supports_size_field():
    """True, если обёртка tetgen принимает ``bgmesh`` (поле размера, >= 0.8.0).

    Возвращает False при заведомо старой версии, True — если параметр есть,
    None — если сигнатуру не удалось прочитать (тогда решает try/except).
    """
    try:
        import inspect
        sig = inspect.signature(getattr(TetGen, "tetrahedralize"))
        return "bgmesh" in sig.parameters
    except Exception:
        return None


def _tetgen_tetrahedralize(tg, switches, bgmesh=None, log=print):
    """Вызвать TetGen, по возможности с полем размера (``bgmesh`` -> ``-m``).

    Если установленная обёртка ``tetgen`` не принимает ``bgmesh`` (или поле
    не применилось), падаем на обычный вызов — никакой регрессии.
    Возвращает True, если поле размера применилось, иначе False.
    """
    if bgmesh is not None:
        supported = _tetgen_supports_size_field()
        if supported is False:
            log("   Внимание: установленная версия tetgen не поддерживает "
                "поле размера (-m, понадобится tetgen >= 0.8.0) — строю без "
                "него; скачок размера ячеек у тела сохранится.")
        else:
            try:
                # metric=1 -> добавит '-m'; bgmesh -> подставит фоновую сетку.
                tg.tetrahedralize(order=1, verbose=0, switches=switches,
                                  bgmesh=bgmesh, metric=1)
                return True
            except TypeError:
                log("   Внимание: установленная версия tetgen не поддерживает "
                    "поле размера (-m, понадобится tetgen >= 0.8.0) — строю "
                    "без него; скачок размера ячеек у тела сохранится.")
            except Exception as e:                                # pragma: no cover
                log("   Внимание: поле размера не применилось (%s: %s) — "
                    "строю без него." % (type(e).__name__, e))
    tg.tetrahedralize(order=1, verbose=0, switches=switches)
    return False


def tetrahedralize_plc(body_pts, body_faces, bounds,
                       min_ratio=2.0, max_volume=None, h_box=None,
                       bgmesh=None, log=print):
    """Триангулировать PLC «тело + короб».

    Ключ Y (nobisect) обязателен — без него теряются грани тела.
    ``bgmesh`` — фоновая сетка поля размера (для ``-m``), см. выше.
    Возвращает (points, tets) или None.
    """
    if not HAS_TETGEN:
        return None, None
    extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2],
                 bounds[5] - bounds[4])
    if not h_box or h_box <= 0:
        h_box = extent / 12.0
    box_pts, box_faces = box_surface(bounds, h_box)
    if box_pts is None or len(box_faces) == 0:
        return None, None

    n_body = len(body_pts)
    pts = np.vstack([body_pts, box_pts])
    faces = np.vstack([body_faces, box_faces + n_body])

    switches = "pY" if not min_ratio else "pq%gY" % float(min_ratio)
    if max_volume and max_volume > 0:
        switches += "a%.6g" % float(max_volume)

    try:
        tg = TetGen(pts, faces)
        used_field = _tetgen_tetrahedralize(tg, switches, bgmesh, log)
    except Exception as e:
        log("   Внимание: TetGen не смог построить сетку (%s: %s)"
            % (type(e).__name__, e))
        return None, None

    tets = np.asarray(tg.grid.cells).reshape(-1, 5)[:, 1:]
    points = np.asarray(tg.grid.points, dtype=float)
    _info = " с полем размера (-m)" if used_field else ""
    log("   Готово: TetGen (%s%s): %d тетраэдров, %d узлов"
        % (switches, _info, len(tets), len(points)))
    return points, np.ascontiguousarray(tets, dtype=np.int64)


def tet_faces(tets):
    """Все 4 грани каждого тетраэдра, отсортированные по узлам.

    Возвращает (flat[K,3], owner[K]), где owner[k] — номер тетраэдра.
    Порядок в flat: тетраэдр 0 грани 0..3, тетраэдр 1 грани 0..3, ...
    """
    tets = np.asarray(tets, dtype=np.int64)
    faces = np.stack([tets[:, [0, 1, 2]],
                      tets[:, [1, 2, 3]],
                      tets[:, [0, 2, 3]],
                      tets[:, [0, 1, 3]]], axis=1)
    flat = np.sort(faces, axis=2).reshape(-1, 3)
    return flat, np.arange(len(flat)) // 4


def _tet_face_keys(tet):
    """Сортированные ключи 4 граней тетраэдра (для поиска по словарю)."""
    a, b, c, d = (int(x) for x in tet)
    return [
        tuple(sorted((a, b, c))),
        tuple(sorted((a, b, d))),
        tuple(sorted((a, c, d))),
        tuple(sorted((b, c, d))),
    ]


def classify_exterior_tets(points, tets, body_face_keys, body_bbox):
    """Топологическая заливка снаружи: какие тетраэдры ВНЕ тела.

    В отличие от геометрического ``select_enclosed_points`` (точка-в-объёме,
    который ошибается на стыках крыло/фюзеляж и оставляет дыры в стенке),
    здесь наружность определяется ТОПОЛОГИЧЕСКИ:

      1. Семена — тетраэдры, чей центроид лежит вне ограничивающего
         параллелепипеда тела (``body_bbox``). Они гарантированно снаружи:
         тело целиком внутри своего bbox, а тетраэдр не пересекает
         поверхность тела (TetGen с ключом Y не даёт crossing-тетраэдров).
      2. Заливка идёт от семян по граням тетраэдров, но НЕ пересекает грани
         тела (``body_face_keys``). Если поверхность тела замкнута, заливка
         останавливается на ней и внутренность не помечается.

    Возвращает булев массив ``exterior`` размера ``len(tets)``.

    Плюс способа: на замкнутой поверхности ~100% граней тела выходит на
    границу (маркер airfoil), чего геометрический метод не даёт. Если же
    поверхность тела дырявая, заливка просачивается внутрь — тогда почти
    все тетраэдры помечаются «снаружи», и вышестоящий код это отвергнет.
    """
    from collections import deque
    n = len(tets)

    # adjacency: ключ грани -> [номера тетраэдров]
    face_map = {}
    for ti in range(n):
        for f in _tet_face_keys(tets[ti]):
            face_map.setdefault(f, []).append(ti)

    # Семена: центроиды вне bbox тела.
    centroids = np.asarray(points, dtype=float)[np.asarray(tets)].mean(axis=1)
    bmin, bmax = body_bbox[0], body_bbox[1]
    outside_seed = (
        (centroids[:, 0] < bmin[0]) | (centroids[:, 0] > bmax[0]) |
        (centroids[:, 1] < bmin[1]) | (centroids[:, 1] > bmax[1]) |
        (centroids[:, 2] < bmin[2]) | (centroids[:, 2] > bmax[2])
    )
    seeds = np.where(outside_seed)[0]

    blocked = {tuple(int(x) for x in row) for row in body_face_keys}

    exterior = np.zeros(n, dtype=bool)
    dq = deque()
    for s in seeds:
        if not exterior[s]:
            exterior[s] = True
            dq.append(s)
    while dq:
        ti = dq.popleft()
        for f in _tet_face_keys(tets[ti]):
            if f in blocked:      # не пересекаем поверхность тела
                continue
            for nb in face_map.get(f, ()):
                if nb != ti and not exterior[nb]:
                    exterior[nb] = True
                    dq.append(nb)
    return exterior


def count_recovered(points, tets, body_pts, body_faces):
    """Сколько входных граней тела присутствует в объёмной сетке.

    TetGen переставляет нумерацию узлов, поэтому входные вершины
    сопоставляются выходным геометрически (cKDTree). Отклонение должно
    быть нулевым: входные точки constrained-триангуляции не двигаются.

    Возвращает (n_recovered, n_total, max_deviation, vertex_map).
    """
    flat, _ = tet_faces(tets)
    keys = {tuple(int(x) for x in row) for row in flat}
    dist, vmap = cKDTree(points).query(body_pts)
    n_rec = 0
    for f in body_faces:
        if tuple(sorted(int(vmap[x]) for x in f)) in keys:
            n_rec += 1
    return n_rec, int(len(body_faces)), float(dist.max()), vmap


def _geometric_interior(points, tets, body_pts, body_faces):
    """Прежний способ: точка-в-объёме через VTK select_enclosed_points."""
    flat = np.hstack([np.full((len(body_faces), 1), 3, dtype=np.int64),
                      body_faces]).ravel()
    body = pv.PolyData(body_pts, flat)

    cells = np.hstack([np.full((len(tets), 1), 4, dtype=np.int64),
                       tets]).ravel()
    ctypes = np.full(len(tets), pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, ctypes, points)
    centers = grid.cell_centers().points

    enclosed = pv.PolyData(centers).select_enclosed_points(
        body, tolerance=1e-5, check_surface=False)
    inside = np.asarray(enclosed["SelectedPoints"]).astype(bool)
    return inside


def remove_interior_tets(points, tets, body_pts, body_faces, log=print,
                         recovered_face_keys=None, body_bbox=None):
    """Убрать тетраэдры, попавшие внутрь тела.

    Основной способ — **топологическая заливка снаружи**
    (``classify_exterior_tets``): она не пересекает грани тела, поэтому на
    замкнутой поверхности ~100% граней выходит на границу (маркер airfoil),
    и стенка не дырявая. Строго геометрический ``select_enclosed_points``
    ошибается на стыках крыло/фюзеляж (19% граней «утонули» внутрь, SU2
    расходился). Если топология дала сбой (нет семян, почти всё снаружи —
    тело дырявое и заливка просочилась) — фолбэк на геометрию.

    ``recovered_face_keys`` — множество сортированных троек индексов узлов
    сетки, образующих поверхность тела (в координатах ``tets``).
    ``body_bbox`` — (min_xyz, max_xyz) габарит тела для выбора семян.

    Возвращает индексы наружных тетраэдров.
    """
    # --- топологическая заливка (основной путь) ------------------------
    if recovered_face_keys is not None and body_bbox is not None:
        try:
            exterior = classify_exterior_tets(
                points, tets, recovered_face_keys, body_bbox)
            inside = ~exterior
            n_in = int(inside.sum())
            n_out = int(exterior.sum())
            # Заливка «протекла» внутрь (тело дырявое) или нет семян —
            # почти всё помечено снаружи. Такой результат не даёт стенки,
            # уходим на геометрию (а она потом будет отвергнута по доле
            # граней на границе).
            if n_in > 0 and n_out > 0:
                log("   Готово (топологическая заливка): внутри тела %d, "
                    "снаружи %d тетраэдров" % (n_in, n_out))
                return np.where(exterior)[0]
        except Exception as e:                                   # pragma: no cover
            log("   Внимание: топологическая заливка не удалась (%s) — "
                "фолбэк на геометрию" % e)

    # --- геометрический фолбэк (прежний путь) -------------------------
    inside = _geometric_interior(points, tets, body_pts, body_faces)
    ext = np.where(~inside)[0]
    log("   Готово (геометрический, точка-в-объёме): удалено внутри тела "
        "%d, осталось %d" % (int(inside.sum()), len(ext)))
    return ext


def collect_airfoil_facets(points, tets, ext, vmap, body_faces):
    """Грани тела, ставшие границей наружной сетки (будущий маркер airfoil).

    points/tets — ПОЛНАЯ объёмная сетка (включая внутренние тетраэдры);
    ext — индексы наружных тетраэдров; vmap — соответствие «индекс вершины
    входной поверхности -> индекс узла объёмной сетки»; body_faces — грани
    входной поверхности в координатах ВХОДНЫХ вершин.

    ВАЖНО: грани тела переводятся в координаты объёмной сетки ЧЕРЕЗ vmap.
    Раньше индексы входной поверхности подставлялись как индексы узлов
    сетки напрямую (remap[x] для x в body_faces), а TetGen перенумеровывает
    узлы — входные точки не обязаны оставаться в начале массива. Из-за
    этого грани тела не совпадали с фактической границей наружной сетки,
    маркер airfoil терял до ~20% граней, и телооблекающая сетка ложно
    отвергалась как дырявая (замерено: 22894 восстановлено из 22898, а на
    границу «выходило» лишь 18493).

    Возвращает (marker, boundary_ratio): список граней-троек (в индексах
    наружной сетки) и их долю от всех входных граней тела.
    """
    from collections import defaultdict

    points = np.asarray(points, dtype=float)
    tets = np.asarray(tets, dtype=np.int64)
    ext = np.asarray(ext, dtype=np.int64)
    vmap = np.asarray(vmap, dtype=np.int64)

    ext_tets = tets[ext]
    used = np.unique(ext_tets.ravel())
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    tets_new = remap[ext_tets]

    # Границы наружной области — грани, встречающиеся ровно один раз.
    flat, _ = tet_faces(tets_new)
    uniq, cnt = np.unique(flat, axis=0, return_counts=True)
    boundary = {tuple(int(x) for x in row) for row in uniq[cnt == 1]}

    # Индекс входной вершины -> индекс узла наружной сетки: vmap (объём) затем
    # remap (наружная часть). Без vmap index входной поверхности терялся.
    body_facets = [tuple(sorted(int(remap[vmap[x]]) for x in f))
                   for f in body_faces]
    marker = [f for f in body_facets if f in boundary]
    boundary_ratio = len(marker) / max(len(body_faces), 1)
    return marker, boundary_ratio


def build_body_fitted_grid(body_meshes, body_min, body_max, margin,
                           min_ratio=2.0, max_volume=None,
                           min_recovery=DEFAULT_MIN_RECOVERY, log=print,
                           target_edge=None, max_surface_faces=400_000,
                           size_field=False):
    """Построить телообтекающую сетку.

    ``target_edge`` — целевой шаг у тела (приходит из пресета качества).
    ``size_field`` — изотропное поле размера через TetGen ``-m`` (фоновую
    сетку с ``target_size``). **По умолчанию выключено и игнорируется даже
    при явном ``True``**: tetgen 0.8.4 сегфолтит (exit 139) на
    плоскостном PLC расчётного короба в «Interpolating mesh size», а
    сегфолт из Python не перехватывается. Сетка поэтому всегда строится
    **без** ``-m`` — стабильно (облегание поверхности сохраняется, просто
    без плавного роста ячеек к полю).

    Разрешение у тела задаётся плотностью входной поверхности: если она
    грубее ``target_edge`` и ``surface_needs_refinement`` это подтверждает,
    поверхность уплотняется (`refine_to_edge_length`) — это безопасный
    путь к более мелким ячейкам у тела (TetGen ключ ``Y`` не дробит
    входные грани, поэтому размер у тела повторяет размер треугольника
    поверхности).

    Возвращает dict:
        grid         — pyvista.UnstructuredGrid (только наружные тетраэдры)
        body_facets  — грани тела как тройки индексов узлов этой сетки
        recovery     — доля входных граней тела, попавших в сетку
        bounds       — габариты расчётной области
    или None, если путь недоступен либо сетка не облегает поверхность.
    """
    if not tetgen_available():
        _miss = tetgen_missing()
        log("   Внимание: телооблекающая сетка недоступна — не хватает %s. "
            "Строится картезианская сетка фона."
            % (", ".join(_miss) if _miss else "зависимостей"))
        return None

    body_pts, body_faces = union_surfaces(body_meshes, log=log)
    if body_pts is None:
        return None
    log("   Готово: объединённая поверхность: %d граней" % len(body_faces))

    # Плотность поверхности задаёт размер ячеек у тела: TetGen сохраняет
    # входные грани как есть и не добавляет точки на крупные плоскости.
    # Ограничение на объём ячейки (-a) тут не годится — оно действует на
    # всю область, а дальнее поле в 3-4 габарита модели при шаге у тела
    # дало бы сотни миллионов тетраэдров. Поэтому уплотняется только
    # поверхность и только если она действительно грубее цели.
    if target_edge and surface_needs_refinement(body_pts, body_faces,
                                                target_edge):
        _sr = _load_surface_refine()
        if _sr is not None:
            try:
                v2, f2, info = _sr.refine_to_edge_length(
                    body_pts, body_faces, target_edge,
                    max_faces=max_surface_faces)
                log("   Поверхность уплотнена до шага %.4f м: %d -> %d граней"
                    % (target_edge, len(body_faces), len(f2)))
                if info.get("capped"):
                    log("   Внимание: достигнут предел %d граней, шаг %.4f м "
                        "не выдержан" % (max_surface_faces, target_edge))
                body_pts, body_faces = v2, f2
            except Exception as e:
                log("   Внимание: уплотнить поверхность не удалось (%s), "
                    "сетка строится по исходной триангуляции" % e)

    bounds = farfield_bounds(body_min, body_max, margin)

    # Поле размера (изотропное) через TetGen -m ОТКЛЮЧЕНО. Проверено на
    # tetgen 0.8.4: '-m' с фоновой сеткой на плоскостном PLC расчётного
    # короба даёт детерминированный SIGSEGV (exit 139) в 'Interpolating
    # mesh size', независимо от построения фона и от ключей. Сегфолт из
    # Python не перехватывается, поэтому даже при явном size_field=True мы
    # НЕ строим фоновую сетку и НЕ передаём её в TetGen — иначе собранное
    # приложение падало бы на каждом реальном расчёте. Сетка строится без
    # -m: облегание поверхности сохраняется, а переход «мелко у тела ->
    # крупно у короба» остаётся резким (цена, которую платим за стабильность).
    bgmesh = None
    if size_field and target_edge and target_edge > 0:
        log("   Внимание: поле размера (-m) отключено — tetgen 0.8.4 "
            "сегфолтит на плоскостном PLC короба в 'Interpolating mesh "
            "size'. Строится телооблекающая сетка без поля (облегание "
            "сохраняется, ячейки у тела определяет плотность поверхности).")
        bgmesh = None

    points, tets = tetrahedralize_plc(body_pts, body_faces, bounds,
                                      min_ratio=min_ratio,
                                      max_volume=max_volume,
                                      bgmesh=bgmesh, log=log)
    if points is None or len(tets) == 0:
        return None

    n_rec, n_tot, dev, vmap = count_recovered(points, tets, body_pts, body_faces)
    recovery = n_rec / max(n_tot, 1)
    log("   Граней тела сохранено в сетке: %d из %d (%.2f%%), "
        "отклонение узлов %.1e" % (n_rec, n_tot, 100.0 * recovery, dev))
    if recovery < min_recovery:
        log("   Внимание: сохранено менее %.0f%% граней тела, сетка не "
            "облегает поверхность — строится картезианская сетка фона"
            % (100.0 * min_recovery))
        return None

    # Ключи граней тела в координатах объёмной сетки (для топологической
    # заливки): сопоставляем входные вершины выходным через vmap из
    # count_recovered. Поверхность тела будем «не пересекать» при заливке.
    recovered_face_keys = None
    try:
        recovered_face_keys = {
            tuple(sorted(int(vmap[x]) for x in f)) for f in body_faces}
    except Exception:
        recovered_face_keys = None
    body_bbox = (np.asarray(body_pts, dtype=float).min(axis=0),
                 np.asarray(body_pts, dtype=float).max(axis=0))

    ext = remove_interior_tets(
        points, tets, body_pts, body_faces, log=log,
        recovered_face_keys=recovered_face_keys, body_bbox=body_bbox)
    if len(ext) < 100:
        log("   Внимание: после вырезания тела осталось %d тетраэдров"
            % len(ext))
        return None

    ext_tets = tets[ext]
    used = np.unique(ext_tets.ravel())
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    tets_new = remap[ext_tets]

    # Грани тела, ставшие границей наружной сетки: это и есть будущий
    # маркер airfoil. Собираются через vmap (индексы входной поверхности ->
    # узлы объёмной сетки) и remap (-> узлы наружной части). Раньше здесь
    # шло remap[x] для x в body_faces, но x — индекс входной поверхности, а
    # не узла сетки: при перенумерации узлов TetGen грани не совпадали с
    # границей, и маркер терял ~20% граней (ложный вердикт «дыры в стенке»).
    marker, boundary_ratio = collect_airfoil_facets(
        points, tets, ext, vmap, body_faces)
    log("   Готово: граней тела на границе сетки: %d из %d (%.2f%%)"
        % (len(marker), len(body_faces), 100.0 * boundary_ratio))

    # Дыры в стенке. ``recovery`` выше проверяет, что грани тела есть ГДЕ-ТО
    # в тетраэдральной сетке (в т.ч. внутри объёма), а SU2 требует, чтобы
    # маркер airfoil был собран именно с границы. Если на границу вышло
    # меньше ``min_recovery`` граней, поверхность тела дырявая: SU2
    # разойдётся (Residual > 10^20) независимо от пресета. Возвращаем None и
    # уходим на картезианский путь, а вызывающий код честно сообщит причину.
    if boundary_ratio < min_recovery:
        log("   Внимание: на границу сетки вышло %.1f%% граней тела — "
            "поверхность дырявая, SU2 разойдётся. Возврат на картезианскую "
            "сетку фона." % (100.0 * boundary_ratio))
        return None

    cells = np.hstack([np.full((len(tets_new), 1), 4, dtype=np.int64),
                       tets_new]).ravel()
    ctypes = np.full(len(tets_new), pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, ctypes, points[used])

    return {"grid": grid,
            "body_facets": np.asarray(marker, dtype=np.int64)
            if marker else np.zeros((0, 3), dtype=np.int64),
            "recovery": recovery,
            "bounds": bounds,
            "n_tets": int(len(tets_new))}
