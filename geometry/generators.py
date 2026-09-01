from __future__ import annotations

import math
import os

import numpy as np
import pyvista as pv

from physics.airfoils import generate_naca4_section


# ---------------------------------------------------------------------------
# Параметры крыла (значения по умолчанию как в UI)
# ---------------------------------------------------------------------------

class WingParameters:
    def __init__(self):
        self.airfoil_root = "NACA2412"
        self.airfoil_tip = "NACA0012"
        self.span = 10.0
        self.chord_root = 1.8
        self.chord_tip = 0.9
        self.sweep = 12.0
        self.dihedral = 0.0
        self.twist = 2.0
        self.taper = None
        self.x_ref = 3.0
        self.flap_kink = False
        self.kink_pos = 0.4
        self.n_sections = 15
        self.n_chord = 40

    @property
    def semi_span(self):
        return 0.5 * self.span

    @property
    def taper_ratio(self):
        if self.taper is not None:
            return float(self.taper)
        return self.chord_tip / max(self.chord_root, 1e-9)


# ---------------------------------------------------------------------------
# Вспомогательные утилиты лофтинга
# ---------------------------------------------------------------------------

def _centroid(points, start, count):
    pts = points[start:start + count]
    cx = sum(p[0] for p in pts) / count
    cy = sum(p[1] for p in pts) / count
    cz = sum(p[2] for p in pts) / count
    return [cx, cy, cz]


def _cap_faces(start_idx, n_points, center_idx):
    """Веерная триангуляция крышки вокруг центроида."""
    faces = []
    for i in range(n_points):
        j = (i + 1) % n_points
        faces.append([3, center_idx, start_idx + i, start_idx + j])
    return faces


def _loft_sections(n, section_count, with_caps=True):
    """Каркас faces для лофта одинаковых секций по n точек."""
    faces = []
    for k in range(section_count - 1):
        for i in range(n - 1):
            a = k * n + i
            faces.append([4, a, a + 1, a + 1 + n, a + n])
        # стык контура (задняя кромка)
        faces.append([4, k * n + n - 1, k * n,
                      (k + 1) * n, (k + 1) * n + n - 1])
    return faces


def horizontal_area(mesh) -> float:
    """Площадь проекции тела на горизонтальную плоскость (OXY)."""
    if mesh is None or mesh.n_points == 0:
        return 0.0
    try:
        surf = mesh.extract_surface().triangulate()
        faces = surf.faces.reshape(-1, 4)[:, 1:]
        p = surf.points
        a = p[faces[:, 1]] - p[faces[:, 0]]
        b = p[faces[:, 2]] - p[faces[:, 0]]
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        return 0.5 * float(np.abs(cross).sum())
    except Exception:
        return 0.0


def _project_outline_to_plane(mesh, normal):
    normal = np.asarray(normal, dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(normal, u)
    coords = np.column_stack([mesh.points @ u, mesh.points @ v])
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(coords)
        outline = coords[hull.vertices]
    except Exception:
        outline = coords
    return {"points": outline, "origin": coords.mean(axis=0), "u": u, "v": v}


def _compute_closed_area_centroid(pts):
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return 0.0, (0.0, 0.0)
    x, y = pts[:, 0], pts[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    A = 0.5 * np.sum(cross)
    if abs(A) < 1e-12:
        return 0.0, (float(x.mean()), float(y.mean()))
    cx = np.sum((x + np.roll(x, -1)) * cross) / (6 * A)
    cy = np.sum((y + np.roll(y, -1)) * cross) / (6 * A)
    return abs(float(A)), (float(cx), float(cy))


def _signed_polygon_area(pts):
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# ---------------------------------------------------------------------------
# Примитивы
# ---------------------------------------------------------------------------

def create_primitive(shape_type: str, params: dict, pos):
    """Куб / Цилиндр / Сфера с параметрами и позицией."""
    x, y, z = pos
    if shape_type == "Куб":
        lx = params.get("lx", 2.0)
        ly = params.get("ly", 1.0)
        lz = params.get("lz", 0.5)
        mesh = pv.Cube(x_length=lx, y_length=ly, z_length=lz,
                       center=(x, y, z + lz / 2)).triangulate()
    elif shape_type == "Цилиндр":
        r = params.get("radius", 0.5)
        h = params.get("height", 3.0)
        mesh = pv.Cylinder(center=(x, y, z + h / 2), direction=(0, 0, 1),
                           radius=r, height=h).triangulate()
    else:  # Сфера
        r = params.get("radius", 1.0)
        mesh = pv.Sphere(radius=r, center=(x, y, z + r))
    return mesh.extract_surface().triangulate().clean()


def create_vtail_support(airfoil_manager, length=1.0, chord=0.5):
    """Простая опора/пилон (цилиндр)."""
    return pv.Cylinder(center=(0, 0, length / 2), direction=(0, 0, 1),
                       radius=0.08 * chord, height=length).triangulate()


generate_vtail_support = create_vtail_support


# ---------------------------------------------------------------------------
# Крыло (с изломом), механизация
# ---------------------------------------------------------------------------

def generate_wing_mesh(span, chord_root, chord_tip, sweep_deg, twist_deg,
                       naca_code, pos_x, pos_y, pos_z,
                       kink_pos_ratio=None, chord_kink=None, sweep_outer_deg=None):
    """Параметрическая поверхность крыла (две консоли) с опциональным изломом.
    Возвращает (mesh, sweep_offset_total[x-сдвиг ПК концевого сечения])."""
    half_span = span / 2.0
    sweep_in = math.radians(sweep_deg)

    stations = []  # (eta, chord, x_le, twist_local)
    stations.append((0.0, chord_root, 0.0, 0.0))
    if kink_pos_ratio is not None:
        eta_k = float(np.clip(kink_pos_ratio, 0.05, 0.95))
        x_kink = eta_k * half_span * math.tan(sweep_in)
        stations.append((eta_k, chord_kink or chord_root, x_kink,
                         twist_deg * 0.5))
        sweep_out = math.radians(sweep_outer_deg if sweep_outer_deg is not None
                                 else sweep_deg)
        x_tip = x_kink + (1.0 - eta_k) * half_span * math.tan(sweep_out)
    else:
        x_tip = half_span * math.tan(sweep_in)
    stations.append((1.0, chord_tip, x_tip, twist_deg))

    loops = []
    for eta, chord, x_le, tw in stations:
        rx, rz = generate_naca4_section(chord, naca_code, twist=tw)
        n = len(rx)
        # две консоли: влево и вправо от оси симметрии
        y_left = -eta * half_span
        y_right = +eta * half_span
        loops.append((n, x_le, y_left, y_right))

    points = []
    all_faces = []
    # строим лофт по секциям: каждая секция — левая и правая точки
    # структура: [sec0_L, sec0_R, sec1_L, sec1_R, ...]
    sec_meshes_pts = []
    for n, x_le, yL, yR in loops:
        chord = loops[len(sec_meshes_pts)][3] if False else None
    # проще: пересобираем заново с известными хордами
    sec_pts = []
    for (eta, chord, x_le, tw), (n, _xl, yL, yR) in zip(stations, loops):
        rx, rz = generate_naca4_section(chord, naca_code, twist=tw)
        left = [[x_le + rx[i], yL + pos_y * 0, pos_z + rz[i]] for i in range(n)]
        right = [[x_le + rx[i], yR, pos_z + rz[i]] for i in range(n)]
        for p in left:
            p[0] += pos_x
            p[1] += 0.0
        for p in right:
            p[0] += pos_x
        sec_pts.append((left, right, n))

    points = []
    for left, right, n in sec_pts:
        points.extend(left)
        points.extend(right)

    nsec = len(sec_pts)
    n = sec_pts[0][2]
    # локальные индексы: секция k -> начало 2*k*n
    def LI(k, side):  # side 0=left,1=right
        return 2 * k * n + side * n

    for k in range(nsec - 1):
        # левая консоль: от секции k к k+1 (движение к корню, т.е. отрицательному y)
        for i in range(n - 1):
            all_faces.append([4, LI(k + 1, 0) + i, LI(k + 1, 0) + i + 1,
                              LI(k, 0) + i + 1, LI(k, 0) + i])
        all_faces.append([4, LI(k + 1, 0) + n - 1, LI(k + 1, 0),
                          LI(k, 0), LI(k, 0) + n - 1])
        # правая консоль
        for i in range(n - 1):
            all_faces.append([4, LI(k, 1) + i, LI(k, 1) + i + 1,
                              LI(k + 1, 1) + i + 1, LI(k + 1, 1) + i])
        all_faces.append([4, LI(k, 1) + n - 1, LI(k, 1),
                          LI(k + 1, 1), LI(k + 1, 1) + n - 1])

    # крышки на концах (левый конец первой секции, правый конец последней)
    def LI(k, side):
        return 2 * k * n + side * n

    c1 = len(points)
    points.append(_centroid(points, LI(0, 0), n))
    all_faces.extend(_cap_faces(LI(0, 0), n, c1))
    c2 = len(points)
    points.append(_centroid(points, LI(nsec - 1, 1), n))
    all_faces.extend(_cap_faces(LI(nsec - 1, 1), n, c2))

    # стык секций в центре (корневая нервюра): крышка между секцией 0 L и R
    # нужна, только если крыло разомкнуто в центре — у нас две консоли, закрываем
    c3 = len(points)
    points.append(_centroid(points, LI(0, 1), n))
    all_faces.extend(_cap_faces(LI(0, 1), n, c3))
    c4 = len(points)
    points.append(_centroid(points, LI(nsec - 1, 0), n))
    all_faces.extend(_cap_faces(LI(nsec - 1, 0), n, c4))

    flat = [v for f in all_faces for v in f]
    mesh = pv.PolyData(np.array(points), np.array(flat)).triangulate().clean(tolerance=1e-6)
    mesh.compute_normals(auto_orient_normals=True, inplace=True)
    return mesh, x_tip


def _mech_panel_mesh(span_part, chord_ref, sweep_offset_end, naca_code,
                     deflection_deg, pos_x, pos_y, pos_z, y_start_ratio,
                     y_end_ratio, half_span, hinge_ratio, slide_ratio,
                     is_slat=False):
    """Панель механизации (закрылок/предкрылок) на части размаха.
    Простое приближение: отдельная поверхность с сдвигом/поворотом относительно шарнира."""
    chord = chord_ref * (0.30 if not is_slat else 0.18)

    def make_half(sign):
        rx, rz = generate_naca4_section(chord, naca_code, twist=0.0)
        n = len(rx)
        pts_in, pts_out = [], []
        for x_le_local, y_abs in ((y_start_ratio, y_start_ratio), (y_end_ratio, y_end_ratio)):
            y = sign * y_abs * half_span
            if not is_slat:
                # закрылок: шарнир на hinge_ratio хорды, отклонение вниз + слайд назад
                hinge_x = pos_x + hinge_ratio * chord_ref + slide_ratio * chord_ref
                base_z = pos_z
                th = math.radians(deflection_deg)
                for i in range(n):
                    dx = rx[i]
                    dz = rz[i]
                    xr = dx * math.cos(th) - dz * math.sin(th)
                    zr = -dx * math.sin(th) + dz * math.cos(th)
                    pts_row = [hinge_x + xr, y, base_z + zr]
                    (pts_in if y_abs == y_start_ratio else pts_out).append(pts_row)
            else:
                # предкрылок: выдвижение вперёд/вниз
                th = math.radians(deflection_deg)
                for i in range(n):
                    dx = rx[i]
                    dz = rz[i]
                    xr = dx * math.cos(th) + dz * math.sin(th)
                    zr = -dx * math.sin(th) + dz * math.cos(th)
                    pts_row = [pos_x - slide_ratio * chord_ref + xr, y,
                               pos_z - 0.02 * chord_ref + zr]
                    (pts_in if y_abs == y_start_ratio else pts_out).append(pts_row)
        points = pts_in + pts_out
        faces = []
        for i in range(n - 1):
            faces.append([4, i, i + 1, n + i + 1, n + i])
        faces.append([4, n - 1, 0, n, 2 * n - 1])
        c1 = len(points)
        points.append(_centroid(points, 0, n))
        faces.extend(_cap_faces(0, n, c1))
        c2 = len(points)
        points.append(_centroid(points, n, n))
        faces.extend(_cap_faces(n, n, c2))
        return points, faces

    p1, f1 = make_half(+1)
    p2, f2 = make_half(-1)
    off = len(p1)
    points = p1 + p2
    faces = f1 + [[f[0]] + [idx + off for idx in f[1:]] for f in f2]
    flat = [v for f in faces for v in f]
    mesh = pv.PolyData(np.array(points), np.array(flat)).triangulate().clean(tolerance=1e-6)
    mesh.compute_normals(auto_orient_normals=True, inplace=True)
    return mesh


def generate_flaps_mesh(span, chord_root, chord_tip, flap_deflection,
                        flap_span_ratio, flap_chord_ratio, pos_x, pos_y, pos_z,
                        sweep_offset, hinge_depth_ratio=0.12, slide_ratio=0.06):
    """Закрылки: панель от оси симметрии до flap_span_ratio полуразмаха."""
    half_span = span / 2.0
    y_end = min(max(flap_span_ratio, 0.1), 0.9)
    return _mech_panel_mesh(span_part=flap_span_ratio, chord_ref=chord_root,
                            sweep_offset_end=sweep_offset, naca_code="0012",
                            deflection_deg=flap_deflection, pos_x=pos_x,
                            pos_y=pos_y, pos_z=pos_z, y_start_ratio=0.05,
                            y_end_ratio=y_end, half_span=half_span,
                            hinge_ratio=1.0 - flap_chord_ratio,
                            slide_ratio=slide_ratio, is_slat=False)


def generate_slats_mesh(span, chord_root, chord_tip, slat_deflection,
                        slat_span_ratio, slat_chord_ratio, pos_x, pos_y, pos_z,
                        sweep_offset, slide_ratio=0.04):
    """Предкрылки: панель от 10% до slat_span_ratio полуразмаха."""
    half_span = span / 2.0
    y_end = min(max(slat_span_ratio, 0.2), 0.95)
    return _mech_panel_mesh(span_part=slat_span_ratio, chord_ref=chord_root,
                            sweep_offset_end=sweep_offset, naca_code="0012",
                            deflection_deg=slat_deflection, pos_x=pos_x,
                            pos_y=pos_y, pos_z=pos_z, y_start_ratio=0.10,
                            y_end_ratio=y_end, half_span=half_span,
                            hinge_ratio=0.0, slide_ratio=slide_ratio, is_slat=True)


def generate_fuselage_mesh(param_group: dict, n_len=80, n_circ=48) -> pv.PolyData:
    """Запасной (быстрый) фюзеляж по n1/n2/n3, если UI не переопределяет."""
    L = float(param_group.get("n1", 5.0))
    R = float(param_group.get("n2", 0.6))
    nose_frac = float(param_group.get("n3", 0.35))
    xs = np.linspace(0.0, 1.0, n_len)
    r = np.ones_like(xs)
    nose = xs < nose_frac
    r[nose] = np.sqrt(np.clip(1.0 - ((nose_frac - xs[nose]) / nose_frac) ** 2, 0.0, 1.0))
    tail = xs > 0.75
    de = (xs[tail] - 0.75) / 0.25
    r[tail] = (1.0 - (de ** 0.8) / (de ** 0.8 + (1 - de) ** 2))
    theta = np.linspace(0.0, 2 * np.pi, n_circ)
    X = np.repeat(xs * L, n_circ)
    T = np.tile(theta, n_len)
    Y = np.repeat(r * R, n_circ) * np.cos(T)
    Z = np.repeat(r * R, n_circ) * np.sin(T)
    pts = np.column_stack([X, Y, Z])
    faces = []
    for i in range(n_len - 1):
        for j in range(n_circ):
            a = i * n_circ + j
            b = i * n_circ + (j + 1) % n_circ
            c = (i + 1) * n_circ + (j + 1) % n_circ
            d = (i + 1) * n_circ + j
            faces += [4, a, b, c, d]
    return pv.PolyData(pts, np.array(faces)).clean()


def generate_tail_surface(airfoil_manager, airfoil_name, span, chord_root,
                          chord_tip, sweep_deg, dihedral_deg, x_offset=0.0,
                          z_offset=0.0, n_chord=40) -> pv.PolyData:
    """ГО: лофт двух консолей (без руля — для совместимости со старым API)."""
    half_span = span / 2.0
    sweep = math.radians(sweep_deg)
    sweep_offset = half_span * math.tan(sweep)
    rx, rz = generate_naca4_section(chord_root, airfoil_name.replace("NACA", ""), 0.0)
    tx, tz = generate_naca4_section(chord_tip, airfoil_name.replace("NACA", ""), 0.0)
    n = len(rx)
    points = []
    for i in range(n):
        points.append([tx[i] + sweep_offset + x_offset, -half_span, tz[i] + z_offset])
    for i in range(n):
        points.append([rx[i] + x_offset, 0.0, rz[i] + z_offset])
    for i in range(n):
        points.append([tx[i] + sweep_offset + x_offset, +half_span, tz[i] + z_offset])
    faces = _loft_sections(n, 3)
    left_c = len(points)
    points.append(_centroid(points, 0, n))
    faces.extend(_cap_faces(0, n, left_c))
    right_c = len(points)
    points.append(_centroid(points, 2 * n, n))
    faces.extend(_cap_faces(2 * n, n, right_c))
    flat = [v for f in faces for v in f]
    mesh = pv.PolyData(np.array(points), np.array(flat)).triangulate().clean(tolerance=1e-6)
    mesh.compute_normals(auto_orient_normals=True, inplace=True)
    return mesh


def generate_vertical_stabilizer_geometry(airfoil_manager, airfoil_name, height,
                                          chord_root, chord_tip, sweep_deg,
                                          z_offset=0.0, n_chord=40) -> pv.PolyData:
    """ВО: лофт корень→конец по высоте."""
    sweep = math.radians(sweep_deg)
    sweep_offset = height * math.tan(sweep)
    # ry/ty — координата толщины профиля. Для киля она откладывается
    # по Y, а лофт идёт по Z (высоте).
    #
    # Раньше координата толщины уходила в глобальный Z, а Y у обоих
    # сечений был жёстко нулём:
    #     points.append([rx[i], 0.0, rz[i] + z_offset])
    #     points.append([tx[i] + sweep_offset, 0.0, tz[i] + z_offset + height])
    # Киль получался плоским листом нулевой толщины. Последствия:
    #   - размах по Y ровно 0, поэтому при выборе шага уплотнения такое
    #     тело давало min_dim=0 и в refine_within_budget забирало весь
    #     бюджет граней (501932 из 600000), оставляя фюзеляж без
    #     уплотнения;
    #   - лист не является объёмом, поэтому trimesh.boolean.union падает
    #     с "Not all meshes are volumes!" — объединить компоненты в одну
    #     замкнутую поверхность для сеточника не удаётся;
    #   - у киля нет объёма и, значит, никаких структурных характеристик.
    rx, ry = generate_naca4_section(chord_root, airfoil_name.replace("NACA", ""), 0.0)
    tx, ty = generate_naca4_section(chord_tip, airfoil_name.replace("NACA", ""), 0.0)
    n = len(rx)
    points = []
    for i in range(n):
        points.append([rx[i], ry[i], z_offset])
    for i in range(n):
        points.append([tx[i] + sweep_offset, ty[i], z_offset + height])
    faces = _loft_sections(n, 2)
    bottom_c = len(points)
    points.append(_centroid(points, 0, n))
    faces.extend(_cap_faces(0, n, bottom_c))
    top_c = len(points)
    points.append(_centroid(points, n, n))
    faces.extend(_cap_faces(n, n, top_c))
    flat = [v for f in faces for v in f]
    mesh = pv.PolyData(np.array(points), np.array(flat)).triangulate().clean(tolerance=1e-6)
    mesh.compute_normals(auto_orient_normals=True, inplace=True)
    return mesh


def generate_wing(params: WingParameters, airfoil_manager=None) -> pv.PolyData:
    """Совместимый вызов нового генератора через WingParameters."""
    mesh, _ = generate_wing_mesh(
        span=params.span, chord_root=params.chord_root, chord_tip=params.chord_tip,
        sweep_deg=params.sweep, twist_deg=params.twist,
        naca_code=params.airfoil_root.replace("NACA", ""),
        pos_x=params.x_ref, pos_y=0.0, pos_z=0.0,
        kink_pos_ratio=params.kink_pos if params.flap_kink else None,
    )
    return mesh


# ---------------------------------------------------------------------------
# Direct CAD Import: STEP/IGES/BREP/… → STL через gmsh (OCC-ядро)
# ---------------------------------------------------------------------------
# gmsh уже есть в зависимостях приложения (используется для объёмной сетки),
# поэтому новых тяжёлых библиотек не добавляется. Конвертация идёт через
# OpenCascade: модель открывается, поверхности триангулируются и пишутся в STL.

CAD_EXTENSIONS = (".step", ".stp", ".iges", ".igs", ".x_t", ".x_b", ".sat",
                  ".brep", ".bdf", ".nas", ".ply", ".obj", ".off")

def cad_to_stl(src_path: str, out_path: str, log=None) -> str:
    """Конвертирует CAD-модель в STL триангуляцией поверхностей через gmsh.

    Параметры:
        src_path — исходный CAD-файл (STEP/IGES/BREP/…)
        out_path — куда писать STL (расширение .stl)
        log      — опциональный callable(msg) для лога

    Возвращает out_path. При ошибке поднимает RuntimeError.
    """
    if log:
        log(f"  Конвертация CAD → STL: {os.path.basename(src_path)}")
    try:
        import gmsh
    except Exception as e:
        raise RuntimeError(
            "Модуль gmsh недоступен — Direct CAD Import требует gmsh. "
            f"({e})") from e

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.open(src_path)            # OCC читает STEP/IGES/BREP/…
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)    # триангуляция поверхностей
        gmsh.write(out_path)           # STL
    except Exception as e:
        raise RuntimeError(f"Не удалось импортировать CAD-модель: {e}") from e
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass

    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError("gmsh не создал STL-файл (пустой результат).")
    return out_path


# ---------------------------------------------------------------------------
# Direct CAD Import: многодетальные сборки (ТЗ, п. 4)
# ---------------------------------------------------------------------------
# STEP/IGES-сборка обычно содержит несколько тел (solid). Если триангулировать
# файл целиком, все детали попадают в один STL и теряют индивидуальные имена —
# их нельзя ни скрыть, ни назначить им разные граничные условия. Функции ниже
# раскладывают сборку на отдельные STL: по одному на тело.

def count_stl_triangles(path: str) -> int:
    """Число треугольников в STL (бинарном или текстовом).

    Чистый Python, без внешних зависимостей — используется и для контроля
    результата конвертации CAD, и в тестах.
    """
    if not os.path.isfile(path):
        return 0
    size = os.path.getsize(path)
    if size < 84:
        # точно не бинарный STL — считаем как текстовый
        n = 0
        with open(path, "r", encoding="ascii", errors="ignore") as f:
            for ln in f:
                if ln.strip().lower().startswith("facet"):
                    n += 1
        return n
    with open(path, "rb") as f:
        head = f.read(84)
        n_bin = int.from_bytes(head[80:84], "little")
        # бинарный STL: 84 байта заголовка + 50 байт на треугольник
        if size == 84 + 50 * n_bin:
            return n_bin
    n = 0
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        for ln in f:
            if ln.strip().lower().startswith("facet"):
                n += 1
    return n


def _stl_name_for_solid(base: str, index: int, name: str, tag) -> str:
    """Имя файла STL для отдельного тела сборки (без gmsh — тестируемо)."""
    clean = "".join(ch if (ch.isalnum() or ch in "-_") else "_"
                    for ch in str(name or "")).strip("_")
    clean = clean[:48]
    if not clean:
        clean = f"solid_{tag}"
    return f"{base}_{index:02d}_{clean}.stl"


def cad_inspect(src_path: str, log=None) -> list:
    """Состав CAD-сборки: список тел (твёрдых тел) с объёмами и габаритами.

    Возвращает список словарей ``{"tag", "name", "volume", "bbox",
    "n_surfaces"}``. Не требует триангуляции — только чтение геометрии,
    поэтому работает быстро даже на больших сборках.
    """
    try:
        import gmsh
    except Exception as e:
        raise RuntimeError(
            "Модуль gmsh недоступен — разбор CAD-сборки требует gmsh. "
            f"({e})") from e
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.open(src_path)
        gmsh.model.occ.synchronize()
        out = []
        for dim, tag in gmsh.model.getEntities(3):
            try:
                volume = float(gmsh.model.occ.getMass(3, tag))
            except Exception:
                volume = 0.0
            try:
                bbox = [float(x) for x in gmsh.model.getBoundingBox(3, tag)]
            except Exception:
                bbox = [0.0] * 6
            try:
                n_surf = len(gmsh.model.getBoundary([(3, tag)], oriented=False))
            except Exception:
                n_surf = 0
            out.append({"tag": int(tag),
                        "name": gmsh.model.getEntityName(3, tag) or "",
                        "volume": volume, "bbox": bbox,
                        "n_surfaces": int(n_surf)})
        if log:
            log(f"  В сборке тел: {len(out)}")
        return out
    except Exception as e:
        raise RuntimeError(f"Не удалось разобрать CAD-сборку: {e}") from e
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass


def cad_split_to_stl(src_path: str, out_dir: str, log=None,
                     min_volume: float = 1e-9,
                     lin_size: float = 0.0) -> list:
    """Раскладывает многодетальную CAD-сборку на отдельные STL.

    Для каждого твёрдого тела сборки пишется свой файл
    ``<имя>_NN_<имя_тела>.stl``; тела объёмом меньше ``min_volume``
    (крепёж, точки, мусор) пропускаются.

    Возвращает список ``{"tag", "name", "stl", "triangles", "volume"}``.
    Если тело в сборке одно, результат эквивалентен :func:`cad_to_stl`.
    """
    try:
        import gmsh
    except Exception as e:
        raise RuntimeError(
            "Модуль gmsh недоступен — Direct CAD Import требует gmsh. "
            f"({e})") from e

    solids = cad_inspect(src_path, log=log)
    keep = [s for s in solids if s.get("volume", 0.0) > min_volume]
    if not keep:
        keep = list(solids)
    if not keep:
        raise RuntimeError("В CAD-файле не найдено ни одного твёрдого тела.")

    os.makedirs(out_dir or ".", exist_ok=True)
    base = os.path.splitext(os.path.basename(src_path))[0]
    results = []
    for i, info in enumerate(keep, start=1):
        tag = info["tag"]
        out_path = os.path.join(
            out_dir, _stl_name_for_solid(base, i, info.get("name", ""), tag))
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Verbosity", 1)
            gmsh.open(src_path)
            gmsh.model.occ.synchronize()
            # удаляем все тела, кроме нужного
            others = [(3, t) for _d, t in gmsh.model.getEntities(3) if t != tag]
            if others:
                gmsh.model.removeEntities(others, deleteMesh=False)
            gmsh.model.occ.synchronize()
            if lin_size and lin_size > 0:
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax",
                                      float(lin_size))
            gmsh.model.mesh.generate(2)
            gmsh.write(out_path)
        except Exception as e:
            try:
                gmsh.finalize()
            except Exception:
                pass
            if log:
                log(f"  Внимание: Тело #{tag} ({info.get('name') or 'без имени'}) "
                    f"не конвертировано: {e}")
            continue
        finally:
            try:
                gmsh.finalize()
            except Exception:
                pass
        n_tri = count_stl_triangles(out_path)
        if n_tri == 0:
            if log:
                log(f"  Внимание: Тело #{tag}: пустая триангуляция, пропущено")
            continue
        results.append({"tag": tag, "name": info.get("name", ""),
                        "stl": out_path, "triangles": n_tri,
                        "volume": info.get("volume", 0.0)})
        if log:
            log(f"  Готово: {os.path.basename(out_path)}: {n_tri} треугольников, "
                f"V={info.get('volume', 0.0):.4g} м³")

    if not results:
        raise RuntimeError("Не удалось триангулировать ни одно тело сборки.")
    return results
