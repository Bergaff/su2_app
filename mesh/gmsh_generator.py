"""
mesh/gmsh_generator.py — ИСПРАВЛЕННАЯ КОПИЯ.

Изменения относительно исходного файла (без потери функционала):

1. write_su2(): жёсткая защита — если маркер 'airfoil' пустой, бросаем
   RuntimeError с понятным сообщением. Иначе SU2 падает тихо и history.csv
   пишется без surface_flow.vtu.

2. write_su2(): запись NPOIN + точек + маркеров обёрнута в try/except —
   если файл обрезается посередине, мы увидим ошибку в логе, а не
   «успешный» SU2 на обрезанной сетке.

3. write_su2(): явный сброс буфера f.flush() + f.fsync() перед закрытием —
   на Windows иногда файл обрывается на 64 КБ без fsync.

4. Добавлена функция quick_mesh_check(path) — можно дёрнуть из main_window
   после генерации, чтобы удостовериться, что mesh.su2 содержит и
   NPOIN, и оба маркера.
"""
from __future__ import annotations

import os
import sys
import json
import traceback
import numpy as np
import pyvista as pv

# Настройки проекта
try:
    from config.settings import (WORK_DIR_BASE, PREVIEW_MESH, MESH_FILE,
                                 MESH_QUALITY)
except ImportError:
    WORK_DIR_BASE = os.path.join(os.getcwd(), "cases")
    PREVIEW_MESH = os.path.join(WORK_DIR_BASE, "preview.vtk")
    MESH_FILE = os.path.join(WORK_DIR_BASE, "mesh.su2")
    MESH_QUALITY = ["Грубая (быстро)", "Средняя", "Точная (медленно)"]

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class MeshCancelled(Exception):
    """Бросается, когда пользователь отменил генерацию сетки."""
    pass


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------
def extract_cells(mesh, cell_type):
    """Надёжный экстрактор ячеек заданного VTK-типа."""
    try:
        cd = mesh.cells_dict
        if cell_type in cd:
            return cd[cell_type].tolist()
    except Exception:
        pass
    try:
        if hasattr(mesh, 'cell_connectivity'):
            connect = np.asarray(mesh.cell_connectivity)
            types = np.asarray(mesh.celltypes)
            offs = np.asarray(mesh.offset)
            mask = types == cell_type
            indices = np.where(mask)[0]
            if len(indices) > 0:
                result = []
                for i in indices:
                    start = int(offs[i])
                    end = int(offs[i + 1]) if i + 1 < len(offs) else len(connect)
                    if start <= end <= len(connect):
                        result.append(connect[start:end].tolist())
                return result
    except Exception:
        pass
    try:
        faces_raw = np.asarray(mesh.faces)
        n_cells = mesh.n_cells
        if n_cells > 0 and len(faces_raw) > 0:
            first_n = int(faces_raw[0])
            if first_n > 0 and len(faces_raw) == n_cells * (first_n + 1):
                reshaped = faces_raw.reshape(n_cells, first_n + 1)
                expected_n = {10: 4, 5: 3, 9: 3, 7: 3}
                if first_n in expected_n.values():
                    return reshaped[:, 1:].tolist()
    except Exception:
        pass
    return []


def _extract_boundary_faces(tets):
    """Грани тетра-сетки, встречающиеся ровно один раз (истинная граница
    объёма). Утилита также используется в тестах."""
    t = np.asarray(tets, dtype=np.int64)
    faces = np.vstack([t[:, [0, 1, 2]], t[:, [0, 1, 3]],
                       t[:, [0, 2, 3]], t[:, [1, 2, 3]]])
    keys = np.sort(faces, axis=1)
    uniq, inv, counts = np.unique(keys, axis=0,
                                  return_inverse=True, return_counts=True)
    return faces[counts[inv] == 1]


# Алиас для обратной совместимости
_extract_boundary_faces = _extract_boundary_faces  # чтобы import не ругался
_extract_boundary_faces = _extract_boundary_faces  # на всякий случай


# ---------------------------------------------------------------------------
# Запись файла формата SU2 (ИСПРАВЛЕННАЯ)
# ---------------------------------------------------------------------------
def write_su2(grid, surface, filename, markers_info=None, **kwargs):
    """Запись mesh.su2 с гарантированной совместимостью.

    ИСПРАВЛЕНО (без потери функционала):
      - явная проверка, что маркер 'airfoil' не пустой (иначе SU2 падает
        тихо и surface_flow.vtu не пишется);
      - запись NPOIN/точек/маркеров под try/except с понятным сообщением
        в логе, если файловая система обрезает файл;
      - явный flush+fsync на Windows перед закрытием.

    T1: kwargs['use_symmetry'] — добавить маркер symmetry_plane на Y=0.
    """
    vol_pts = np.asarray(grid.points)
    n_points = len(vol_pts)
    tetras = extract_cells(grid, cell_type=10)
    if not tetras:
        raise RuntimeError("Не найдены тетраэдры в сетке")

    surf_pts = np.asarray(surface.points)
    if markers_info is None:
        markers_info = [("airfoil", 0), ("farfield", 0)]

    # === T1: список плоскостей симметрии (XY, XZ, YZ) =================
    # Поддержка нескольких плоскостей одновременно. Каждая плоскость
    # даёт отдельный маркер в mesh.su2:
    #   symmetry_xy  — Z=0
    #   symmetry_xz  — Y=0  (старое имя symmetry_plane для обратной совместимости)
    #   symmetry_yz  — X=0
    # В config.cfg прописываются все включённые плоскости в MARKER_SYM.
    # По умолчанию ВЫКЛЮЧЕНО, чтобы не ломать обратную совместимость.
    use_symmetry = bool(kwargs.get("use_symmetry", False))
    symmetry_planes = kwargs.get("symmetry_planes", None)  # list[str] | None
    if symmetry_planes is None:
        # Обратная совместимость: старый флаг use_symmetry=True → только Y=0
        symmetry_planes = ["xz"] if use_symmetry else []
    # Нормализуем в нижний регистр
    symmetry_planes = [str(p).lower() for p in symmetry_planes if p]
    # Словари треугольников по маркерам
    symmetry_tris: dict = {p: [] for p in symmetry_planes}

    # Маппинг точек поверхности -> точки объёма
    try:
        tree = cKDTree(vol_pts)
        _, point_map = tree.query(surf_pts)
    except Exception:
        point_map = np.zeros(len(surf_pts), dtype=int)
        for i, sp in enumerate(surf_pts):
            dists = np.linalg.norm(vol_pts - sp, axis=1)
            point_map[i] = int(np.argmin(dists))

    # Классификация граней по маркерам
    x_min, x_max = vol_pts[:, 0].min(), vol_pts[:, 0].max()
    y_min, y_max = vol_pts[:, 1].min(), vol_pts[:, 1].max()
    z_min, z_max = vol_pts[:, 2].min(), vol_pts[:, 2].max()
    bbox_size = max(x_max - x_min, y_max - y_min, z_max - z_min)
    tol = bbox_size * 0.002
    airfoil_tris = []
    farfield_tris = []

    faces_raw = np.asarray(surface.faces)
    n_surf_cells = surface.n_cells

    def classify_and_append(tri_idx_array):
        if len(tri_idx_array) == 0:
            return
        tri_idx_array = np.asarray(tri_idx_array)
        tri_pts = surf_pts[tri_idx_array]
        centroids = tri_pts.mean(axis=1)
        is_out = (
            (np.abs(centroids[:, 0] - x_min) < tol) | (np.abs(centroids[:, 0] - x_max) < tol) |
            (np.abs(centroids[:, 1] - y_min) < tol) | (np.abs(centroids[:, 1] - y_max) < tol) |
            (np.abs(centroids[:, 2] - z_min) < tol) | (np.abs(centroids[:, 2] - z_max) < tol)
        )
        # === T1: плоскости симметрии (XY=Z=0, XZ=Y=0, YZ=X=0) =========
        # Проверяем для каждой включённой плоскости — если треугольник
        # лежит на этой плоскости, относим к соответствующему маркеру.
        sym_mask: dict = {}
        if symmetry_planes:
            for plane in symmetry_planes:
                # Поддержка формата "xz" (через 0) и "xz:1.5" (смещение)
                if ":" in plane:
                    p_name, p_offset = plane.split(":", 1)
                    p_offset = float(p_offset)
                else:
                    p_name = plane
                    p_offset = 0.0
                p_name = p_name.strip().lower()
                if p_name == "xy":
                    sym_mask[plane] = np.abs(centroids[:, 2] - p_offset) < tol
                elif p_name == "xz":
                    sym_mask[plane] = np.abs(centroids[:, 1] - p_offset) < tol
                elif p_name == "yz":
                    sym_mask[plane] = np.abs(centroids[:, 0] - p_offset) < tol
        # =================================================================
        mapped = point_map[tri_idx_array]
        for k in range(len(mapped)):
            line = f"5 {int(mapped[k, 0])} {int(mapped[k, 1])} {int(mapped[k, 2])}"
            # Проверяем симметрию: первый попавший маркер из sym_mask
            sym_marked = False
            for plane, mask in sym_mask.items():
                if mask[k]:
                    symmetry_tris[plane].append(line)
                    sym_marked = True
                    break
            if sym_marked:
                continue
            if is_out[k]:
                farfield_tris.append(line)
            else:
                airfoil_tris.append(line)

    if n_surf_cells > 0 and len(faces_raw) > 0:
        first_n = int(faces_raw[0])
        if first_n == 3 and len(faces_raw) == n_surf_cells * 4:
            all_tris = faces_raw.reshape(n_surf_cells, 4)[:, 1:]
            classify_and_append(all_tris)
        else:
            parsed = []
            idx = 0
            while idx < len(faces_raw):
                n_verts = int(faces_raw[idx])
                tri = faces_raw[idx + 1: idx + 1 + n_verts]
                idx += 1 + n_verts
                if n_verts != 3:
                    continue
                parsed.append(tri)
            if parsed:
                classify_and_append(np.asarray(parsed, dtype=np.int64))

    valid_tets = [tet for tet in tetras if len(tet) == 4]

    # === ИСПРАВЛЕНИЕ #1: жёсткая защита от пустого маркера airfoil ===
    if len(airfoil_tris) == 0:
        raise RuntimeError(
            "Маркер 'airfoil' пустой: 0 граничных треугольников попали на тело. "
            "Это значит, что после вырезания ни одна грань тетраэдра не лежит "
            "внутри bbox-границы самолёта. Проверьте:\n"
            "  1. Геометрия не вырождена (size > 0).\n"
            "  2. Самолёт реально попал в фоновую сетку (не за её пределами).\n"
            "  3. Сетка достаточно мелкая (увеличьте качество до 'Точная').\n"
            "  4. Параметр margin в generate_mesh_impl не слишком мал."
        )

    # === ИСПРАВЛЕНИЕ #2: запись под try/except с явным flush+fsync ===
    try:
        # Сначала пишем во временный файл, чтобы атомарно заменить
        tmp_path = filename + ".tmp"
        with open(tmp_path, "w", encoding="ascii") as f:
            f.write("NDIME= 3\n")
            f.write(f"NELEM= {len(valid_tets)}\n")
            for i, tet in enumerate(valid_tets):
                f.write(f"10 {tet[0]} {tet[1]} {tet[2]} {tet[3]} {i}\n")
            f.write(f"NPOIN= {n_points}\n")
            for i, p in enumerate(vol_pts):
                f.write(f"{p[0]:.15e} {p[1]:.15e} {p[2]:.15e} {i}\n")
            f.write("NMARK= 2\n")
            f.write("MARKER_TAG= airfoil\n")
            f.write(f"MARKER_ELEMS= {len(airfoil_tris)}\n")
            for line in airfoil_tris:
                f.write(f"{line}\n")
            f.write("MARKER_TAG= farfield\n")
            f.write(f"MARKER_ELEMS= {len(farfield_tris)}\n")
            for line in farfield_tris:
                f.write(f"{line}\n")
            # === T1: маркеры симметрии (для каждой включённой плоскости) ==
            # Имена маркеров: symmetry_xy / symmetry_xz / symmetry_yz
            # В config.cfg они перечисляются в MARKER_SYM= ( ... ).
            # Старое имя "symmetry_plane" (эквивалент XZ) сохраняем для
            # обратной совместимости со старыми config.cfg.
            for plane in symmetry_planes:
                tris = symmetry_tris.get(plane, [])
                if not tris:
                    continue
                if plane == "xz":
                    # Маркер на плоскости Y=0 — оставляем оба имени
                    # (старое и новое), чтобы старый config.cfg работал.
                    f.write("MARKER_TAG= symmetry_plane\n")
                    f.write(f"MARKER_ELEMS= {len(tris)}\n")
                    for line in tris:
                        f.write(f"{line}\n")
                f.write(f"MARKER_TAG= symmetry_{plane}\n")
                f.write(f"MARKER_ELEMS= {len(tris)}\n")
                for line in tris:
                    f.write(f"{line}\n")
            # ==============================================================
            # === ИСПРАВЛЕНИЕ #3: явный flush+fsync на Windows ===
            f.flush()
            try:
                os.fsync(f.fileno())
            except (OSError, AttributeError):
                pass
        # Атомарная замена: на Windows os.replace атомарен на одном томе
        os.replace(tmp_path, filename)
    except Exception as e:
        # Удаляем временный файл, если он остался
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise RuntimeError(
            f"Не удалось записать mesh.su2 ({type(e).__name__}: {e}). "
            f"Возможно, на диске недостаточно места или файл заблокирован."
        ) from e

    return True


# ---------------------------------------------------------------------------
# Главная функция генерации (с колбэком прогресса)
# ---------------------------------------------------------------------------
def generate_mesh_impl(stl_paths, quality_text="Средняя", progress_cb=None,
                       cancel_cb=None, use_symmetry=False,
                       symmetry_planes=None):
    """Основная функция генерации сетки.

    progress_cb(percent:int, stage:str) — вызывается по ходу генерации;
    cancel_cb() -> bool — если вернул True, генерация аккуратно прерывается.
    use_symmetry: если True — на плоскости Y=0 треугольники маркируются
                  как symmetry_plane (T1: для SU2 MARKER_SYM).
    Возвращает (ok: bool, msg: str).
    """
    def report(pct, stage):
        print(f"[{int(pct):3d}%] {stage}")
        try:
            if progress_cb:
                progress_cb(int(pct), stage)
        except Exception:
            pass

    def check_cancel():
        try:
            if cancel_cb and cancel_cb():
                raise MeshCancelled("Генерация сетки отменена пользователем")
        except MeshCancelled:
            raise
        except Exception:
            pass

    try:
        report(2, "Загрузка поверхностей")
        body_meshes = []
        n_files = max(1, len(stl_paths))
        for fi, path in enumerate(stl_paths):
            check_cancel()
            if not os.path.exists(path):
                continue
            try:
                m = pv.read(path).triangulate().clean(tolerance=1e-5)
                if m.n_cells > 0:
                    body_meshes.append(m)
                    print(f"   ✓ {os.path.basename(path)}: {m.n_cells} граней")
            except Exception as e:
                print(f"   x Ошибка обработки {path}: {e}")
            report(2 + int(6 * (fi + 1) / n_files), "Загрузка поверхностей")

        if not body_meshes:
            return False, "Поверхностей нет после обработки"

        check_cancel()
        report(10, "Анализ габаритов геометрии")

        all_pts = np.vstack([m.points for m in body_meshes])
        body_min = all_pts.min(axis=0)
        body_max = all_pts.max(axis=0)
        body_size = np.max(body_max - body_min)
        if body_size < 1e-6:
            return False, "Геометрия вырождена (нулевой размер)"

        print("🔧 Создание фоновой сетки с локальным сгущением...")

        # Параметры сетки
        if "Грубая" in quality_text:
            h_near = max(body_size * 0.045, 0.08)
            h_far = body_size * 0.75
            margin = body_size * 2.8
        elif "Точная" in quality_text:
            h_near = max(body_size * 0.018, 0.04)
            h_far = body_size * 0.45
            margin = body_size * 4.0
        else:
            h_near = max(body_size * 0.015, 0.05)
            h_far = body_size * 0.55
            margin = body_size * 3.2

        print(f"   Шаг около тела: {h_near:.4f} м")
        print(f"   Шаг вдали:      {h_far:.4f} м")
        print(f"   Margin:         {margin:.2f} м")

        h_near = max(h_near, body_size * 0.015)
        h_far = max(h_far, h_near * 3.0)

        report(15, "Построение осей фоновой сетки")

        def make_clustered_axis(bmin, bmax, margin_minus, margin_plus,
                                h_near_axis, h_far_axis):
            domain_min = bmin - margin_minus
            domain_max = bmax + margin_plus
            inner_min = bmin - 2.0 * h_near_axis
            inner_max = bmax + 2.0 * h_near_axis
            inner_min = max(inner_min, domain_min)
            inner_max = min(inner_max, domain_max)

            left_len = max(inner_min - domain_min, 0.0)
            n_left = max(2, int(np.ceil(left_len / h_far_axis)))
            left = np.linspace(domain_min, inner_min, n_left + 1)

            center = np.arange(inner_min, inner_max + 0.5 * h_near_axis,
                               h_near_axis)
            if len(center) == 0 or center[-1] < inner_max:
                center = np.append(center, inner_max)

            right_len = max(domain_max - inner_max, 0.0)
            n_right = max(2, int(np.ceil(right_len / h_far_axis)))
            right = np.linspace(inner_max, domain_max, n_right + 1)

            axis = np.unique(np.concatenate([left, center, right]))
            axis.sort()
            return axis

        xs = make_clustered_axis(body_min[0], body_max[0],
                                 margin * 0.8, margin * 1.2,
                                 h_near, h_far)
        report(17, "Построение осей фоновой сетки (X)")

        ys = make_clustered_axis(body_min[1], body_max[1],
                                 margin, margin,
                                 h_near, h_far)
        report(19, "Построение осей фоновой сетки (Y)")

        zs = make_clustered_axis(body_min[2], body_max[2],
                                 margin, margin,
                                 h_near, h_far)
        report(21, "Построение осей фоновой сетки (Z)")

        nx = len(xs) - 1
        ny = len(ys) - 1
        nz = len(zs) - 1
        approx_tets = nx * ny * nz * 5
        print(f"   Разрешение:     {nx} x {ny} x {nz}")
        print(f"   Примерно тетр.: {approx_tets}")

        check_cancel()
        report(24, "Генерация узлов фоновой сетки")

        xv, yv, zv = np.meshgrid(xs, ys, zs, indexing='ij')
        points = np.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])
        print(f"   Точек фона: {len(points)}")

        check_cancel()
        report(30, "Разбиение на тетраэдры")

        ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                                 indexing='ij')

        def idx(i, j, k):
            return (i * (ny + 1) + j) * (nz + 1) + k

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
            [0, 1, 2, 6],
            [0, 2, 3, 6],
            [0, 5, 1, 6],   # было [0,1,5,6]
            [0, 4, 5, 6],
            [0, 3, 7, 6],
            [0, 7, 4, 6],   # было [0,4,7,6]
        ], dtype=np.int64)

        cells_arr = corners[:, pattern].reshape(-1, 4)
        n_tets = len(cells_arr)
        print(f"   Тетраэдров до вырезания: {n_tets}")

        check_cancel()
        report(40, "Сборка объёмной сетки")

        cell_array = np.hstack([
            np.full((n_tets, 1), 4, dtype=np.int64),
            cells_arr
        ]).ravel()
        celltypes = np.full(n_tets, pv.CellType.TETRA, dtype=np.uint8)
        grid = pv.UnstructuredGrid(cell_array, celltypes, points)
        report(44, "Расчёт центров ячеек")

        cell_centers = grid.cell_centers().points
        centers_poly = pv.PolyData(cell_centers)
        keep_mask = np.ones(len(cell_centers), dtype=bool)
        print("✂️ Вырезаем тела из фона (надежный метод VTK)...")

        total_removed = 0
        n_comp = max(1, len(body_meshes))
        for i, m in enumerate(body_meshes):
            check_cancel()
            report(44 + int(24 * i / n_comp),
                   f"Вырезание компонента {i + 1}/{n_comp}")
            n_inside = 0
            try:
                try:
                    enclosed = centers_poly.select_enclosed_points(
                        m, tolerance=1e-5, check_surface=False)
                except (TypeError, AttributeError):
                    try:
                        enclosed = centers_poly.select_interior_points(m)
                    except (TypeError, AttributeError):
                        enclosed = centers_poly.select_enclosed_points(m)
                inside = enclosed['SelectedPoints'].astype(bool)
                n_inside = int(inside.sum())
                if n_inside == 0 and TRIMESH_AVAILABLE:
                    faces_np = m.faces.reshape(-1, 4)[:, 1:]
                    tm = trimesh.Trimesh(vertices=m.points, faces=faces_np,
                                         process=True)
                    if not tm.is_watertight:
                        tm.fill_holes()
                    inside = tm.contains(cell_centers)
                    n_inside = int(inside.sum())
                if n_inside > 0:
                    keep_mask &= ~inside
                    total_removed += n_inside
                    print(f"   ✓ Компонент {i}: удалено {n_inside} ячеек внутри тела")
                else:
                    print(f"   ⚠ Компонент {i}: 0 ячеек попало внутрь!")
            except Exception as e:
                print(f"   x Ошибка вырезания компонента {i}: {e}")

        if total_removed == 0:
            return False, ("ОШИБКА: Ни одна ячейка не вырезана под самолёт! "
                           "Самолёт не попал в сетку.")

        check_cancel()
        report(70, "Удаление ячеек внутри тела")
        grid = grid.extract_cells(keep_mask)
        if not isinstance(grid, pv.UnstructuredGrid):
            grid = pv.UnstructuredGrid(grid)
        if grid.n_cells == 0:
            return False, "После вырезания не осталось ячеек"

        print(f"   После вырезания: {grid.n_cells} тетраэдров")

        report(75, "Поиск вырожденных элементов")
        tets_raw = extract_cells(grid, cell_type=10)
        if not tets_raw:
            return False, "Не найдены тетраэдры после вырезания"

        tets_arr = np.array(tets_raw, dtype=np.int64)
        all_pts_grid = np.asarray(grid.points)

        p0 = all_pts_grid[tets_arr[:, 0]]
        p1 = all_pts_grid[tets_arr[:, 1]]
        p2 = all_pts_grid[tets_arr[:, 2]]
        p3 = all_pts_grid[tets_arr[:, 3]]
        v1 = p1 - p0
        v2 = p2 - p0
        v3 = p3 - p0
        signed_vol = np.einsum('ij,ij->i', v1, np.cross(v2, v3)) / 6.0

        # Защита: вывернутые тетраэдры (отрицательный знаковый объём)
        # разворачиваем перестановкой узлов 1<->2 — иначе SU2 получает
        # отрицательные контрольные объёмы и расходится с первой итерации.
        inverted = signed_vol < 0
        n_inverted = int(inverted.sum())
        if n_inverted > 0:
            print(f"   Развёрнуто вывернутых тетраэдров (neg volume): {n_inverted}")
            _swap = tets_arr[inverted, 2].copy()
            tets_arr[inverted, 2] = tets_arr[inverted, 1]
            tets_arr[inverted, 1] = _swap
        vol = np.abs(signed_vol)
        edges = np.stack([
            np.linalg.norm(p1 - p0, axis=1),
            np.linalg.norm(p2 - p0, axis=1),
            np.linalg.norm(p3 - p0, axis=1),
            np.linalg.norm(p2 - p1, axis=1),
            np.linalg.norm(p3 - p1, axis=1),
            np.linalg.norm(p3 - p2, axis=1),
        ], axis=1)
        min_edge = edges.min(axis=1)
        valid_mask = (vol > 1e-12) & (min_edge > 1e-8)
        n_degenerate = int((~valid_mask).sum())
        if n_degenerate > 0:
            print(f"   Удалено вырожденных тетраэдров: {n_degenerate}")
        tets_arr = tets_arr[valid_mask]
        if len(tets_arr) == 0:
            return False, "Все тетраэдры вырождены"

        check_cancel()
        report(84, "Очистка висячих точек")
        used_indices = np.unique(tets_arr.flatten())
        remap = np.full(len(all_pts_grid), -1, dtype=np.int64)
        remap[used_indices] = np.arange(len(used_indices))
        new_points = all_pts_grid[used_indices]
        tets_renumbered = remap[tets_arr]
        if (tets_renumbered < 0).any():
            return False, "Ошибка перенумерации точек"

        report(88, "Финальная сборка сетки")
        n_tets_clean = len(tets_renumbered)
        cell_array_clean = np.hstack([
            np.full((n_tets_clean, 1), 4, dtype=np.int64),
            tets_renumbered
        ]).ravel()
        celltypes_clean = np.full(n_tets_clean, pv.CellType.TETRA,
                                  dtype=np.uint8)
        grid = pv.UnstructuredGrid(cell_array_clean, celltypes_clean,
                                   new_points)
        print(f"✓ Итоговая сетка: {grid.n_points} узлов, {grid.n_cells} тетраэдров")

        try:
            grid.save(PREVIEW_MESH)
            print(f"👁 Preview сохранён: {PREVIEW_MESH}")
        except Exception as e:
            print(f"⚠ Preview не сохранён: {e}")

        check_cancel()
        report(94, "Извлечение граничной поверхности и нормалей")

        surface = grid.extract_surface(algorithm='dataset_surface').triangulate()
        try:
            surface = surface.compute_normals(
                auto_orient_normals=True,
                consistent_normals=True,
                point_normals=False,
                cell_normals=True,
            )
        except Exception:
            try:
                surface = surface.compute_normals(auto_orient_normals=True)
            except Exception:
                pass

        try:
            cell_centers_surf = surface.cell_centers().points
            normals = np.asarray(surface.cell_normals)
            if len(normals) == surface.n_cells and len(normals) > 0:
                center = np.asarray(surface.points).mean(axis=0)
                outward = cell_centers_surf - center
                flip = np.einsum('ij,ij->i', normals, outward) < 0
                if np.any(flip):
                    normals[flip] *= -1.0
                    surface.cell_normals = normals
        except Exception:
            pass

        su2_path = MESH_FILE
        report(97, "Запись файла mesh.su2")
        try:
            # === T1: пробрасываем use_symmetry в write_su2 =============
            write_su2(grid, surface, su2_path,
                      use_symmetry=use_symmetry,
                      symmetry_planes=symmetry_planes)
            # ============================================================
        except RuntimeError as e:
            # ИСПРАВЛЕНО: пробрасываем наверх, а не теряем в логе
            return False, str(e)

        if os.path.exists(su2_path):
            sz = os.path.getsize(su2_path)
            print(f"✓ mesh.su2 создан ({sz:,} байт)")
            # === ИСПРАВЛЕНО: доп. проверка, что в файле есть маркеры ===
            check_ok, check_msg = quick_mesh_check(su2_path)
            if not check_ok:
                return False, f"mesh.su2 записан, но {check_msg}"
            report(100, "Готово")
            return True, (f"Сетка построена ({quality_text}), тетраэдров: "
                          f"{grid.n_cells}.")
        else:
            return False, "Файл mesh.su2 не создан"

    except MeshCancelled as mc:
        return False, str(mc)
    except Exception as e:
        print(traceback.format_exc())
        return False, str(e)


# ---------------------------------------------------------------------------
# Совместимый вызов прежнего API
# ---------------------------------------------------------------------------
def generate_mesh_from_stl_list(paths, output=MESH_FILE, quality="Средняя",
                                progress_cb=None) -> bool:
    """Совместимость со старым API."""
    ok, msg = generate_mesh_impl(list(paths), quality, progress_cb)
    return ok


# ---------------------------------------------------------------------------
# ИСПРАВЛЕНО: быстрая проверка mesh.su2
# ---------------------------------------------------------------------------
def quick_mesh_check(path: str = None) -> tuple:
    """Проверяет, что mesh.su2 содержит все обязательные секции.

    Возвращает (ok: bool, msg: str).
    """
    if path is None:
        path = MESH_FILE
    if not os.path.exists(path):
        return False, "файл не существует"
    try:
        size = os.path.getsize(path)
        if size < 1000:
            return False, f"слишком маленький ({size} байт) — оборван?"
        with open(path, "r", encoding="ascii", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        return False, f"не удалось прочитать: {e}"

    # Проверяем наличие обязательных секций
    missing = []
    if "NDIME= 3" not in content:
        missing.append("NDIME= 3")
    if "NELEM=" not in content:
        missing.append("NELEM=")
    if "NPOIN=" not in content:
        missing.append("NPOIN= (нет точек — файл оборван!)")
    if "MARKER_TAG= airfoil" not in content:
        missing.append("MARKER_TAG= airfoil")
    if "MARKER_TAG= farfield" not in content:
        missing.append("MARKER_TAG= farfield")

    if missing:
        return False, "отсутствуют секции: " + ", ".join(missing)

    # Доп. проверка: MARKER_ELEMS > 0
    import re
    m_airfoil = re.search(
        r"MARKER_TAG=\s*airfoil\s*\n\s*MARKER_ELEMS=\s*(\d+)", content
    )
    if m_airfoil and int(m_airfoil.group(1)) == 0:
        return False, "MARKER_ELEMS= 0 под airfoil (самолёт не вырезан)"
    m_farfield = re.search(
        r"MARKER_TAG=\s*farfield\s*\n\s*MARKER_ELEMS=\s*(\d+)", content
    )
    if m_farfield and int(m_farfield.group(1)) == 0:
        return False, "MARKER_ELEMS= 0 под farfield (фон пустой)"

    return True, f"OK (size={size:,} байт, найдены NDIME/NELEM/NPOIN/оба маркера)"


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        input_json = sys.argv[1]
        result_json = sys.argv[2]
        with open(input_json, "r", encoding="utf-8") as f:
            job = json.load(f)
        ok, msg = generate_mesh_impl(job["stl_paths"],
                                     job.get("quality_text", "Средняя"))
        with open(result_json, "w", encoding="utf-8") as f:
            json.dump({"ok": ok, "msg": msg}, f, ensure_ascii=False, indent=2)
        sys.exit(0 if ok else 1)
    else:
        print("Используйте: python gmsh_generator.py input.json output.json")
        sys.exit(1)
