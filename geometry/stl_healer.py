import os


def heal_stl_mesh(path: str, log_callback=None, work_dir: str = None):
    """Лечение STL: очистка, удаление вырожденных граней, заливка дыр,
    ремонт нормалей. Возвращает (ok, msg, healed_path)."""
    def log(m):
        if log_callback:
            log_callback(m)

    import pyvista as pv
    import numpy as np

    work_dir = work_dir or os.path.dirname(path) or "."
    os.makedirs(work_dir, exist_ok=True)
    healed_path = os.path.join(
        work_dir, os.path.splitext(os.path.basename(path))[0] + "_healed.stl")

    try:
        mesh = pv.read(path).extract_surface().triangulate()
        log(f"  Исходно: {mesh.n_points} точек, {mesh.n_cells} граней")

        before = mesh.n_points
        mesh = mesh.clean(tolerance=1e-6)
        log(f"  clean: -{before - mesh.n_points} дублей точек")

        # вырожденные треугольники
        areas = mesh.compute_cell_sizes(length=False, volume=False).cell_data.get("Area")
        if areas is not None:
            bad = int((areas <= 1e-12).sum())
            if bad:
                good = np.where(areas > 1e-12)[0]
                mesh = mesh.extract_cells(good).extract_surface().triangulate()
                log(f"  удалено вырожденных граней: {bad}")

        # заливка дыр
        try:
            n_open = mesh.n_open_edges
            if n_open > 0:
                mesh = mesh.fill_holes(1000).clean()
                log(f"  залиты отверстия (открытых рёбер было: {n_open})")
        except Exception as e:
            log(f"  предупреждение заливки: {e}")

        # trimesh-ремонт
        try:
            import trimesh
            tm = trimesh.Trimesh(vertices=mesh.points,
                                 faces=mesh.faces.reshape(-1, 4)[:, 1:])
            trimesh.repair.fix_normals(tm)
            trimesh.repair.fill_holes(tm)
            tm.remove_degenerate_faces()
            tm.merge_vertices()
            watertight = bool(tm.is_watertight)
            import numpy as _np
            faces = _np.hstack([_np.full((len(tm.faces), 1), 3), tm.faces]).astype(_np.int64)
            mesh = pv.PolyData(_np.asarray(tm.vertices), faces)
            log(f"  watertight: {'да' if watertight else 'нет'}")
        except Exception as e:
            log(f"  trimesh-ремонт пропущен: {e}")

        mesh.save(healed_path)
        log(f"  сохранено: {healed_path}")
        return True, "Лечение завершено успешно.", healed_path
    except Exception as e:
        return False, f"Ошибка лечения: {e}", None


class HealReportDialog:
    """Диалог с отчётом о лечении STL (ленивая обёртка над QDialog)."""

    def __new__(cls, title: str, text: str, parent=None):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
        dlg = QDialog(parent)
        dlg.setWindowTitle(title)
        dlg.setMinimumSize(520, 420)
        lay = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setPlainText(text)
        lay.addWidget(te)
        btn = QPushButton("Закрыть")
        btn.clicked.connect(dlg.accept)
        lay.addWidget(btn)
        return dlg


# --- совместимость с прежним классом ---
class STLHealer:
    def __init__(self, input_path: str, output_dir: str = None):
        self.input_path = input_path
        self.output_dir = output_dir
        self.healed_path = None

    def heal(self):
        ok, msg, healed = heal_stl_mesh(self.input_path, work_dir=self.output_dir)
        self.healed_path = healed
        return ok, msg
