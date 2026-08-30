"""
mesh/mesh_worker.py — НЕблокирующий воркер генерации сетки (QThread).

T1: добавлен параметр use_symmetry — пробрасывается в generate_mesh_impl
для создания маркера symmetry_plane в mesh.su2 (для SU2 MARKER_SYM).
"""

from __future__ import annotations

import os
import traceback
from PyQt5.QtCore import QThread, pyqtSignal

try:
    from mesh.gmsh_generator import generate_mesh_impl
except Exception:  # pragma: no cover
    generate_mesh_impl = None


class MeshCancelled(Exception):
    """Исключение, выбрасываемое при отмене генерации сетки."""


class MeshWorker(QThread):
    """Запускает generate_mesh_impl в отдельном потоке.

    Параметры:
        stl_paths    — список путей к STL-файлам
        quality_text — "Грубая" / "Средняя" / "Точная"
        parent       — QObject-родитель (обычно MainWindow)
        use_symmetry — T1: добавить маркер symmetry_plane для SU2 MARKER_SYM
    """
    progress_signal = pyqtSignal(int, str)   # процент 0..100, этап
    finished_signal = pyqtSignal(bool, str)  # ok, message

    def __init__(self, stl_paths, quality_text="Средняя", parent=None,
                 use_symmetry: bool = False,
                 symmetry_planes: list = None):
        super().__init__(parent)
        self.stl_paths = list(stl_paths) if stl_paths else []
        self.quality_text = quality_text
        self._cancel_requested = False
        # === T1: плоскости симметрии (XY, XZ, YZ) ====================
        # Список плоскостей. Для обратной совместимости: если передан
        # старый флаг use_symmetry=True без списка — добавляем "xz".
        self.use_symmetry = bool(use_symmetry)
        self.symmetry_planes = list(symmetry_planes) if symmetry_planes else None
        if use_symmetry and not self.symmetry_planes:
            self.symmetry_planes = ["xz"]
        # ==============================================================

    def cancel(self):
        self._cancel_requested = True

    def _check_cancel(self):
        if self._cancel_requested:
            raise MeshCancelled("Генерация сетки отменена пользователем")

    def _progress_cb(self, percent, stage):
        try:
            self.progress_signal.emit(int(percent), str(stage))
        except Exception:
            pass

    def run(self):
        if generate_mesh_impl is None:
            self.finished_signal.emit(
                False, "Не удалось импортировать mesh.gmsh_generator.")
            return
        try:
            # === T1: пробрасываем плоскости симметрии ==================
            ok, msg = generate_mesh_impl(
                self.stl_paths,
                quality_text=self.quality_text,
                progress_cb=self._progress_cb,
                cancel_cb=self._check_cancel,
                use_symmetry=self.use_symmetry,
                symmetry_planes=self.symmetry_planes,
            )
            # ==========================================================
        except MeshCancelled as e:
            self.finished_signal.emit(False, str(e))
            return
        except Exception as e:
            tb = traceback.format_exc(limit=3)
            self.finished_signal.emit(False, f"Ошибка генерации сетки: {e}\n{tb}")
            return
        self.finished_signal.emit(bool(ok), str(msg))


class MeshAdaptWorker(QThread):
    """Адаптация сетки по решению (SU2_ADAPT) в фоновом потоке.

    Использует solver.workers.run_su2_adapt: берёт mesh.su2 и restart.dat
    из готового расчёта и строит mesh_adapt.su2 (локальное сгущение в
    областях высоких градиентов). Результат кладётся обратно в рабочую
    сетку (MESH_FILE) при подтверждении пользователем.
    """
    progress_signal = pyqtSignal(int, str)   # процент, этап
    finished_signal = pyqtSignal(bool, str)  # ok, message

    def __init__(self, case_dir, mesh_path, restart_path, parent=None,
                 adapt_markers=("airfoil",), abs_error: float = 1e-6):
        super().__init__(parent)
        self.case_dir = case_dir
        self.mesh_path = mesh_path
        self.restart_path = restart_path
        self.adapt_markers = list(adapt_markers)
        self.abs_error = float(abs_error)

    def _log(self, m):
        try:
            self.progress_signal.emit(0, str(m))
        except Exception:
            pass

    def run(self):
        try:
            from solver.workers import run_su2_adapt
        except Exception as e:  # pragma: no cover
            self.finished_signal.emit(False, f"Не удалось импортировать run_su2_adapt: {e}")
            return
        try:
            out = run_su2_adapt(
                case_dir=self.case_dir,
                mesh_path=self.mesh_path,
                restart_path=self.restart_path,
                adapt_markers=self.adapt_markers,
                abs_error=self.abs_error,
                log_cb=self._log,
            )
        except Exception as e:
            tb = traceback.format_exc(limit=3)
            self.finished_signal.emit(False, f"Ошибка адаптации: {e}\n{tb}")
            return
        self.finished_signal.emit(True, out)
