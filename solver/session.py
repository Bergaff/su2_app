import json
import os
import shutil
from datetime import datetime


class CalculationSession:
    """Сессия расчёта (single или sweep): состояние, точки, продолжение с паузы."""

    SESSION_META = "session_meta.json"

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        self.session_id = ""
        self.mode = "single"
        self.solver = "EULER"
        self.physics = {}
        self.ref_data = (1.0, 1.0, 0.25, 0.0, 0.0)
        self.active_markers = []
        self.aoa_list = []
        self.cpu_cores = 1
        self.current_index = 0        # индекс ТЕКУЩЕЙ (считающейся) точки
        self.next_index = 0           # индекс следующей несчитанной точки
        self.paused = False
        self.cancelled = False
        self.finished = False
        self.results = []
        self.case_dirs = []
        # True — писать в config.cfg ENABLE_CUDA= YES. Выставляется только
        # когда SU2_CFD реально собран с поддержкой CUDA: иначе SU2
        # завершится с ошибкой «ENABLE_CUDA is set to YES».
        self.enable_cuda = False
        # Данные для оценки потолка итераций и выбора CFL. Нули и False
        # означают «не задано» — тогда сборщик конфига работает по прежней
        # схеме, привязанной только к качеству сетки.
        self.n_bodies = 0
        self.n_points = 0
        self.cfl_aggressive = False

    # ------------------------------------------------------------------
    def start_new(self, mode, solver, physics, ref_data, active_markers,
                  aoa_list, cpu_cores=1, n_bodies=None, n_points=None,
                  cfl_aggressive=None):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.mode = mode
        self.solver = solver
        self.physics = dict(physics)
        self.ref_data = tuple(ref_data)
        self.active_markers = list(active_markers)
        self.aoa_list = list(aoa_list)
        self.cpu_cores = max(1, int(cpu_cores))
        self.current_index = 0
        self.next_index = 0
        self.paused = False
        self.cancelled = False
        self.finished = False
        self.results = []
        if n_bodies is not None:
            self.n_bodies = int(n_bodies)
        if n_points is not None:
            self.n_points = int(n_points)
        if cfl_aggressive is not None:
            self.cfl_aggressive = bool(cfl_aggressive)
        self.case_dirs = [os.path.join(self.work_dir,
                                       f"case_{self.session_id}",
                                       f"aoa_{a:+.2f}")
                          for a in self.aoa_list]
        self.save()

    # ------------------------------------------------------------------
    def current_case_dir(self):
        if 0 <= self.current_index < len(self.case_dirs):
            return self.case_dirs[self.current_index]
        return None

    def case_dir_for(self, idx):
        if 0 <= idx < len(self.case_dirs):
            return self.case_dirs[idx]
        return None

    @property
    def current_aoa(self):
        if 0 <= self.next_index < len(self.aoa_list):
            return self.aoa_list[self.next_index]
        return None

    @property
    def is_complete(self):
        return self.next_index >= len(self.aoa_list)

    # ------------------------------------------------------------------
    def mark_point_processed(self, idx):
        self.current_index = idx
        self.next_index = idx + 1
        self.save()

    def mark_finished(self):
        self.finished = True
        self.paused = False
        self.save()

    def mark_paused(self):
        self.paused = True
        self.save()

    def mark_cancelled(self):
        self.cancelled = True
        self.paused = False
        self.save()

    # ------------------------------------------------------------------
    def meta_path(self):
        return os.path.join(self.work_dir, self.SESSION_META)

    def save(self):
        data = {
            "session_id": self.session_id,
            "mode": self.mode,
            "solver": self.solver,
            "physics": self.physics,
            "ref_data": list(self.ref_data),
            "active_markers": self.active_markers,
            "aoa_list": self.aoa_list,
            "cpu_cores": self.cpu_cores,
            "current_index": self.current_index,
            "next_index": self.next_index,
            "paused": self.paused,
            "cancelled": self.cancelled,
            "finished": self.finished,
            "results": self.results,
            "case_dirs": self.case_dirs,
        }
        try:
            with open(self.meta_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def exists_on_disk(self) -> bool:
        return os.path.exists(self.meta_path())

    def load(self) -> bool:
        meta = self.meta_path()
        if not os.path.exists(meta):
            return False
        try:
            with open(meta, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            return False
        self.session_id = d.get("session_id", "")
        self.mode = d.get("mode", "single")
        self.solver = d.get("solver", "EULER")
        self.physics = d.get("physics", {})
        self.ref_data = tuple(d.get("ref_data", (1.0, 1.0, 0.25, 0.0, 0.0)))
        self.active_markers = d.get("active_markers", [])
        self.aoa_list = d.get("aoa_list", [])
        self.cpu_cores = d.get("cpu_cores", 1)
        self.current_index = d.get("current_index", 0)
        self.next_index = d.get("next_index", 0)
        self.paused = d.get("paused", False)
        self.cancelled = d.get("cancelled", False)
        self.finished = d.get("finished", False)
        self.results = d.get("results", [])
        self.case_dirs = d.get("case_dirs", [])
        return True

    def clear(self):
        root = os.path.join(self.work_dir, f"case_{self.session_id}")
        if os.path.isdir(root):
            shutil.rmtree(root, ignore_errors=True)
        try:
            os.remove(self.meta_path())
        except OSError:
            pass
