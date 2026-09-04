import json
import os

CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".aeroopt_config.json")

# Базовые рабочие каталоги проекта
WORK_DIR_BASE = os.path.abspath("./work")
RESULTS_DIR = os.path.abspath("./results")
MESH_FILE = os.path.abspath("./mesh.su2")
PREVIEW_MESH = os.path.abspath("./mesh_preview.vtk")
os.makedirs(WORK_DIR_BASE, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Вердикт о последней построенной сетке. Заполняет mesh/gmsh_generator.py,
# читает solver/workers.py.
#
# Зачем: автоконфиг после неудачного прогона перезапускает точку с другим
# пресетом, но пресеты меняют только CFL_NUMBER, CFL_ADAPT, MUSCL и entropy
# fix — ни сетку, ни INNER_ITER. Если прогон встал из-за того, что сетка не
# описывает геометрию, повтор с другими численными настройками дать ничего
# не может, а стоит полного прогона. На пластине plane_wing.step это
# 1100 итераций за 951 с, на полном самолёте — десятки минут за попытку.
# Словарь изменяется на месте, поэтому переопределение имени не требуется.
MESH_DIAGNOSIS = {
    "body_fitted": True,   # поверхность тела вошла в сетку как есть
    "unresolved": [],      # компоненты тоньше шага у тела
    "flat": [],            # компоненты нулевой толщины
    "reason": "",          # почему поверхность не облегается
}

# Качество сетки (как в комбо-боксе UI)
MESH_QUALITY = ["Грубая (быстро)", "Средняя", "Точная (медленно)"]

# Роли компонентов: ключ (англ.) → подпись (рус.)
ROLES = {
    "import": "Импорт",
    "fuselage": "Фюзеляж",
    "wing": "Крыло",
    "flap": "Закрылок",
    "slat": "Предкрылок",
    "h_stab": "ГО (стабилизатор)",
    "elevator": "Руль высоты",
    "v_stab": "ВО (киль)",
    "other": "Другое",
}

ROLE_COLORS = {
    "import": (0.9, 0.8, 0.6),
    "fuselage": (0.62, 0.77, 0.90),
    "wing": (0.55, 0.85, 0.55),
    "flap": (0.95, 0.75, 0.40),
    "slat": (0.95, 0.55, 0.40),
    "h_stab": (0.85, 0.80, 0.50),
    "elevator": (0.95, 0.65, 0.55),
    "v_stab": (0.80, 0.65, 0.90),
    "other": (0.65, 0.65, 0.65),
}


class UserConfig:
    """Файловое хранилище настроек (путь к SU2 и т.п.)."""

    DEFAULTS = {
        "su2_exe": r"C:\Program Files\SU2\SU2_CFD.exe",
        "mpiexec": "mpiexec",
        "first_launch_done": False,
        "last_project": "",
    }

    def __init__(self, path: str = CONFIG_FILE):
        self.path = path
        self.data = dict(self.DEFAULTS)
        self.load()

    def load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                if isinstance(saved, dict):
                    self.data.update(saved)
        except Exception:
            pass

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @property
    def su2_exe(self) -> str:
        return self.data.get("su2_exe", "")

    @su2_exe.setter
    def su2_exe(self, value: str):
        self.data["su2_exe"] = value
        self.save()

    @property
    def mpiexec(self) -> str:
        return self.data.get("mpiexec", "mpiexec")


config = UserConfig()
