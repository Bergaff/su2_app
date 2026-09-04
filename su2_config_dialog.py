# -*- coding: utf-8 -*-
"""
su2_config_dialog.py - окно настроек расчёта SU2 для AeroOpt.

Возможности:
  * Редактирование config.cfg через интерфейс (без ручного правки текста).
  * У каждого параметра - русское название и ПОДСКАЗКА при наведении
    (tooltip): что это, на что влияет, что крутить при расхождении.
  * Именные ПРЕСЕТЫ: встроенные ("Стандартный", "Устойчивый safe",
    "Ультра-устойчивый ultra") + свои пресеты, которые сохраняются в
    %APPDATA%\\AeroOpt\\su2_presets.json и доступны во всех проектах.
  * Авто-предложение после расхождения (offer_recovery_after_failure) -
    вызывается из раннера SU2.

Зависимости: PyQt5 (с запасным вариантом PySide2), su2_autoconfig.py
(лежит рядом). Тяжёлых пакетов не требует.

Быстрая интеграция (в su2_gui.py, сразу после создания главного окна):

    import su2_config_dialog
    su2_config_dialog.install_menu(main_window)

Это добавит в меню бар отдельное меню "SU2" с настройками, пресетами
и справкой. Тонкая проводка авто-предложения после падения расчёта -
в файле, который запускает SU2_CFD.exe (см. offer_recovery_after_failure).
"""

from __future__ import annotations

import glob
import json
import os
import re

try:
    from PyQt5.QtWidgets import (
        QApplication, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
        QFormLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel,
        QLineEdit, QMainWindow, QMenu, QMessageBox, QPushButton,
        QScrollArea, QSpinBox, QVBoxLayout, QWidget, QAction)
    from PyQt5.QtCore import Qt
except ImportError:  # PySide2
    from PySide2.QtWidgets import (
        QApplication, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
        QFormLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel,
        QLineEdit, QMainWindow, QMenu, QMessageBox, QPushButton,
        QScrollArea, QSpinBox, QVBoxLayout, QWidget, QAction)
    from PySide2.QtCore import Qt

import su2_autoconfig

# Официальные кейсы SU2 (конфиги + 3D-сетки) — вспомогательный stdlib-пакет.
# Если его нет (например, при частичной сборке), диалог просто скроет
# соответствующую группу.
try:
    import official_cases
except Exception:                                        # pragma: no cover
    official_cases = None


# ---------------------------------------------------------------------------
# Где хранятся пользовательские пресеты
# ---------------------------------------------------------------------------

def _appdata_dir():
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    d = os.path.join(base, "AeroOpt")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return d


PRESETS_JSON = os.path.join(_appdata_dir(), "su2_presets.json")

# Встроенные пресеты — ровно два, ultra и safe. Значения и подписи берутся
# из su2_autoconfig: раньше здесь лежала своя копия чисел, и она расходилась
# с тем, что реально подставлял автоконфиг.
try:
    BUILTIN_PRESETS = {k: dict(v) for k, v in su2_autoconfig.PRESETS.items()}
except Exception:                                        # pragma: no cover
    BUILTIN_PRESETS = {
        "ultra": {"TIME_DISCRE_FLOW": "EULER_IMPLICIT", "MUSCL_FLOW": "YES"},
        "safe": {"TIME_DISCRE_FLOW": "EULER_IMPLICIT", "MUSCL_FLOW": "NO"},
    }

try:
    import su2_preset_format
except Exception:                                        # pragma: no cover
    su2_preset_format = None

# Пресеты с официальных кейсов SU2 (заполняются после определения PARAMS).
# Каждый пресет — это настройки, которые реально стоят в официальном
# config.cfg, сверенные с полями диалога. Нельзя удалить/переименовать.
OFFICIAL_PRESETS = {}     # имя -> {ключ: значение}


# ---------------------------------------------------------------------------
# Каталог параметров: группа, ключ, подпись, тип, опции, подсказка.
# Типы: 'yesno', 'combo', 'double', 'int', 'str'
# ---------------------------------------------------------------------------

PARAMS = [
    # --- Поток ---
    ("Поток (условия обтекания)", "SOLVER", "Тип уравнений", "combo",
     ["EULER", "NAVIER_STOKES", "RANS", "INC_EULER",
      "INC_NAVIER_STOKES", "INC_RANS"], None,
     "EULER — невязкий газ: без пограничного слоя, быстро и максимально "
     "устойчиво; подъёмная сила считается хорошо, сопротивление — только "
     "волновое/давление.\n"
     "NAVIER_STOKES — вязкий ламинарный поток.\n"
     "RANS — вязкий турбулентный: нужна модель турбулентности и очень "
     "мелкая сетка у стенки (y+≈1).\n"
     "INC_* — несжимаемые варианты для малых скоростей (M<0.3)."),
    ("Поток (условия обтекания)", "MATH_PROBLEM", "Тип задачи", "combo",
     ["DIRECT", "CONTINUOUS_ADJOINT"], None,
     "DIRECT — обычный прямой расчёт обтекания.\n"
     "CONTINUOUS_ADJOINT — сопряжённая задача: нужна для градиентов при "
     "оптимизации формы, а не для просмотра полей."),
    ("Поток (условия обтекания)", "MACH_NUMBER", "Число Маха", "double",
     None, (0.0, 5.0, 3, 0.01),
     "Число Маха набегающего потока (скорость / скорость звука).\n"
     "0.176 ≈ 60 м/с у земли. Выше ~0.7 появляются сверхзвуковые зоны и "
     "схема становится чувствительнее."),
    ("Поток (условия обтекания)", "AOA", "Угол атаки, °", "double",
     None, (-90.0, 90.0, 2, 0.1),
     "Угол атаки в градусах. На больших углах (срыв потока, ~15°+) "
     "установившийся расчёт может не сходиться."),
    ("Поток (условия обтекания)", "FREESTREAM_PRESSURE",
     "Давление потока, Па", "double", None, (1000.0, 1e7, 1, 100.0),
     "Статическое давление набегающего потока. 101325 Па — стандартная "
     "атмосфера на уровне моря."),
    ("Поток (условия обтекания)", "FREESTREAM_TEMPERATURE",
     "Температура потока, K", "double", None, (100.0, 1000.0, 2, 1.0),
     "Температура набегающего потока. 288.15 K — стандартная атмосфера "
     "у земли."),

    # --- Геометрия / силы ---
    ("Геометрия и силы", "REF_LENGTH", "Характерная длина, м", "double",
     None, (0.001, 1000.0, 4, 0.001),
     "Масштаб длины для безразмерных коэффициентов (обычно средняя "
     "аэродинамическая хорда крыла)."),
    ("Геометрия и силы", "REF_AREA", "Площадь отсчёта, м²", "double",
     None, (0.001, 1e6, 4, 0.01),
     "Площадь, на которую нормируются силы (площадь крыла в плане). "
     "Влияет только на CL/CD/CM, не на сам поток."),
    ("Геометрия и силы", "REF_ORIGIN_MOMENT_X",
     "Точка момента X, м", "double", None, (-100.0, 100.0, 4, 0.01),
     "X-координата точки, относительно которой считается тангажный "
     "момент CM (обычно 25% хорды)."),

    # --- Маркеры ---
    ("Граничные условия (маркеры)", "MARKER_EULER",
     "Стенка (профиль/самолёт)", "str", None, None,
     "Имена границ сетки с условием «твёрдая стенка». Должны ДОСЛОВНО "
     "совпадать с MARKER_TAG в mesh.su2 (по умолчанию: airfoil).\n"
     "Если имя не совпадёт — SU2 напишет 'marker not found', граничное "
     "условие не наложится и расчёт разойдётся."),
    ("Граничные условия (маркеры)", "MARKER_FAR",
     "Дальнее поле", "str", None, None,
     "Имя границы дальнего поля в mesh.su2 (по умолчанию: farfield)."),

    # --- Численная схема ---
    ("Численная схема", "CONV_NUM_METHOD_FLOW", "Схема конвекции", "combo",
     ["ROE", "JST", "AUSM", "HLLC", "LAX-FRIEDRICH", "ROE_SW"], None,
     "ROE — стандарт, точнее на гладких потоках.\n"
     "JST — центрально-разностная с искусственной вязкостью: мягче и "
     "устойчивее на грубых/рваных сетках. Если ROE расходится — попробуй JST."),
    ("Численная схема", "MUSCL_FLOW", "2-й порядок (MUSCL)", "yesno",
     None, None,
     "YES — второй порядок точности: точнее поля и силы, но на грубой "
     "сетке возможны осцилляции и расхождение.\n"
     "NO — первый порядок: максимально устойчиво, решение более «вязкое». "
     "При отладке и расхождении ставить NO."),
    ("Численная схема", "SLOPE_LIMITER_FLOW", "Ограничитель наклона",
     "combo", ["VENKATAKRISHNAN", "BARTH_JESPERSEN", "NONE",
               "SHARP_EDGES", "WALL_DISTANCE"], None,
     "Гасит осцилляции у крутых градиентов (для MUSCL=YES).\n"
     "VENKATAKRISHNAN — мягкий, устойчивый (рекомендуется).\n"
     "BARTH_JESPERSEN — строже.\n"
     "NONE — без ограничителя: точнее на гладких решениях, но риск "
     "расхождения."),
    ("Численная схема", "VENKAT_LIMITER_COEFF",
     "К-т ограничителя Venkat", "double", None, (0.0, 1.0, 3, 0.01),
     "Меньше значение = сильнее подавление осцилляций (устойчивее, но "
     "решение «размазаннее»); больше = точнее, но риск взрыва невязки. "
     "Штатно 0.05."),
    ("Численная схема", "LOW_MACH_PREC", "Прекондиционер Roe-Turkel",
     "yesno", None, None,
     "Нужен, когда число Маха мало (M < 0.3, то есть примерно до 100 м/с).\n"
     "Сжимаемый решатель на таком режиме жёсток по акустике: звуковые волны\n"
     "идут в 1/M раз быстрее потока, поэтому невязка плотности встаёт и не\n"
     "падает, сколько ни считай. Прекондиционер снимает эту жёсткость.\n"
     "В SU2 по умолчанию выключено; имеет смысл только при схеме ROE."),
    ("Численная схема", "LOW_MACH_CORR", "Поправка Роу на низкий мах",
     "yesno", None, None,
     "Правит избыточную численную диссипацию схемы Роу после реконструкции\n"
     "MUSCL на малых числах Маха. Лишняя диссипация на низких махах — это\n"
     "завышенное сопротивление при почти правильной подъёмной силе.\n"
     "Имеет смысл включать вместе с LOW_MACH_PREC."),
    ("Численная схема", "ENTROPY_FIX_COEFF", "Энтропийная поправка",
     "double", None, (0.0, 1.0, 3, 0.01),
     "Добавляет диссипацию схеме Роу в окрестности звуковых/тормозных "
     "точек (нос профиля, зоны M≈1).\n"
     "0.05 — штатно; 0.1–0.2 — заметно устойчивее при расхождениях."),
    ("Численная схема", "NUM_METHOD_GRAD", "Метод градиентов", "combo",
     ["WEIGHTED_LEAST_SQUARES", "GREEN_GAUSS"], None,
     "Как считаются градиенты на неструктурированной сетке.\n"
     "WEIGHTED_LEAST_SQUARES — точнее (рекомендуется).\n"
     "GREEN_GAUSS — проще, грубее на скошенных ячейках."),

    # --- Время / CFL ---
    ("Шаг по времени (CFL)", "TIME_DISCRE_FLOW", "Схема по времени",
     "combo", ["EULER_IMPLICIT", "EULER_EXPLICIT",
               "RUNGE-KUTTA_EXPLICIT"], None,
     "EULER_IMPLICIT — неявная: допускает большие CFL, быстрая сходимость, "
     "устойчивее (рекомендуется).\n"
     "Явные схемы требуют очень малых CFL (≤1) и сотен тысяч итераций."),
    ("Шаг по времени (CFL)", "CFL_NUMBER", "Число CFL", "double",
     None, (0.01, 1000.0, 3, 0.1),
     "Размер шага по псевдовремени.\n"
     "Малое (0.5–2) — медленно, но устойчиво; большое (10–100) — быстро, "
     "но риск расхождения.\n"
     "ГЛАВНЫЙ регулятор: если невязка взрывается — снижать CFL."),
    ("Шаг по времени (CFL)", "CFL_ADAPT", "Автоподстройка CFL", "yesno",
     None, None,
     "YES — SU2 сам растит CFL при спокойной невязке и уменьшает при "
     "росте (быстро на хороших сетках).\n"
     "NO — фиксированный CFL_NUMBER: предсказуемо на капризных сетках "
     "(рекомендуется при отладке)."),

    # --- Линейный решатель ---
    ("Линейный решатель (неявный шаг)", "LINEAR_SOLVER",
     "Линейный решатель", "combo", ["FGMRES", "GMRES", "BCGSTAB"], None,
     "Решает линейную систему на каждом неявном шаге. FGMRES — стандарт "
     "для сжимаемых задач."),
    ("Линейный решатель (неявный шаг)", "LINEAR_SOLVER_PREC",
     "Предобуславливатель", "combo", ["ILU", "JACOBI", "LINELET"], None,
     "ILU — неполная LU-факторизация: мощный стандарт.\n"
     "JACOBI — простой, слабее.\n"
     "LINELET — для сильно вытянутых призматических ячеек у стенки."),
    ("Линейный решатель (неявный шаг)", "LINEAR_SOLVER_ERROR",
     "Точность лин. решателя", "double", None, (1e-10, 1e-1, 12, 1e-8),
     "Критерий сходимости линейного решателя на шаге. Меньше = точнее "
     "каждый шаг, но дороже. Штатно 1e-6."),
    ("Линейный решатель (неявный шаг)", "LINEAR_SOLVER_ITER",
     "Итераций лин. решателя", "int", None, (1, 200, 1),
     "Максимум итераций линейного решателя на одном нелинейном шаге. "
     "Больше = надёжнее на сложных сетках (10–20 штатно)."),

    # --- Сходимость ---
    ("Итерации и сходимость", "INNER_ITER", "Максимум итераций", "int",
     None, (10, 200000, 100),
     "Верхний предел нелинейных итераций. Расчёт остановится раньше, если "
     "невязка упадёт до целевого уровня."),
    ("Итерации и сходимость", "CONV_RESIDUAL_MINVAL",
     "Целевая невязка (log10)", "double", None, (-12.0, -1.0, 1, 0.5),
     "Целевое падение log10 невязки для авто-остановки. -6 — отличная "
     "сходимость; -4 обычно достаточно для стабильных сил CL/CD."),
    ("Итерации и сходимость", "CONV_STARTITER",
     "Итераций до контроля", "int", None, (0, 1000, 10),
     "С какой итерации SU2 начинает проверять критерий сходимости "
     "(чтобы транзиент старта не считался)."),
]


# ---------------------------------------------------------------------------
# Официальные пресеты SU2 (из встроенных официальных config.cfg)
# ---------------------------------------------------------------------------
# Ключи, которые диалог умеет показывать/применять. Для официального пресета
# берём только их — в официальном config.cfg сотни строк (в т.ч. DV_*,
# FFD_* для оптимизации), которые диалог не умеет показать и применять
# не должен (они меняют постановку, а не численную схему).
_DIALOG_PARAM_KEYS = {row[1] for row in PARAMS}


def read_config_text_keys(text: str) -> dict:
    """Разбирает текст config.cfg в словарь активных ``KEY= value``.

    Отличается от ``read_config`` тем, что принимает строку, а не файл, и
    работает с '%' как единственным комментарием SU2 (как и сам SU2).
    """
    out = {}
    for raw in (text or "").splitlines():
        s = raw.split("%", 1)[0].strip()
        if not s or "=" not in s:
            continue
        key, _, val = s.partition("=")
        k = key.strip()
        if k:
            out[k] = val.strip()
    return out


def _map_official_cfg_to_preset(text: str) -> dict:
    """Собирает из официального config.cfg пресет только из диалоговых ключей."""
    cfg = read_config_text_keys(text)
    return {k: v for k, v in cfg.items() if k in _DIALOG_PARAM_KEYS}


def _build_official_presets():
    """Наполняет ``OFFICIAL_PRESETS`` из встроенных официальных кейсов SU2.

    Каждый пресет — это настройки, которые реально стоят в официальном
    config.cfg, но только те, которые диалог умеет применить. Значения
    «заморожены» на момент импорта: правки официальных файлов в
    ``official_cases/configs/`` подхватятся при следующем запуске.
    """
    if official_cases is None:
        return
    try:
        for cid in official_cases.list_cases():
            case = official_cases.get_case(cid)
            text = official_cases.bundled_config_text(case.config_file)
            params = _map_official_cfg_to_preset(text)
            if not params:
                continue
            name = "Официальный: %s" % case.name
            OFFICIAL_PRESETS[name] = {
                "case_id": cid,
                "description": case.description,
                "solver": case.solver,
                "params": params,
            }
    except Exception:                                        # pragma: no cover
        OFFICIAL_PRESETS.clear()


# Заполняем после того, как PARAMS определён.
_build_official_presets()


# ---------------------------------------------------------------------------
# Чтение/запись config.cfg (активные строки KEY= value; '%' - комментарий SU2)
# ---------------------------------------------------------------------------

def read_config(path):
    cfg = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("%") or s.startswith("#"):
                    continue
                if "=" not in s:
                    continue
                key, _, val = s.partition("=")
                cfg[key.strip()] = val.strip()
    except OSError:
        pass
    return cfg


def write_config_values(path, values):
    """Пишет значения в существующий config.cfg (строки на месте),
    отсутствующие ключи добавляет блоком в конец."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines(keepends=True)
    except OSError:
        lines = []

    active_pat = {k: re.compile(r'^[ \t]*' + re.escape(k) + r'[ \t]*=.*$')
                  for k in values}
    done = set()
    out = []
    for ln in lines:
        bare = ln.rstrip("\r\n")
        matched = None
        for k, pat in active_pat.items():
            if k not in done and pat.match(bare):
                matched = k
                break
        if matched is not None:
            out.append(f"{matched}= {values[matched]}\n")
            done.add(matched)
        else:
            out.append(ln if ln.endswith("\n") else ln + "\n")

    missing = [k for k in values if k not in done]
    if missing:
        # Комментарий в config.cfg у SU2 - только '%' (не '#'): строка с '#'
        # и без '=' роняет SU2 в TokenizeString().
        out.append("\n% ===== AeroOpt: дополнительные параметры =====\n")
        for k in missing:
            out.append(f"{k}= {values[k]}\n")

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.writelines(out)
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, AttributeError):
            pass
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Пользовательские пресеты
# ---------------------------------------------------------------------------

def load_user_presets():
    try:
        with open(PRESETS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): dict(v) for k, v in data.items()
                    if isinstance(v, dict)}
    except (OSError, ValueError):
        pass
    return {}


def save_user_presets(presets):
    try:
        with open(PRESETS_JSON, "w", encoding="utf-8") as f:
            json.dump(presets, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def all_presets():
    """Все пресеты: встроенные + официальные + пользовательские.

    Официальные пресеты раскрываются до плоского ``{ключ: значение}``,
    чтобы вызывающий код (`_apply_preset_to_fields`) работал одинаково.
    Метаданные (описание, SOLVER, id кейса) лежат в ``OFFICIAL_PRESETS`` и
    доступны через ``official_preset_meta``.
    """
    p = dict(BUILTIN_PRESETS)
    for name, meta in OFFICIAL_PRESETS.items():
        p[name] = dict(meta["params"])
    p.update(load_user_presets())
    return p


def official_preset_meta(name):
    """Метаданные официального пресета или None."""
    return OFFICIAL_PRESETS.get(name)


def _is_reserved_preset(name):
    """Встроенные (ultra/safe) и официальные пресеты нельзя переименовывать
    или удалять — но задать поля по ним можно."""
    return name in BUILTIN_PRESETS or name in OFFICIAL_PRESETS


# ---------------------------------------------------------------------------
# Диалог настроек
# ---------------------------------------------------------------------------

class Su2ConfigDialog(QDialog):
    def __init__(self, config_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки расчёта SU2 — AeroOpt")
        self.resize(720, 720)
        self.config_path = config_path
        self.widgets = {}   # key -> (widget, kind)
        self._build_ui()
        self._load_into_widgets()

    # ---- построение интерфейса ----
    def _build_ui(self):
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.path_label = QLabel("")
        self.path_label.setStyleSheet("color: #666;")
        top.addWidget(self.path_label, 1)
        btn_open = QPushButton("Открыть другой config.cfg…")
        btn_open.clicked.connect(self._pick_config)
        top.addWidget(btn_open)
        root.addLayout(top)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)

        groups = {}
        for group, key, label, kind, options, limits, tip in PARAMS:
            if group not in groups:
                box = QGroupBox(group)
                form = QFormLayout(box)
                form.setLabelAlignment(Qt.AlignRight)
                groups[group] = (box, form)
                inner_layout.addWidget(box)
            box, form = groups[group]
            w = self._make_widget(kind, options, limits)
            w.setToolTip(tip)
            lbl = QLabel(label + ":")
            lbl.setToolTip(tip)
            lbl.setMinimumWidth(190)
            form.addRow(lbl, w)
            self.widgets[key] = (w, kind)

        inner_layout.addStretch(1)
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        # ---- строка пресетов ----
        pres = QHBoxLayout()
        pres.addWidget(QLabel("Пресет:"))
        self.preset_combo = QComboBox()
        self._reload_preset_combo()
        pres.addWidget(self.preset_combo, 1)
        b_load = QPushButton("Загрузить")
        b_load.setToolTip("Применить значения выбранного пресета к полям "
                          "(config.cfg изменится после «Сохранить»).")
        b_load.clicked.connect(self._apply_preset_to_fields)
        pres.addWidget(b_load)
        b_save = QPushButton("Сохранить как мой пресет…")
        b_save.setToolTip("Сохранить текущие значения полей под именем — "
                          "пресет будет доступен во всех проектах.")
        b_save.clicked.connect(self._save_current_as_preset)
        pres.addWidget(b_save)
        b_del = QPushButton("Удалить мой пресет")
        b_del.setToolTip("Удалить выбранный пользовательский пресет. "
                         "Встроенные ultra и safe не удаляются.")
        b_del.clicked.connect(self._delete_user_preset)
        pres.addWidget(b_del)
        b_ren = QPushButton("Переименовать…")
        b_ren.setToolTip("Переименовать выбранный пользовательский пресет.")
        b_ren.clicked.connect(self._rename_user_preset)
        pres.addWidget(b_ren)
        b_exp = QPushButton("Экспорт…")
        b_exp.setToolTip("Выгрузить выбранный пресет в файл .su2preset, "
                         "чтобы перенести на другую машину.")
        b_exp.clicked.connect(self._export_preset)
        pres.addWidget(b_exp)
        root.addLayout(pres)

        # Какие именно ключи записаны в выбранном пресете.
        self.lbl_preset_params = QLabel("")
        self.lbl_preset_params.setWordWrap(True)
        self.lbl_preset_params.setTextInteractionFlags(
            Qt.TextSelectableByMouse)
        self.lbl_preset_params.setStyleSheet(
            "color: #4A4A4A; font-size: 10px; font-family: Consolas, monospace;")
        self.preset_combo.currentTextChanged.connect(self._show_preset_params)
        root.addWidget(self.lbl_preset_params)
        self._show_preset_params()

        # ---- официальные кейсы SU2 (конфиги + 3D-сетки) ----
        self._build_official_group(root)

        # ---- кнопки ----
        btns = QHBoxLayout()
        btns.addStretch(1)
        b_cancel = QPushButton("Отмена")
        b_cancel.clicked.connect(self.reject)
        btns.addWidget(b_cancel)
        b_ok = QPushButton("Сохранить в config.cfg")
        b_ok.setDefault(True)
        b_ok.clicked.connect(self._save_and_close)
        btns.addWidget(b_ok)
        root.addLayout(btns)

    def _make_widget(self, kind, options, limits):
        if kind == "yesno":
            w = QComboBox()
            w.addItems(["YES", "NO"])
            return w
        if kind == "combo":
            w = QComboBox()
            w.addItems(options)
            w.setEditable(True)
            return w
        if kind == "int":
            w = QSpinBox()
            lo, hi, step = limits
            w.setRange(int(lo), int(hi))
            w.setSingleStep(int(step))
            w.setMaximum(200000)
            return w
        if kind == "double":
            w = QDoubleSpinBox()
            lo, hi, decimals, step = limits
            w.setRange(lo, hi)
            w.setDecimals(int(decimals))
            w.setSingleStep(step)
            w.setGroupSeparatorShown(False)
            return w
        return QLineEdit()  # 'str'

    # ---- загрузка/сохранение значений ----
    def _load_into_widgets(self):
        self.cfg = read_config(self.config_path)
        self.path_label.setText(self.config_path)
        for key, (w, kind) in self.widgets.items():
            if key not in self.cfg:
                continue
            val = self.cfg[key]
            try:
                if kind == "yesno":
                    idx = 0 if val.strip().upper().startswith("Y") else 1
                    w.setCurrentIndex(idx)
                elif kind == "combo":
                    items = [w.itemText(i) for i in range(w.count())]
                    up = val.strip()
                    match = next((x for x in items if x.upper() == up.upper()),
                                 None)
                    if match:
                        w.setCurrentText(match)
                    else:
                        w.setEditText(up)
                elif kind == "int":
                    w.setValue(int(float(str(val).split()[0])))
                elif kind == "double":
                    w.setValue(float(str(val).split()[0]))
                else:
                    w.setText(val)
            except (ValueError, TypeError):
                if kind == "str":
                    w.setText(val)

    def _collect_values(self):
        out = {}
        for key, (w, kind) in self.widgets.items():
            if kind == "yesno":
                out[key] = w.currentText()
            elif kind == "combo":
                out[key] = w.currentText().strip()
            elif kind in ("int", "double"):
                out[key] = str(w.value())
            else:
                out[key] = w.text().strip()
        return out

    def _save_and_close(self):
        values = self._collect_values()
        try:
            write_config_values(self.config_path, values)
        except OSError as e:
            QMessageBox.critical(self, "Настройки SU2",
                                 f"Не удалось записать config.cfg:\n{e}")
            return
        QMessageBox.information(
            self, "Настройки SU2",
            "config.cfg сохранён.\n\n"
            "Следующий расчёт пойдёт с этими настройками.\n"
            "Откат к исходному — меню SU2 → Восстановить config.cfg.orig.")
        self.accept()

    # ---- пресеты ----
    def _reload_preset_combo(self):
        self.preset_combo.clear()
        self.preset_combo.addItems(sorted(all_presets().keys()))

    def _apply_preset_to_fields(self):
        name = self.preset_combo.currentText()
        presets = all_presets()
        if name not in presets:
            return
        for key, val in presets[name].items():
            if key not in self.widgets:
                continue
            w, kind = self.widgets[key]
            try:
                if kind == "yesno":
                    w.setCurrentIndex(0 if str(val).upper().startswith("Y")
                                      else 1)
                elif kind == "combo":
                    w.setEditText(str(val))
                elif kind == "int":
                    w.setValue(int(float(str(val).split()[0])))
                elif kind == "double":
                    w.setValue(float(str(val).split()[0]))
                else:
                    w.setText(str(val))
            except (ValueError, TypeError):
                pass
        QMessageBox.information(
            self, "Пресет",
            f"Пресет «{name}» загружен в поля.\n"
            f"Нажми «Сохранить в config.cfg», чтобы применить.")

    def _save_current_as_preset(self):
        name, ok = QInputDialog.getText(
            self, "Мой пресет", "Название пресета:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in BUILTIN_PRESETS:
            QMessageBox.warning(self, "Мой пресет",
                                "Это имя зарезервировано за встроенным "
                                "пресетом. Выбери другое.")
            return
        if name in OFFICIAL_PRESETS:
            QMessageBox.warning(self, "Мой пресет",
                                "Это имя занято официальным пресетом SU2. "
                                "Выбери другое.")
            return
        presets = load_user_presets()
        presets[name] = self._collect_values()
        save_user_presets(presets)
        self._reload_preset_combo()
        self.preset_combo.setCurrentText(name)
        QMessageBox.information(self, "Мой пресет",
                                f"Пресет «{name}» сохранён.")

    def _show_preset_params(self, name=None):
        """Показывает, какие ключи записаны в выбранном пресете."""
        name = name if name is not None else self.preset_combo.currentText()
        p = all_presets().get(name)
        if not p:
            self.lbl_preset_params.setText("")
            return
        lines = "\n".join("  %s= %s" % (k, p[k]) for k in sorted(p))
        if name in BUILTIN_PRESETS:
            info = {}
            try:
                info = su2_autoconfig.PRESET_INFO.get(name, ("", ""))
            except Exception:
                info = ("", "")
            head = "Встроенный «%s» — %s. Изменить нельзя: сохраните под своим именем." % (
                name, info[0] or "пресет")
            if info[1]:
                head += "\n" + info[1]
        elif official_preset_meta(name) is not None:
            meta = official_preset_meta(name)
            head = "Официальный пресет «%s» — %s. Настройки взяты из " \
                   "официального config.cfg кейса '%s'. Изменить нельзя: " \
                   "сохраните под своим именем." % (
                name, meta["solver"], meta["case_id"])
            if meta["description"]:
                head += "\n" + meta["description"]
        else:
            head = "Мой пресет «%s»." % name
        self.lbl_preset_params.setText("%s\nКлючей: %d\n%s" % (head, len(p), lines))

    def _rename_user_preset(self):
        """Переименовывает пользовательский пресет."""
        name = self.preset_combo.currentText()
        if name in BUILTIN_PRESETS:
            QMessageBox.information(self, "Переименование",
                                    "Встроенные пресеты ultra и safe "
                                    "переименовывать нельзя.")
            return
        if name in OFFICIAL_PRESETS:
            QMessageBox.information(self, "Переименование",
                                    "Официальные пресеты SU2 переименовывать "
                                    "нельзя — они соответствуют эталонным "
                                    "config.cfg.")
            return
        presets = load_user_presets()
        if name not in presets:
            return
        new, ok = QInputDialog.getText(self, "Переименовать пресет",
                                       "Новое название:", text=name)
        if not ok:
            return
        new = new.strip()
        if not new or new == name:
            return
        if new in BUILTIN_PRESETS or new in OFFICIAL_PRESETS or new in presets:
            QMessageBox.warning(self, "Переименование",
                                "Имя «%s» уже занято." % new)
            return
        presets[new] = presets.pop(name)
        save_user_presets(presets)
        self._reload_preset_combo()
        self.preset_combo.setCurrentText(new)

    def _export_preset(self):
        """Выгружает выбранный пресет в .su2preset."""
        name = self.preset_combo.currentText()
        p = all_presets().get(name)
        if not p:
            return
        if su2_preset_format is None:
            QMessageBox.warning(self, "Экспорт пресета",
                                "Модуль su2_preset_format не загружен.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт пресета", name + su2_preset_format.EXTENSION,
            "Пресет AeroOpt (*" + su2_preset_format.EXTENSION + ")")
        if not path:
            return
        try:
            su2_preset_format.export_preset(
                path, name, dict(p),
                description="Экспорт из AeroOpt",
                based_on=name if name in BUILTIN_PRESETS else None)
        except Exception as e:
            QMessageBox.critical(self, "Экспорт пресета",
                                 "Не удалось записать файл:\n%s" % e)
            return
        QMessageBox.information(self, "Экспорт пресета",
                                "Пресет «%s» сохранён:\n%s" % (name, path))

    def _delete_user_preset(self):
        name = self.preset_combo.currentText()
        if name in BUILTIN_PRESETS:
            QMessageBox.information(
                self, "Мой пресет",
                "Встроенные пресеты удалять нельзя.")
            return
        if name in OFFICIAL_PRESETS:
            QMessageBox.information(
                self, "Мой пресет",
                "Официальные пресеты SU2 удалять нельзя.")
            return
        presets = load_user_presets()
        if name in presets:
            del presets[name]
            save_user_presets(presets)
            self._reload_preset_combo()

    # ---- официальные кейсы SU2 (эталонные конфиги и 3D-сетки) ----
    def _build_official_group(self, root):
        """Добавляет группу «Официальные кейсы SU2» в диалог.

        Полностью вспомогательная: не меняет существующие пресеты и поля,
        а даёт доступ к официальным config.cfg SU2, скачиванию официальных
        3D-сеток и диагностике «неправдоподобных значений».
        """
        box = QGroupBox("Официальные кейсы SU2 (эталонные конфиги)")
        lay = QVBoxLayout(box)

        if official_cases is None:
            lay.addWidget(QLabel(
                "Пакет official_cases не загружен — эталонные кейсы SU2 "
                "недоступны."))
            root.addWidget(box)
            return

        row = QHBoxLayout()
        row.addWidget(QLabel("Кейс:"))
        self.official_combo = QComboBox()
        self._reload_official_combo()
        self.official_combo.currentTextChanged.connect(
            self._on_official_selected)
        row.addWidget(self.official_combo, 1)
        lay.addLayout(row)

        self.lbl_official = QLabel("")
        self.lbl_official.setWordWrap(True)
        self.lbl_official.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_official.setStyleSheet(
            "color: #4A4A4A; font-size: 10px; "
            "font-family: Consolas, monospace;")
        lay.addWidget(self.lbl_official)
        self._on_official_selected()

        btns = QHBoxLayout()
        b_dl = QPushButton("Скачать 3D-сетку")
        b_dl.setToolTip("Скачать официальную сетку SU2 (3D-модель) в "
                        "official_cases/meshes/.")
        b_dl.clicked.connect(self._on_official_download)
        btns.addWidget(b_dl)
        b_wr = QPushButton("Записать конфиг рядом…")
        b_wr.setToolTip("Сохранить официальный config.cfg в каталог "
                        "текущего config.cfg (файл official_<id>.cfg), не "
                        "трогая ваш config.cfg.")
        b_wr.clicked.connect(self._on_official_write)
        btns.addWidget(b_wr)
        b_cmp = QPushButton("Сравнить с моим")
        b_cmp.setToolTip("Разобрать ваш config.cfg и объяснить, почему "
                         "значения могут быть неправдоподобно большими.")
        b_cmp.clicked.connect(self._on_official_compare)
        btns.addWidget(b_cmp)
        lay.addLayout(btns)

        root.addWidget(box)

    def _reload_official_combo(self):
        if official_cases is None:
            return
        self.official_combo.clear()
        for cid in official_cases.list_cases():
            case = official_cases.get_case(cid)
            self.official_combo.addItem("%s — %s" % (cid, case.name), cid)

    def _selected_official_id(self):
        if official_cases is None:
            return None
        return self.official_combo.currentData()

    def _on_official_selected(self):
        cid = self._selected_official_id()
        if not cid:
            return
        try:
            case = official_cases.get_case(cid)
        except Exception:
            self.lbl_official.setText("")
            return
        parts = [
            "%s (%sD)" % (case.solver, case.dimension),
        ]
        if case.mach is not None:
            parts.append("M=%.3f" % case.mach)
        if case.aoa is not None:
            parts.append("AoA=%.2f°" % case.aoa)
        if case.reynolds is not None:
            parts.append("Re=%.3g" % case.reynolds)
        lines = [case.name]
        lines.append(" · ".join(parts) if parts else "")
        if case.ref_cl is not None:
            lines.append("Эталон SU2 (итер. %s): CL=%s, CD=%s" % (
                case.ref_iter or "?", case.ref_cl, case.ref_cd))
        mesh_note = "Сетка: %s" % case.mesh_filename if case.mesh_filename else \
            "Сетка: нет"
        if case.mesh_size:
            mesh_note += " (%s байт)" % case.mesh_size
        lines.append(mesh_note)
        if case.notes:
            lines.append("  " + case.notes)
        self.lbl_official.setText("\n".join(lines))

    def _on_official_download(self):
        cid = self._selected_official_id()
        if not cid:
            return
        try:
            path = official_cases.download_mesh(cid)
        except Exception as e:
            QMessageBox.critical(
                self, "Официальные кейсы SU2",
                "Не удалось скачать сетку «%s»:\n%s\n\n"
                "Нужен доступ к api.github.com (сетки SU2 качаются по "
                "требованию)." % (cid, e))
            return
        QMessageBox.information(
            self, "Официальные кейсы SU2",
            "Сетка скачана:\n%s\n\n"
            "Положите её рядом с config.cfg кейса, чтобы SU2 её нашёл." % path)

    def _on_official_write(self):
        cid = self._selected_official_id()
        if not cid:
            return
        try:
            case = official_cases.get_case(cid)
        except Exception as e:
            QMessageBox.critical(self, "Официальные кейсы SU2", str(e))
            return
        case_dir = os.path.dirname(os.path.abspath(self.config_path)) \
            if self.config_path else os.getcwd()
        out = os.path.join(case_dir, "official_%s.cfg" % cid)
        text = official_cases.bundled_config_text(case.config_file)
        try:
            with open(out, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
        except OSError as e:
            QMessageBox.critical(self, "Официальные кейсы SU2",
                                 "Не удалось записать файл:\n%s" % e)
            return
        QMessageBox.information(
            self, "Официальные кейсы SU2",
            "Официальный config.cfg записан:\n%s\n\n"
            "Ваш config.cfg не изменён. Этот файл можно открыть и сравнить "
            "с текущим (например, через «Сравнить с моим»)." % out)

    def _on_official_compare(self):
        if not self.config_path or not os.path.exists(self.config_path):
            QMessageBox.warning(self, "Официальные кейсы SU2",
                                "Сначала откройте config.cfg кейса.")
            return
        try:
            res = official_cases.compare_file(self.config_path)
        except Exception as e:
            QMessageBox.critical(self, "Официальные кейсы SU2",
                                 "Не удалось разобрать config.cfg:\n%s" % e)
            return
        QMessageBox.information(
            self, "Официальные кейсы SU2",
            official_cases.render_diagnosis(res))

    # ---- прочее ----
    def _pick_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбери config.cfg", "", "config.cfg (config.cfg);;Все файлы (*)")
        if path:
            self.config_path = path
            self._load_into_widgets()


# ---------------------------------------------------------------------------
# Поиск свежего config.cfg (work/case_*/aoa_*/config.cfg)
# ---------------------------------------------------------------------------

def find_latest_config():
    roots = []
    cwd = os.getcwd()
    for pattern in ("work", os.path.join("dist", "AeroOpt", "work")):
        roots.append(os.path.join(cwd, pattern))
    candidates = []
    for r in roots:
        candidates += glob.glob(os.path.join(
            r, "case_*", "aoa_*", "config.cfg"))
        candidates += glob.glob(os.path.join(r, "case_*", "config.cfg"))
    if not candidates:
        return None
    try:
        return max(candidates, key=os.path.getmtime)
    except OSError:
        return candidates[-1]


def _ask_config(parent, title="Выбери config.cfg кейса"):
    latest = find_latest_config()
    start = latest or os.getcwd()
    path, _ = QFileDialog.getOpenFileName(
        parent, title, start, "config.cfg (config.cfg);;Все файлы (*)")
    return path or None


# ---------------------------------------------------------------------------
# Авто-предложение после падения расчёта (вызывать из раннера SU2)
# ---------------------------------------------------------------------------

def offer_recovery_after_failure(parent, case_dir, screen_text=""):
    """Показать диалог восстановления после ошибки/расхождения SU2.

    Возвращает:
      'rerun_safe'   - применён устойчивый пресет, раннер должен
                       перезапустить SU2 в той же папке;
      'settings'     - пользователь открыл ручные настройки (сам решит);
      'rerun_ultra'  - safe уже был, применён ultra;
      'abort'/'none' - ничего не делать.
    """
    res = su2_autoconfig.detect_result(case_dir, screen_text)
    if res["status"] == "converged":
        return "none"

    detail = res.get("detail", "Расчёт не сошёлся.")
    box = QMessageBox(parent)
    box.setWindowTitle("Расчёт не сошёлся")
    box.setIcon(QMessageBox.Warning)
    box.setText(
        f"<b>{detail}</b>\n\n"
        "Что можно сделать:\n"
        "• <b>Устойчивый пресет</b> — CFL 2.0 без автоподстройки, 1-й "
        "порядок, усиленная энтропийная поправка: в большинстве случаев "
        "расхождение уходит (расчёт медленнее, но доходит до конца).\n"
        "• <b>Настройки SU2…</b> — открыть полный диалог со всеми "
        "параметрами и подсказками.\n"
        "• Если не поможет даже ultra-пресет — дело в сетке: перегенерируй "
        "её качеством «Точная».")
    box.setTextFormat(Qt.RichText)

    b_safe = box.addButton("Применить устойчивый пресет и пересчитать",
                           QMessageBox.AcceptRole)
    b_set = box.addButton("Открыть настройки SU2…", QMessageBox.ActionRole)
    box.addButton("Отмена", QMessageBox.RejectRole)
    box.exec_()

    clicked = box.clickedButton()
    if clicked is b_set:
        cfg = os.path.join(case_dir, "config.cfg")
        dlg = Su2ConfigDialog(cfg, parent)
        dlg.exec_()
        return "settings"
    if clicked is b_safe:
        cfg = os.path.join(case_dir, "config.cfg")
        # Если safe уже применялся (есть .orig и в конфиге стоит CFL 2.0
        # без адаптации) - сразу даём ultra.
        cur = read_config(cfg)
        already_safe = (cur.get("CFL_ADAPT", "").upper() == "NO"
                        and cur.get("MUSCL_FLOW", "").upper() == "NO")
        preset = "ultra" if already_safe else "safe"
        try:
            su2_autoconfig.apply_preset(cfg, preset)
        except Exception as e:
            QMessageBox.critical(parent, "Автоконфиг",
                                 f"Не удалось применить пресет:\n{e}")
            return "abort"
        QMessageBox.information(
            parent, "Автоконфиг",
            f"Применён пресет «{preset}». Перезапускаю расчёт.\n"
            "Оригинальный config.cfg сохранён как config.cfg.orig.")
        return "rerun_ultra" if preset == "ultra" else "rerun_safe"
    return "abort"


# ---------------------------------------------------------------------------
# Установка меню в главное окно (одна строка интеграции)
# ---------------------------------------------------------------------------

def install_menu(main_window):
    """Добавляет меню «SU2» в QMainWindow. Безопасно вызывать один раз."""
    try:
        menubar = main_window.menuBar()
    except AttributeError:
        return None

    menu = QMenu("SU2", main_window)

    def _open_settings():
        cfg = _ask_config(main_window)
        if cfg:
            Su2ConfigDialog(cfg, main_window).exec_()

    def _apply_safe():
        cfg = _ask_config(main_window)
        if not cfg:
            return
        _, changes = su2_autoconfig.apply_preset(cfg, "safe")
        QMessageBox.information(
            main_window, "Устойчивый пресет",
            "Применено к:\n" + cfg + "\n\n" + "\n".join(changes) +
            "\n\nОригинал: config.cfg.orig")

    def _restore():
        cfg = _ask_config(main_window)
        if not cfg:
            return
        ok = su2_autoconfig.restore_original(cfg)
        QMessageBox.information(
            main_window, "Откат",
            "config.cfg.orig восстановлен." if ok
            else "Бэкап config.cfg.orig не найден (пресет не применялся?).")

    act_settings = QAction("Настройки расчёта SU2…", main_window)
    act_settings.setToolTip("Открыть config.cfg с подсказками по параметрам "
                            "и пресетами.")
    act_settings.triggered.connect(_open_settings)
    menu.addAction(act_settings)

    act_safe = QAction("Устойчивый пресет (safe) для кейса…", main_window)
    act_safe.triggered.connect(_apply_safe)
    menu.addAction(act_safe)

    act_restore = QAction("Восстановить исходный config.cfg", main_window)
    act_restore.triggered.connect(_restore)
    menu.addAction(act_restore)

    menubar.addMenu(menu)
    return menu


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    cfg = find_latest_config() or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not cfg:
        cfg, _ = QFileDialog.getOpenFileName(
            None, "Выбери config.cfg", os.getcwd(),
            "config.cfg (config.cfg);;Все файлы (*)")
    if cfg:
        Su2ConfigDialog(cfg).exec_()
