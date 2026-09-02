"""
ui/main_window.py — главное окно AeroOpt v4.0 (патч-версия).

Изменения относительно исходного файла (без потери функционала):
 1. Полётные условия (скорость/высота/AoA/ISA/пресеты) перенесены в
    config/flight_conditions.py. Узел дерева 'Flight Conditions' удалён.
    Вся информация теперь отображается в 'Global Definitions' как
    QGroupBox 'Полётные условия'.
 2. Убран узел 'Active Parts' в дереве (Active Parts больше не дублируют
    таблицу bodies_table).
 3. Все вызовы self.plotter.reset_camera() удалены — камера больше НЕ
    перескакивает при действиях пользователя.
 4. Добавлена кнопка 'Готово: Применить' для выбора количества ядер CPU.
    Значение применяется только после нажатия.
 5. Память отображается через ctypes (Windows) / resource (Unix) —
    без зависимости от psutil.
 6. Длинные подсказки в QFormLayout (Fuselage/Wing/Stabilizers) перенесены
    в setToolTip каждого виджета; текст лейблов сокращён.

Все остальные классы и методы (CADNavigationEventFilter, SU2FirstLaunchDialog,
SU2InstallWorker, AeroPlotCanvas, оптимизация, sweep, mesh, лечение STL,
балансировка, экспорт CSV/Paraview, диалоги, ...) сохранены без изменений.
"""

from __future__ import annotations

import os
import sys
import csv
import math
import json
import time
import shutil
import ctypes
# `resource` существует только на Unix; на Windows PyInstaller-бандл падает
# на этом импорте, поэтому импортируем опционально.
try:
    import resource as _resource  # type: ignore
except ImportError:  # Windows и прочие без POSIX resource
    _resource = None
from datetime import datetime

# High DPI (ТЗ 2.4 / п.10): включаем до создания QApplication
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QGroupBox, QFileDialog, QMessageBox,
    QProgressBar, QTableWidget, QTableWidgetItem, QComboBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QFormLayout, QRadioButton, QButtonGroup,
    QTabWidget, QLineEdit, QDialog, QDialogButtonBox, QMenu,
    QSplitter, QScrollArea, QTreeWidget, QTreeWidgetItem, QStackedWidget,
    QToolTip, QSlider, QInputDialog, QSizePolicy, QFrame,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject, QLockFile, QStandardPaths
from PyQt5.QtGui import QColor, QFont, QMouseEvent

from config.settings import (
    ROLES, ROLE_COLORS, MESH_QUALITY, WORK_DIR_BASE, RESULTS_DIR,
    MESH_FILE, PREVIEW_MESH, config,
)
from config.flight_conditions import (
    FlightConditions,
    FLIGHT_PRESETS,
    list_flight_presets,
)
from optimization.rules import (
    Rule, RuleSet, RuleType, RuleOperator, RuleSeverity, PRESETS,
)
from physics.atmosphere import isa_atmosphere, sutherland_viscosity
from physics.airfoils import generate_naca4_section
from geometry.stl_healer import heal_stl_mesh, HealReportDialog
from geometry.generators import (
    create_primitive, generate_wing_mesh, generate_flaps_mesh,
    generate_slats_mesh, cad_to_stl, CAD_EXTENSIONS,
)
from mesh.gmsh_generator import generate_mesh_impl
from mesh.mesh_worker import MeshWorker, MeshAdaptWorker
from solver.workers import (
    SU2Worker, SweepWorker, OptimizationWorker, SessionRunner,
    hidden_subprocess_kwargs, _mesh_npoin,
)
from solver.session import CalculationSession
# === T6: лицензирование (опционально — не падает, если модуль недоступен)
_LICENSE_ERROR = ""
try:
    from license_client.license_checker import LicenseChecker, LicenseStatus
    _LICENSE_AVAILABLE = True
    print("[AeroOpt] license_client import OK")
except Exception as _lic_err:
    _LICENSE_AVAILABLE = False
    LicenseChecker = None
    LicenseStatus = None
    _LICENSE_ERROR = str(_lic_err)
    print(f"[AeroOpt] license_checker ERROR: {_lic_err}")
    import traceback
    traceback.print_exc()
# ====================================================================
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

SU2_EXE = config.su2_exe


# ---------------------------------------------------------------------------
# ТЗ п.6 / 5 — CAD-навигация: ПКМ = Pan, СКМ = Zoom, ЛКМ = вращение
# ---------------------------------------------------------------------------
class CADNavigationEventFilter(QObject):
    """Фильтр событий для переопределения навигации в 3D (ТЗ 5):

    ЛКМ — вращение; ПКМ — панорама (превращаем в среднюю кнопку VTK);
    СКМ/колесо — зум (превращаем в правую кнопку VTK)."""

    def __init__(self, interactor):
        super().__init__()
        self.interactor = interactor
        self._inside = False  # защита от рекурсии

    def eventFilter(self, obj, event):
        if self._inside:
            return False
        if event.type() in (QMouseEvent.MouseButtonPress,
                            QMouseEvent.MouseButtonRelease,
                            QMouseEvent.MouseMove):
            if hasattr(event, "button") and hasattr(event, "buttons"):
                button = event.button()
                buttons = event.buttons()
                new_button = button
                if button == Qt.RightButton:
                    new_button = Qt.MiddleButton
                elif button == Qt.MiddleButton:
                    new_button = Qt.RightButton
                new_buttons = buttons
                if buttons & Qt.RightButton:
                    new_buttons = (buttons & ~Qt.RightButton) | Qt.MiddleButton
                elif buttons & Qt.MiddleButton:
                    new_buttons = (buttons & ~Qt.MiddleButton) | Qt.RightButton
                try:
                    new_event = QMouseEvent(
                        event.type(), event.localPos(), event.windowPos(),
                        event.screenPos(), new_button, new_buttons,
                        event.modifiers(), event.source())
                except TypeError:
                    new_event = QMouseEvent(
                        event.type(), event.localPos(), event.windowPos(),
                        event.screenPos(), new_button, new_buttons,
                        event.modifiers())
                self._inside = True
                try:
                    QApplication.sendEvent(obj, new_event)
                finally:
                    self._inside = False
                return True
        return super().eventFilter(obj, event)


# ---------------------------------------------------------------------------
# Диалог первого запуска (ТЗ 2.2)
# ---------------------------------------------------------------------------
class SU2FirstLaunchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AeroOpt v4.0 — Первоначальная настройка решателя")
        self.setMinimumWidth(550)
        self.setMinimumHeight(280)
        layout = QVBoxLayout(self)
        banner = QLabel("Добро пожаловать в AeroOpt v4.0!")
        banner.setStyleSheet("font-size: 16px; font-weight: bold; color: #22384A;")
        layout.addWidget(banner)
        desc = QLabel(
            "Для проведения численного аэродинамического анализа (CFD) программе требуется\n"
            "расчетный модуль SU2 (SU2_CFD.exe). Пожалуйста, выберите действие:")
        desc.setStyleSheet("font-size: 11px; line-height: 1.4;")
        layout.addWidget(desc)
        layout.addSpacing(15)
        self.btn_auto = QPushButton("СКАЧАТЬ И УСТАНОВИТЬ АВТОМАТИЧЕСКИ (Рекомендуется)")
        self.btn_auto.setStyleSheet("background-color: #2E5A78; color: #FBFBFC; font-weight: bold; font-size: 11px; padding: 10px;")
        self.btn_auto.clicked.connect(self.on_auto_clicked)
        layout.addWidget(self.btn_auto)
        self.btn_manual = QPushButton("Указать путь к существующему SU2_CFD.exe на диске")
        self.btn_manual.setStyleSheet("background-color: #2E5A78; color: #FBFBFC; font-weight: bold; font-size: 11px; padding: 10px;")
        self.btn_manual.clicked.connect(self.on_manual_clicked)
        layout.addWidget(self.btn_manual)
        layout.addSpacing(15)
        self.lbl_status = QLabel("Статус: не настроено")
        self.lbl_status.setStyleSheet("color: #9B2C2C; font-weight: bold;")
        layout.addWidget(self.lbl_status)
        self.choice = None
        self.manual_path = None

    def on_auto_clicked(self):
        self.choice = "auto"
        self.accept()

    def on_manual_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите исполняемый файл SU2_CFD.exe", "",
            "Исполняемые файлы (*SU2_CFD.exe *SU2_CFD)")
        if path:
            self.manual_path = path
            self.choice = "manual"
            self.accept()


class SU2InstallWorker(QThread):
    progress_signal = pyqtSignal(int, str)   # процент, статус
    finished_signal = pyqtSignal(bool, str)  # успех, путь или ошибка

    def __init__(self, install_dir):
        super().__init__()
        self.install_dir = install_dir

    def run(self):
        try:
            import urllib.request
            import zipfile
            url = "https://github.com/su2code/SU2/releases/download/v7.5.1/SU2-v7.5.1-win64.zip"
            zip_path = os.path.join(self.install_dir, "su2_temp.zip")
            self.progress_signal.emit(5, "Скачивание SU2 v7.5.1...")

            def report_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = int((block_num * block_size / total_size) * 80) + 5
                    self.progress_signal.emit(percent, f"Скачивание SU2: {percent - 5}%")

            urllib.request.urlretrieve(url, zip_path, reporthook=report_hook)
            self.progress_signal.emit(85, "Распаковка архива...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.install_dir)
            try:
                os.remove(zip_path)
            except Exception:
                pass
            self.progress_signal.emit(95, "Поиск исполняемого файла...")
            su2_cfd_path = None
            for root, dirs, files in os.walk(self.install_dir):
                for f in files:
                    if f.lower() in ("su2_cfd.exe", "su2_cfd"):
                        su2_cfd_path = os.path.abspath(os.path.join(root, f))
                        break
                if su2_cfd_path:
                    break
            if su2_cfd_path:
                self.finished_signal.emit(True, su2_cfd_path)
            else:
                self.finished_signal.emit(False, "Не удалось найти SU2_CFD в распакованной папке.")
        except Exception as e:
            self.finished_signal.emit(False, str(e))


# ---------------------------------------------------------------------------
# Память процесса (без psutil)
# ---------------------------------------------------------------------------
def _get_process_rss_bytes():
    """Возвращает RSS текущего процесса в байтах. Без зависимостей."""
    try:
        if sys.platform == "win32":
            # Windows: GetProcessMemoryInfo
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            psapi = ctypes.windll.psapi
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters),
                                          counters.cb):
                return int(counters.WorkingSetSize)
            return 0
        else:
            # Unix: ru_maxrss в килобайтах
            if _resource is None:
                return 0
            usage = _resource.getrusage(_resource.RUSAGE_SELF)
            return int(usage.ru_maxrss) * 1024
    except Exception:
        return 0


def format_memory_size(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "Memory: --"
    mb = num_bytes / (1024 * 1024)
    if mb > 1024:
        return f"Memory: {mb / 1024:.2f} GB"
    return f"Memory: {mb:.1f} MB"


# ---------------------------------------------------------------------------
# Реальное количество ядер CPU
# ---------------------------------------------------------------------------
def _format_duration(seconds: float) -> str:
    """Человеческий формат длительности: '45 с', '2 мин 15 с', '1 ч 12 мин'."""
    try:
        seconds = max(0, int(round(float(seconds))))
    except Exception:
        return "—"
    if seconds < 60:
        return f"{seconds} с"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m} мин {s} с"
    h, m = divmod(m, 60)
    return f"{h} ч {m} мин"


def _detect_cpu_cores() -> dict:
    """Возвращает dict с информацией о доступных ядрах.

    Поля:
      physical       — физические ядра (без гиперпоточности)
      logical        — логические ядра (с HT/SMT)
      mpi_max        — безопасный максимум для mpiexec
      recommended    — рекомендуемое значение по умолчанию
    """
    physical = os.cpu_count() or 1
    logical = physical
    # psutil: точные физические
    try:
        import psutil  # type: ignore
        physical = psutil.cpu_count(logical=False) or physical
        logical = psutil.cpu_count(logical=True) or logical
    except Exception:
        # psutil нет — фоллбэк: считаем, что HT нет, physical == logical
        pass
    # mpiexec / Microsoft MPI (Windows): по умолчанию до 64 процессов
    # без HPC Pack. На практике безопаснее physical.
    # На Linux OpenMPI / MPICH — без жёсткого лимита, но physical
    # почти всегда лучше.
    mpi_max = min(physical, 64)
    recommended = max(1, min(physical, 8))
    return {
        "physical": physical,
        "logical": logical,
        "mpi_max": mpi_max,
        "recommended": recommended,
    }


# ---------------------------------------------------------------------------
# 2D-графики результатов
# ---------------------------------------------------------------------------
class AeroPlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#FAFAFB')
        self.axes1 = fig.add_subplot(121)
        self.axes2 = fig.add_subplot(122)
        super().__init__(fig)
        self.setParent(parent)
        self.setup_plot()

    def setup_plot(self):
        self.axes1.set_title("Поляра: CL vs CD", fontsize=9, fontweight='bold', color='#22384A')
        self.axes1.set_xlabel("CD (сопротивление)", fontsize=8)
        self.axes1.set_ylabel("CL (подъемная сила)", fontsize=8)
        self.axes1.grid(True, linestyle='--', alpha=0.5)
        self.axes1.tick_params(labelsize=8)
        self.axes2.set_title("Подъемная сила: CL vs AoA", fontsize=9, fontweight='bold', color='#22384A')
        self.axes2.set_xlabel("AoA (угол атаки, град)", fontsize=8)
        self.axes2.set_ylabel("CL (подъемная сила)", fontsize=8)
        self.axes2.grid(True, linestyle='--', alpha=0.5)
        self.axes2.tick_params(labelsize=8)
        self.figure.tight_layout()

    def update_plots(self, results):
        self.axes1.clear()
        self.axes2.clear()
        self.setup_plot()
        valid_res = [r for r in results if not r.get("error", True)]
        if not valid_res:
            self.draw()
            return
        aoas = [r["aoa"] for r in valid_res]
        cls = [r["cl"] for r in valid_res]
        cds = [r["cd"] for r in valid_res]
        order = np.argsort(aoas)
        aoas_sorted = np.array(aoas)[order]
        cls_sorted = np.array(cls)[order]
        cds_sorted = np.array(cds)[order]
        self.axes1.plot(cds_sorted, cls_sorted, 'o-r', linewidth=1.5, label="Поляра")
        self.axes2.plot(aoas_sorted, cls_sorted, 'o-b', linewidth=1.5, label="CL")
        self.axes1.legend(fontsize=7)
        self.axes2.legend(fontsize=7)
        self.draw()


# ---------------------------------------------------------------------------
# Главное окно
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AeroOpt v4.0")
        self.setGeometry(50, 50, 1750, 1050)
        self.setStyleSheet("""
            QMainWindow { background-color: #F2F3F5; }
            QTreeWidget {
                background-color: #FBFBFC; border: 1px solid #C3CBD5;
                font-family: Segoe UI, sans-serif; font-size: 11px;
            }
            QTreeWidget::item { padding: 4px; }
            QTreeWidget::item:selected { background-color: #D9DDE2; color: #1A1A1A; }
            QStackedWidget { background-color: #FBFBFC; border: 1px solid #C3CBD5; }
            QGroupBox {
                border: 1px solid #D1D9E0; margin-top: 15px;
                font-family: Segoe UI, sans-serif; font-size: 11px;
                font-weight: bold; background-color: #FAFAFB;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px;
                color: #2c4257;
            }
            QLabel { font-family: Segoe UI, sans-serif; font-size: 11px; color: #2c4257; }
            QPushButton {
                background-color: #ECEEF1; border: 1px solid #B0B9C5;
                padding: 5px 12px;
                font-family: Segoe UI, sans-serif; font-size: 11px; color: #1a2d37;
            }
            QPushButton:hover { background-color: #E2E5E9; border-color: #2E5A78; }
            QPushButton:pressed { background-color: #D2D7DC; border-color: #26485F; }
            QPushButton:disabled {
                /* Контраст текста к фону держим не ниже 4.5:1 (WCAG AA):
                   прежний #94A3B8 по #E2E5E9 давал 2.03:1 — надпись на
                   отключённой кнопке было не разобрать. */
                background-color: #E9EBEE !important;
                border-color: #C3CBD5 !important;
                color: #5C6B7A !important;
            }
            QTableWidget {
                background-color: #FBFBFC; border: 1px solid #C3CBD5;
                gridline-color: #E6E8EB; font-family: Segoe UI, sans-serif;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #E9EBEE; padding: 4px; border: 1px solid #C3CBD5;
                font-family: Segoe UI, sans-serif; font-size: 11px;
                font-weight: bold; color: #2c4257;
            }
            QTabWidget::pane { border: 1px solid #C3CBD5; background-color: #FBFBFC; }
            QTabBar::tab {
                background-color: #E5E7EA; border: 1px solid #B0B9C5;
                border-bottom-color: none; border-top-left-radius: 4px;
                border-top-right-radius: 4px; padding: 6px 14px; margin-right: 2px;
                font-family: Segoe UI, sans-serif; font-size: 11px; color: #2c4257;
            }
            QTabBar::tab:selected {
                background-color: #FBFBFC; border-bottom-color: #FBFBFC; font-weight: bold;
            }
        """)

        # ------------------------ данные -------------------------------
        # Полётные условия — теперь единый объект
        self.flight = FlightConditions()

        self.bodies = []
        self.next_body_id = 0
        self.all_results = []
        self.worker = None
        self.sweep_worker = None
        self.opt_worker = None
        self.session = None
        self.session_runner = None
        self.current_selected_body_index = -1
        self.flow_arrow_actor = None
        self.current_surface_mesh = None
        # Камера запоминается между перерисовками карты поля. Сбрасывается
        # только при загрузке НОВОГО результата: тогда вид надо подобрать
        # заново, а при смене отображаемой величины — оставить как есть.
        self._flow_scene_ready = False
        self.current_volume_mesh = None
        self.latest_case_dir = None
        self.wing_box_actor = None
        self.rule_set = RuleSet()
        self.mesh_ready = False
        self.paused_case_dir = None
        self.pause_requested = False
        self.generation_history = []
        self._mesh_worker = None
        self._meshing = False
        self._mesh_start_time = None
        self._opt_running = False
        self._eta_ema = None
        # Значение ядер, реально применяемое в расчёте.
        # Источник истины — физические ядра, не os.cpu_count() (тот
        # возвращает логические и завышает для HT-машин).
        cpu_info = _detect_cpu_cores()
        self._cpu_cores_pending = cpu_info["recommended"]
        # Гибридный GPU-режим: какой вычислитель и доля GPU.
        # Применяются через кнопку «Готово: Применить» (apply_load_level) и
        # передаются в CalculationSession при следующем запуске.
        self._compute_device_pending = "cpu"     # "cpu" | "cpu_gpu"
        self._gpu_percent_pending = 0            # 0..100
        self._gpu_percent_last_applied = 0       # для лога «откат на CPU»

        # === T6: инициализация LicenseChecker (если доступен) ===========
        # Сервер лицензий и HMAC-ключ берутся из переменных окружения
        # или из config.settings (если добавите). По умолчанию — публичный
        # сервер разработки. В прод-сборке подмените через .env.
        self._license = None
        if _LICENSE_AVAILABLE:
            try:
# URL по умолчанию: raw.githubusercontent.com (в license_checker.py)
                # Кастомный URL можно задать через переменную окружения
                server = os.environ.get("AEROOPT_LICENSE_SERVER", None)
                self._license = LicenseChecker(
                    server_url=server,
                    app_version="4.1.0",
                )
                status = self._license.bootstrap()
                if status == LicenseStatus.ACTIVE:
                    pass  # всё ок — без шума
                elif status == LicenseStatus.GRACE:
                    print(f"[AeroOpt] License: grace period "
                          f"({self._license.get_status_text()})")
                else:
                    print(f"[AeroOpt] License: {self._license.get_status_text()}")
            except Exception as _e:
                print(f"[AeroOpt] License init error: {_e}")
                self._license = None
        # =================================================================

        # ==================== МЕНЮ (Файл) ==============================
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("Файл")
        open_action = file_menu.addAction("Открыть проект")
        open_action.triggered.connect(self.load_project)
        save_action = file_menu.addAction("Сохранить проект")
        save_action.triggered.connect(self.save_project)
        save_as_action = file_menu.addAction("Сохранить как...")
        save_as_action.triggered.connect(self.save_project)
        file_menu.addSeparator()
        reset_action = file_menu.addAction("Сбросить интерфейс")
        reset_action.triggered.connect(self.reset_interface)

        # === T6: меню «Лицензия» ========================================
        license_menu = menu_bar.addMenu("Лицензия")
        act_activate = license_menu.addAction("Активировать ключ...")
        act_activate.triggered.connect(self._show_activate_dialog)
        act_status = license_menu.addAction("Статус лицензии")
        act_status.triggered.connect(self._show_license_status)
        act_deactivate = license_menu.addAction("Отвязать эту машину")
        act_deactivate.triggered.connect(self._deactivate_license)
        # =================================================================

        main_central_widget = QWidget()
        self.setCentralWidget(main_central_widget)
        main_layout = QVBoxLayout(main_central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # ==================== РИББОН ==================================
        self.ribbon = QTabWidget()
        self.ribbon.setMaximumHeight(115)
        self.ribbon.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #C3CBD5; background-color: #F7F8F9; }
            QTabBar::tab { padding: 4px 15px; font-weight: bold; }
        """)

        # 1. Главная
        tab_home = QWidget()
        home_lay = QHBoxLayout(tab_home)
        home_lay.setContentsMargins(5, 5, 5, 5)
        btn_open = QPushButton("Открыть проект")
        btn_open.clicked.connect(self.load_project)
        btn_save = QPushButton("Сохранить")
        btn_save.clicked.connect(self.save_project)
        btn_mesh_build = QPushButton("Построить сетку")
        btn_mesh_build.clicked.connect(self.make_mesh_from_bodies)
        btn_run_sol = QPushButton("Запустить расчёт")
        btn_run_sol.clicked.connect(self.start_calculation)
        self.ribbon_btn_run = btn_run_sol
        btn_pause_sol = QPushButton("Пауза")
        btn_pause_sol.clicked.connect(self.pause_calculation)
        self.ribbon_btn_pause = btn_pause_sol
        btn_resume_sol = QPushButton("Продолжить")
        btn_resume_sol.clicked.connect(self.resume_calculation)
        self.ribbon_btn_resume = btn_resume_sol
        btn_cancel_sol = QPushButton("Отмена")
        btn_cancel_sol.clicked.connect(self.cancel_calculation)
        self.ribbon_btn_cancel = btn_cancel_sol
        for btn in [btn_open, btn_save, btn_mesh_build, btn_run_sol,
                    btn_pause_sol, btn_resume_sol, btn_cancel_sol]:
            home_lay.addWidget(btn)
        self.ribbon_btn_mesh = btn_mesh_build
        home_lay.addStretch()
        self.ribbon.addTab(tab_home, "Главная (Home)")

        # 2. Геометрия
        tab_geometry_ribbon = QWidget()
        geom_rib_lay = QHBoxLayout(tab_geometry_ribbon)
        geom_rib_lay.setContentsMargins(5, 5, 5, 5)
        btn_gen_aircraft = QPushButton("Полный самолет")
        btn_gen_aircraft.clicked.connect(self.generate_full_aircraft)
        btn_gen_w = QPushButton("Создать крыло")
        btn_gen_w.clicked.connect(lambda: self.generate_wing_mesh_parametric(
            self.w_span.value(), self.w_chord_root.value(), self.w_chord_tip.value()))
        btn_gen_f = QPushButton("Создать фюзеляж")
        btn_gen_f.clicked.connect(self.generate_fuselage)
        btn_gen_hs_r = QPushButton("Создать ГО")
        btn_gen_hs_r.clicked.connect(self.generate_horizontal_stabilizer)
        btn_gen_vk_r = QPushButton("Создать ВО")
        btn_gen_vk_r.clicked.connect(self.generate_vertical_stabilizer)
        btn_auto_w_r = QPushButton("Автоподбор крыла")
        btn_auto_w_r.clicked.connect(self.auto_suggest_wing_params)
        for btn in [btn_gen_aircraft, btn_gen_w, btn_gen_f,
                    btn_gen_hs_r, btn_gen_vk_r, btn_auto_w_r]:
            geom_rib_lay.addWidget(btn)
        geom_rib_lay.addStretch()
        self.ribbon.addTab(tab_geometry_ribbon, "Геометрия (Geometry)")

        # 3. Физика и правила
        tab_physics_ribbon = QWidget()
        phys_rib_lay = QHBoxLayout(tab_physics_ribbon)
        phys_rib_lay.setContentsMargins(5, 5, 5, 5)
        btn_check_r = QPushButton("Проверить правила")
        btn_check_r.clicked.connect(self.validate_current_design)
        btn_check_conflict = QPushButton("Проверить конфликты")
        btn_check_conflict.clicked.connect(self.check_rules_consistency)
        phys_rib_lay.addWidget(btn_check_r)
        phys_rib_lay.addWidget(btn_check_conflict)
        phys_rib_lay.addStretch()
        self.ribbon.addTab(tab_physics_ribbon, "Физика и Правила")

        main_layout.addWidget(self.ribbon)

        # ==================== ОСНОВНОЙ СПЛИТТЕР =======================
        self.outer_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.outer_splitter)

        self.left_panel_splitter = QSplitter(Qt.Horizontal)
        self.outer_splitter.addWidget(self.left_panel_splitter)

        # 1. Дерево Model Builder (слева)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Model Builder")
        self.tree.setMinimumWidth(260)
        self.tree.setMaximumWidth(350)
        self.left_panel_splitter.addWidget(self.tree)
        root_item = QTreeWidgetItem(self.tree, ["aeroopt_project.aop (root)"])
        self.tree.addTopLevelItem(root_item)
        root_item.setExpanded(True)
        # Global Definitions — теперь содержит И полётные условия, И правила
        global_defs = QTreeWidgetItem(root_item, ["Global Definitions"])
        self.item_global_defs = global_defs
        # Узел 'Flight Conditions' больше не нужен — страница объединена
        self.item_rules = QTreeWidgetItem(global_defs, ["Design Rules (Правила проектирования)"])
        # «Формат конфигурации» — это настройка расчёта, а не результат,
        # поэтому узел живёт в Global Definitions.
        self.item_presets = QTreeWidgetItem(
            global_defs, ["Config Presets (Формат конфигурации)"])
        global_defs.setExpanded(True)
        component = QTreeWidgetItem(root_item, ["Component 1 (comp1)"])
        self.item_component = component
        component.setExpanded(True)
        geom = QTreeWidgetItem(component, ["Geometry 1"])
        self.geom_node = geom
        self.item_components = QTreeWidgetItem(geom, ["Component List (Список деталей)"])
        self.item_fuselage = QTreeWidgetItem(geom, ["Fuselage Generator"])
        self.item_wing = QTreeWidgetItem(geom, ["Wing Generator"])
        self.item_flaps_slats = QTreeWidgetItem(geom, ["Flaps & Slats"])
        self.item_stabilizers = QTreeWidgetItem(geom, ["Stabilizers & Tail"])
        # Узел 'Active Parts' убран — дублировал таблицу bodies_table
        geom.setExpanded(True)
        self.item_mesh = QTreeWidgetItem(component, ["Mesh 1"])
        study = QTreeWidgetItem(root_item, ["Study 1"])
        self.item_study = study
        self.item_solver = QTreeWidgetItem(study, ["Solver Settings"])
        self.item_opt = QTreeWidgetItem(study, ["Multipoint Optimization"])
        # ТЗ: аэроупругость (средний приоритет) и прочность (низкий)
        self.item_aeroelastic = QTreeWidgetItem(
            study, ["Aeroelasticity (Флатер и дивергенция)"])
        self.item_structural = QTreeWidgetItem(
            study, ["Strength (Прочность корневого сечения)"])
        study.setExpanded(True)
        results_node = QTreeWidgetItem(root_item, ["Results"])
        self.item_results = results_node
        self.item_trim = QTreeWidgetItem(results_node, ["Trim & Balancing (Балансировка)"])
        self.item_flow_viz = QTreeWidgetItem(results_node, ["Flow Visualization (Поле обтекания)"])
        self.item_history = QTreeWidgetItem(results_node, ["Generation History"])
        # ТЗ: «Спецфункции» и «Формат конфигурации»
        self.item_specials = QTreeWidgetItem(
            results_node, ["Special Functions (Поляра и отчёты)"])
        results_node.setExpanded(True)

        # 2. Панель настроек (посередине)
        self.settings_container = QWidget()
        self.settings_container.setMinimumWidth(320)
        self.settings_container.setMaximumWidth(420)
        settings_outer_lay = QVBoxLayout(self.settings_container)
        settings_outer_lay.setContentsMargins(0, 0, 0, 0)
        settings_outer_lay.setSpacing(0)
        header_widget = QWidget()
        header_widget.setStyleSheet("background-color: #E9EBEE; border-bottom: 1px solid #C3CBD5;")
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(10, 6, 10, 6)
        self.lbl_settings_header = QLabel("Settings")
        self.lbl_settings_header.setStyleSheet("font-weight: bold; color: #2c4257; font-size: 11px;")
        header_layout.addWidget(self.lbl_settings_header)
        settings_outer_lay.addWidget(header_widget)
        self.settings_stack = QStackedWidget()
        # Страницы с длинными подписями кнопок (например «Формат
        # конфигурации») растягивали стек, и вся панель настроек
        # разъезжалась при переключении. Стек больше не влияет на
        # ширину панели, а широкое содержимое уходит в прокрутку.
        self.settings_stack.setSizePolicy(QSizePolicy.Ignored,
                                          QSizePolicy.Preferred)
        self._settings_scroll = QScrollArea()
        self._settings_scroll.setWidgetResizable(True)
        self._settings_scroll.setFrameShape(QFrame.NoFrame)
        self._settings_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAsNeeded)
        self._settings_scroll.setWidget(self.settings_stack)
        settings_outer_lay.addWidget(self._settings_scroll)
        self.left_panel_splitter.addWidget(self.settings_container)
        self.left_panel_splitter.setSizes([260, 340])

        # ==================== СТРАНИЦЫ НАСТРОЕК =======================

        # Page Global Definitions: Полётные условия + Design Rules
        self.page_global_defs = QWidget()
        gd_lay = QVBoxLayout(self.page_global_defs)
        gd_lay.setContentsMargins(10, 10, 10, 10)

        # --- QGroupBox: Полётные условия (всё, что раньше было на page_conditions)
        fc_group = QGroupBox("Полётные условия (Flight Conditions)")
        fc_lay = QFormLayout(fc_group)

        self.combo_flight_preset = QComboBox()
        for name in list_flight_presets():
            self.combo_flight_preset.addItem(name)
        self.combo_flight_preset.currentTextChanged.connect(self.on_flight_preset_changed)
        fc_lay.addRow("Пресет:", self.combo_flight_preset)

        self.input_speed = QDoubleSpinBox()
        self.input_speed.setRange(1, 500)
        self.input_speed.setValue(self.flight.speed_m_s)
        self.input_speed.setSuffix(" м/с")
        self.input_speed.setToolTip(
            "Скорость набегающего потока в м/с. Из неё вместе с высотой "
            "автоматически считается число Маха и параметры ISA-атмосферы."
        )
        fc_lay.addRow("Скорость:", self.input_speed)

        self.input_alt = QSpinBox()
        self.input_alt.setRange(0, 20000)
        self.input_alt.setValue(int(self.flight.altitude_m))
        self.input_alt.setSuffix(" м")
        self.input_alt.setToolTip(
            "Высота полёта над уровнем моря (м). До 11 000 м используется "
            "стандартная атмосфера ISA; выше — экстраполяция по тропопаузе."
        )
        fc_lay.addRow("Высота:", self.input_alt)

        self.input_aoa = QDoubleSpinBox()
        self.input_aoa.setRange(-15, 25)
        self.input_aoa.setValue(self.flight.aoa_deg)
        self.input_aoa.setSuffix(" °")
        self.input_aoa.setToolTip(
            "Угол атаки в градусах — угол между хордой крыла и направлением "
            "набегающего потока. Используется как AOA в config.cfg SU2."
        )
        fc_lay.addRow("Угол атаки:", self.input_aoa)

        self.lbl_isa = QLabel("—")
        self.lbl_isa.setStyleSheet("color:#2E5A78; font-weight:bold;")
        self.lbl_isa.setToolTip(
            "Параметры стандартной атмосферы для текущей высоты: "
            "T — температура, P — давление, ρ — плотность, M — число Маха."
        )
        fc_lay.addRow("Атмосфера (ISA):", self.lbl_isa)

        self.input_speed.valueChanged.connect(self.on_flight_field_changed)
        self.input_alt.valueChanged.connect(self.on_flight_field_changed)
        self.input_aoa.valueChanged.connect(self.on_flight_field_changed)

        # Кнопка применения — без неё изменения в self.flight не пишутся
        # (раньше изменения подхватывались сразу; теперь явный «commit»)
        self.btn_apply_flight = QPushButton("Готово: Применить условия полёта")
        self.btn_apply_flight.setToolTip(
            "Применить введённые скорость/высоту/AoA к текущей сессии расчёта. "
            "До нажатия изменения только отображаются в полях."
        )
        self.btn_apply_flight.clicked.connect(self.apply_flight_conditions)
        fc_lay.addRow(self.btn_apply_flight)

        gd_lay.addWidget(fc_group)

        # --- QGroupBox: Design Rules (содержимое бывшей page_rules)
        rules_group = QGroupBox("Правила проектирования (Design Rules)")
        rules_lay = QVBoxLayout(rules_group)

        preset_group = QGroupBox("Готовые наборы (пресеты)")
        pg_lay = QHBoxLayout(preset_group)
        self.combo_preset = QComboBox()
        for name in PRESETS.keys():
            self.combo_preset.addItem(name)
        btn_load_preset_rules = QPushButton("Загрузить")
        btn_load_preset_rules.clicked.connect(self.load_rule_preset)
        pg_lay.addWidget(self.combo_preset)
        pg_lay.addWidget(btn_load_preset_rules)
        rules_lay.addWidget(preset_group)

        self.rules_table = QTableWidget(0, 6)
        self.rules_table.setHorizontalHeaderLabels(["Имя", "Параметр", "Условие", "Значение", "Тип", "Вкл"])
        self.rules_table.horizontalHeader().setStretchLastSection(True)
        # Множественное выделение — Ctrl/Shift по строкам, чтобы можно было
        # удалить сразу несколько правил одной кнопкой.
        self.rules_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.rules_table.setSelectionMode(QTableWidget.ExtendedSelection)
        rules_lay.addWidget(self.rules_table)

        rules_btn_lay = QHBoxLayout()
        btn_add_rule = QPushButton("Добавить")
        btn_add_rule.clicked.connect(self.add_rule_dialog)
        btn_rem_rule = QPushButton("Удалить")
        btn_rem_rule.clicked.connect(self.remove_selected_rule)
        btn_check_rules = QPushButton("Проверить")
        btn_check_rules.clicked.connect(self.check_rules_consistency)
        rules_btn_lay.addWidget(btn_add_rule)
        rules_btn_lay.addWidget(btn_rem_rule)
        rules_btn_lay.addWidget(btn_check_rules)
        rules_lay.addLayout(rules_btn_lay)

        rules_io_lay = QHBoxLayout()
        btn_save_rules = QPushButton("Сохранить")
        btn_save_rules.clicked.connect(self.save_rule_set)
        btn_load_rules = QPushButton("Открыть")
        btn_load_rules.clicked.connect(self.load_rule_set)
        btn_validate_now = QPushButton("Готово: Проверить")
        btn_validate_now.clicked.connect(self.validate_current_design)
        rules_io_lay.addWidget(btn_save_rules)
        rules_io_lay.addWidget(btn_load_rules)
        rules_io_lay.addWidget(btn_validate_now)
        rules_lay.addLayout(rules_io_lay)

        gd_lay.addWidget(rules_group)
        gd_lay.addStretch()

        self.settings_stack.addWidget(self.page_global_defs)

        # Page 3: Component List
        self.page_components = QWidget()
        lay3 = QVBoxLayout(self.page_components)
        lay3.setContentsMargins(10, 10, 10, 10)
        wizard_group = QGroupBox("Помощник импорта (Import Wizard)")
        wiz_lay = QVBoxLayout(wizard_group)
        # Сокращённый текст + кнопка «? Помощь» с подробной подсказкой
        wiz_short = QLabel("Загрузите геометрию, задайте роль и переходите к Mesh.")
        wiz_short.setStyleSheet("font-size: 10px; color: #2c4257;")
        wiz_lay.addWidget(wiz_short)
        btn_wiz_help = QPushButton("Помощь")
        btn_wiz_help.setToolTip(
            "Шаг 1: Нажмите «Фюзеляж» или «STL» и выберите файл. Помимо "
            "STL принимаются CAD-форматы — STEP, IGES, Parasolid, ACIS, "
            "BREP, NASTRAN, PLY, OBJ, OFF; они триангулируются через "
            "gmsh автоматически.\n"
            "Шаг 2: В таблице ниже выберите роль детали из выпадающего списка "
            "(для полной модели используйте «Произвольный самолет»/«Другое»).\n"
            "Шаг 3: Выберите в дереве слева узел «Mesh 1» и нажмите "
            "«Построить расчётную сетку»."
        )
        btn_wiz_help.clicked.connect(
            lambda checked=False: QToolTip.showText(
                btn_wiz_help.mapToGlobal(btn_wiz_help.rect().bottomRight()),
                btn_wiz_help.toolTip()
            )
        )
        wiz_lay.addWidget(btn_wiz_help)
        lay3.addWidget(wizard_group)
        btn_load_lay = QHBoxLayout()
        self.btn_add_fuselage = QPushButton("Фюзеляж")
        self.btn_add_fuselage.clicked.connect(self.load_stl_fuselage)
        self.btn_add_body = QPushButton("STL")
        self.btn_add_body.clicked.connect(self.add_bodies)
        self.btn_add_primitive = QPushButton("Примитив")
        menu = QMenu()
        menu.addAction("Куб", lambda: self._create_primitive("Куб"))
        menu.addAction("Цилиндр", lambda: self._create_primitive("Цилиндр"))
        menu.addAction("Сфера", lambda: self._create_primitive("Сфера"))
        self.btn_add_primitive.setMenu(menu)
        btn_load_lay.addWidget(self.btn_add_fuselage)
        btn_load_lay.addWidget(self.btn_add_body)
        btn_load_lay.addWidget(self.btn_add_primitive)
        lay3.addLayout(btn_load_lay)
        simplify_layout = QHBoxLayout()
        self.btn_simplify_simple = QPushButton("Упростить (грубо)")
        self.btn_simplify_simple.clicked.connect(lambda: self.simplify_geometry(level="simple"))
        self.btn_simplify_medium = QPushButton("Упростить (средне)")
        self.btn_simplify_medium.clicked.connect(lambda: self.simplify_geometry(level="medium"))
        simplify_layout.addWidget(self.btn_simplify_simple)
        simplify_layout.addWidget(self.btn_simplify_medium)
        lay3.addLayout(simplify_layout)
        self.btn_heal_stl = QPushButton("Лечить выбранный STL")
        self.btn_heal_stl.clicked.connect(self.heal_selected_stl)
        self.btn_heal_stl.setEnabled(False)
        lay3.addWidget(self.btn_heal_stl)
        self.bodies_table = QTableWidget(0, 2)
        self.bodies_table.setHorizontalHeaderLabels(["Компонент", "Роль"])
        self.bodies_table.horizontalHeader().setStretchLastSection(True)
        self.bodies_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.bodies_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.bodies_table.cellClicked.connect(self.on_table_click)
        lay3.addWidget(self.bodies_table)
        tool_layout = QHBoxLayout()
        self.btn_remove = QPushButton("Удалить")
        self.btn_remove.clicked.connect(self.remove_body)
        tool_layout.addWidget(self.btn_remove)
        for axis in ['x', 'y', 'z']:
            btn = QPushButton(f"{axis.upper()} +90°")
            btn.clicked.connect(lambda checked, a=axis: self.rotate_selected(a, 90))
            tool_layout.addWidget(btn)
        lay3.addLayout(tool_layout)
        cad_group = QGroupBox("Direct CAD Import (STEP/IGES/BREP…)")
        cad_lay = QVBoxLayout(cad_group)
        self.chk_cad_split = QCheckBox("Разбирать сборку на отдельные детали")
        self.chk_cad_split.setChecked(True)
        self.chk_cad_split.setToolTip(
            "Многодетальная сборка (несколько тел в STEP/IGES) будет "
            "импортирована как несколько компонентов с собственными "
            "именами — их можно скрывать и назначать им разные роли.\n"
            "Если выключить — вся сборка импортируется одной деталью "
            "(«сборка как деталь»).")
        self.lbl_cad_info = QLabel("Требует gmsh. Поддерживаются: STEP, "
                                   "IGES, BREP, Parasolid, ACIS, PLY, OBJ.")
        self.lbl_cad_info.setWordWrap(True)
        self.lbl_cad_info.setStyleSheet("color:#666; font-size:10px;")
        cad_lay.addWidget(self.chk_cad_split)
        cad_lay.addWidget(self.lbl_cad_info)
        lay3.addWidget(cad_group)
        self.settings_stack.addWidget(self.page_components)

        # Page 4: Fuselage
        self.page_fuselage = QWidget()
        lay4 = QFormLayout(self.page_fuselage)
        lay4.setContentsMargins(10, 10, 10, 10)
        self.f_length = QDoubleSpinBox(); self.f_length.setRange(2, 30); self.f_length.setValue(8); self.f_length.setSuffix(" м")
        self.f_length.setToolTip("Полная длина фюзеляжа от носа до хвоста в метрах.")
        self.f_diameter = QDoubleSpinBox(); self.f_diameter.setRange(0.5, 5); self.f_diameter.setValue(1.2); self.f_diameter.setSuffix(" м")
        self.f_diameter.setToolTip("Максимальный диаметр (или высота) поперечного сечения фюзеляжа.")
        self.f_nose_ratio = QDoubleSpinBox(); self.f_nose_ratio.setRange(0.1, 0.5); self.f_nose_ratio.setValue(0.25)
        self.f_nose_ratio.setToolTip("Доля длины, занимаемая носовой частью (скругление).")
        self.f_tail_ratio = QDoubleSpinBox(); self.f_tail_ratio.setRange(0.1, 0.5); self.f_tail_ratio.setValue(0.3)
        self.f_tail_ratio.setToolTip("Доля длины, занимаемая хвостовым сужением.")
        self.f_pos_x = QDoubleSpinBox(); self.f_pos_x.setRange(-50, 50); self.f_pos_x.setValue(0)
        self.f_pos_x.setToolTip("Смещение фюзеляжа по X относительно начала координат (м).")
        self.f_pos_y = QDoubleSpinBox(); self.f_pos_y.setRange(-50, 50); self.f_pos_y.setValue(0)
        self.f_pos_y.setToolTip("Смещение фюзеляжа по Y (м).")
        self.f_pos_z = QDoubleSpinBox(); self.f_pos_z.setRange(-50, 50); self.f_pos_z.setValue(0)
        self.f_pos_z.setToolTip("Смещение фюзеляжа по Z (м); обычно совпадает с осью симметрии модели.")
        lay4.addRow("Длина:", self.f_length)
        lay4.addRow("Диаметр:", self.f_diameter)
        lay4.addRow("Доля носа:", self.f_nose_ratio)
        lay4.addRow("Доля хвоста:", self.f_tail_ratio)
        lay4.addRow("X:", self.f_pos_x)
        lay4.addRow("Y:", self.f_pos_y)
        lay4.addRow("Z:", self.f_pos_z)
        self.btn_gen_fuselage = QPushButton("Сгенерировать фюзеляж")
        self.btn_gen_fuselage.clicked.connect(self.generate_fuselage)
        self.btn_export_fuselage = QPushButton("Экспорт фюзеляжа")
        self.btn_export_fuselage.clicked.connect(self.export_fuselage)
        lay4.addRow(self.btn_gen_fuselage)
        lay4.addRow(self.btn_export_fuselage)
        self.settings_stack.addWidget(self.page_fuselage)

        # Page 5: Wing
        scroll_wing_inner = QWidget()
        lay5 = QFormLayout(scroll_wing_inner)
        lay5.setContentsMargins(10, 10, 10, 10)
        self.w_span = QDoubleSpinBox(); self.w_span.setRange(0.5, 50); self.w_span.setValue(10)
        self.w_span.setToolTip("Полный размах крыла от законцовки до законцовки (м).")
        self.w_chord_root = QDoubleSpinBox(); self.w_chord_root.setRange(0.1, 15); self.w_chord_root.setValue(1.8)
        self.w_chord_root.setToolTip("Хорда крыла в корневом сечении (м).")
        self.w_chord_tip = QDoubleSpinBox(); self.w_chord_tip.setRange(0.1, 15); self.w_chord_tip.setValue(0.9)
        self.w_chord_tip.setToolTip("Хорда крыла в концевом сечении (м).")
        self.w_sweep = QDoubleSpinBox(); self.w_sweep.setRange(-45, 60); self.w_sweep.setValue(12)
        self.w_sweep.setToolTip("Стреловидность по передней кромке, градусы. >0 — назад.")
        self.w_twist = QDoubleSpinBox(); self.w_twist.setRange(-10, 10); self.w_twist.setValue(2)
        self.w_twist.setToolTip("Геометрическая крутка концевого сечения относительно корневого, градусы.")
        self.w_naca = QLineEdit("2412")
        self.w_naca.setToolTip("Профиль NACA 4-значного кода. Примеры: 0012, 2412, 4412.")
        self.w_pos_x = QDoubleSpinBox(); self.w_pos_x.setRange(-50, 50); self.w_pos_x.setValue(3)
        self.w_pos_x.setToolTip("X-координата корневой хорды крыла (м).")
        self.w_pos_y = QDoubleSpinBox(); self.w_pos_y.setRange(-50, 50); self.w_pos_y.setValue(0)
        self.w_pos_y.setToolTip("Y-координата корневой хорды (м).")
        self.w_pos_z = QDoubleSpinBox(); self.w_pos_z.setRange(-50, 50); self.w_pos_z.setValue(0)
        self.w_pos_z.setToolTip("Z-координата корневой хорды (м).")
        self.chk_kink = QCheckBox("Включить излом крыла")
        self.chk_kink.setChecked(False)
        self.chk_kink.setToolTip("Включить излом крыла (двухсекционная консоль с переменным углом стреловидности).")
        self.w_kink_pos = QDoubleSpinBox(); self.w_kink_pos.setRange(0.1, 0.9); self.w_kink_pos.setValue(0.4); self.w_kink_pos.setEnabled(False)
        self.w_kink_pos.setToolTip("Положение излома в долях полуразмаха.")
        self.w_chord_kink = QDoubleSpinBox(); self.w_chord_kink.setRange(0.1, 15); self.w_chord_kink.setValue(1.3); self.w_chord_kink.setEnabled(False)
        self.w_chord_kink.setToolTip("Хорда в сечении излома (м).")
        self.w_sweep_outer = QDoubleSpinBox(); self.w_sweep_outer.setRange(-45, 60); self.w_sweep_outer.setValue(8.0); self.w_sweep_outer.setEnabled(False)
        self.w_sweep_outer.setToolTip("Стреловидность внешней секции (после излома), град.")
        self.chk_kink.stateChanged.connect(self._toggle_kink_controls)
        lay5.addRow("Размах:", self.w_span)
        lay5.addRow("Корневая хорда:", self.w_chord_root)
        lay5.addRow("Концевая хорда:", self.w_chord_tip)
        lay5.addRow("Стреловидность (внутр):", self.w_sweep)
        lay5.addRow("Крутка:", self.w_twist)
        lay5.addRow("Профиль NACA:", self.w_naca)
        lay5.addRow("X:", self.w_pos_x)
        lay5.addRow("Y:", self.w_pos_y)
        lay5.addRow("Z:", self.w_pos_z)
        lay5.addRow(self.chk_kink)
        lay5.addRow("Положение излома (%):", self.w_kink_pos)
        lay5.addRow("Хорда в изломе:", self.w_chord_kink)
        lay5.addRow("Стреловидность (внеш):", self.w_sweep_outer)
        wbox_group = QGroupBox("Область генерации (Wing Box)")
        wb_lay = QFormLayout(wbox_group)
        self.wbox_cx = QDoubleSpinBox(); self.wbox_cx.setRange(-100, 100); self.wbox_cx.setValue(2.5)
        self.wbox_cy = QDoubleSpinBox(); self.wbox_cy.setRange(-100, 100); self.wbox_cy.setValue(0.0)
        self.wbox_cz = QDoubleSpinBox(); self.wbox_cz.setRange(-100, 100); self.wbox_cz.setValue(0.0)
        self.wbox_lx = QDoubleSpinBox(); self.wbox_lx.setRange(0.1, 100); self.wbox_lx.setValue(2.0); self.wbox_lx.setSuffix(" м")
        self.wbox_ly = QDoubleSpinBox(); self.wbox_ly.setRange(0.1, 200); self.wbox_ly.setValue(10.0); self.wbox_ly.setSuffix(" м")
        self.wbox_lz = QDoubleSpinBox(); self.wbox_lz.setRange(0.1, 100); self.wbox_lz.setValue(1.0); self.wbox_lz.setSuffix(" м")
        self.chk_wing_auto_from_box = QCheckBox("Авторазмеры крыла из области")
        self.chk_wing_auto_from_box.setChecked(True)
        btn_box_from_fuselage = QPushButton("Взять область из фюзеляжа")
        btn_box_from_fuselage.clicked.connect(self.fill_wing_box_from_fuselage)
        btn_box_preview = QPushButton("Показать область")
        btn_box_preview.clicked.connect(self.preview_wing_box)
        wb_lay.addRow("Центр X:", self.wbox_cx)
        wb_lay.addRow("Центр Y:", self.wbox_cy)
        wb_lay.addRow("Центр Z:", self.wbox_cz)
        wb_lay.addRow("Размер X:", self.wbox_lx)
        wb_lay.addRow("Размер Y:", self.wbox_ly)
        wb_lay.addRow("Размер Z:", self.wbox_lz)
        wb_lay.addRow(self.chk_wing_auto_from_box)
        wb_lay.addRow(btn_box_from_fuselage)
        wb_lay.addRow(btn_box_preview)
        lay5.addRow(wbox_group)
        self.btn_gen_wing = QPushButton("Сгенерировать крыло")
        self.btn_gen_wing.clicked.connect(lambda: self.generate_wing_mesh_parametric(
            self.w_span.value(), self.w_chord_root.value(), self.w_chord_tip.value()))
        self.btn_auto_wing = QPushButton("Автоподбор крыла по фюзеляжу")
        self.btn_auto_wing.clicked.connect(self.auto_suggest_wing_params)
        self.btn_export_wing = QPushButton("Экспорт крыла")
        self.btn_export_wing.clicked.connect(self.export_wing)
        btn_full_aircraft = QPushButton("Сгенерировать ВЕСЬ САМОЛЁТ")
        btn_full_aircraft.clicked.connect(self.generate_full_aircraft)
        lay5.addRow(self.btn_gen_wing)
        lay5.addRow(self.btn_auto_wing)
        lay5.addRow(self.btn_export_wing)
        lay5.addRow(btn_full_aircraft)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(scroll_wing_inner)
        self.page_wing = scroll
        self.settings_stack.addWidget(self.page_wing)

        # Page 6: Flaps & Slats
        self.page_flaps_slats = QWidget()
        lay6 = QVBoxLayout(self.page_flaps_slats)
        lay6.setContentsMargins(10, 10, 10, 10)
        flap_group = QGroupBox("Закрылки")
        flap_lay = QFormLayout(flap_group)
        self.flap_enabled = QCheckBox("Включить закрылки")
        self.flap_enabled.setChecked(False)
        self.flap_deflection = QDoubleSpinBox(); self.flap_deflection.setRange(0, 45); self.flap_deflection.setValue(15); self.flap_deflection.setSuffix("°"); self.flap_deflection.setEnabled(False)
        self.flap_deflection.setToolTip("Угол отклонения закрылка вниз, градусы.")
        self.flap_span_ratio = QDoubleSpinBox(); self.flap_span_ratio.setRange(0.1, 0.8); self.flap_span_ratio.setValue(0.5); self.flap_span_ratio.setEnabled(False)
        self.flap_span_ratio.setToolTip("Доля размаха, на которой установлен закрылок (от корня).")
        self.flap_chord_ratio = QDoubleSpinBox(); self.flap_chord_ratio.setRange(0.1, 0.4); self.flap_chord_ratio.setValue(0.25); self.flap_chord_ratio.setEnabled(False)
        self.flap_chord_ratio.setToolTip("Доля хорды, занимаемая закрылком (от задней кромки).")
        self.flap_hinge_depth = QDoubleSpinBox(); self.flap_hinge_depth.setRange(0.01, 0.5); self.flap_hinge_depth.setValue(0.12); self.flap_hinge_depth.setEnabled(False)
        self.flap_hinge_depth.setToolTip("Глубина оси вращения закрылка (доля хорды).")
        self.flap_slide = QDoubleSpinBox(); self.flap_slide.setRange(0.0, 0.3); self.flap_slide.setValue(0.06); self.flap_slide.setEnabled(False)
        self.flap_slide.setToolTip("Выдвижение закрылка типа Фаулер назад (доля хорды).")
        flap_lay.addRow(self.flap_enabled)
        flap_lay.addRow("Угол отклонения:", self.flap_deflection)
        flap_lay.addRow("Размах (%):", self.flap_span_ratio)
        flap_lay.addRow("Хорда (%):", self.flap_chord_ratio)
        flap_lay.addRow("Глубина оси вращения:", self.flap_hinge_depth)
        flap_lay.addRow("Выдвижение Фаулера (%):", self.flap_slide)
        self.flap_enabled.stateChanged.connect(lambda checked: self._toggle_flap_controls(checked))
        lay6.addWidget(flap_group)
        slat_group = QGroupBox("Предкрылки")
        slat_lay = QFormLayout(slat_group)
        self.slat_enabled = QCheckBox("Включить предкрылки")
        self.slat_enabled.setChecked(False)
        self.slat_deflection = QDoubleSpinBox(); self.slat_deflection.setRange(0, 30); self.slat_deflection.setValue(10); self.slat_deflection.setSuffix("°"); self.slat_deflection.setEnabled(False)
        self.slat_deflection.setToolTip("Угол отклонения предкрылка, градусы.")
        self.slat_span_ratio = QDoubleSpinBox(); self.slat_span_ratio.setRange(0.1, 0.95); self.slat_span_ratio.setValue(0.8); self.slat_span_ratio.setEnabled(False)
        self.slat_span_ratio.setToolTip("Доля размаха, на которой установлен предкрылок.")
        self.slat_chord_ratio = QDoubleSpinBox(); self.slat_chord_ratio.setRange(0.05, 0.25); self.slat_chord_ratio.setValue(0.12); self.slat_chord_ratio.setEnabled(False)
        self.slat_chord_ratio.setToolTip("Доля хорды, занимаемая предкрылком (от передней кромки).")
        self.slat_slide = QDoubleSpinBox(); self.slat_slide.setRange(0.0, 0.2); self.slat_slide.setValue(0.04); self.slat_slide.setEnabled(False)
        self.slat_slide.setToolTip("Выдвижение предкрылка вперёд (доля хорды).")
        slat_lay.addRow(self.slat_enabled)
        slat_lay.addRow("Угол отклонения:", self.slat_deflection)
        slat_lay.addRow("Размах (%):", self.slat_span_ratio)
        slat_lay.addRow("Хорда (%):", self.slat_chord_ratio)
        slat_lay.addRow("Выдвижение (%):", self.slat_slide)
        self.slat_enabled.stateChanged.connect(lambda checked: self._toggle_slat_controls(checked))
        lay6.addWidget(slat_group)
        self.settings_stack.addWidget(self.page_flaps_slats)

        # Page 7: Stabilizers
        self.page_stabilizers = QWidget()
        lay7 = QVBoxLayout(self.page_stabilizers)
        lay7.setContentsMargins(10, 10, 10, 10)
        hs_group = QGroupBox("Горизонтальное оперение (ГО)")
        hs_lay = QFormLayout(hs_group)
        self.hs_span = QDoubleSpinBox(); self.hs_span.setRange(0.5, 30); self.hs_span.setValue(3.0)
        self.hs_span.setToolTip("Полный размах горизонтального оперения (м).")
        self.hs_chord = QDoubleSpinBox(); self.hs_chord.setRange(0.1, 5); self.hs_chord.setValue(0.8)
        self.hs_chord.setToolTip("Хорда стабилизатора (м).")
        self.hs_sweep = QDoubleSpinBox(); self.hs_sweep.setRange(-30, 45); self.hs_sweep.setValue(15)
        self.hs_sweep.setToolTip("Стреловидность ГО по передней кромке, градусы.")
        self.hs_pos_x = QDoubleSpinBox(); self.hs_pos_x.setRange(-50, 50); self.hs_pos_x.setValue(6.5)
        self.hs_pos_x.setToolTip("X-координата корневой хорды ГО (м).")
        self.hs_pos_z = QDoubleSpinBox(); self.hs_pos_z.setRange(-10, 10); self.hs_pos_z.setValue(0.0)
        self.hs_pos_z.setToolTip("Z-координата ГО (м).")
        self.elev_deflection = QDoubleSpinBox(); self.elev_deflection.setRange(-30, 30); self.elev_deflection.setValue(0.0); self.elev_deflection.setSuffix("°")
        self.elev_deflection.setToolTip("Угол отклонения руля высоты, градусы.")
        self.hs_auto = QCheckBox("Автоподбор по фюзеляжу")
        btn_gen_hs = QPushButton("Сгенерировать ГО")
        btn_gen_hs.clicked.connect(self.generate_horizontal_stabilizer)
        btn_export_hs = QPushButton("Экспорт")
        btn_export_hs.clicked.connect(lambda: self.export_component("h_stab"))
        hs_lay.addRow("Размах:", self.hs_span)
        hs_lay.addRow("Хорда:", self.hs_chord)
        hs_lay.addRow("Стреловидность °:", self.hs_sweep)
        hs_lay.addRow("X:", self.hs_pos_x)
        hs_lay.addRow("Z:", self.hs_pos_z)
        hs_lay.addRow("Руль высоты °:", self.elev_deflection)
        hs_lay.addRow(self.hs_auto)
        hs_lay.addRow(btn_gen_hs)
        hs_lay.addRow(btn_export_hs)
        lay7.addWidget(hs_group)
        vk_group = QGroupBox("Вертикальное оперение (ВО) / Киль")
        vk_lay = QFormLayout(vk_group)
        self.vk_height = QDoubleSpinBox(); self.vk_height.setRange(0.3, 15); self.vk_height.setValue(1.5)
        self.vk_height.setToolTip("Высота вертикального оперения от фюзеляжа (м).")
        self.vk_chord = QDoubleSpinBox(); self.vk_chord.setRange(0.1, 5); self.vk_chord.setValue(0.7)
        self.vk_chord.setToolTip("Хорда ВО (м).")
        self.vk_sweep = QDoubleSpinBox(); self.vk_sweep.setRange(-30, 60); self.vk_sweep.setValue(20)
        self.vk_sweep.setToolTip("Стреловидность ВО, градусы.")
        self.vk_pos_x = QDoubleSpinBox(); self.vk_pos_x.setRange(-50, 50); self.vk_pos_x.setValue(6.0)
        self.vk_pos_x.setToolTip("X-координата корневой хорды ВО (м).")
        self.vk_pos_z = QDoubleSpinBox(); self.vk_pos_z.setRange(-10, 10); self.vk_pos_z.setValue(0.0)
        self.vk_pos_z.setToolTip("Z-координата ВО (м).")
        btn_gen_vk = QPushButton("Сгенерировать ВО")
        btn_gen_vk.clicked.connect(self.generate_vertical_stabilizer)
        btn_export_vk = QPushButton("Экспорт")
        btn_export_vk.clicked.connect(lambda: self.export_component("v_stab"))
        vk_lay.addRow("Высота:", self.vk_height)
        vk_lay.addRow("Хорда:", self.vk_chord)
        vk_lay.addRow("Стреловидность °:", self.vk_sweep)
        vk_lay.addRow("X:", self.vk_pos_x)
        vk_lay.addRow("Z:", self.vk_pos_z)
        vk_lay.addRow(btn_gen_vk)
        vk_lay.addRow(btn_export_vk)
        lay7.addWidget(vk_group)
        self.settings_stack.addWidget(self.page_stabilizers)

        # Page 8: Mesh
        self.page_mesh = QWidget()
        lay8 = QVBoxLayout(self.page_mesh)
        lay8.setContentsMargins(10, 10, 10, 10)
        q_group = QGroupBox("Качество сетки")
        q_lay = QFormLayout(q_group)
        self.combo_mesh_quality = QComboBox()
        self.combo_mesh_quality.addItems(MESH_QUALITY)
        self.combo_mesh_quality.setCurrentIndex(1)
        self.lbl_mesh_info = QLabel("~30-60 сек")
        self.combo_mesh_quality.currentTextChanged.connect(self.on_mesh_quality_changed)
        q_lay.addRow("Качество:", self.combo_mesh_quality)
        q_lay.addRow("Инфо:", self.lbl_mesh_info)
        lay8.addWidget(q_group)
        self.btn_make_mesh = QPushButton("Построить расчётную сетку")
        self.btn_make_mesh.clicked.connect(self.make_mesh_from_bodies)
        lay8.addWidget(self.btn_make_mesh)
        self.btn_adapt_mesh = QPushButton("Адаптировать сетку (SU2_ADAPT)")
        self.btn_adapt_mesh.setToolTip(
            "Локально сгустить сетку по решению (restart.dat) из последнего "
            "расчёта: точнее в областях высоких градиентов, чем при "
            "глобальном сгущении. Требует SU2_ADAPT из дистрибутива SU2.")
        self.btn_adapt_mesh.clicked.connect(self.adapt_current_mesh)
        lay8.addWidget(self.btn_adapt_mesh)
        adapt2 = QGroupBox("Адаптация по распределению Cp (gmsh)")
        a2 = QFormLayout(adapt2)
        self.adapt_h_min = QDoubleSpinBox()
        self.adapt_h_min.setRange(1e-5, 1.0)
        self.adapt_h_min.setDecimals(5)
        self.adapt_h_min.setValue(0.002)
        self.adapt_h_min.setSuffix(" м")
        self.adapt_h_max = QDoubleSpinBox()
        self.adapt_h_max.setRange(1e-4, 10.0)
        self.adapt_h_max.setDecimals(4)
        self.adapt_h_max.setValue(0.05)
        self.adapt_h_max.setSuffix(" м")
        self.adapt_power = QDoubleSpinBox()
        self.adapt_power.setRange(0.2, 4.0)
        self.adapt_power.setSingleStep(0.1)
        self.adapt_power.setValue(1.0)
        self.adapt_power.setToolTip(
            "Показатель сгущения: 1 — линейно по градиенту Cp, "
            "больше — резче контраст между носком и остальной поверхностью.")
        self.btn_adapt_cp = QPushButton("Перестроить сетку по Cp")
        self.btn_adapt_cp.setToolTip(
            "Взять surface_flow.csv последнего расчёта, построить поле "
            "целевых размеров (мельче там, где больше |dCp/ds|) и "
            "перестроить поверхностную сетку через gmsh.\n"
            "Не требует SU2_ADAPT, но требует gmsh.")
        self.btn_adapt_cp.clicked.connect(self.adapt_mesh_by_cp)
        a2.addRow("Мин. размер:", self.adapt_h_min)
        a2.addRow("Макс. размер:", self.adapt_h_max)
        a2.addRow("Показатель:", self.adapt_power)
        a2.addRow(self.btn_adapt_cp)
        lay8.addWidget(adapt2)
        self.settings_stack.addWidget(self.page_mesh)

        # Page 9: Solver — теперь с кнопкой «Готово: Применить» для ядер
        self.page_solver = QWidget()
        lay9 = QVBoxLayout(self.page_solver)
        lay9.setContentsMargins(10, 10, 10, 10)
        su2_group = QGroupBox("Настройка пути к решателю SU2")
        su2_lay = QVBoxLayout(su2_group)
        self.txt_su2_path = QLineEdit(config.su2_exe)
        self.txt_su2_path.setToolTip("Полный путь к исполняемому файлу SU2_CFD.exe")
        su2_lay.addWidget(QLabel("Путь к SU2_CFD.exe:"))
        su2_lay.addWidget(self.txt_su2_path)
        su2_btn_lay = QHBoxLayout()
        btn_browse_su2 = QPushButton("Обзор...")
        btn_browse_su2.clicked.connect(self.browse_su2_exe)
        btn_check_su2 = QPushButton("Проверить связь")
        btn_check_su2.clicked.connect(self.check_su2_connection)
        btn_save_su2 = QPushButton("Сохранить путь")
        btn_save_su2.clicked.connect(self.save_su2_path)
        btn_install_su2 = QPushButton("Авто-установка SU2")
        btn_install_su2.setStyleSheet("background-color: #2E5A78; color: #FBFBFC; font-weight: bold;")
        btn_install_su2.clicked.connect(self.install_su2_automatically)
        su2_btn_lay.addWidget(btn_browse_su2)
        su2_btn_lay.addWidget(btn_check_su2)
        su2_btn_lay.addWidget(btn_save_su2)
        su2_btn_lay.addWidget(btn_install_su2)
        su2_lay.addLayout(su2_btn_lay)
        self.lbl_su2_status = QLabel("Статус: не проверено")
        self.lbl_su2_status.setStyleSheet("color: #6B7280; font-style: italic;")
        su2_lay.addWidget(self.lbl_su2_status)
        lay9.addWidget(su2_group)
        solver_group = QGroupBox("Решатель")
        s_lay = QVBoxLayout(solver_group)
        self.combo_solver = QComboBox()
        # === T4: выбор турбомодели (SA / SST) =============================
        self.combo_solver.addItems([
            "Euler (невязкий)",
            "RANS SA (вязкий, Спаларт-Аллмарас)",
            "RANS SST (вязкий, Menter k-ω)",
        ])
        s_lay.addWidget(self.combo_solver)
        lay9.addWidget(solver_group)

        # Производительность: слайдеры нагрузки (CPU и GPU), вычислитель.
        # Пользователь задаёт проценты — программа сама считает,
        # сколько ядер CPU и долю GPU выделить.
        perf_group = QGroupBox("Производительность и ресурсы")
        perf_lay = QFormLayout(perf_group)
        cpu_info = _detect_cpu_cores()
        self._cpu_cores_max = cpu_info["physical"]
        self._cpu_cores_logical = cpu_info["logical"]
        self._cpu_cores_mpi_max = cpu_info["mpi_max"]

        # Слайдер «Нагрузка CPU» (10..100%).
        # cores = max(1, round(physical * percent / 100))
        cores_info_text = (
            f"Физических ядер CPU: {self._cpu_cores_max} "
            f"(логических: {self._cpu_cores_logical}). "
            f"SU2 использует mpiexec — безопасно до {self._cpu_cores_mpi_max}. "
            f"Ползунок задаёт процент использования физических ядер. "
            f"Само число MPI-процессов считается автоматически."
        )

        self.slider_cpu_load = QSlider(Qt.Horizontal)
        self.slider_cpu_load.setRange(10, 100)
        self.slider_cpu_load.setValue(50)  # 50% = половина физических
        self.slider_cpu_load.setTickPosition(QSlider.TicksBelow)
        self.slider_cpu_load.setTickInterval(10)
        self.slider_cpu_load.setToolTip(cores_info_text)
        self.lbl_cpu_load_value = QLabel("50%")
        self.lbl_cpu_load_value.setStyleSheet(
            "color: #2E5A78; font-weight: bold; min-width: 40px;"
        )
        # Точный ручной override (спинбокс, 0 = «по слайдеру»)
        self.spin_cpu_cores = QSpinBox()
        self.spin_cpu_cores.setRange(0, max(1, self._cpu_cores_mpi_max))
        self.spin_cpu_cores.setValue(0)
        self.spin_cpu_cores.setSpecialValueText("авто")
        self.spin_cpu_cores.setToolTip(
            "0 = «авто» (по слайдеру). "
            "1..{} — принудительное число ядер (перекрывает слайдер).".format(
                self._cpu_cores_mpi_max
            )
        )
        cpu_load_lay = QHBoxLayout()
        cpu_load_lay.addWidget(self.slider_cpu_load, 1)
        cpu_load_lay.addWidget(self.lbl_cpu_load_value)
        cpu_load_lay.addWidget(QLabel("  вручную:"))
        cpu_load_lay.addWidget(self.spin_cpu_cores)
        perf_lay.addRow("Нагрузка CPU:", cpu_load_lay)

        # === Вычислитель и нагрузка GPU из интерфейса убраны ============
        #
        # Официальные сборки SU2 (win64-omp, win64-mpi, linux64-omp,
        # linux64-mpi, macos64, macos64-mpi) видеокарту не используют: в
        # release-конфигурации SU2 опция -Denable-cuda не включена ни для
        # одной платформы. А сама поддержка CUDA в SU2 ограничена одним
        # местом — config_template.cfg: «Use CUDA GPU Acceleration for
        # FGMRES Linear Solver Only», то есть только произведение матрицы
        # на вектор во внутреннем цикле линейного решателя.
        #
        # Показывать пользователю выбор, который не работает, бессмысленно.
        # При этом весь служебный слой оставлен на месте: solver/gpu_launcher.py,
        # GPU-ветки в solver/workers.py, проводка ENABLE_CUDA в
        # solver/config_builder.py и поля _compute_device_pending /
        # _gpu_percent_pending. _current_device() жёстко возвращает "cpu",
        # поэтому код просто не выходит на GPU-ветки. Если в SU2 появится
        # рабочая GPU-сборка, вернуть интерфейс можно, не переписывая логику.
        #
        # self.combo_device, self.slider_gpu_load, self.lbl_gpu_load_value
        # и self.lbl_gpu_load_row больше не создаются; все обращения к ним
        # ниже защищены getattr().

        # === Опции расчёта: RAMP-разгон ================================
        # Галочки «Плоскость симметрии» больше нет: она дублировала список
        # плоскостей на странице Mesh 1. Симметрия включена тогда и только
        # тогда, когда добавлена хотя бы одна плоскость.
        opts_lay = QHBoxLayout()
        self.chk_use_ramp_aoa = QCheckBox("RAMP-разгон AoA")
        self.chk_use_ramp_aoa.setChecked(False)
        self.chk_use_ramp_aoa.setToolTip(
            "Плавно наращивать угол атаки от 0° до нужного за 100 итераций.\n"
            "Улучшает сходимость на жёстких моделях (высокие AoA, закрылки)."
        )
        opts_lay.addWidget(self.chk_use_ramp_aoa)

        self.chk_cfl_aggressive = QCheckBox("Быстрый CFL (до 1000)")
        self.chk_cfl_aggressive.setChecked(False)
        self.chk_cfl_aggressive.setToolTip(
            "Поднять потолок адаптивного CFL с 5 до 1000.\n"
            "Это главный рычаг скорости: неявная схема EULER_IMPLICIT при\n"
            "CFL 5 идёт очень мелким шагом. Официальный туториал SU2 по\n"
            "невязкому обтеканию ONERA M6 использует ( 0.1, 2.0, 100.0, 1e10 ).\n\n"
            "Включать только когда расчёт устойчиво сходится: на сетке,\n"
            "которая не разрешает геометрию, высокий CFL не ускоряет\n"
            "расчёт, а роняет его быстрее. При расходимости детектор\n"
            "застоя прервёт точку и предложит вернуться к осторожному CFL."
        )
        opts_lay.addWidget(self.chk_cfl_aggressive)
        opts_lay.addStretch()
        perf_lay.addRow(opts_lay)
        # =================================================================

        # === T1-визуал: Плоскости симметрии (XY/XZ/YZ) ================
        sym_group = QGroupBox("Плоскости симметрии")
        sym_lay = QVBoxLayout(sym_group)
        sym_info = QLabel(
            "Плоскости рисуются в 3D-окне. Добавленная плоскость — это и\n"
            "есть включённая симметрия: сетка обрежется по ней, и SU2\n"
            "посчитает только половину модели. Плоскость нужно добавить\n"
            "ДО построения сетки, иначе резка не выполнится."
        )
        sym_info.setStyleSheet("color: #4A4A4A; font-size: 10px;")
        sym_lay.addWidget(sym_info)

        # Кнопки добавления
        sym_btns = QHBoxLayout()
        for plane in ("xy", "xz", "yz"):
            btn = QPushButton(f"＋ {plane.upper()}")
            btn.setToolTip(f"Добавить плоскость {plane.upper()} (через начало координат)")
            btn.clicked.connect(lambda checked=False, p=plane: self._add_symmetry_plane(p))
            sym_btns.addWidget(btn)
        sym_btns.addStretch()
        sym_lay.addLayout(sym_btns)

        # Список плоскостей с кнопками удаления
        self.sym_list_layout = QVBoxLayout()
        self.sym_list_layout.setSpacing(2)
        sym_lay.addLayout(self.sym_list_layout)
        # Список плоскостей в виде списка словарей:
        # [{"axis": "xy", "enabled": True, "actor": ...}, ...]
        self._symmetry_planes = []
        lay8.addWidget(sym_group)
        # ================================================================

        # Кнопка применения + индикатор
        cores_apply_lay = QHBoxLayout()
        self.btn_apply_cores = QPushButton("Готово: Применить")
        self.btn_apply_cores.setToolTip(
            "Подтвердить выбранную нагрузку CPU. "
            "Используется при следующем запуске расчёта."
        )
        self.btn_apply_cores.clicked.connect(self.apply_load_level)
        self.lbl_cores_status = QLabel("—")
        self.lbl_cores_status.setStyleSheet("color: #2c4257; font-style: italic;")
        cores_apply_lay.addWidget(self.btn_apply_cores)
        cores_apply_lay.addWidget(self.lbl_cores_status)
        cores_apply_lay.addStretch()
        perf_lay.addRow(cores_apply_lay)
        lay9.addWidget(perf_group)

        # Подключаем сигналы и инициализируем индикатор
        self._connect_load_signals()
        self._refresh_load_status_label()

        mode_group = QGroupBox("Режим продувки")
        m_lay = QVBoxLayout(mode_group)
        self.rb_single = QRadioButton("Одиночный расчёт")
        self.rb_single.setChecked(True)
        self.rb_sweep = QRadioButton("Поляра (Sweep)")
        self.mode_buttons = QButtonGroup()
        self.mode_buttons.addButton(self.rb_single)
        self.mode_buttons.addButton(self.rb_sweep)
        m_lay.addWidget(self.rb_single)
        m_lay.addWidget(self.rb_sweep)
        lay9.addWidget(mode_group)
        sweep_group = QGroupBox("Параметры Sweep")
        swf = QFormLayout(sweep_group)
        self.input_aoa_start = QDoubleSpinBox(); self.input_aoa_start.setRange(-15, 25); self.input_aoa_start.setValue(-2)
        self.input_aoa_end = QDoubleSpinBox(); self.input_aoa_end.setRange(-15, 25); self.input_aoa_end.setValue(12)
        self.input_aoa_step = QDoubleSpinBox(); self.input_aoa_step.setRange(0.1, 5); self.input_aoa_step.setValue(2)
        swf.addRow("Старт AoA:", self.input_aoa_start)
        swf.addRow("Конец AoA:", self.input_aoa_end)
        swf.addRow("Шаг:", self.input_aoa_step)
        lay9.addWidget(sweep_group)
        self.rb_single.toggled.connect(self.update_mode_ui)
        self.rb_sweep.toggled.connect(self.update_mode_ui)
        self.btn_run = QPushButton("ЗАПУСТИТЬ РАСЧЁТ")
        self.btn_run.clicked.connect(self.start_calculation)
        self.btn_run.setEnabled(False)
        lay9.addWidget(self.btn_run)
        pause_resume_lay = QHBoxLayout()
        self.btn_pause = QPushButton("ПАУЗА")
        self.btn_pause.clicked.connect(self.pause_calculation)
        self.btn_pause.setEnabled(False)
        self.btn_resume = QPushButton("ВОЗОБНОВИТЬ")
        self.btn_resume.clicked.connect(self.resume_calculation)
        self.btn_resume.setEnabled(False)
        pause_resume_lay.addWidget(self.btn_pause)
        pause_resume_lay.addWidget(self.btn_resume)
        lay9.addLayout(pause_resume_lay)
        self.btn_cancel = QPushButton("ОТМЕНА")
        self.btn_cancel.clicked.connect(self.cancel_calculation)
        self.btn_cancel.setEnabled(False)
        lay9.addWidget(self.btn_cancel)
        self.settings_stack.addWidget(self.page_solver)

        # Page 10: Multipoint Optimization
        self.page_opt = QWidget()
        lay10 = QVBoxLayout(self.page_opt)
        lay10.setContentsMargins(10, 10, 10, 10)
        self.btn_start_opt = QPushButton("ЗАПУСТИТЬ ОПТИМИЗАЦИЮ")
        self.btn_start_opt.clicked.connect(self.run_geometric_optimization)
        lay10.addWidget(self.btn_start_opt)
        mp_group = QGroupBox("Многоточечная оптимизация")
        mp_lay = QVBoxLayout(mp_group)
        self.points_table = QTableWidget(0, 3)
        self.points_table.setHorizontalHeaderLabels(["Режим", "AoA °", "Вес"])
        mp_lay.addWidget(self.points_table)
        mp_btns = QHBoxLayout()
        btn_add_pt = QPushButton("Добавить точку")
        btn_add_pt.clicked.connect(self.add_opt_point)
        btn_load_preset_opt = QPushButton("Пресет")
        btn_load_preset_opt.clicked.connect(lambda: self.load_opt_points_preset("cruise"))
        mp_btns.addWidget(btn_add_pt)
        mp_btns.addWidget(btn_load_preset_opt)
        mp_lay.addLayout(mp_btns)
        lay10.addWidget(mp_group)
        self.opt_target_cl = QDoubleSpinBox(); self.opt_target_cl.setRange(0.01, 2.5); self.opt_target_cl.setValue(0.45)
        self.opt_target_k = QDoubleSpinBox(); self.opt_target_k.setRange(1, 100); self.opt_target_k.setValue(15)
        lay10.addWidget(QLabel("Целевой Cl:"))
        lay10.addWidget(self.opt_target_cl)
        lay10.addWidget(QLabel("Целевое K:"))
        lay10.addWidget(self.opt_target_k)
        self.lbl_opt_status = QLabel("Ожидание...")
        self.lbl_opt_status.setStyleSheet("color: #666; font-style: italic;")
        lay10.addWidget(self.lbl_opt_status)

        # ---- Табличная оптимизация (DOE): перебор вариантов по таблице ----
        doe_group = QGroupBox("Табличная оптимизация (перебор вариантов)")
        doe_lay = QVBoxLayout(doe_group)
        self.doe_table = QTableWidget(0, len(self._doe_param_names()))
        self.doe_table.setHorizontalHeaderLabels(self._doe_param_labels())
        self.doe_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.doe_table.setSelectionMode(QTableWidget.ExtendedSelection)
        doe_lay.addWidget(self.doe_table)
        doe_btns = QHBoxLayout()
        btn_doe_from_current = QPushButton("Из текущих параметров")
        btn_doe_from_current.setToolTip(
            "Добавить строку со значениями из генератора крыла.")
        btn_doe_from_current.clicked.connect(self.add_doe_row_from_current)
        btn_doe_add = QPushButton("Ввести вручную…")
        btn_doe_add.clicked.connect(self.add_doe_row_dialog)
        btn_doe_del = QPushButton("Удалить выбранное")
        btn_doe_del.clicked.connect(self.remove_doe_rows)
        btn_doe_clear = QPushButton("Очистить")
        btn_doe_clear.clicked.connect(self.clear_doe_table)
        btn_doe_grid = QPushButton("Сетка вариантов…")
        btn_doe_grid.setToolTip(
            "Сгенерировать таблицу по диапазонам параметров: полный "
            "факторный план, варьирование по одному параметру или "
            "латинский гиперкуб.")
        btn_doe_grid.clicked.connect(self.show_doe_grid_dialog)
        doe_btns.addWidget(btn_doe_from_current)
        doe_btns.addWidget(btn_doe_add)
        doe_btns.addWidget(btn_doe_grid)
        doe_btns.addWidget(btn_doe_del)
        doe_btns.addWidget(btn_doe_clear)
        doe_btns.addStretch()
        doe_lay.addLayout(doe_btns)
        gen_lay = QHBoxLayout()
        gen_lay.addWidget(QLabel("Поколений:"))
        self.doe_generations = QSpinBox()
        self.doe_generations.setRange(1, 10)
        self.doe_generations.setValue(1)
        self.doe_generations.setToolTip(
            "Сколько поколений перебора выполнить. Со 2-го поколения "
            "диапазоны сужаются вдвое вокруг лучшего варианта "
            "предыдущего поколения.")
        gen_lay.addWidget(self.doe_generations)
        gen_lay.addWidget(QLabel("Сужение диапазона:"))
        self.doe_shrink = QDoubleSpinBox()
        self.doe_shrink.setRange(0.1, 1.0)
        self.doe_shrink.setSingleStep(0.1)
        self.doe_shrink.setValue(0.5)
        gen_lay.addWidget(self.doe_shrink)
        gen_lay.addStretch()
        doe_lay.addLayout(gen_lay)
        self.doe_results = QTableWidget(0, 4)
        self.doe_results.setHorizontalHeaderLabels(
            ["Вариант", "Cl", "K", "Статус"])
        self.doe_results.horizontalHeader().setStretchLastSection(True)
        doe_lay.addWidget(self.doe_results)
        doe_run_lay = QHBoxLayout()
        self.btn_start_doe = QPushButton("ЗАПУСТИТЬ ПЕРЕБОР")
        self.btn_start_doe.setToolTip(
            "Прогнать все строки таблицы: каждая строка = один расчёт "
            "(геометрия перестраивается, сетка перегенерируется).")
        self.btn_start_doe.clicked.connect(self.run_doe_optimization)
        self.lbl_doe_status = QLabel("—")
        self.lbl_doe_status.setStyleSheet("color: #666; font-style: italic;")
        doe_run_lay.addWidget(self.btn_start_doe)
        doe_run_lay.addWidget(self.lbl_doe_status, 1)
        doe_lay.addLayout(doe_run_lay)
        lay10.addWidget(doe_group)
        self.settings_stack.addWidget(self.page_opt)

        # Page 11: Trim & Balancing
        self.page_trim = QWidget()
        lay11 = QVBoxLayout(self.page_trim)
        lay11.setContentsMargins(10, 10, 10, 10)
        self.trim_arm = QDoubleSpinBox(); self.trim_arm.setRange(0.1, 100.0); self.trim_arm.setValue(5.0); self.trim_arm.setSuffix(" м")
        self.trim_eff = QDoubleSpinBox(); self.trim_eff.setRange(0.001, 1.0); self.trim_eff.setValue(0.015); self.trim_eff.setDecimals(3)
        self.lbl_trim_result = QLabel("Расчет не производился")
        self.lbl_trim_result.setStyleSheet("color: #2E6B45; font-weight: bold; font-size: 13px;")
        btn_calc_trim = QPushButton("Рассчитать балансировку")
        btn_calc_trim.clicked.connect(self.calculate_aerodynamic_trim)
        lay11.addWidget(QLabel("Плечо ГО (L_ht):"))
        lay11.addWidget(self.trim_arm)
        lay11.addWidget(QLabel("Эффективность руля (dCm/d_de):"))
        lay11.addWidget(self.trim_eff)
        lay11.addWidget(btn_calc_trim)
        lay11.addWidget(self.lbl_trim_result)
        self.settings_stack.addWidget(self.page_trim)

        # Page 12: Flow Visualization
        self.page_flow_viz = QWidget()
        lay_viz = QVBoxLayout(self.page_flow_viz)
        lay_viz.setContentsMargins(10, 10, 10, 10)
        vg = QGroupBox("Параметры визуализации")
        v_lay = QFormLayout(vg)
        self.combo_scalar = QComboBox()
        self.combo_scalar.setEnabled(False)
        self.combo_scalar.currentIndexChanged.connect(self.render_flow_scene)
        self.chk_show_volume = QCheckBox("Объёмная сетка")
        self.chk_show_volume.stateChanged.connect(self.render_flow_scene)
        self.chk_show_arrows = QCheckBox("Стрелки скорости")
        self.chk_show_arrows.stateChanged.connect(self.render_flow_scene)
        self.btn_show_flow = QPushButton("Визуализировать обтекание")
        self.btn_show_flow.clicked.connect(self.show_flow_field)
        self.btn_show_flow.setEnabled(False)
        v_lay.addRow("Карта поля:", self.combo_scalar)
        v_lay.addRow(self.chk_show_volume)
        v_lay.addRow(self.chk_show_arrows)
        v_lay.addRow(self.btn_show_flow)
        lay_viz.addWidget(vg)
        self.settings_stack.addWidget(self.page_flow_viz)

        # Page 13: Generation History
        self.page_history = QWidget()
        lay12 = QVBoxLayout(self.page_history)
        lay12.setContentsMargins(10, 10, 10, 10)
        self.history_table = QTableWidget(0, 8)
        self.history_table.setHorizontalHeaderLabels([
            "ID", "Размах (м)", "Root хорда", "Tip хорда",
            "Стрел-ть (°)", "Крутка (°)", "Качество K", "Действие",
        ])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        lay12.addWidget(self.history_table)
        btn_clear_history = QPushButton("Очистить историю")
        btn_clear_history.clicked.connect(self.clear_generation_history)
        lay12.addWidget(btn_clear_history)
        self.settings_stack.addWidget(self.page_history)

        # Page 14-17: аэроупругость, прочность, спецфункции, пресеты (ТЗ)
        from ui.analysis_pages import (
            build_aeroelastic_page, build_presets_page, build_info_page,
            build_specials_page, build_structural_page)
        self.page_aeroelastic, self.ae_w = build_aeroelastic_page(
            on_check=self.run_aeroelastic_check,
            on_plot=self.plot_vg_diagram)
        self.settings_stack.addWidget(self.page_aeroelastic)
        self.page_structural, self.st_w = build_structural_page(
            on_calc=self.run_structural_check)
        self.settings_stack.addWidget(self.page_structural)
        self.page_specials, self.sp_w = build_specials_page(
            on_polar=self.build_polar_from_results,
            on_report=self.export_analysis_report,
            on_csv=self.export_polar_csv)
        self.settings_stack.addWidget(self.page_specials)
        # Корневые узлы дерева показывают поясняющий текст, а не поля:
        # настройки живут в дочерних узлах, дублировать их кнопками здесь
        # незачем. Кнопки перехода стоят в самом низу страницы.
        def _goto(node):
            def _h():
                self.tree.setCurrentItem(node)
                self.tree.scrollToItem(node)
            return _h
        self.page_info_global = build_info_page("global_defs", [
            ("Условия полёта и правила", "Полётные условия и правила "
             "проектирования", _goto(self.item_rules)),
            ("Формат конфигурации", "Именованные пресеты config.cfg",
             _goto(self.item_presets)),
        ])
        self.settings_stack.addWidget(self.page_info_global)
        self.page_info_component = build_info_page("component", [
            ("Состав модели", "Список деталей и их роли",
             _goto(self.item_components)),
            ("Расчётная сетка", "Построение объёмной сетки",
             _goto(self.item_mesh)),
        ])
        self.settings_stack.addWidget(self.page_info_component)
        self.page_info_study = build_info_page("study", [
            ("Настройки решателя", "Тип уравнений, итерации, ядра",
             _goto(self.item_solver)),
            ("Перебор вариантов", "Многоточечная оптимизация",
             _goto(self.item_opt)),
            ("Аэроупругость", "Флаттер и дивергенция",
             _goto(self.item_aeroelastic)),
        ])
        self.settings_stack.addWidget(self.page_info_study)
        self.page_info_results = build_info_page("results", [
            ("Балансировка", "Подбор отклонения руля высоты",
             _goto(self.item_trim)),
            ("Поле обтекания", "Распределение давлений по поверхности",
             _goto(self.item_flow_viz)),
            ("Поляра и отчёты", "Поляра, наилучшее качество, выгрузка",
             _goto(self.item_specials)),
        ])
        self.settings_stack.addWidget(self.page_info_results)

        self.page_presets, self.pr_w = build_presets_page(
            on_export=self.export_config_preset,
            on_import=self.import_config_preset,
            on_apply=self.apply_imported_preset)
        self.settings_stack.addWidget(self.page_presets)
        self._imported_preset = None

        self.tree.itemSelectionChanged.connect(self.on_tree_selection_changed)

        # ==================== ПРАВАЯ ЧАСТЬ ============================
        self.right_panel_container = QWidget()
        right_outer_lay = QVBoxLayout(self.right_panel_container)
        right_outer_lay.setContentsMargins(0, 0, 0, 0)
        right_outer_lay.setSpacing(5)
        self.right_splitter = QSplitter(Qt.Vertical)
        right_outer_lay.addWidget(self.right_splitter)

        # 3D-сцена
        self.plotter_widget = QWidget()
        plotter_layout = QVBoxLayout(self.plotter_widget)
        plotter_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self.plotter_widget)
        self.plotter.add_axes()
        try:
            self.plotter.enable_point_picking(
                callback=self.on_3d_pick, show_message=False,
                picker='point', left_clicking=True)
        except Exception:
            pass
        self._plot_interactor = getattr(self.plotter, "interactor", None) \
            or self.plotter
        plotter_layout.addWidget(self._plot_interactor)
        self.right_splitter.addWidget(self.plotter_widget)

        # нижние вкладки: лог / таблица / графики
        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.setMinimumHeight(180)
        self.right_splitter.addWidget(self.bottom_tabs)
        tab_log_inner = QWidget()
        log_lay_in = QVBoxLayout(tab_log_inner)
        log_lay_in.setContentsMargins(3, 3, 3, 3)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_lay_in.addWidget(self.log_text)
        self.bottom_tabs.addTab(tab_log_inner, "Сообщения / Лог (Messages)")
        tab_results_inner = QWidget()
        res_lay_in = QVBoxLayout(tab_results_inner)
        res_lay_in.setContentsMargins(3, 3, 3, 3)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["AoA", "Cl", "Cd", "Cm", "K", "Статус"])
        res_lay_in.addWidget(self.table)
        self.btn_save_csv = QPushButton("Экспорт поляры CSV")
        self.btn_save_csv.clicked.connect(self.save_polar_csv)
        self.btn_save_csv.setEnabled(False)
        res_lay_in.addWidget(self.btn_save_csv)
        self.bottom_tabs.addTab(tab_results_inner, "Таблица результатов (Results)")
        self.plot_canvas = AeroPlotCanvas(self, width=5, height=3)
        self.bottom_tabs.addTab(self.plot_canvas, "2D Аэро Графики (Aero Plots)")
        self.right_splitter.setSizes([600, 300])
        self.outer_splitter.addWidget(self.left_panel_splitter)
        self.outer_splitter.addWidget(self.right_panel_container)
        self.outer_splitter.setSizes([650, 1100])

        # ==================== СТАТУС-БАР ==============================
        sb = self.statusBar()
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(15)
        self.progress.setMaximumWidth(200)
        self.progress.setVisible(False)
        sb.addPermanentWidget(self.progress)
        self.lbl_status_time = QLabel("")
        self.lbl_status_time.setStyleSheet("font-weight: bold; color: #2E5A78; margin-right: 15px;")
        sb.addPermanentWidget(self.lbl_status_time)
        # Живые показатели как в диспетчере задач: ЦПУ и память.
        # ГПУ убран: SU2 считает только на CPU, и счётчик загрузки
        # видеокарты рядом с показателями расчёта вводил в заблуждение.
        _mono = "font-family: Consolas, monospace; color: #2c4257; margin-right: 12px;"
        self.lbl_status_cpu = QLabel("ЦПУ н/д")
        self.lbl_status_cpu.setStyleSheet(_mono)
        sb.addPermanentWidget(self.lbl_status_cpu)
        self.lbl_status_memory = QLabel("Память н/д")
        self.lbl_status_memory.setStyleSheet(_mono)
        sb.addPermanentWidget(self.lbl_status_memory)

        # CAD-навигация
        self.plotter.enable_trackball_style()
        try:
            self._nav_filter = CADNavigationEventFilter(self._plot_interactor)
            self._plot_interactor.installEventFilter(self._nav_filter)
        except Exception:
            self._nav_filter = None

        self.project_saved = True
        self.mem_timer = QTimer(self)
        self.mem_timer.timeout.connect(self.update_system_status)
        self.mem_timer.start(2000)
        # Часы тикают собственным таймером: раньше надпись времени
        # перерисовывалась только по событиям прогресса, поэтому шла рывками.
        self._clock_start = None
        self._clock_eta = None
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._tick_clock)
        self.clock_timer.start(250)
        self._system_monitor = None
        QTimer.singleShot(400, self.update_system_status)
        # По умолчанию открываем Global Definitions (там теперь и условия, и правила)
        self.tree.setCurrentItem(self.item_global_defs)
        self.update_isa()
        self.update_mode_ui()
        self.log_text.append("Готово: AeroOpt v4.0 запущен.")
        if hasattr(self, 'points_table'):
            self.load_opt_points_preset("cruise")
        self.set_calculation_buttons_enabled(run=False, pause=False,
                                             resume=False, cancel=False)
        self._check_pending_session()
        QTimer.singleShot(300, self._check_first_launch)

    # =============================================================
    # ПАМЯТЬ (без psutil: ctypes на Windows, resource на Unix)
    # =============================================================
    # ------------------------------------------------------------------
    # Живые показатели ЦПУ / ГПУ / памяти
    # ------------------------------------------------------------------
    def update_system_status(self):
        """Обновляет ЦПУ, ГПУ и память в панели состояния.

        Память процесса берётся через GetProcessMemoryInfo с явными
        argtypes: прежний вызов без объявления типов на 64-битной Windows
        возвращал ноль, поэтому индикатор всегда был пустым.
        """
        try:
            from ui.system_monitor import SystemMonitor
            if self._system_monitor is None:
                self._system_monitor = SystemMonitor()
            cores_used = self._cpu_cores_override() or None
            snap = self._system_monitor.snapshot(cores_used=cores_used)
            labels = SystemMonitor.labels(snap)
            self.lbl_status_cpu.setText(labels["cpu"])
            pass  # индикатора ГПУ в статус-баре больше нет
            self.lbl_status_memory.setText(labels["mem"])
            if snap["rss"] is None and not getattr(
                    self, "_psutil_install_attempted", False):
                self._psutil_install_attempted = True
                self._try_install_psutil()
        except Exception:
            self.lbl_status_cpu.setText("ЦПУ н/д")
            pass  # индикатора ГПУ в статус-баре больше нет
            self.lbl_status_memory.setText("Память н/д")

    # ------------------------------------------------------------------
    # Часы: собственный таймер, не зависит от событий прогресса
    # ------------------------------------------------------------------
    def _clock_begin(self, eta=None):
        self._clock_start = time.time()
        self._clock_eta = eta
        self._tick_clock()

    def _clock_set_eta(self, eta):
        """Обновляет оценку остатка; часы перерисует собственный таймер."""
        self._clock_eta = eta
        if self._clock_start is None:
            self._clock_start = time.time()

    def _clock_end(self):
        self._clock_start = None
        self._clock_eta = None
        self.lbl_status_time.setText("")

    # ------------------------------------------------------------------
    # Полоса прогресса: шаг не мельче 2%
    # ------------------------------------------------------------------
    def _set_progress(self, percent):
        """Двигает полосу шагами не мельче 2%.

        Воркер присылает процент по каждой записи SU2 (при INNER_ITER=6000
        и частоте вывода 50 это ~120 событий, шаг меньше процента). Без
        порога полоса дрожит; с порогом идёт заметными шагами 2-5%.
        Крайние значения 0 и 100 применяются всегда.
        """
        try:
            value = max(0, min(100, int(round(float(percent)))))
        except (TypeError, ValueError):
            return
        if value in (0, 100) or value - self.progress.value() >= 2 \
                or value < self.progress.value():
            self.progress.setValue(value)

    def _tick_clock(self):
        if not self._clock_start:
            return
        mins, secs = divmod(int(time.time() - self._clock_start), 60)
        text = f"{mins:02d}:{secs:02d}"
        eta = self._clock_eta
        if eta and eta > 0:
            em, es = divmod(int(eta), 60)
            text += f"  ·  осталось {em}м {es:02d}с"
        self.lbl_status_time.setText(text)

    # Совместимость со старыми вызовами.
    def update_memory_status(self):
        self.update_system_status()

    def _try_install_psutil(self):
        """Докачка psutil при первом запуске (только в режиме разработки).

        В собранном .exe НИЧЕГО не ставим: sys.executable там — сам
        AeroOpt.exe, и вызов 'python -m pip' перезапустил бы копию
        приложения. psutil либо вшит в сборку, либо работает ctypes-fallback.
        """
        if getattr(sys, "frozen", False):
            return

        class _Installer(QThread):
            def run(_self):
                try:
                    import subprocess
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--quiet",
                         "psutil", "--disable-pip-version-check"],
                        capture_output=True, timeout=90,
                    )
                except Exception:
                    pass

        try:
            self._mem_install_worker = _Installer(self)
            self._mem_install_worker.start()
        except Exception:
            pass

    def set_calculation_buttons_enabled(self, run=False, pause=False,
                                        resume=False, cancel=False):
        """Синхронизация визуального состояния кнопок расчёта (ТЗ 2.3 / п.2)."""
        self.btn_run.setEnabled(run)
        self.ribbon_btn_run.setEnabled(run)
        self.btn_pause.setEnabled(pause)
        self.ribbon_btn_pause.setEnabled(pause)
        self.btn_resume.setEnabled(resume)
        self.ribbon_btn_resume.setEnabled(resume)
        self.btn_cancel.setEnabled(cancel)
        self.ribbon_btn_cancel.setEnabled(cancel)
        busy = pause or cancel
        self.btn_make_mesh.setEnabled(not busy and not self._meshing)
        self.ribbon_btn_mesh.setEnabled(not busy and not self._meshing)

    # =============================================================
    # ПОЛЁТНЫЕ УСЛОВИЯ (новое)
    # =============================================================
    def on_flight_preset_changed(self, preset_name: str):
        if not preset_name or preset_name not in FLIGHT_PRESETS:
            return
        fc = FLIGHT_PRESETS[preset_name]
        # Подставляем значения в поля (без рекурсивного срабатывания)
        for w in (self.input_speed, self.input_alt, self.input_aoa):
            w.blockSignals(True)
        try:
            self.input_speed.setValue(fc.speed_m_s)
            self.input_alt.setValue(int(fc.altitude_m))
            self.input_aoa.setValue(fc.aoa_deg)
        finally:
            for w in (self.input_speed, self.input_alt, self.input_aoa):
                w.blockSignals(False)
        self.update_isa()
        self.log_text.append(
            f"Пресет полётных условий: {preset_name} "
            f"(примените кнопкой «Готово: Применить условия полёта»)"
        )

    def on_flight_field_changed(self, *_args):
        # При ручном изменении просто обновляем ISA-метку,
        # но НЕ применяем self.flight (для этого отдельная кнопка)
        self.update_isa()

    def apply_flight_conditions(self):
        self.flight.speed_m_s = float(self.input_speed.value())
        self.flight.altitude_m = float(self.input_alt.value())
        self.flight.aoa_deg = float(self.input_aoa.value())
        # Имя — из пресета, если выбран; иначе «Ручной режим»
        preset = self.combo_flight_preset.currentText()
        if preset and preset in FLIGHT_PRESETS and \
                abs(FLIGHT_PRESETS[preset].speed_m_s - self.flight.speed_m_s) < 1e-6 and \
                abs(FLIGHT_PRESETS[preset].altitude_m - self.flight.altitude_m) < 1e-6 and \
                abs(FLIGHT_PRESETS[preset].aoa_deg - self.flight.aoa_deg) < 1e-6:
            self.flight.name = FLIGHT_PRESETS[preset].name
            self.flight.preset_name = preset
        else:
            self.flight.name = "Ручной режим"
            self.flight.preset_name = ""
        self.log_text.append(
            f"Готово: Применены условия полёта: V={self.flight.speed_m_s:.1f} м/с, "
            f"H={self.flight.altitude_m:.0f} м, AoA={self.flight.aoa_deg:.2f}° "
            f"({self.flight.name})"
        )

    def apply_cpu_cores(self, new_value: int):
        """Совместимостный wrapper: применяет заданное количество ядер.

        Сейчас UI использует выпадающий список уровня нагрузки; этот
        метод оставлен, чтобы старые вызовы (если где-то есть) работали."""
        new_value = max(1, int(new_value))
        max_cores = getattr(self, "_cpu_cores_max", None)
        if max_cores and new_value > max_cores:
            self.log_text.append(
                f"Внимание: Запрошено {new_value} ядер, "
                f"но физически доступно {max_cores}. "
                f"Применяю {max_cores}."
            )
            new_value = max_cores
        self._cpu_cores_pending = new_value
        suffix = (
            f"Применено: {new_value} ядер (доступно: {max_cores})"
            if max_cores else f"Применено: {new_value} ядер"
        )
        try:
            self.lbl_cores_status.setText(suffix)
        except Exception:
            pass
        self.log_text.append(f"Готово: Применено количество ядер CPU: {new_value}")

    # ------------------------------------------------------------------
    # Нагрузка CPU/GPU через слайдеры процентов
    # ------------------------------------------------------------------
    # Пользователь задаёт процент нагрузки CPU (10..100%) и GPU (0..100%).
    # Программа сама считает число MPI-процессов и долю GPU-ресурса.
    # Спинбокс «вручную» (0=авто, N>0=принудительно) перекрывает слайдер CPU.
    def _cpu_load_percent(self) -> int:
        """Текущее значение слайдера нагрузки CPU (10..100)."""
        try:
            return int(self.slider_cpu_load.value())
        except Exception:
            return 50

    def _gpu_load_percent(self) -> int:
        """Текущее значение слайдера нагрузки GPU (0..100)."""
        try:
            # Слайдера нагрузки GPU в интерфейсе нет — доля GPU всегда 0.
            return 0
        except Exception:
            return 0

    def _cpu_cores_override(self) -> int:
        """Спинбокс ручного override: 0 = «по слайдеру», >0 = принудительно."""
        try:
            return int(self.spin_cpu_cores.value())
        except Exception:
            return 0

    def _resolve_cores_for_level(self) -> int:
        """Считает число MPI-процессов CPU из слайдера/спинбокса.

        Приоритет:
          1. spin_cpu_cores > 0 (ручной override).
          2. Иначе: physical * percent / 100.
        Ограничивает по mpiexec_max. В CPU+GPU — оставляет минимум
        одно ядро на CPU-часть (FFT, preconditioner).
        """
        physical = max(1, int(self._cpu_cores_max or 1))
        mpi_max = int(self._cpu_cores_mpi_max or physical)
        override = self._cpu_cores_override()
        if override > 0:
            cores = override
        else:
            percent = max(10, min(100, self._cpu_load_percent()))
            cores = max(1, int(round(physical * percent / 100.0)))
        cores = min(max(1, cores), mpi_max)
        # CPU+GPU: не даём занять все ядра — минимум 1 на CPU-часть,
        # максимум 75% ядер отдаём CPU, остальное можно GPU.
        if self._current_device() == "cpu_gpu" and cores > 1:
            cores = max(1, min(cores, max(1, int(physical * 0.75))))
        return max(1, cores)

    def _current_device(self) -> str:
        """Всегда "cpu": выбора вычислителя в интерфейсе нет.

        GPU-ветки в solver/workers.py оставлены, но недостижимы, пока
        _current_device() возвращает "cpu".
        """
        try:
            return "cpu"
            idx = 0
        except Exception:
            idx = 0
        return "cpu_gpu" if idx == 1 else "cpu"

    def _refresh_load_status_label(self):
        """Обновляет lbl_cores_status: 'Применено: 4 ядер CPU (50%)'.

        GPU в индикаторе не показывается: выбора вычислителя в
        интерфейсе нет, _current_device() всегда возвращает "cpu".
        """
        if not hasattr(self, "lbl_cores_status"):
            return
        try:
            cpu_pct = self._cpu_load_percent()
            self.lbl_cpu_load_value.setText(f"{cpu_pct}%")
        except Exception:
            cpu_pct = 50
        cores = self._resolve_cores_for_level()
        try:
            self.lbl_cores_status.setText(
                f"Применено: {cores} ядер CPU ({cpu_pct}%)"
            )
        except Exception:
            pass

    def _connect_load_signals(self):
        """Подключает автообновление индикатора при изменении слайдеров/combo/spinbox."""
        if getattr(self, "_load_signals_connected", False):
            return
        try:
            self.slider_cpu_load.valueChanged.connect(
                lambda *_: self._refresh_load_status_label()
            )
            # Слайдера нагрузки GPU и выбора вычислителя в интерфейсе
            # нет, сигналы от них не подключаются.
            self.spin_cpu_cores.valueChanged.connect(
                lambda *_: self._refresh_load_status_label()
            )
        except Exception:
            pass
        self._load_signals_connected = True

    def _detect_gpus(self) -> list:
        """Возвращает список найденных GPU (словари {name, mem_mb}).

        Использует nvidia-smi (NVIDIA), rocm-smi (AMD ROCm) или WMI.
        При отсутствии всех — возвращает [] (CPU-only fallback)."""
        result = []
        # NVIDIA — nvidia-smi
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
                **hidden_subprocess_kwargs(),
            )
            if r.returncode == 0 and r.stdout.strip():
                for line in r.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        try:
                            mem = int(float(parts[1]))
                        except ValueError:
                            mem = 0
                        result.append({"name": parts[0], "mem_mb": mem, "vendor": "NVIDIA"})
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        except Exception:
            pass
        # AMD — rocm-smi
        if not result:
            try:
                import subprocess
                r = subprocess.run(
                    ["rocm-smi", "--showproductname", "--csv"],
                    capture_output=True, text=True, timeout=5,
                    **hidden_subprocess_kwargs(),
                )
                if r.returncode == 0 and r.stdout.strip():
                    for line in r.stdout.strip().splitlines()[1:]:
                        name = line.split(",")[0].strip()
                        if name:
                            result.append({"name": name, "mem_mb": 0, "vendor": "AMD"})
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                pass
            except Exception:
                pass
        # Windows — WMI (fallback)
        if not result and sys.platform == "win32":
            try:
                import subprocess
                ps = (
                    "Get-WmiObject Win32_VideoController | "
                    "Select-Object -Property Name, AdapterRAM | "
                    "ConvertTo-Csv -NoTypeInformation"
                )
                r = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps],
                    capture_output=True, text=True, timeout=10,
                    **hidden_subprocess_kwargs(),
                )
                if r.returncode == 0:
                    for line in r.stdout.strip().splitlines()[1:]:
                        if "," not in line:
                            continue
                        try:
                            name = line.split(",")[0].strip().strip('"')
                            ram = int(line.split(",")[1].strip().strip('"') or 0)
                            if name and "Microsoft Basic" not in name:
                                result.append({
                                    "name": name, "mem_mb": ram // (1024 * 1024),
                                    "vendor": "Unknown"
                                })
                        except (ValueError, IndexError):
                            continue
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                pass
            except Exception:
                pass
        return result

    def apply_load_level(self):
        """Применяет выбранные нагрузки CPU/GPU. Без диалогов.

        Фиксирует:
          * self._cpu_cores_pending     — число MPI-процессов
          * self._compute_device_pending — "cpu" или "cpu_gpu"
          * self._gpu_percent_pending    — 0..100 (доля GPU)

        Если GPU не обнаружены в системе — автоматический откат на чистый
        CPU с подробным логом. Старые вызовы и проекты остаются рабочими
        (по умолчанию device = "cpu", gpu_percent = 0)."""
        device = self._current_device()
        cores = self._resolve_cores_for_level()
        self._cpu_cores_pending = cores
        # Выбора вычислителя в интерфейсе нет: всегда CPU. Служебные
        # поля GPU-режима обнулены и оставлены только потому, что их
        # читает solver/workers.py и файл проекта.
        gpu_info = ""
        device = "cpu"
        self._compute_device_pending = "cpu"
        self._gpu_percent_pending = 0
        self._gpu_percent_last_applied = 0
        self.log_text.append(
            f"Готово: Применено: {cores} ядер CPU ({self._cpu_load_percent()}%)"
        )
        # Обновляем индикатор
        self._refresh_load_status_label()

    # =============================================================
    # ПЕРВЫЙ ЗАПУСК / SU2 (ТЗ 2.2)
    # =============================================================
    # =============================================================
    # T6: МЕТОДЫ ЛИЦЕНЗИРОВАНИЯ
    # =============================================================
    def _show_activate_dialog(self):
        """Диалог активации лицензионного ключа через сервер AeroOpt.

        Ввод ключа -> POST /v1/activate на воркере (привязка HWID в D1).
        Никакого локального разбора licenses.json и ручного HMAC здесь нет —
        всё это делает LicenseChecker.activate().
        """
        import logging
        try:
            from PyQt5.QtWidgets import (QApplication, QInputDialog,
                                         QMessageBox)
            from PyQt5.QtCore import Qt
        except ImportError:  # запасной вариант, если проект на PySide2
            from PySide2.QtWidgets import (QApplication, QInputDialog,
                                           QMessageBox)
            from PySide2.QtCore import Qt

        logger = logging.getLogger("aeroopt")

        try:
            key, ok = QInputDialog.getText(
                self,
                "Активация лицензии AeroOpt",
                "Введите лицензионный ключ:\n\n"
                "(формат: AERO-XXXX-XXXX-XXXX-XXXXX)\n\n"
                "Нет ключа? Напишите на sales@aeroopt.app",
            )
            if not ok:
                logger.info("Активация отменена пользователем")
                return

            key = (key or "").strip().upper()
            if not key:
                QMessageBox.warning(
                    self, "Активация лицензии",
                    "Ключ не введён.")
                return

            logger.info("Попытка активации ключа: %s", key)

            # Сетевой запрос (до ~20 с) — показываем курсор ожидания
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                success, message = self._license.activate(key)
            except Exception as e:
                logger.exception("Исключение при активации")
                success, message = False, f"Ошибка запроса к серверу: {e}"
            finally:
                QApplication.restoreOverrideCursor()

            if success:
                logger.info("Активация успешна: %s", message)
                QMessageBox.information(
                    self, "Активация лицензии",
                    f"Готово: Лицензия успешно активирована.\n\n{message}")
                # Обновить индикатор лицензии в главном окне, если такой
                # метод есть (имена в разных версиях UI могут отличаться)
                for meth in ("_update_license_status",
                             "_refresh_license_status",
                             "_update_license_ui",
                             "_show_license_status"):
                    fn = getattr(self, meth, None)
                    if callable(fn):
                        try:
                            fn()
                        except Exception:
                            logger.exception(
                                "Не удалось обновить индикатор лицензии "
                                "(метод %s)", meth)
                        break
            else:
                logger.warning("Активация отклонена: %s", message)
                QMessageBox.warning(
                    self, "Активация лицензии",
                    f"Не удалось активировать ключ.\n\n{message}\n\n"
                    "Если ключ введён верно, но ошибка повторяется — "
                    "напишите на sales@aeroopt.app "
                    "(с приложением лога сессии).")
        except Exception:
            logger.exception("Необработанное исключение в _show_activate_dialog")
            try:
                QMessageBox.critical(
                    self, "Активация лицензии",
                    "Внутренняя ошибка диалога активации.\n"
                    "Подробности — в логе сессии: %APPDATA%\\AeroOpt\\logs")
            except Exception:
                pass

    def _show_license_status(self):
        """Показывает текущий статус лицензии."""
        if self._license is None:
            QMessageBox.information(
                self, "Лицензия",
                "Модуль лицензирования недоступен."
            )
            return
        info = self._license.get_activation_info()
        text = (
            f"Статус: {info['status_text']}\n"
            f"Ключ: {info['license_key'] or '—'}\n"
            f"Продукт: {info['product'] or '—'}\n"
            f"HWID: {info['hwid']}\n"
            f"Привязок: {info['hwid_count']} / {info['hwid_max']}\n"
        )
        if info['expires_at']:
            text += f"Истекает: {time.strftime('%Y-%m-%d %H:%M', time.localtime(info['expires_at']))}\n"
        if info['last_heartbeat']:
            text += f"Последний heartbeat: {time.strftime('%Y-%m-%d %H:%M', time.localtime(info['last_heartbeat']))}\n"
        # Пробуем сделать heartbeat прямо сейчас
        ok, hb_msg = self._license.heartbeat()
        text += f"\nHeartbeat: {'Готово: ' + hb_msg if ok else '' + hb_msg}"
        QMessageBox.information(self, "Статус лицензии", text)

    def _deactivate_license(self):
        """Отвязывает HWID от license_key."""
        if self._license is None or not self._license.license_key:
            QMessageBox.information(
                self, "Лицензия", "Нет активной привязки для отвязки."
            )
            return
        reply = QMessageBox.question(
            self, "Отвязка лицензии",
            f"Отвязать эту машину (HWID {self._license.hwid[:12]}...)\n"
            f"от ключа {self._license.license_key}?\n\n"
            "После этого приложение перестанет работать "
            "до повторной активации.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        ok, msg = self._license.deactivate()
        QApplication.restoreOverrideCursor()
        if ok:
            QMessageBox.information(
                self, "Отвязка",
                "Готово: Лицензия отвязана. Закройте приложение."
            )
            self.log_text.append("Лицензия отвязана")
        else:
            QMessageBox.warning(
                self, "Ошибка",
                f"Сервер вернул ошибку: {msg}\n\n"
                "Локальная привязка всё равно сброшена."
            )

    # =============================================================
    # T1-ВИЗУАЛ: Плоскости симметрии (XY / XZ / YZ)
    # =============================================================
    # Нормали и буквы осей для плоскостей симметрии. Вынесены в чистую
    # функцию, чтобы состояние видимости можно было проверить тестом
    # без отрисовки.
    _SYM_VIEW_NORMAL = {"xz": (0.0, 1.0, 0.0),
                        "xy": (0.0, 0.0, 1.0),
                        "yz": (1.0, 0.0, 0.0)}
    _SYM_VIEW_AXIS = {"xz": "y", "xy": "z", "yz": "x"}

    @staticmethod
    def symmetry_view_planes(planes):
        """Список плоскостей -> плоскости отсечения VTK ``(normal, origin)``.

        Состояние ``view`` у плоскости: 0 — видны обе стороны,
        1 — скрыта отрицательная половина, 2 — скрыта положительная.
        VTK отрезает ту часть, где значение плоскости отрицательно,
        поэтому для состояния 1 нормаль берётся как есть, а для
        состояния 2 — с обратным знаком.
        """
        idx = {"x": 0, "y": 1, "z": 2}
        out = []
        for p in planes or ():
            try:
                state = int(p.get("view", 0) or 0)
            except (TypeError, ValueError):
                state = 0
            if state not in (1, 2):
                continue
            axis = str(p.get("axis", "")).lower()
            normal = MainWindow._SYM_VIEW_NORMAL.get(axis)
            letter = MainWindow._SYM_VIEW_AXIS.get(axis)
            if normal is None or letter is None:
                continue
            sign = 1.0 if state == 1 else -1.0
            origin = [0.0, 0.0, 0.0]
            try:
                origin[idx[letter]] = float(p.get("offset", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
            # «+ 0.0» убирает -0.0, чтобы подпись и лог не путали.
            out.append((tuple(c * sign + 0.0 for c in normal),
                        tuple(origin)))
        return out

    @staticmethod
    def symmetry_view_label(axis, state):
        """Подпись кнопки видимости: «Всё», «y+» или «y-»."""
        if state == 1 or state == 2:
            letter = MainWindow._SYM_VIEW_AXIS.get(str(axis).lower(), "")
            return "%s%s" % (letter, "+" if state == 1 else "-")
        return "Всё"

    def _cycle_symmetry_view(self, axis: str):
        """Кнопка видимости: обе стороны -> без отрицательной половины ->
        без положительной половины -> снова обе."""
        for p in self._symmetry_planes:
            if p["axis"] == axis:
                try:
                    cur = int(p.get("view", 0) or 0)
                except (TypeError, ValueError):
                    cur = 0
                p["view"] = (cur + 1) % 3
                break
        self._apply_symmetry_view()
        self._rebuild_symmetry_list()

    def _apply_symmetry_view(self):
        """Накладывает плоскости отсечения на актёры тел.

        Сами меши не пересчитываются: отсечение живёт в свойстве актёра,
        поэтому состояние обратимо и не портит геометрию.
        """
        specs = self.symmetry_view_planes(self._symmetry_planes)
        try:
            import vtk
        except Exception:
            return
        self._sym_clip_planes = []
        for normal, origin in specs:
            try:
                pl = vtk.vtkPlane()
                pl.SetOrigin(*origin)
                pl.SetNormal(*normal)
                self._sym_clip_planes.append(pl)
            except Exception:
                pass
        for b in getattr(self, "bodies", []) or []:
            actor = b.get("actor") if isinstance(b, dict) else None
            if actor is None:
                continue
            # Плоскости отсечения в VTK 9 живут на маппере: у
            # vtkOpenGLProperty метода AddClippingPlane нет, поэтому
            # через свойство актёра это молча не сработало бы.
            try:
                mapper = actor.GetMapper()
            except Exception:
                mapper = None
            if mapper is None or not hasattr(mapper, "AddClippingPlane"):
                continue
            try:
                mapper.RemoveAllClippingPlanes()
                for pl in self._sym_clip_planes:
                    mapper.AddClippingPlane(pl)
            except Exception:
                pass
        try:
            self.plotter.render()
        except Exception:
            pass

    def _add_symmetry_plane(self, axis: str):
        """Добавляет плоскость симметрии с опциональным смещением."""
        axis = axis.lower()
        if axis not in ("xy", "xz", "yz"):
            return
        if any(p["axis"] == axis for p in self._symmetry_planes):
            self.log_text.append(f"Внимание: Плоскость {axis.upper()} уже добавлена.")
            return

        # Диалог смещения
        offset, ok = QInputDialog.getDouble(
            self, f"Смещение {axis.upper()}",
            f"Смещение плоскости {axis.upper()} от начала координат (м):\n"
            f"(0 = через центр, положительное = вправо/вверх/вперёд)",
            0.0, -100.0, 100.0, 2
        )
        if not ok:
            return

        plane = {"axis": axis, "enabled": True, "actor": None,
                 "offset": offset, "view": 0}
        self._symmetry_planes.append(plane)
        self._rebuild_symmetry_list()
        self._update_symmetry_3d()
        suffix = f" (смещение {offset:+.2f}м)" if abs(offset) > 0.001 else ""
        self.log_text.append(f"Готово: Добавлена плоскость {axis.upper()}{suffix}")

        # Резка выполняется при генерации сетки, поэтому уже построенная
        # сетка плоскости не знает: считалась бы полная модель.
        if getattr(self, "mesh_ready", False):
            self.invalidate_mesh("добавлена плоскость симметрии")

    def _remove_symmetry_plane(self, axis: str):
        """Удаляет плоскость симметрии."""
        for i, p in enumerate(self._symmetry_planes):
            if p["axis"] == axis:
                if p.get("actor"):
                    try:
                        self.plotter.remove_actor(p["actor"])
                    except Exception:
                        pass
                self._symmetry_planes.pop(i)
                break
        self._rebuild_symmetry_list()
        self._update_symmetry_3d()
        self._apply_symmetry_view()
        self.log_text.append(f"Удалена плоскость {axis.upper()}")
        if getattr(self, "mesh_ready", False):
            self.invalidate_mesh("удалена плоскость симметрии")

    def _rebuild_symmetry_list(self):
        while self.sym_list_layout.count():
            item = self.sym_list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        from PyQt5.QtWidgets import QHBoxLayout as _QH, QLabel as _L, QPushButton as _PB
        for p in self._symmetry_planes:
            row = _QH()
            axis = p["axis"].upper()
            offset = p.get("offset", 0.0)
            if abs(offset) > 0.001:
                lbl_text = f"  • {axis}  (смещение {offset:+.2f}м)"
            else:
                lbl_text = f"  • {axis}  (через начало координат)"
            lbl = _L(lbl_text)
            lbl.setStyleSheet("color: #1A1A1A;")
            row.addWidget(lbl)
            row.addStretch()
            state = int(p.get("view", 0) or 0)
            letter = MainWindow._SYM_VIEW_AXIS.get(p["axis"], "")
            btn_view = _PB(self.symmetry_view_label(p["axis"], state))
            btn_view.setFixedWidth(46)
            if state == 1:
                tip = ("Скрыта отрицательная половина по %s.\n"
                       "Ещё раз — скрыть положительную." % letter)
            elif state == 2:
                tip = ("Скрыта положительная половина по %s.\n"
                       "Ещё раз — показать обе стороны." % letter)
            else:
                tip = ("Показаны обе стороны.\n"
                       "Нажмите, чтобы скрыть отрицательную половину по %s."
                       % letter)
            btn_view.setToolTip(tip)
            btn_view.clicked.connect(lambda checked=False, a=p["axis"]:
                                     self._cycle_symmetry_view(a))
            row.addWidget(btn_view)
            btn_del = _PB("×")
            btn_del.setFixedWidth(28)
            btn_del.setToolTip("Удалить")
            btn_del.clicked.connect(lambda checked=False, a=p["axis"]:
                                    self._remove_symmetry_plane(a))
            row.addWidget(btn_del)
            w = QWidget()
            w.setLayout(row)
            self.sym_list_layout.addWidget(w)

    def _update_symmetry_3d(self):
        for p in self._symmetry_planes:
            if p.get("actor"):
                try:
                    self.plotter.remove_actor(p["actor"])
                    p["actor"] = None
                except Exception:
                    pass
        bounds = [b["mesh"].bounds for b in self.bodies if b.get("mesh") is not None]
        if bounds:
            x_min = min(b[0] for b in bounds); x_max = max(b[1] for b in bounds)
            y_min = min(b[2] for b in bounds); y_max = max(b[3] for b in bounds)
            z_min = min(b[4] for b in bounds); z_max = max(b[5] for b in bounds)
            size = 1.2 * max(x_max - x_min, y_max - y_min, z_max - z_min, 1.0)
        else:
            size = 5.0
        for p in self._symmetry_planes:
            try:
                offset = p.get("offset", 0.0)
                if p["axis"] == "xy":
                    plane = pv.Plane(center=(0, 0, offset), direction=(0, 0, 1),
                                     i_size=size, j_size=size)
                elif p["axis"] == "xz":
                    plane = pv.Plane(center=(0, offset, 0), direction=(0, 1, 0),
                                     i_size=size, j_size=size)
                else:
                    plane = pv.Plane(center=(offset, 0, 0), direction=(1, 0, 0),
                                     i_size=size, j_size=size)
                p["actor"] = self.plotter.add_mesh(
                    plane, color="#C75D2C", opacity=0.25,
                    show_edges=True, edge_color="#1A1A1A",
                    name=f"symmetry_{p['axis']}")
            except Exception as e:
                self.log_text.append(f"Внимание: Не удалось отрисовать плоскость {p['axis']}: {e}")
        try:
            self.plotter.render()
        except Exception:
            pass

    def get_symmetry_plane_axes(self) -> list:
        """Плоскости симметрии без смещения: только оси ``xy/xz/yz``.

        :meth:`get_symmetry_planes` может вернуть ``"xz:0.5"`` (плоскость
        со смещением) — такой формат понимают генератор сетки и GUI, но
        ``MARKER_SYM`` в ``config.cfg`` пишется по имени маркера, поэтому
        оптимизации нужны чистые имена осей.
        """
        out = []
        for spec in self.get_symmetry_planes():
            axis = str(spec).split(":", 1)[0].strip().lower()
            if axis in ("xy", "xz", "yz") and axis not in out:
                out.append(axis)
        return out

    def get_symmetry_planes(self) -> list:
        """Возвращает список плоскостей с учётом смещения."""
        result = []
        for p in self._symmetry_planes:
            if p.get("enabled", True):
                offset = p.get("offset", 0.0)
                if abs(offset) > 0.001:
                    result.append(f"{p['axis']}:{offset}")
                else:
                    result.append(p["axis"])
        return result

    # =============================================================
    # ПЕРВЫЙ ЗАПУСК / SU2 (ТЗ 2.2)
    # =============================================================
    def _check_first_launch(self):
        path = config.su2_exe
        if not path or not os.path.exists(path):
            dlg = SU2FirstLaunchDialog(self)
            if dlg.exec_() == QDialog.Accepted:
                if dlg.choice == "auto":
                    self.tree.setCurrentItem(self.item_solver)
                    self.install_su2_automatically()
                elif dlg.choice == "manual":
                    self.txt_su2_path.setText(dlg.manual_path)
                    self.save_su2_path()
                    self.check_su2_connection()
            else:
                self.log_text.append(
                    "Внимание: Первоначальная настройка пропущена. SU2 можно настроить позже "
                    "в разделе Solver Settings.")

    def browse_su2_exe(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите SU2_CFD.exe", "",
            "Исполняемые файлы (*SU2_CFD.exe *SU2_CFD);;Все файлы (*)")
        if path:
            self.txt_su2_path.setText(path)

    def save_su2_path(self):
        config.su2_exe = self.txt_su2_path.text().strip()
        config.save()
        self.lbl_su2_status.setText("Статус: путь сохранён")
        self.lbl_su2_status.setStyleSheet("color: #2c4257; font-style: italic;")
        self.log_text.append(f"Путь к SU2 сохранён: {config.su2_exe}")

    def check_su2_connection(self):
        import subprocess
        path = self.txt_su2_path.text().strip()
        if not path or not os.path.exists(path):
            self.lbl_su2_status.setText("Статус: файл не найден Ошибка: ")
            self.lbl_su2_status.setStyleSheet("color: #9B2C2C; font-weight: bold;")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.lbl_su2_status.setText("Статус: проверка...")
        QApplication.processEvents()
        try:
            startup_time = time.time()
            subprocess.run([path], capture_output=True, timeout=25,
                           **hidden_subprocess_kwargs())
            ok = True
        except subprocess.TimeoutExpired:
            ok = True
        except Exception:
            ok = False
        QApplication.restoreOverrideCursor()
        if ok:
            self.lbl_su2_status.setText("Статус: SU2 доступен Готово: ")
            self.lbl_su2_status.setStyleSheet("color: #2E6B45; font-weight: bold;")
        else:
            self.lbl_su2_status.setText("Статус: SU2 не запускается Ошибка: ")
            self.lbl_su2_status.setStyleSheet("color: #9B2C2C; font-weight: bold;")

    def install_su2_automatically(self):
        install_dir = os.path.join(os.path.expanduser("~"), ".aeroopt", "SU2")
        os.makedirs(install_dir, exist_ok=True)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self._install_worker = SU2InstallWorker(install_dir)
        self._install_worker.progress_signal.connect(self.on_su2_install_progress)
        self._install_worker.finished_signal.connect(
            lambda ok, msg: self.on_su2_install_finished(ok, msg, install_dir))
        self._install_worker.start()

    def on_su2_install_progress(self, percent, msg):
        self._set_progress(percent)
        self.lbl_su2_status.setText(msg)

    def on_su2_install_finished(self, ok, path_or_msg, install_dir):
        QApplication.restoreOverrideCursor()
        self.progress.setVisible(False)
        if self._install_worker:
            self._install_worker.deleteLater()
            self._install_worker = None
        if ok:
            self.txt_su2_path.setText(path_or_msg)
            config.su2_exe = path_or_msg
            config.save()
            self.lbl_su2_status.setText("Статус: SU2 установлен Готово: ")
            self.lbl_su2_status.setStyleSheet("color: #2E6B45; font-weight: bold;")
            self.log_text.append(f"Готово: SU2 установлен: {path_or_msg}")
            self.check_su2_connection()
        else:
            self.lbl_su2_status.setText("Статус: ошибка установки Ошибка: ")
            self.lbl_su2_status.setStyleSheet("color: #9B2C2C; font-weight: bold;")
            QMessageBox.critical(self, "Установка SU2", path_or_msg)

    # =============================================================
    # MODEL BUILDER — без Active Parts (они дублировали таблицу)
    # =============================================================
    def update_model_builder_bodies(self):
        # Узел self.item_active_bodies удалён, метод оставлен как no-op,
        # чтобы не падать в местах, где он вызывается из update_bodies_table.
        return

    # =============================================================
    # ДЕРЕВО → СТРАНИЦЫ
    # =============================================================
    def on_tree_selection_changed(self):
        selected_items = self.tree.selectedItems()
        if not selected_items:
            return
        item = selected_items[0]

        # узел 'Active Parts' больше не существует, но оставленная
        # проверка на parent == self.item_active_bodies теперь не сработает.
        mapping = [
            (self.item_global_defs, self.page_info_global,
             "Settings - Global Definitions"),
            (self.item_component, self.page_info_component,
             "Settings - Component 1"),
            (self.item_study, self.page_info_study,
             "Settings - Study 1"),
            (self.item_results, self.page_info_results,
             "Settings - Results"),
            (self.item_rules, self.page_global_defs,
             "Settings - Design Rules"),
            (self.item_components, self.page_components,
             "Settings - Component List"),
            (self.item_fuselage, self.page_fuselage,
             "Settings - Fuselage Generator"),
            (self.item_wing, self.page_wing, "Settings - Wing Generator"),
            (self.item_flaps_slats, self.page_flaps_slats,
             "Settings - Flaps & Slats"),
            (self.item_stabilizers, self.page_stabilizers,
             "Settings - Stabilizers & Tail"),
            (self.item_mesh, self.page_mesh, "Settings - Mesh 1"),
            (self.item_solver, self.page_solver, "Settings - Solver Settings"),
            (self.item_opt, self.page_opt, "Settings - Multipoint Optimization"),
            (self.item_trim, self.page_trim, "Settings - Trim & Balancing"),
            (self.item_flow_viz, self.page_flow_viz,
             "Settings - Flow Visualization"),
            (self.item_history, self.page_history,
             "Settings - Generation History"),
            (self.item_aeroelastic, self.page_aeroelastic,
             "Settings - Aeroelasticity"),
            (self.item_structural, self.page_structural,
             "Settings - Strength"),
            (self.item_specials, self.page_specials,
             "Settings - Special Functions"),
            (self.item_presets, self.page_presets,
             "Settings - Config Presets"),
        ]
        for node, page, title in mapping:
            if item == node:
                self.settings_stack.setCurrentWidget(page)
                self.lbl_settings_header.setText(title)
                return

    # =============================================================
    # УТИЛИТЫ
    # =============================================================
    @staticmethod
    def _create_triangular_cap_faces(start_idx, n_points, center_idx=None):
        """Веерная триангуляция для замыкания полигона."""
        faces = []
        if center_idx is not None:
            for i in range(n_points):
                j = (i + 1) % n_points
                faces.append([3, center_idx, start_idx + i, start_idx + j])
        else:
            for i in range(n_points - 2):
                faces.append([3, start_idx, start_idx + i + 1, start_idx + i + 2])
        return faces

    @staticmethod
    def _compute_centroid(points_list, start, count):
        pts = points_list[start:start + count]
        cx = sum(p[0] for p in pts) / count
        cy = sum(p[1] for p in pts) / count
        cz = sum(p[2] for p in pts) / count
        return [cx, cy, cz]

    # =============================================================
    # ПРОЕКТ: СОХРАНЕНИЕ / ЗАГРУЗКА
    # =============================================================
    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить проект", "",
                                              "JSON (*.json)")
        if not path:
            return
        opt_points = []
        if hasattr(self, 'points_table'):
            for row in range(self.points_table.rowCount()):
                try:
                    opt_points.append({
                        "name": self.points_table.item(row, 0).text(),
                        "aoa": float(self.points_table.item(row, 1).text()),
                        "weight": float(self.points_table.item(row, 2).text()),
                    })
                except Exception:
                    pass
        data = {
            "bodies": [
                {"name": b["name"], "path": b["path"], "role": b["role"],
                 "visible": b.get("visible", True)}
                for b in self.bodies
            ],
            "wing_box_params": {
                "cx": self.wbox_cx.value(), "cy": self.wbox_cy.value(),
                "cz": self.wbox_cz.value(), "lx": self.wbox_lx.value(),
                "ly": self.wbox_ly.value(), "lz": self.wbox_lz.value(),
                "auto_from_box": self.chk_wing_auto_from_box.isChecked(),
            },
            "rule_set": self.rule_set.to_dict(),
            "opt_points": opt_points,
            "doe_rows": self._get_doe_candidates(),
            # Полётные условия — сериализуем как часть проекта
            "flight": self.flight.to_dict(),
            # Настройки нагрузки (слайдеры)
            "cpu_percent": self._cpu_load_percent() if hasattr(self, "slider_cpu_load") else 50,
            # Поля оставлены для совместимости со старыми файлами проекта.
            "gpu_percent": 0,
            "cpu_cores_override": self._cpu_cores_override() if hasattr(self, "spin_cpu_cores") else 0,
            "compute_device": "cpu",
            "cpu_cores": self._cpu_cores_pending,
            # Опции расчёта: симметрия, RAMP и турбомодель
            # Галочки больше нет: симметрия включена, если есть плоскости.
            "use_symmetry": bool(self.get_symmetry_planes()),
            # T1-визуал: список плоскостей симметрии (XY/XZ/YZ) из 3D-инструмента
            "symmetry_planes": list(self.get_symmetry_planes()),
            "use_ramp_aoa": bool(getattr(self, "chk_use_ramp_aoa", None) and
                                self.chk_use_ramp_aoa.isChecked()),
            "turb_model": self.get_turb_model() if hasattr(self, "combo_solver") else "SA",
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.log_text.append(f"Проект сохранён: {path}")
        self.project_saved = True

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть проект", "",
                                              "JSON (*.json)")
        if not path:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.plotter.clear()
        self.plotter.add_axes()
        self.bodies.clear()
        self.next_body_id = 0
        for item in data.get("bodies", []):
            if os.path.exists(item.get("path", "")):
                try:
                    mesh = pv.read(item["path"])
                    role = item.get("role", "other")
                    color = ROLE_COLORS.get(role, (0.5, 0.5, 0.5))
                    actor = self.plotter.add_mesh(mesh, color=color, opacity=0.8,
                                                  show_edges=True)
                    self.bodies.append({
                        "id": self.next_body_id, "name": item["name"],
                        "path": item["path"], "role": role,
                        "visible": item.get("visible", True), "color": color,
                        "mesh": mesh, "actor": actor,
                    })
                    self.next_body_id += 1
                except Exception as e:
                    self.log_text.append(f"Внимание: Не загружено {item.get('name')}: {e}")
        wbp = data.get("wing_box_params", {})
        if wbp:
            self.wbox_cx.setValue(wbp.get("cx", 2.5))
            self.wbox_cy.setValue(wbp.get("cy", 0.0))
            self.wbox_cz.setValue(wbp.get("cz", 0.0))
            self.wbox_lx.setValue(wbp.get("lx", 2.0))
            self.wbox_ly.setValue(wbp.get("ly", 10.0))
            self.wbox_lz.setValue(wbp.get("lz", 1.0))
            self.chk_wing_auto_from_box.setChecked(wbp.get("auto_from_box", True))
            self.preview_wing_box()
        rs_data = data.get("rule_set")
        if rs_data:
            try:
                self.rule_set = RuleSet.from_dict(rs_data)
                self.update_rules_table()
            except Exception as e:
                self.log_text.append(f"Внимание: Ошибка загрузки правил: {e}")
        if hasattr(self, 'points_table'):
            self.points_table.setRowCount(0)
            for pt in data.get("opt_points", []):
                row = self.points_table.rowCount()
                self.points_table.insertRow(row)
                self.points_table.setItem(row, 0, QTableWidgetItem(pt.get("name", "")))
                self.points_table.setItem(row, 1, QTableWidgetItem(f"{pt.get('aoa', 3.0):.2f}"))
                self.points_table.setItem(row, 2, QTableWidgetItem(f"{pt.get('weight', 1.0):.2f}"))
        # Таблица DOE (перебор вариантов)
        if hasattr(self, 'doe_table'):
            self.doe_table.setRowCount(0)
            for drow in data.get("doe_rows", []):
                row = self.doe_table.rowCount()
                self.doe_table.insertRow(row)
                for col, key in enumerate(self._doe_param_names()):
                    val = drow.get(key, self._doe_current_values().get(key, 0.0))
                    item = QTableWidgetItem(f"{float(val):.3f}")
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.doe_table.setItem(row, col, item)
        # Полётные условия
        fc_data = data.get("flight")
        if fc_data:
            self.flight = FlightConditions.from_dict(fc_data)
            self.input_speed.setValue(self.flight.speed_m_s)
            self.input_alt.setValue(int(self.flight.altitude_m))
            self.input_aoa.setValue(self.flight.aoa_deg)
            if self.flight.preset_name and self.flight.preset_name in FLIGHT_PRESETS:
                idx = self.combo_flight_preset.findText(self.flight.preset_name)
                if idx >= 0:
                    self.combo_flight_preset.setCurrentIndex(idx)
            self.update_isa()
        # Настройки нагрузки и вычислителя
        device = data.get("compute_device")
        # Новый формат (слайдеры процентов)
        cpu_percent = data.get("cpu_percent")
        gpu_percent = data.get("gpu_percent")
        override = data.get("cpu_cores_override")
        # Старый формат (load_level из 4 пресетов) — для обратной совместимости
        legacy_level = data.get("load_level")
        legacy_cpu_cores = data.get("cpu_cores")
        # Маппинг старых уровней на проценты
        legacy_to_percent = {
            "minimal": 25,
            "balanced": 50,
            "high": 100,
            "maximum": 100,
        }
        if cpu_percent is None:
            if legacy_level in legacy_to_percent:
                cpu_percent = legacy_to_percent[legacy_level]
            else:
                cpu_percent = 50
        if gpu_percent is None:
            gpu_percent = 0
        if override is None:
            # Старые проекты: если был «manual» — восстановим override
            if legacy_level == "manual" and legacy_cpu_cores:
                override = int(legacy_cpu_cores)
            else:
                override = 0
        try:
            self.slider_cpu_load.setValue(max(10, min(100, int(cpu_percent))))
        except Exception:
            pass
        try:
            pass  # слайдера нагрузки GPU нет, значение из файла не применяем
        except Exception:
            pass
        try:
            self.spin_cpu_cores.setValue(int(override))
        except Exception:
            pass
        if device in ("cpu", "cpu_gpu"):
            try:
                pass  # выбора вычислителя в интерфейсе нет
            except Exception:
                pass
        # === Восстановление опций расчёта: симметрия / RAMP
        #               и выбор турбомодели (SA/SST) =====================
        # data["use_symmetry"] из старых проектов читаем только как
        # признак того, что плоскости были; сам список восстанавливается
        # ниже из data["symmetry_planes"].
        # T1-визуал: восстановление плоскостей симметрии (XY/XZ/YZ)
        # Плоскости берём только из сохранённого списка: раньше при
        # use_symmetry=True без списка подставлялся "xz", и проект
        # открывался уже с разрезанной пополам моделью.
        planes_to_restore = data.get("symmetry_planes")
        if planes_to_restore and hasattr(self, "_symmetry_planes"):
            self._symmetry_planes = []  # очищаем актёров
            for ax in planes_to_restore:
                if ax in ("xy", "xz", "yz") and \
                        not any(p["axis"] == ax for p in self._symmetry_planes):
                    self._symmetry_planes.append(
                        {"axis": ax, "enabled": True, "actor": None})
            self._rebuild_symmetry_list()
            self._update_symmetry_3d()
        if "use_ramp_aoa" in data and hasattr(self, "chk_use_ramp_aoa"):
            try:
                self.chk_use_ramp_aoa.setChecked(bool(data["use_ramp_aoa"]))
            except Exception:
                pass
        turb_model = data.get("turb_model")
        if turb_model in ("SA", "SST") and hasattr(self, "combo_solver"):
            try:
                # Euler=0, RANS SA=1, RANS SST=2
                self.combo_solver.setCurrentIndex(2 if turb_model == "SST" else 1)
            except Exception:
                pass
        # ================================================================
        # Обновим индикатор
        try:
            self._refresh_load_status_label()
            self.apply_load_level()
        except Exception:
            pass
        self.update_bodies_table()
        # камеру НЕ сбрасываем — оставляем как было до загрузки
        QApplication.restoreOverrideCursor()
        self.log_text.append(f"Проект загружен: {path}")
        self.project_saved = True

    # =============================================================
    # UI-ОБРАБОТЧИКИ КОМПОНЕНТОВ
    # =============================================================
    def on_mesh_quality_changed(self, *args):
        q = self.combo_mesh_quality.currentText()
        if "Грубая" in q:
            self.lbl_mesh_info.setText("~15-30 сек")
        elif "Точная" in q:
            self.lbl_mesh_info.setText("~2-5 мин")
        else:
            self.lbl_mesh_info.setText("~30-60 сек")

    def select_and_highlight_body(self, index):
        self.current_selected_body_index = index
        for i, b in enumerate(self.bodies):
            if b.get("actor"):
                try:
                    if i == index:
                        b["actor"].GetProperty().SetColor(1.0, 0.5, 0.0)
                        b["actor"].GetProperty().SetOpacity(1.0)
                    else:
                        color = ROLE_COLORS.get(b["role"], ROLE_COLORS["other"])
                        b["actor"].GetProperty().SetColor(*pv.Color(color).float_rgb)
                        b["actor"].GetProperty().SetOpacity(0.6)
                except Exception:
                    pass
        self.plotter.render()

    def on_3d_pick(self, point):
        if point is None:
            return
        min_dist = float('inf')
        closest_idx = -1
        for i, b in enumerate(self.bodies):
            if b.get("actor") is None:
                continue
            bounds = b["mesh"].bounds
            if (bounds[0] <= point[0] <= bounds[1] and
                    bounds[2] <= point[1] <= bounds[3] and
                    bounds[4] <= point[2] <= bounds[5]):
                dist = np.linalg.norm(b["mesh"].points - point, axis=1).min()
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
        if closest_idx >= 0:
            self.bodies_table.selectRow(closest_idx)
            self.select_and_highlight_body(closest_idx)

    def on_table_click(self, row, col):
        self.select_and_highlight_body(row)
        self.btn_heal_stl.setEnabled(0 <= row < len(self.bodies))

    def _get_fuselage_body(self):
        return next((b for b in self.bodies if b["role"] == "fuselage"), None)

    # =============================================================
    # WING-BOX
    # =============================================================
    def fill_wing_box_from_fuselage(self):
        fuselage = self._get_fuselage_body()
        if not fuselage:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите или сгенерируйте фюзеляж.")
            return
        x_min, x_max, y_min, y_max, z_min, z_max = fuselage["mesh"].bounds
        length = max(x_max - x_min, 1.0)
        height = max(z_max - z_min, 0.5)
        self.wbox_cx.setValue(x_min + 0.42 * length)
        self.wbox_cy.setValue((y_min + y_max) * 0.5)
        self.wbox_cz.setValue((z_min + z_max) * 0.5)
        self.wbox_lx.setValue(max(0.8, 0.22 * length))
        self.wbox_ly.setValue(max(2.0, 1.15 * length))
        self.wbox_lz.setValue(max(0.4, 1.2 * height))
        self.preview_wing_box()
        self.log_text.append("Область крыла заполнена по фюзеляжу.")

    def _get_wing_box_bounds(self):
        cx, cy, cz = self.wbox_cx.value(), self.wbox_cy.value(), self.wbox_cz.value()
        lx, ly, lz = self.wbox_lx.value(), self.wbox_ly.value(), self.wbox_lz.value()
        return (cx - lx / 2, cx + lx / 2,
                cy - ly / 2, cy + ly / 2,
                cz - lz / 2, cz + lz / 2)

    def preview_wing_box(self):
        if self.wing_box_actor:
            try:
                self.plotter.remove_actor(self.wing_box_actor)
            except Exception:
                pass
            self.wing_box_actor = None
        bounds = self._get_wing_box_bounds()
        box = pv.Box(bounds=bounds)
        self.wing_box_actor = self.plotter.add_mesh(
            box, color="yellow", style="wireframe", line_width=2)
        self.plotter.render()

    def _resolve_wing_params_from_box(self, span, chord_root, chord_tip):
        x_min, x_max, y_min, y_max, z_min, z_max = self._get_wing_box_bounds()
        lx = x_max - x_min
        ly = y_max - y_min
        half_span = span / 2.0
        sweep_rad = math.radians(self.w_sweep.value())
        sweep_offset = half_span * math.tan(sweep_rad)
        span = min(span, ly * 0.98)
        max_x_footprint = sweep_offset + chord_tip
        if max_x_footprint > lx * 0.98:
            scale = (lx * 0.98) / max_x_footprint
            chord_root *= scale
            chord_tip *= scale
        pos_x = x_min + 0.02 * lx
        pos_y = (y_min + y_max) * 0.5
        pos_z = (z_min + z_max) * 0.5
        self.w_span.setValue(span)
        self.w_chord_root.setValue(chord_root)
        self.w_chord_tip.setValue(chord_tip)
        self.w_pos_x.setValue(pos_x)
        self.w_pos_y.setValue(pos_y)
        self.w_pos_z.setValue(pos_z)
        return span, chord_root, chord_tip, pos_x, pos_y, pos_z

    # =============================================================
    # СПРАВОЧНЫЕ ДАННЫЕ
    # =============================================================
    def calculate_reference_data(self):
        wing = next((b for b in self.bodies if b["role"] == "wing"), None)
        if wing:
            span = max(self.w_span.value(), 1e-6)
            cr = max(self.w_chord_root.value(), 1e-6)
            ct = max(self.w_chord_tip.value(), 1e-6)
            taper = ct / cr
            ref_area = 0.5 * (cr + ct) * span
            ref_length = (2.0 / 3.0) * cr * ((1 + taper + taper ** 2) / (1 + taper))
            sweep_offset = 0.5 * span * math.tan(math.radians(self.w_sweep.value()))
            ox = self.w_pos_x.value() + 0.25 * ref_length + 0.5 * sweep_offset
            oy = self.w_pos_y.value()
            oz = self.w_pos_z.value()
            self.log_text.append(
                f"RefData: Lref={ref_length:.3f}, Sref={ref_area:.3f}, "
                f"O=({ox:.3f}, {oy:.3f}, {oz:.3f})")
            return (ref_length, ref_area, ox, oy, oz)
        bounds = [b["mesh"].bounds for b in self.bodies if b.get("mesh") is not None]
        if bounds:
            x_min = min(b[0] for b in bounds)
            x_max = max(b[1] for b in bounds)
            y_min = min(b[2] for b in bounds)
            y_max = max(b[3] for b in bounds)
            z_min = min(b[4] for b in bounds)
            z_max = max(b[5] for b in bounds)
            ref_length = max(x_max - x_min, 1.0)
            ref_area = max((y_max - y_min) * (z_max - z_min), 1.0)
            ox = 0.5 * (x_min + x_max)
            oy = 0.5 * (y_min + y_max)
            oz = 0.5 * (z_min + z_max)
            self.log_text.append(
                f"Внимание: Крыла нет, RefData по габаритам: Lref={ref_length:.3f}, "
                f"Sref={ref_area:.3f}")
            return (ref_length, ref_area, ox, oy, oz)
        self.log_text.append("Внимание: RefData не вычислены, по умолчанию 1.0/1.0")
        return (1.0, 1.0, 0.25, 0.0, 0.0)

    # =============================================================
    # ЗАГРУЗКА STL
    # =============================================================
    def load_stl_fuselage(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Загрузить фюзеляж", "", self._geometry_file_filter())
        for p in paths:
            self._add_body(p, "fuselage")
        # КАМЕРУ НЕ СБРАСЫВАЕМ
        self.update_flow_arrow()

    def _cad_try_split(self, path: str, role: str) -> bool:
        """Импорт CAD-сборки по телам. True — если импорт выполнен.

        Возвращает False, если в файле одно тело или gmsh недоступен:
        тогда вызывающий код идёт обычным путём (одна триангуляция).
        """
        from geometry.generators import cad_inspect, cad_split_to_stl
        try:
            solids = cad_inspect(path, log=lambda m: self.log_text.append(m))
        except Exception as e:
            self.log_text.append(f"  Разбор сборки недоступен: {e}")
            return False
        if len(solids) <= 1:
            self.log_text.append("  В файле одно тело — импорт без разбора")
            return False
        vols = "; ".join(f"{s_.get('name') or ('тело ' + str(s_['tag']))}"
                         f" V={s_.get('volume', 0.0):.4g}" for s_ in solids[:6])
        self.log_text.append(f"  Сборка из {len(solids)} тел: {vols}")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            parts = cad_split_to_stl(
                path, WORK_DIR_BASE,
                log=lambda m: self.log_text.append(m))
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self.log_text.append(f"  Внимание: Не удалось разобрать сборку: {e}")
            return False
        QApplication.restoreOverrideCursor()
        if not parts:
            return False
        for part in parts:
            self._add_body(part["stl"], role)
            if self.bodies:
                self.bodies[-1]["name"] = (
                    f"{part.get('name') or os.path.basename(part['stl'])}"
                    f" ({part['triangles']} тр.)")
        self.update_bodies_table()
        self.update_flow_arrow()
        self.log_text.append(f"Готово: Сборка импортирована по телам: {len(parts)}")
        return True

    @staticmethod
    def _geometry_file_filter() -> str:
        """Фильтр диалога импорта геометрии.

        Приложение поддерживает не только STL: ``cad_to_stl`` триангулирует
        CAD-модель через gmsh. Прежний фильтр ``"STL (*.stl)"`` это скрывал,
        и пользователь не мог выбрать STEP или IGES, хотя код их принимал.
        """
        cad = " ".join("*" + e for e in CAD_EXTENSIONS)
        return ("Все поддерживаемые (*.stl %s);;"
                "STL (*.stl);;CAD (%s);;Все файлы (*)" % (cad, cad))

    def add_bodies(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Добавить компоненты", "", self._geometry_file_filter())
        for p in paths:
            self._add_body(p, "other")
        # КАМЕРУ НЕ СБРАСЫВАЕМ
        self.update_flow_arrow()

    def _add_body(self, path, role):
        ext = os.path.splitext(path)[1].lower()
        source_path = path
        if ext in CAD_EXTENSIONS:
            # Direct CAD Import: конвертируем модель в STL через gmsh.
            # ТЗ п.4: многодетальная сборка раскладывается на отдельные
            # тела — иначе детали сливаются в одну и теряют имена.
            self.log_text.append(
                f"Direct CAD Import: {os.path.basename(path)} ({ext})")
            os.makedirs(WORK_DIR_BASE, exist_ok=True)
            if self.chk_cad_split.isChecked() and self._cad_try_split(path, role):
                return
            stl_name = f"_cad_{self.next_body_id}_{os.path.splitext(os.path.basename(path))[0]}.stl"
            stl_path = os.path.join(WORK_DIR_BASE, stl_name)
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                cad_to_stl(path, stl_path,
                           log=lambda m: self.log_text.append(m))
                self.log_text.append(f"  Готово: CAD → STL: {stl_path}")
                path = stl_path
            except Exception as e:
                QApplication.restoreOverrideCursor()
                self.log_text.append(f"Ошибка импорта CAD: {e}")
                QMessageBox.critical(
                    self, "Ошибка чтения CAD-формата",
                    f"Не удалось импортировать {ext}-файл: {e}\n\n"
                    "Проверьте, что файл не повреждён. Как запасной вариант "
                    "экспортируйте модель в STL из CAD-системы.")
                return
            QApplication.restoreOverrideCursor()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            mesh = pv.read(path).triangulate().clean(tolerance=1e-6)
            color = ROLE_COLORS.get(role, ROLE_COLORS["other"])
            actor = self.plotter.add_mesh(mesh, color=color, opacity=0.8,
                                          show_edges=True)
            self.bodies.append({
                "id": self.next_body_id, "name": os.path.basename(source_path),
                "path": path, "role": role, "visible": True, "color": color,
                "mesh": mesh, "actor": actor,
            })
            self.next_body_id += 1
            self.update_bodies_table()
            self.log_text.append(f"Готово: Загружен: {os.path.basename(source_path)}")
            self.invalidate_mesh("загружен новый STL")
        except Exception as e:
            self.log_text.append(f"Ошибка загрузки: {e}")
            if ext in CAD_EXTENSIONS:
                QMessageBox.critical(self, "Ошибка чтения CAD-формата",
                                     f"Не удалось импортировать {ext}-файл напрямую. "
                                     "Экспортируйте его в STL из CAD-системы.")
        QApplication.restoreOverrideCursor()

    # =============================================================
    # ТАБЛИЦА ТЕЛ
    # =============================================================
    def update_bodies_table(self):
        self.bodies_table.setRowCount(0)
        for body in self.bodies:
            row = self.bodies_table.rowCount()
            self.bodies_table.insertRow(row)
            self.bodies_table.setItem(row, 0, QTableWidgetItem(body["name"]))
            combo = QComboBox()
            for k, v in ROLES.items():
                combo.addItem(v, k)
            combo.setCurrentText(ROLES.get(body["role"], "Другое"))
            combo.currentIndexChanged.connect(
                lambda idx, c=combo, bid=body["id"]:
                self.on_role_changed(bid, c.currentData()))
            self.bodies_table.setCellWidget(row, 1, combo)
        # update_model_builder_bodies — теперь no-op (узла в дереве нет)
        self.update_model_builder_bodies()
        # T1-визуал: пересчитать размер плоскостей симметрии по новому bbox
        if hasattr(self, "_symmetry_planes") and self._symmetry_planes:
            self._update_symmetry_3d()
        self.project_saved = False

    def on_role_changed(self, body_id, new_role):
        for b in self.bodies:
            if b["id"] == body_id:
                b["role"] = new_role
                b["color"] = ROLE_COLORS.get(new_role, ROLE_COLORS["other"])
                if b["actor"]:
                    try:
                        b["actor"].GetProperty().SetColor(*pv.Color(b["color"]).float_rgb)
                    except Exception:
                        b["actor"].GetProperty().SetColor(b["color"])
                    self.plotter.render()
                if new_role == "wing":
                    bounds = b["mesh"].bounds
                    x_min, x_max, y_min, y_max, z_min, z_max = bounds
                    self.wbox_cx.setValue((x_min + x_max) / 2)
                    self.wbox_cy.setValue((y_min + y_max) / 2)
                    self.wbox_cz.setValue((z_min + z_max) / 2)
                    self.wbox_lx.setValue(x_max - x_min)
                    self.wbox_ly.setValue(y_max - y_min)
                    self.wbox_lz.setValue(z_max - z_min)
                    self.preview_wing_box()
                    self.log_text.append(
                        f"Область генерации адаптирована под деталь '{b['name']}'.")
                break

    def remove_body(self):
        selected_ranges = self.bodies_table.selectedRanges()
        if not selected_ranges:
            return
        rows_to_delete = set()
        for r in selected_ranges:
            for row in range(r.topRow(), r.bottomRow() + 1):
                rows_to_delete.add(row)
        for row in sorted(list(rows_to_delete), reverse=True):
            if 0 <= row < len(self.bodies):
                body = self.bodies.pop(row)
                if body.get("actor"):
                    try:
                        self.plotter.remove_actor(body["actor"])
                    except Exception:
                        pass
        self.current_selected_body_index = -1
        self.update_bodies_table()
        self.plotter.render()
        self.invalidate_mesh("удалены компоненты")

    def rotate_selected(self, axis, angle):
        idx = self.bodies_table.currentRow()
        if idx < 0 or idx >= len(self.bodies):
            return
        body = self.bodies[idx]
        c = body["mesh"].center
        if axis == 'x':
            rotated = body["mesh"].rotate_x(angle, point=c, inplace=False)
        elif axis == 'y':
            rotated = body["mesh"].rotate_y(angle, point=c, inplace=False)
        else:
            rotated = body["mesh"].rotate_z(angle, point=c, inplace=False)
        body["mesh"] = rotated
        if body.get("actor"):
            self.plotter.remove_actor(body["actor"])
        body["actor"] = self.plotter.add_mesh(body["mesh"], color=body["color"],
                                              opacity=0.8, show_edges=True)
        # КАМЕРУ НЕ СБРАСЫВАЕМ — пользователь останется на своём виде
        self.plotter.render()
        self.invalidate_mesh("компонент повёрнут")

    def simplify_geometry(self, level="medium"):
        idx = self.bodies_table.currentRow()
        if idx < 0 or idx >= len(self.bodies):
            QMessageBox.warning(self, "Ошибка", "Выберите компонент в таблице для упрощения.")
            return
        body = self.bodies[idx]
        mesh = body["mesh"]
        original_count = mesh.n_cells
        self.log_text.append(f"Упрощение '{body['name']}' ({original_count} граней)...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if level == "simple":
                simplified = mesh.decimate(target_reduction=0.95)
                if simplified.n_points > 10:
                    simplified = simplified.smooth(n_iter=5, relaxation_factor=0.1)
            else:
                simplified = mesh.decimate(target_reduction=0.70)
                if simplified.n_points > 10:
                    simplified = simplified.smooth(n_iter=3, relaxation_factor=0.05)
            self.log_text.append(f"  Готово: {original_count} → {simplified.n_cells} граней")
            if simplified.n_cells == 0:
                self.log_text.append("  Ошибка: После упрощения 0 граней")
                return
            if body.get("actor"):
                self.plotter.remove_actor(body["actor"])
            body["mesh"] = simplified
            body["path"] = os.path.join(
                WORK_DIR_BASE, f"_body_{body['id']}_{body['role']}_simplified.stl")
            os.makedirs(WORK_DIR_BASE, exist_ok=True)
            simplified.save(body["path"])
            body["actor"] = self.plotter.add_mesh(
                simplified, color=body["color"], opacity=0.6, show_edges=True)
            # КАМЕРУ НЕ СБРАСЫВАЕМ
            self.plotter.render()
            self.invalidate_mesh("геометрия упрощена")
        except Exception as e:
            self.log_text.append(f"  Ошибка упрощения: {e}")
        QApplication.restoreOverrideCursor()

    # =============================================================
    # ПРИМИТИВЫ
    # =============================================================
    def _create_primitive(self, shape_type):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Создать: {shape_type}")
        dialog.setMinimumWidth(350)
        form = QFormLayout(dialog)
        param_widgets = {}
        if shape_type == "Куб":
            lx = QDoubleSpinBox(); lx.setRange(0.01, 100); lx.setValue(2.0); lx.setSuffix(" м")
            ly = QDoubleSpinBox(); ly.setRange(0.01, 100); ly.setValue(1.0); ly.setSuffix(" м")
            lz = QDoubleSpinBox(); lz.setRange(0.01, 100); lz.setValue(0.5); lz.setSuffix(" м")
            form.addRow("Длина X:", lx); param_widgets['lx'] = lx
            form.addRow("Ширина Y:", ly); param_widgets['ly'] = ly
            form.addRow("Высота Z:", lz); param_widgets['lz'] = lz
        elif shape_type == "Цилиндр":
            r = QDoubleSpinBox(); r.setRange(0.01, 50); r.setValue(0.5); r.setSuffix(" м")
            h = QDoubleSpinBox(); h.setRange(0.01, 100); h.setValue(3.0); h.setSuffix(" м")
            form.addRow("Радиус:", r); param_widgets['radius'] = r
            form.addRow("Высота:", h); param_widgets['height'] = h
        else:
            r = QDoubleSpinBox(); r.setRange(0.01, 100); r.setValue(1.0); r.setSuffix(" м")
            form.addRow("Радиус:", r); param_widgets['radius'] = r
        pos_x = QDoubleSpinBox(); pos_x.setRange(-50, 50); pos_x.setValue(0); pos_x.setSuffix(" м")
        pos_y = QDoubleSpinBox(); pos_y.setRange(-50, 50); pos_y.setValue(0); pos_y.setSuffix(" м")
        pos_z = QDoubleSpinBox(); pos_z.setRange(-50, 50); pos_z.setValue(0); pos_z.setSuffix(" м")
        form.addRow("X:", pos_x)
        form.addRow("Y:", pos_y)
        form.addRow("Z:", pos_z)
        role_combo = QComboBox()
        for k, v in ROLES.items():
            role_combo.addItem(v, k)
        form.addRow("Роль:", role_combo)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        form.addRow(btns)
        if dialog.exec_() != QDialog.Accepted:
            return
        params = {k: w.value() for k, w in param_widgets.items()}
        pos = [pos_x.value(), pos_y.value(), pos_z.value()]
        try:
            mesh = create_primitive(shape_type, params, pos)
        except Exception as e:
            self.log_text.append(f"Ошибка создания примитива: {e}")
            return
        os.makedirs(WORK_DIR_BASE, exist_ok=True)
        prim_id = self.next_body_id
        prim_path = os.path.join(WORK_DIR_BASE, f"_primitive_{prim_id}.stl")
        mesh.save(prim_path)
        role = role_combo.currentData()
        color = ROLE_COLORS.get(role, ROLE_COLORS["other"])
        actor = self.plotter.add_mesh(mesh, color=color, opacity=0.6, show_edges=True)
        self.bodies.append({
            "id": prim_id, "name": f"{shape_type}_{prim_id}", "path": prim_path,
            "role": role, "visible": True, "color": color, "mesh": mesh, "actor": actor,
        })
        self.next_body_id += 1
        self.update_bodies_table()
        # КАМЕРУ НЕ СБРАСЫВАЕМ
        self.log_text.append(f"Готово: Примитив '{shape_type}' создан ({role})")

    # =============================================================
    # ГЕНЕРАЦИЯ ГЕОМЕТРИИ
    # =============================================================
    def generate_fuselage(self):
        try:
            self.log_text.append("Генерация фюзеляжа...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            L = self.f_length.value()
            D = self.f_diameter.value()
            nose_ratio = self.f_nose_ratio.value()
            tail_ratio = self.f_tail_ratio.value()
            pos_x = self.f_pos_x.value()
            pos_y = self.f_pos_y.value()
            pos_z = self.f_pos_z.value()
            radius_max = D / 2.0
            nose_len = L * nose_ratio
            tail_len = L * tail_ratio
            n_rings = 28
            n_points = 48
            points = []
            faces = []
            rings = []
            for i in range(1, n_rings):
                t_global = i / n_rings
                x_local = -L / 2.0 + L * t_global
                x_from_nose = x_local + L / 2.0
                x_to_tail = L / 2.0 - x_local
                if x_from_nose < nose_len:
                    t = x_from_nose / nose_len
                    r = radius_max * math.sin(math.pi * t / 2.0)
                elif x_to_tail < tail_len:
                    t = x_to_tail / tail_len
                    r = radius_max * math.sin(math.pi * t / 2.0)
                else:
                    r = radius_max
                if r < 1e-6:
                    continue
                ring = []
                for j in range(n_points):
                    theta = 2.0 * math.pi * j / n_points
                    y = pos_y + r * math.cos(theta)
                    z = pos_z + r * math.sin(theta)
                    ring.append(len(points))
                    points.append([pos_x + x_local, y, z])
                rings.append(ring)
            nose_tip_idx = len(points)
            points.append([pos_x - L / 2.0, pos_y, pos_z])
            tail_tip_idx = len(points)
            points.append([pos_x + L / 2.0, pos_y, pos_z])
            first_ring = rings[0]
            for j in range(n_points):
                faces.append([3, nose_tip_idx, first_ring[(j + 1) % n_points], first_ring[j]])
            for k in range(len(rings) - 1):
                r1, r2 = rings[k], rings[k + 1]
                for j in range(n_points):
                    faces.append([4, r1[j], r1[(j + 1) % n_points],
                                  r2[(j + 1) % n_points], r2[j]])
            last_ring = rings[-1]
            for j in range(n_points):
                faces.append([3, last_ring[j], last_ring[(j + 1) % n_points], tail_tip_idx])
            flat_faces = [v for f in faces for v in f]
            fuselage_mesh = pv.PolyData(np.array(points),
                                        np.array(flat_faces)).triangulate().clean(tolerance=1e-6)
            fuselage_mesh.compute_normals(auto_orient_normals=True, inplace=True)
            for b in self.bodies:
                if b["role"] == "fuselage" and b.get("actor"):
                    self.plotter.remove_actor(b["actor"])
            self.bodies = [b for b in self.bodies if b["role"] != "fuselage"]
            os.makedirs(WORK_DIR_BASE, exist_ok=True)
            path = os.path.join(WORK_DIR_BASE, "generated_fuselage.stl")
            fuselage_mesh.save(path)
            color = ROLE_COLORS["fuselage"]
            actor = self.plotter.add_mesh(fuselage_mesh, color=color, opacity=0.8,
                                          show_edges=True)
            self.bodies.append({
                "id": self.next_body_id, "name": "generated_fuselage.stl",
                "path": path, "role": "fuselage", "visible": True,
                "color": color, "mesh": fuselage_mesh, "actor": actor,
            })
            self.next_body_id += 1
            self.update_bodies_table()
            # КАМЕРУ НЕ СБРАСЫВАЕМ
            self.plotter.render()
            self.log_text.append(f"Готово: Фюзеляж: L={L:.2f} м, D={D:.2f} м")
            self.invalidate_mesh("сгенерирован новый фюзеляж")
        except Exception as e:
            self.log_text.append(f"Ошибка генерации фюзеляжа: {e}")
        QApplication.restoreOverrideCursor()

    def generate_wing_mesh_parametric(self, span, chord_root, chord_tip,
                                      silent=False, flap_deflection=0.0):
        try:
            for b in self.bodies:
                if b["role"] in ("wing", "flap", "slat") and b.get("actor"):
                    self.plotter.remove_actor(b["actor"])
            self.bodies = [b for b in self.bodies
                           if b["role"] not in ("wing", "flap", "slat")]
            span, chord_root, chord_tip, pos_x, pos_y, pos_z = \
                self._resolve_wing_params_from_box(span, chord_root, chord_tip)
            kink_pos_ratio = self.w_kink_pos.value() if self.chk_kink.isChecked() else None
            chord_kink = self.w_chord_kink.value() if self.chk_kink.isChecked() else None
            sweep_outer_deg = self.w_sweep_outer.value() if self.chk_kink.isChecked() else None
            wing_mesh, sweep_offset = generate_wing_mesh(
                span=span, chord_root=chord_root, chord_tip=chord_tip,
                sweep_deg=self.w_sweep.value(), twist_deg=self.w_twist.value(),
                naca_code=self.w_naca.text(), pos_x=pos_x, pos_y=pos_y, pos_z=pos_z,
                kink_pos_ratio=kink_pos_ratio, chord_kink=chord_kink,
                sweep_outer_deg=sweep_outer_deg)
            os.makedirs(WORK_DIR_BASE, exist_ok=True)
            wing_path = os.path.join(WORK_DIR_BASE, "generated_wing.stl")
            wing_mesh.save(wing_path)
            color = ROLE_COLORS["wing"]
            actor = self.plotter.add_mesh(wing_mesh, color=color, opacity=0.9,
                                          show_edges=True)
            self.bodies.append({
                "id": self.next_body_id, "name": "generated_wing.stl",
                "path": wing_path, "role": "wing", "visible": True,
                "color": color, "mesh": wing_mesh, "actor": actor,
            })
            self.next_body_id += 1
            if self.flap_enabled.isChecked():
                flaps_mesh = generate_flaps_mesh(
                    span=span, chord_root=chord_root, chord_tip=chord_tip,
                    flap_deflection=self.flap_deflection.value(),
                    flap_span_ratio=self.flap_span_ratio.value(),
                    flap_chord_ratio=self.flap_chord_ratio.value(),
                    pos_x=pos_x, pos_y=pos_y, pos_z=pos_z,
                    sweep_offset=sweep_offset,
                    hinge_depth_ratio=self.flap_hinge_depth.value(),
                    slide_ratio=self.flap_slide.value())
                if flaps_mesh is not None:
                    flaps_path = os.path.join(WORK_DIR_BASE, "generated_flaps.stl")
                    flaps_mesh.save(flaps_path)
                    flap_color = ROLE_COLORS["flap"]
                    flap_actor = self.plotter.add_mesh(
                        flaps_mesh, color=flap_color, opacity=0.9, show_edges=True)
                    self.bodies.append({
                        "id": self.next_body_id, "name": "generated_flaps.stl",
                        "path": flaps_path, "role": "flap", "visible": True,
                        "color": flap_color, "mesh": flaps_mesh, "actor": flap_actor,
                    })
                    self.next_body_id += 1
            if self.slat_enabled.isChecked():
                slats_mesh = generate_slats_mesh(
                    span=span, chord_root=chord_root, chord_tip=chord_tip,
                    slat_deflection=self.slat_deflection.value(),
                    slat_span_ratio=self.slat_span_ratio.value(),
                    slat_chord_ratio=self.slat_chord_ratio.value(),
                    pos_x=pos_x, pos_y=pos_y, pos_z=pos_z,
                    sweep_offset=sweep_offset, slide_ratio=self.slat_slide.value())
                if slats_mesh is not None:
                    slats_path = os.path.join(WORK_DIR_BASE, "generated_slats.stl")
                    slats_mesh.save(slats_path)
                    slat_color = ROLE_COLORS["slat"]
                    slat_actor = self.plotter.add_mesh(
                        slats_mesh, color=slat_color, opacity=0.9, show_edges=True)
                    self.bodies.append({
                        "id": self.next_body_id, "name": "generated_slats.stl",
                        "path": slats_path, "role": "slat", "visible": True,
                        "color": slat_color, "mesh": slats_mesh, "actor": slat_actor,
                    })
                    self.next_body_id += 1
            self.update_bodies_table()
            # КАМЕРУ НЕ СБРАСЫВАЕМ
            self.plotter.render()
            record_id = len(self.generation_history) + 1
            last_k = 0.0
            if self.all_results:
                last_res = self.all_results[-1]
                cl = last_res.get("cl", 0.0)
                cd = last_res.get("cd", 0.0)
                if cd > 0.0001:
                    last_k = cl / cd
            self.generation_history.append({
                "id": record_id, "span": span, "chord_root": chord_root,
                "chord_tip": chord_tip, "sweep": self.w_sweep.value(),
                "twist": self.w_twist.value(), "k": last_k,
            })
            if hasattr(self, "update_history_table"):
                self.update_history_table()
            if not silent:
                self.log_text.append(
                    f"Готово: Крыло сгенерировано! (размах {span:.2f}м, хорда {chord_root:.2f}м)")
                if self.flap_enabled.isChecked():
                    self.log_text.append("Сгенерированы механизированные закрылки.")
                self.invalidate_mesh("сгенерировано новое крыло")
        except Exception as e:
            self.log_text.append(f"Ошибка генерации крыла: {e}")

    def generate_horizontal_stabilizer(self):
        try:
            fuselage = self._get_fuselage_body()
            if fuselage and self.hs_auto.isChecked():
                bounds = fuselage["mesh"].bounds
                length = bounds[1] - bounds[0]
                self.hs_pos_x.setValue(bounds[1] - length * 0.15)
                self.hs_pos_z.setValue((bounds[4] + bounds[5]) * 0.5)
                self.hs_span.setValue(max(1.5, length * 0.35))
                self.hs_chord.setValue(max(0.3, length * 0.10))
                self.log_text.append("ГО автоподстроено по фюзеляжу")
            for b in self.bodies:
                if b["role"] in ("h_stab", "elevator") and b.get("actor"):
                    self.plotter.remove_actor(b["actor"])
            self.bodies = [b for b in self.bodies
                           if b["role"] not in ("h_stab", "elevator")]
            span = self.hs_span.value()
            chord = self.hs_chord.value()
            sweep = self.hs_sweep.value()
            pos_x = self.hs_pos_x.value()
            pos_z = self.hs_pos_z.value()
            elev_deflection = self.elev_deflection.value()
            half_span = span / 2.0
            sweep_offset = half_span * math.tan(math.radians(sweep))
            rx, rz = generate_naca4_section(chord * 0.70, "0012", twist=0.0)
            tx, tz = generate_naca4_section(chord * 0.5 * 0.70, "0012", twist=0.0)
            n = len(rx)
            points = []
            for i in range(n):
                points.append([tx[i] + sweep_offset + pos_x, -half_span, tz[i] + pos_z])
            for i in range(n):
                points.append([rx[i] + pos_x, 0.0, rz[i] + pos_z])
            for i in range(n):
                points.append([tx[i] + sweep_offset + pos_x, +half_span, tz[i] + pos_z])
            faces = []
            for i in range(n - 1):
                faces.append([4, i, i + 1, n + i + 1, n + i])
            faces.append([4, n - 1, 0, n, 2 * n - 1])
            for i in range(n - 1):
                faces.append([4, n + i, n + i + 1, 2 * n + i + 1, 2 * n + i])
            faces.append([4, 2 * n - 1, n, 2 * n, 3 * n - 1])
            left_c = len(points)
            points.append(self._compute_centroid(points, 0, n))
            faces.extend(self._create_triangular_cap_faces(0, n, center_idx=left_c))
            right_c = len(points)
            points.append(self._compute_centroid(points, 2 * n, n))
            faces.extend(self._create_triangular_cap_faces(2 * n, n, center_idx=right_c))
            flat_faces = [v for f in faces for v in f]
            hs_mesh = pv.PolyData(np.array(points),
                                  np.array(flat_faces)).triangulate().clean(tolerance=1e-6)
            hs_mesh.compute_normals(auto_orient_normals=True, inplace=True)
            os.makedirs(WORK_DIR_BASE, exist_ok=True)
            path = os.path.join(WORK_DIR_BASE, "h_stabilizer.stl")
            hs_mesh.save(path)
            actor = self.plotter.add_mesh(hs_mesh, color=ROLE_COLORS["h_stab"],
                                          opacity=0.9, show_edges=True)
            self.bodies.append({
                "id": self.next_body_id, "name": "h_stabilizer.stl", "path": path,
                "role": "h_stab", "visible": True, "color": ROLE_COLORS["h_stab"],
                "mesh": hs_mesh, "actor": actor,
            })
            self.next_body_id += 1
            rx_el, rz_el = generate_naca4_section(chord * 0.30, "0012",
                                                  twist=elev_deflection)
            tx_el, tz_el = generate_naca4_section(chord * 0.5 * 0.30, "0012",
                                                  twist=elev_deflection)
            hinge_root_x = pos_x + chord * 0.70
            hinge_tip_x = pos_x + sweep_offset + chord * 0.5 * 0.70
            el_points = []
            for i in range(n):
                el_points.append([tx_el[i] + hinge_tip_x, -half_span, tz_el[i] + pos_z])
            for i in range(n):
                el_points.append([rx_el[i] + hinge_root_x, 0.0, rz_el[i] + pos_z])
            for i in range(n):
                el_points.append([tx_el[i] + hinge_tip_x, +half_span, tz_el[i] + pos_z])
            el_faces = []
            for i in range(n - 1):
                el_faces.append([4, i, i + 1, n + i + 1, n + i])
            el_faces.append([4, n - 1, 0, n, 2 * n - 1])
            for i in range(n - 1):
                el_faces.append([4, n + i, n + i + 1, 2 * n + i + 1, 2 * n + i])
            el_faces.append([4, 2 * n - 1, n, 2 * n, 3 * n - 1])
            el_left_c = len(el_points)
            el_points.append(self._compute_centroid(el_points, 0, n))
            el_faces.extend(self._create_triangular_cap_faces(0, n, center_idx=el_left_c))
            el_right_c = len(el_points)
            el_points.append(self._compute_centroid(el_points, 2 * n, n))
            el_faces.extend(self._create_triangular_cap_faces(2 * n, n, center_idx=el_right_c))
            el_flat_faces = [v for f in el_faces for v in f]
            el_mesh = pv.PolyData(np.array(el_points),
                                  np.array(el_flat_faces)).triangulate().clean(tolerance=1e-6)
            el_mesh.compute_normals(auto_orient_normals=True, inplace=True)
            el_path = os.path.join(WORK_DIR_BASE, "elevator.stl")
            el_mesh.save(el_path)
            el_actor = self.plotter.add_mesh(el_mesh, color=ROLE_COLORS["elevator"],
                                             opacity=0.9, show_edges=True)
            self.bodies.append({
                "id": self.next_body_id, "name": "elevator.stl", "path": el_path,
                "role": "elevator", "visible": True, "color": ROLE_COLORS["elevator"],
                "mesh": el_mesh, "actor": el_actor,
            })
            self.next_body_id += 1
            self.update_bodies_table()
            # Не вызываем plotter.render() здесь — пользователь останется
            # на своём ракурсе (ГО/ВО генерируется поверх уже видимой геометрии).
            self.log_text.append(
                f"Готово: ГО и руль высоты сгенерированы! Отклонение руля: {elev_deflection:.1f}°")
            self.invalidate_mesh("сгенерировано горизонтальное оперение")
        except Exception as e:
            self.log_text.append(f"Ошибка генерации ГО: {e}")

    def generate_vertical_stabilizer(self):
        try:
            fuselage = self._get_fuselage_body()
            if fuselage:
                bounds = fuselage["mesh"].bounds
                length = bounds[1] - bounds[0]
                self.vk_pos_x.setValue(bounds[1] - length * 0.12)
                self.vk_pos_z.setValue((bounds[4] + bounds[5]) * 0.5)
                self.vk_height.setValue(max(0.5, (bounds[5] - bounds[4]) * 1.0))
            for b in self.bodies:
                if b["role"] == "v_stab" and b.get("actor"):
                    self.plotter.remove_actor(b["actor"])
            self.bodies = [b for b in self.bodies if b["role"] != "v_stab"]
            height = self.vk_height.value()
            chord = self.vk_chord.value()
            sweep = self.vk_sweep.value()
            pos_x = self.vk_pos_x.value()
            pos_z = self.vk_pos_z.value()
            sweep_offset = height * math.tan(math.radians(sweep))
            rx, rz = generate_naca4_section(chord, "0012", twist=0.0)
            tx, tz = generate_naca4_section(chord * 0.45, "0012", twist=0.0)
            n = len(rx)
            points = []
            for i in range(n):
                points.append([rx[i] + pos_x, 0.0, rz[i] + pos_z])
            for i in range(n):
                points.append([tx[i] + sweep_offset + pos_x, 0.0, tz[i] + pos_z + height])
            faces = []
            for i in range(n - 1):
                faces.append([4, i, i + 1, n + i + 1, n + i])
            faces.append([4, n - 1, 0, n, 2 * n - 1])
            bottom_c = len(points)
            points.append(self._compute_centroid(points, 0, n))
            faces.extend(self._create_triangular_cap_faces(0, n, center_idx=bottom_c))
            top_c = len(points)
            points.append(self._compute_centroid(points, n, n))
            faces.extend(self._create_triangular_cap_faces(n, n, center_idx=top_c))
            flat_faces = [v for f in faces for v in f]
            vk_mesh = pv.PolyData(np.array(points),
                                  np.array(flat_faces)).triangulate().clean(tolerance=1e-6)
            vk_mesh.compute_normals(auto_orient_normals=True, inplace=True)
            os.makedirs(WORK_DIR_BASE, exist_ok=True)
            path = os.path.join(WORK_DIR_BASE, "v_stabilizer.stl")
            vk_mesh.save(path)
            actor = self.plotter.add_mesh(vk_mesh, color=ROLE_COLORS["v_stab"],
                                          opacity=0.9, show_edges=True)
            self.bodies.append({
                "id": self.next_body_id, "name": "v_stabilizer.stl", "path": path,
                "role": "v_stab", "visible": True, "color": ROLE_COLORS["v_stab"],
                "mesh": vk_mesh, "actor": actor,
            })
            self.next_body_id += 1
            self.update_bodies_table()
            # Не вызываем plotter.render() — пользователь остаётся на своём ракурсе.
            self.log_text.append(f"Готово: ВО: H={height:.1f}м, chord={chord:.2f}м")
            self.invalidate_mesh("сгенерировано вертикальное оперение")
        except Exception as e:
            self.log_text.append(f"Ошибка генерации ВО: {e}")

    def export_component(self, role):
        comp = next((b for b in self.bodies if b["role"] == role), None)
        if not comp:
            QMessageBox.warning(self, "Ошибка",
                                f"Компонент '{ROLES.get(role, role)}' не найден.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, f"Экспорт {ROLES.get(role, role)}", f"{role}.stl", "STL (*.stl)")
        if path:
            comp["mesh"].save(path)
            self.log_text.append(f"Экспортировано: {path}")

    def export_fuselage(self):
        fuselage = next((b for b in self.bodies if b["role"] == "fuselage"), None)
        if not fuselage:
            QMessageBox.warning(self, "Ошибка", "Фюзеляж не найден.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт фюзеляжа",
                                              "fuselage.stl", "STL (*.stl)")
        if path:
            fuselage["mesh"].save(path)
            self.log_text.append(f"Фюзеляж экспортирован: {path}")

    def export_wing(self):
        wing = next((b for b in self.bodies if b["role"] == "wing"), None)
        if not wing:
            QMessageBox.warning(self, "Ошибка", "Крыло не найдено.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт крыла", "wing.stl",
                                              "STL (*.stl)")
        if path:
            wing["mesh"].save(path)
            self.log_text.append(f"Крыло экспортировано: {path}")

    def generate_full_aircraft(self):
        try:
            self.log_text.append("Генерация полного самолёта...")
            if not self._get_fuselage_body():
                self.generate_fuselage()
            self.fill_wing_box_from_fuselage()
            self.preview_wing_box()
            self.auto_suggest_wing_params()
            self.generate_horizontal_stabilizer()
            self.generate_vertical_stabilizer()
            # КАМЕРУ НЕ СБРАСЫВАЕМ
            self.log_text.append("Готово: Полный самолёт сгенерирован!")
        except Exception as e:
            self.log_text.append(f"Ошибка генерации самолёта: {e}")

    def auto_suggest_wing_params(self):
        fuselage = self._get_fuselage_body()
        if fuselage is None:
            QMessageBox.warning(self, "Нет фюзеляжа",
                                "Сначала загрузите или сгенерируйте фюзеляж.")
            return
        bounds = fuselage["mesh"].bounds
        length = bounds[1] - bounds[0]
        span = max(length * 1.2, 6.0)
        chord_root = length * 0.18
        chord_tip = chord_root * 0.5
        self.w_span.setValue(span)
        self.w_chord_root.setValue(chord_root)
        self.w_chord_tip.setValue(chord_tip)
        self.w_sweep.setValue(8.0)
        self.w_twist.setValue(2.0)
        self.w_pos_x.setValue(bounds[0] + length * 0.4)
        self.w_pos_y.setValue(0.0)
        self.w_pos_z.setValue((bounds[4] + bounds[5]) / 2.0)
        self.log_text.append(f"Автоподбор крыла: размах {span:.2f} м, "
                             f"корень {chord_root:.2f} м")
        try:
            self.generate_wing_mesh_parametric(span, chord_root, chord_tip)
        except Exception as e:
            self.log_text.append(f"  Внимание: Ошибка генерации: {e}")

    def heal_selected_stl(self):
        idx = self.bodies_table.currentRow()
        if idx < 0 or idx >= len(self.bodies):
            return
        body = self.bodies[idx]
        def log(msg):
            self.log_text.append(msg)
        log(f"\nЛечение: {body['name']}")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        ok, msg, healed_path = heal_stl_mesh(body["path"], log_callback=log,
                                             work_dir=WORK_DIR_BASE)
        QApplication.restoreOverrideCursor()
        if ok and healed_path and os.path.exists(healed_path):
            try:
                new_mesh = pv.read(healed_path)
                if body.get("actor"):
                    self.plotter.remove_actor(body["actor"])
                body["actor"] = self.plotter.add_mesh(
                    new_mesh, color=body["color"], opacity=0.8, show_edges=True)
                body["mesh"] = new_mesh
                body["path"] = healed_path
                self.plotter.render()
            except Exception as e:
                log(f"Внимание: Не удалось обновить сцену: {e}")
        report = f"=== ОТЧЁТ О ЛЕЧЕНИИ ===\n\nФайл: {body['name']}\n{msg}"
        if ok and healed_path and os.path.exists(healed_path):
            nm = pv.read(healed_path)
            report += (f"\n\nПосле лечения:\n  Вершин: {nm.n_points}\n"
                       f"  Граней: {nm.n_cells}\n  Открытых рёбер: {nm.n_open_edges}")
        dlg = HealReportDialog("Отчёт о лечении STL", report, self)
        dlg.exec_()
        log("Готово")

    # =============================================================
    # ПРОВЕРКА СЕТКИ
    # =============================================================
    def _validate_mesh_file(self):
        if not os.path.exists(MESH_FILE):
            return False, "Файл mesh.su2 не найден."
        try:
            size = os.path.getsize(MESH_FILE)
            if size < 5000:
                return False, f"mesh.su2 слишком маленький ({size} байт)."
        except Exception as e:
            return False, f"Не удалось проверить размер mesh.su2: {e}"
        try:
            with open(MESH_FILE, "r", encoding="ascii", errors="ignore") as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            return False, f"Не удалось прочитать mesh.su2: {e}"
        airfoil_elems = None
        farfield_elems = None
        for i, line in enumerate(lines):
            if line.startswith("MARKER_TAG="):
                tag = line.split("=", 1)[1].strip()
                if i + 1 < len(lines) and lines[i + 1].startswith("MARKER_ELEMS="):
                    try:
                        elems = int(lines[i + 1].split("=", 1)[1].strip())
                    except Exception:
                        elems = 0
                    if tag == "airfoil":
                        airfoil_elems = elems
                    elif tag == "farfield":
                        farfield_elems = elems
        if airfoil_elems is None:
            return False, "В mesh.su2 нет маркера airfoil."
        if farfield_elems is None:
            return False, "В mesh.su2 нет маркера farfield."
        if airfoil_elems <= 0:
            return False, ("Маркер airfoil есть, но в нём 0 граничных элементов. "
                           "Самолёт не вырезан из сетки. "
                           "В этом случае SU2 не сможет записать surface_flow.vtu — "
                           "нечего выводить на поверхность. "
                           "Постройте сетку заново, проверьте, что геометрия "
                           "попадает в фоновую сетку (включите 'Точная' качество).")
        if farfield_elems <= 0:
            return False, "Маркер farfield есть, но в нём 0 граничных элементов."
        return True, f"Сетка OK: airfoil={airfoil_elems}, farfield={farfield_elems}"

    def _ensure_mesh_ready(self):
        if not self.mesh_ready:
            QMessageBox.warning(
                self, "Сетка не готова",
                "Сначала постройте расчётную сетку.\n\n"
                "Если вы меняли геометрию, крыло, фюзеляж, оперение "
                "или загружали новый STL — сетку нужно построить заново.")
            return False
        ok, msg = self._validate_mesh_file()
        if not ok:
            QMessageBox.critical(self, "Некорректная сетка",
                                 msg + "\n\nПостройте сетку заново.")
            self.mesh_ready = False
            return False
        self.log_text.append(f"Готово: Проверка сетки: {msg}")
        return True

    def invalidate_mesh(self, reason="геометрия изменена"):
        self.mesh_ready = False
        if hasattr(self, "btn_run"):
            self.btn_run.setEnabled(False)
            self.ribbon_btn_run.setEnabled(False)
        self.log_text.append(f"Внимание: Сетка устарела: {reason}. "
                             "Нужно построить сетку заново.")

    # =============================================================
    # СЕТКА
    # =============================================================
    def make_mesh_from_bodies(self):
        if self._meshing:
            return
        self.btn_make_mesh.setEnabled(False)
        self.ribbon_btn_mesh.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.ribbon_btn_run.setEnabled(False)
        if not self.bodies:
            QMessageBox.warning(self, "Ошибка", "Нет компонентов для построения сетки.")
            self.btn_make_mesh.setEnabled(True)
            self.ribbon_btn_mesh.setEnabled(True)
            return
        os.makedirs(WORK_DIR_BASE, exist_ok=True)
        visible = [b for b in self.bodies if b.get("visible", True)]
        if not visible:
            QMessageBox.warning(self, "Ошибка", "Нет видимых компонентов.")
            self.btn_make_mesh.setEnabled(True)
            self.ribbon_btn_mesh.setEnabled(True)
            return
        stl_paths = []
        t_save_start = time.time()
        self.log_text.append("\n" + "=" * 50)
        self.log_text.append("ПОСТРОЕНИЕ РАСЧЁТНОЙ СЕТКИ")
        self.log_text.append(f"   Старт: {datetime.now().strftime('%H:%M:%S')}, "
                             f"компонентов: {len(visible)}, "
                             f"качество: {self.combo_mesh_quality.currentText()}")
        self.log_text.append("=" * 50)
        for b in visible:
            try:
                path = os.path.join(WORK_DIR_BASE, f"_body_{b['id']}_{b['role']}.stl")
                b["mesh"].save(path)
                stl_paths.append(path)
                self.log_text.append(
                    f"  Готово: {b['name']} ({b['mesh'].n_cells} граней) → {path}"
                )
            except Exception as e:
                self.log_text.append(f"  Ошибка: {b['name']}: {e}")
        if not stl_paths:
            QMessageBox.critical(self, "Ошибка", "Не удалось сохранить геометрию.")
            self.btn_make_mesh.setEnabled(True)
            self.ribbon_btn_mesh.setEnabled(True)
            return
        self.log_text.append(
            f"  Сохранение STL заняло {time.time() - t_save_start:.1f}с"
        )
        quality = self.combo_mesh_quality.currentText()
        self.log_text.append(f"\nКачество: {quality}")
        self.log_text.append("Генерация в фоновом режиме (UI не зависает)…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._meshing = True
        self._mesh_start_time = time.time()
        self._clock_begin()
        self._eta_ema = None
        self._last_logged_pct = -1
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.lbl_su2_status.setText("Генерация расчетной сетки...")
        self.lbl_su2_status.setStyleSheet("color: #2E5A78; font-style: italic;")
        # === Плоскости симметрии пробрасываются в MeshWorker ============
        # Источник — список плоскостей из 3D-инструмента.
        # Симметрия включена тогда и только тогда, когда плоскость
        # добавлена: отдельной галочки больше нет.
        sym_planes_for_mesh = list(self.get_symmetry_planes())
        use_sym_for_mesh = bool(sym_planes_for_mesh)
        try:
            self._mesh_worker = MeshWorker(
                stl_paths, quality, parent=self,
                use_symmetry=use_sym_for_mesh,
                symmetry_planes=sym_planes_for_mesh,
            )
        except TypeError:
            # Старая сигнатура MeshWorker без новых параметров
            try:
                self._mesh_worker = MeshWorker(
                    stl_paths, quality, parent=self,
                    use_symmetry=use_sym_for_mesh,
                )
            except TypeError:
                self._mesh_worker = MeshWorker(stl_paths, quality, parent=self)
        # ================================================================
        self._mesh_worker.setProperty("stl_paths", stl_paths)
        self._mesh_worker.progress_signal.connect(self.on_mesh_progress)
        self._mesh_worker.finished_signal.connect(
            lambda ok, msg: self.on_mesh_finished(ok, msg, stl_paths))
        self._mesh_worker.start()

    def on_mesh_progress(self, percent, stage):
        self._set_progress(percent)
        self.lbl_mesh_info.setText(f"{stage} ({percent}%)")
        elapsed = time.time() - self._mesh_start_time
        if percent > 2:
            total_est = elapsed * 100.0 / percent
            remain = max(0.0, total_est - elapsed)
            self._eta_ema = remain if self._eta_ema is None \
                else 0.3 * remain + 0.7 * self._eta_ema
            mins = int(self._eta_ema // 60)
            secs = int(self._eta_ema % 60)
            self._clock_set_eta(self._eta_ema)
            # Подробный лог: первый раз при <5%, потом каждые ~20%
            if not hasattr(self, "_last_logged_pct"):
                self._last_logged_pct = -1
            if percent - self._last_logged_pct >= 20 or percent < 5:
                self.log_text.append(
                    f"  [{int(elapsed):02d}с] {stage} — {percent}%, "
                    f"осталось ~{mins}м {secs:02d}с"
                )
                self._last_logged_pct = percent

    def on_mesh_finished(self, ok, msg, stl_paths):
        self._meshing = False
        self.progress.setVisible(False)
        self._clock_end()
        if self._mesh_worker:
            self._mesh_worker.deleteLater()
            self._mesh_worker = None
        for p in stl_paths or []:
            try:
                os.remove(p)
            except Exception:
                pass
        QApplication.restoreOverrideCursor()
        if ok:
            self.log_text.append("\n" + "=" * 50)
            elapsed = time.time() - self._mesh_start_time
            self.log_text.append(
                f"Готово: {msg}  (время: {int(elapsed)}с)"
            )
            self.log_text.append("=" * 50)
            valid, valid_msg = self._validate_mesh_file()
            if not valid:
                self.mesh_ready = False
                QMessageBox.critical(self, "Сетка создана, но некорректна", valid_msg)
                self.btn_make_mesh.setEnabled(True)
                self.ribbon_btn_mesh.setEnabled(True)
                return
            self.mesh_ready = True
            self.log_text.append(f"Готово: {valid_msg}")
            # Размер mesh.su2
            try:
                sz = os.path.getsize(MESH_FILE)
                if sz > 1024 * 1024:
                    sz_str = f"{sz / (1024 * 1024):.1f} МБ"
                else:
                    sz_str = f"{sz / 1024:.0f} КБ"
                self.log_text.append(f"Размер mesh.su2: {sz_str}")
            except Exception:
                pass
            try:
                self.plotter.clear()
                self.plotter.add_axes()
                for b in self.bodies:
                    if b.get("visible", True):
                        b["actor"] = self.plotter.add_mesh(
                            b["mesh"], color=b["color"], opacity=0.6, show_edges=True)
                if os.path.exists(PREVIEW_MESH):
                    try:
                        mesh_preview = pv.read(PREVIEW_MESH)
                        self.plotter.add_mesh(
                            mesh_preview, color="lightblue", opacity=0.15,
                            show_edges=True, name="volume_mesh")
                        self.log_text.append(
                            f"Отображена сетка: {mesh_preview.n_cells} ячеек")
                    except Exception as e:
                        self.log_text.append(f"Внимание: Не удалось загрузить preview: {e}")
                # T1-визуал: перерисовать плоскости симметрии после очистки
                if hasattr(self, "_symmetry_planes") and self._symmetry_planes:
                    self._update_symmetry_3d()
                # КАМЕРУ НЕ СБРАСЫВАЕМ
            except Exception as e:
                self.log_text.append(f"Внимание: Ошибка визуализации: {e}")
            self.btn_run.setEnabled(True)
            self.ribbon_btn_run.setEnabled(True)
            self.btn_show_flow.setEnabled(False)
            self.lbl_su2_status.setText("Готово: Сетка успешно построена")
            self.lbl_su2_status.setStyleSheet("color: #2E6B45; font-weight: bold;")
        else:
            self.log_text.append("\n" + "=" * 50)
            self.log_text.append(f"Ошибка: {msg}")
            self.log_text.append("=" * 50)
            self.lbl_su2_status.setText("Ошибка построения сетки!")
            self.lbl_su2_status.setStyleSheet("color: #9B2C2C; font-weight: bold;")
            QMessageBox.critical(self, "Ошибка генерации сетки", msg)
        self.btn_make_mesh.setEnabled(True)
        self.ribbon_btn_mesh.setEnabled(True)

    # =============================================================
    # АДАПТИВНАЯ СЕТКА (SU2_ADAPT по решению)
    # =============================================================
    def adapt_mesh_by_cp(self):
        """Перестройка поверхностной сетки по градиенту Cp (gmsh)."""
        from mesh.adapt_gmsh import (adaptivity_report, format_adaptivity_report,
                                     parse_surface_flow_csv,
                                     pressure_gradient_along_surface,
                                     rebuild_with_metric, surface_size_metric,
                                     write_metric_msh)
        csv_path = self._find_latest_surface_flow_csv()
        if not csv_path:
            QMessageBox.information(
                self, "Адаптация по Cp",
                "Не найден surface_flow.csv — нужен завершённый расчёт с "
                "записью распределения по поверхности.\n\n"
                "В config.cfg должно быть SURFACE_CSV... / OUTPUT_FILES, "
                "либо выполните расчёт штатной кнопкой: приложение пишет "
                "surface_flow.csv в каталог расчёта.")
            return
        src_mesh = self._last_stl_or_mesh_source()
        if not src_mesh:
            QMessageBox.warning(self, "Адаптация по Cp",
                                "Нет исходной геометрии (STL) для "
                                "перестроения сетки. Сначала импортируйте "
                                "или сгенерируйте геометрию.")
            return
        try:
            samples = parse_surface_flow_csv(csv_path)
            grad = pressure_gradient_along_surface(
                samples["x"], samples["y"], samples["cp"],
                samples.get("z"))
            pts, sizes = surface_size_metric(
                samples["x"], samples["y"], grad,
                h_min=self.adapt_h_min.value(),
                h_max=self.adapt_h_max.value(),
                power=self.adapt_power.value(), z=samples.get("z"))
            os.makedirs(WORK_DIR_BASE, exist_ok=True)
            metric = os.path.join(WORK_DIR_BASE, "_adapt_metric.msh")
            write_metric_msh(metric, pts, sizes)
            out_stl = os.path.join(WORK_DIR_BASE, "_adapted_surface.stl")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                rebuild_with_metric(src_mesh, metric, out_stl,
                                    h_min=self.adapt_h_min.value(),
                                    h_max=self.adapt_h_max.value(),
                                    log=lambda m: self.log_text.append(m))
            finally:
                QApplication.restoreOverrideCursor()
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self.log_text.append(f"Ошибка: Адаптация по Cp не выполнена: {e}")
            QMessageBox.critical(self, "Адаптация по Cp", str(e))
            return
        rep = adaptivity_report(samples, sizes)
        text = format_adaptivity_report(rep)
        self.log_text.append("" + text.replace("\n", " | "))
        self.log_text.append(f"  Готово: Адаптивная геометрия: {out_stl}")
        self._add_body(out_stl, "other")
        self.update_flow_arrow()
        QMessageBox.information(
            self, "Адаптация по Cp",
            text + "\n\nАдаптивная поверхность добавлена как новый "
            "компонент. Удалите исходную деталь, если она больше не нужна, "
            "и перестройте расчётную сетку.")

    def _find_latest_surface_flow_csv(self) -> str:
        """Ищет свежайший surface_flow.csv в каталогах расчётов."""
        cands = []
        for root, _dirs, files in os.walk(WORK_DIR_BASE):
            for fn in files:
                if fn.lower() == "surface_flow.csv":
                    p = os.path.join(root, fn)
                    try:
                        cands.append((os.path.getmtime(p), p))
                    except OSError:
                        continue
        if not cands:
            return ""
        return max(cands)[1]

    def _last_stl_or_mesh_source(self) -> str:
        """Путь к STL первой видимой детали — источник для перестроения."""
        for b in getattr(self, "bodies", []):
            if b.get("visible", True) and str(b.get("path", "")).lower().endswith(".stl"):
                p = b["path"]
                if os.path.isfile(p):
                    return p
        return ""

    def adapt_current_mesh(self):
        if getattr(self, "_adapting", False):
            return
        if not os.path.isfile(MESH_FILE):
            QMessageBox.warning(self, "Адаптивная сетка",
                                "Сначала постройте сетку (кнопка «Построить сетку»).")
            return
        # Нужно решение (restart.dat) из завершённого расчёта
        restart = self._find_latest_restart()
        if not restart:
            QMessageBox.information(
                self, "Адаптивная сетка",
                "Для адаптации нужно решение (restart.dat) из завершённого "
                "расчёта.\n\n1. Постройте сетку и выполните расчёт.\n"
                "2. Затем нажмите «Адаптировать сетку» — сетка локально "
                "сгустится в областях высоких градиентов, и её можно "
                "считать ещё раз (точнее и с меньшим числом ячеек, чем "
                "при глобальном сгущении).")
            return
        case_dir = os.path.dirname(restart)
        npoin_before = _mesh_npoin(MESH_FILE) or 0
        reply = QMessageBox.question(
            self, "Адаптивная сетка",
            f"Сетка сейчас: {npoin_before or '?'} точек.\n\n"
            f"Адаптировать по решению из:\n{os.path.basename(case_dir)}\n\n"
            "После адаптации рабочая сетка (mesh.su2) будет заменена "
            "адаптированной. Продолжить?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self._adapting = True
        self.btn_adapt_mesh.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.lbl_su2_status.setText("Адаптация сетки (SU2_ADAPT)...")
        self.lbl_su2_status.setStyleSheet("color: #2E5A78; font-style: italic;")
        self.log_text.append("\n" + "=" * 50)
        self.log_text.append("АДАПТИВНАЯ СЕТКА (SU2_ADAPT)")
        self.log_text.append("=" * 50)
        self._adapt_worker = MeshAdaptWorker(
            case_dir=case_dir,
            mesh_path=MESH_FILE,
            restart_path=restart,
            adapt_markers=("airfoil",),
            parent=self,
        )
        self._adapt_worker.progress_signal.connect(
            lambda p, s: self.statusBar().showMessage(f"Адаптация: {s}", 5000))
        self._adapt_worker.finished_signal.connect(self.on_adapt_finished)
        self._adapt_worker.start()

    def _find_latest_restart(self):
        """Ищет самый свежий restart.dat в work/case_*/aoa_*/."""
        candidates = []
        for root, _dirs, files in os.walk(WORK_DIR_BASE):
            if "restart.dat" in files:
                candidates.append(os.path.join(root, "restart.dat"))
        if not candidates:
            return None
        try:
            return max(candidates, key=os.path.getmtime)
        except OSError:
            return candidates[-1]

    def on_adapt_finished(self, ok, msg):
        self._adapting = False
        self.btn_adapt_mesh.setEnabled(True)
        self.progress.setVisible(False)
        if getattr(self, "_adapt_worker", None):
            self._adapt_worker.deleteLater()
            self._adapt_worker = None
        if not ok:
            self.log_text.append(f"Ошибка: {msg}")
            self.lbl_su2_status.setText("Ошибка адаптации")
            self.lbl_su2_status.setStyleSheet("color: #9B2C2C; font-weight: bold;")
            QMessageBox.critical(self, "Ошибка адаптации сетки", msg)
            return
        # Заменяем рабочую сетку адаптированной
        try:
            npoin_before = _mesh_npoin(MESH_FILE) or 0
            shutil.copy2(msg, MESH_FILE)
            npoin_after = _mesh_npoin(MESH_FILE) or 0
            self.log_text.append(
                f"Готово: Адаптированная сетка: {msg}")
            self.log_text.append(
                f"   Точки: {npoin_before} → {npoin_after} "
                f"({(npoin_after / max(npoin_before, 1)):.2f}×)")
            self.lbl_su2_status.setText("Готово: Сетка адаптирована")
            self.lbl_su2_status.setStyleSheet("color: #2E6B45; font-weight: bold;")
            QMessageBox.information(
                self, "Адаптация завершена",
                f"Сетка адаптирована по решению.\n\n"
                f"Точки: {npoin_before} → {npoin_after}\n\n"
                "Теперь можно запустить расчёт заново — результат будет "
                "точнее в областях высоких градиентов.")
        except Exception as e:
            self.log_text.append(f"Не удалось применить адаптированную сетку: {e}")
            QMessageBox.critical(self, "Ошибка", f"{e}")

    # =============================================================
    # РАСЧЁТ
    # =============================================================
    def start_calculation(self):
        # === T6: проверка лицензии перед расчётом ======================
        if self._license is not None:
            allowed, reason = self._license.is_calculation_allowed()
            if not allowed:
                msg_box = QMessageBox(self)
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setWindowTitle("Лицензия")
                msg_box.setTextFormat(Qt.RichText)
                msg_box.setText(
                    f"<b>Расчёт заблокирован</b><br><br>"
                    f"{reason}<br><br>"
                    'Купить лицензию: '
                    '<a href="https://bergaff.github.io/aeroopt-site/">'
                    'aeroopt.app</a><br><br>'
                    "Введите ключ через меню <b>Лицензия → Активировать</b>."
                )
                msg_box.exec_()
                self.log_text.append(f"{reason}")
                return
            # Запрашиваем одноразовый run_token у сервера
            ok, msg = self._license.acquire_run_token()
            if not ok:
                QMessageBox.warning(
                    self, "Лицензия", f"Не удалось получить run_token:\n{msg}")
                self.log_text.append(f"{msg}")
                return
            # ok=True (или offline в grace) — продолжаем
        # ================================================================
        if not self._ensure_mesh_ready():
            return
        if not self.validate_rules_before_run():
            return
        # Перед расчётом автоматически фиксируем полётные условия,
        # чтобы не было сюрпризов с «забыл нажать Применить».
        self.apply_flight_conditions()
        # Проверка приостановленной сессии: если она есть и геометрия
        # изменилась (или просто есть) — даём выбор: продолжить или начать новую.
        existing = CalculationSession(WORK_DIR_BASE)
        if existing.exists_on_disk() and existing.load() and existing.paused \
                and not existing.is_complete:
            choice = self._ask_resume_or_new(existing)
            if choice == "cancel":
                return
            if choice == "new":
                existing.clear()
                self.log_text.append(
                    "Прошлая приостановленная сессия отменена, "
                    "стартуем новую."
                )
            elif choice == "resume":
                # compute_device и gpu_percent из старой сессии не
                # восстанавливаем: выбора вычислителя в интерфейсе нет,
                # иначе проект с "cpu_gpu" включил бы GPU-ветки в обход UI.
                self._compute_device_pending = "cpu"
                self._gpu_percent_pending = 0
                self._cpu_cores_pending = int(
                    getattr(existing, "cpu_cores", self._cpu_cores_pending) or
                    self._cpu_cores_pending
                )
                self.session = existing
                self.log_text.append(
                    f"Возобновление сессии: точка "
                    f"{existing.current_index + 1}/{len(existing.aoa_list)}"
                )
                self._launch_session_runner()
                return
        # Подсказка по числу ядер. Прежний делитель 150000 точек на ядро
        # для сетки в 174 тысячи точек давал рекомендацию «1 ядро» и тут
        # же советовал нагрузку увеличить — совет противоречил сам себе.
        # Ориентир для SU2 — порядка 25 тысяч узлов на ядро.
        try:
            npoin = _mesh_npoin(MESH_FILE) or 0
            if npoin > 0:
                phys = max(1, int(getattr(self, "_cpu_cores_max", 1) or 1))
                rec = min(phys, max(1, int(round(npoin / 25000.0))))
                cur = self._resolve_cores_for_level()
                if rec > cur:
                    tail = ("Для ускорения увеличьте нагрузку CPU "
                            "в Solver Settings.")
                elif rec < cur:
                    tail = ("Больше ядер здесь не ускорит расчёт: "
                            "нагрузка упрётся в память и обмен.")
                else:
                    tail = "Нагрузка подобрана по размеру сетки."
                self.log_text.append(
                    f"Сетка ~{npoin} узлов: рекомендуется ≈{rec} ядер "
                    f"(сейчас {cur}). {tail}")
        except Exception:
            pass
        physics = self.get_physics()
        solver = self.get_solver()
        ref_data = self.calculate_reference_data()
        active_markers = [b["role"] for b in self.bodies if b.get("visible", True)]
        if self.rb_sweep.isChecked():
            a0, a1 = self.input_aoa_start.value(), self.input_aoa_end.value()
            st = self.input_aoa_step.value() or 1.0
            aoa_list = []
            cur = a0
            while cur <= a1 + 1e-6:
                aoa_list.append(round(cur, 2))
                cur += st
            mode = "sweep"
        else:
            aoa_list = [self.input_aoa.value()]
            mode = "single"
        self.table.setRowCount(0)
        self.all_results = []
        self.session = CalculationSession(WORK_DIR_BASE)
        # Запоминаем качество сетки — это влияет на INNER_ITER в config.cfg
        mesh_quality_now = self.combo_mesh_quality.currentText()
        # Гибридный GPU-режим. Если пользователь выбрал «CPU+GPU»,
        # но GPU не нашлись — apply_load_level уже сделал откат на cpu/0.
        # Здесь просто пробрасываем текущее применённое состояние.
        compute_device_now = getattr(self, "_compute_device_pending", "cpu")
        gpu_percent_now = getattr(self, "_gpu_percent_pending", 0)
        # === Флаги симметрии / RAMP / турбомодель =========================
        # Источник истины для симметрии — список плоскостей из 3D-инструмента.
        # Галочки нет: симметрия включена, если добавлена плоскость.
        symmetry_planes_now = list(self.get_symmetry_planes())
        use_symmetry_now = bool(symmetry_planes_now)
        use_ramp_aoa_now = bool(
            getattr(self, "chk_use_ramp_aoa", None)
            and self.chk_use_ramp_aoa.isChecked()
        )
        turb_model_now = self.get_turb_model()
        # ==================================================================
        # Пытаемся передать через kwargs (если solver/session.py их поддерживает).
        # Если сигнатура старой версии — kwarg просто проигнорируется
        # и мы допишем атрибутами ниже.
        start_new_kwargs = {"cpu_cores": self._cpu_cores_pending}
        # Число компонентов и узлов сетки нужны для оценки потолка
        # итераций: одиночное крыло и самолёт из пяти компонентов с
        # механизацией не должны получать одинаковые 6000.
        n_bodies_now = len([b for b in getattr(self, "bodies", []) or []
                            if b.get("mesh") is not None])
        try:
            n_points_now = _mesh_npoin(MESH_FILE) or 0
        except Exception:
            n_points_now = 0
        cfl_fast_now = bool(getattr(self, "chk_cfl_aggressive", None)
                            and self.chk_cfl_aggressive.isChecked())
        try:
            import inspect
            sig = inspect.signature(self.session.start_new)
            for key, value in [
                ("compute_device", compute_device_now),
                ("gpu_percent", gpu_percent_now),
                ("use_symmetry", use_symmetry_now),
                ("symmetry_planes", symmetry_planes_now),
                ("use_ramp_aoa", use_ramp_aoa_now),
                ("turb_model", turb_model_now),
                ("n_bodies", n_bodies_now),
                ("n_points", n_points_now),
                ("cfl_aggressive", cfl_fast_now),
            ]:
                if key in sig.parameters:
                    start_new_kwargs[key] = value
        except Exception:
            pass
        try:
            self.session.start_new(mode, solver, physics, ref_data,
                                   active_markers, aoa_list, **start_new_kwargs)
        except TypeError:
            # Старая сигнатура: без **kwargs — вызываем как раньше
            self.session.start_new(mode, solver, physics, ref_data,
                                   active_markers, aoa_list,
                                   cpu_cores=self._cpu_cores_pending)
        # Дублируем атрибутами — solver/workers.py и config_builder.py
        # читают их напрямую (см. write_case_config).
        # Старые версии solver/session.py просто их проигнорируют.
        self.session.compute_device = compute_device_now
        self.session.gpu_percent = gpu_percent_now
        self.session.use_symmetry = use_symmetry_now
        self.session.symmetry_planes = symmetry_planes_now
        self.session.use_ramp_aoa = use_ramp_aoa_now
        self.session.turb_model = turb_model_now
        # start_new не принимает mesh_quality — пишем напрямую (атрибут есть
        # в __init__, см. solver/session.py)
        self.session.mesh_quality = mesh_quality_now
        self.log_text.append(
            f"\nЗапуск сессии ({mode}) в {datetime.now().strftime('%H:%M:%S')}, "
            f"точек: {len(aoa_list)}, ядер: {self._cpu_cores_pending}, "
            f"решатель: {solver}, сетка: {mesh_quality_now}"
        )
        # Старт таймера всей сессии — для итогового отчёта «расчёт шёл X секунд»
        self._session_start_time = time.time()
        if aoa_list:
            self.log_text.append(
                f"   Точки: {', '.join(f'{a:.2f}°' for a in aoa_list)}"
            )
        self._launch_session_runner()

    def _ask_resume_or_new(self, existing) -> str:
        """Диалог: «Продолжить прошлую / Начать новую / Отмена». Возвращает
        'resume' | 'new' | 'cancel'."""
        n_done = existing.current_index + 1
        n_total = len(existing.aoa_list)
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowTitle("Приостановленная сессия")
        msg_box.setText(
            f"Найдена приостановленная сессия ({existing.mode}).\n"
            f"Прогресс: точка {n_done}/{n_total}.\n\n"
            f"Что сделать?"
        )
        btn_resume = msg_box.addButton("Продолжить прошлую", QMessageBox.AcceptRole)
        btn_new = msg_box.addButton("Начать новую", QMessageBox.DestructiveRole)
        btn_cancel = msg_box.addButton("Отмена", QMessageBox.RejectRole)
        msg_box.setDefaultButton(btn_resume)
        msg_box.exec_()
        clicked = msg_box.clickedButton()
        if clicked is btn_resume:
            return "resume"
        if clicked is btn_new:
            return "new"
        return "cancel"

    def _launch_session_runner(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.calc_start_time = time.time()
        self._eta_ema = None
        self._clock_begin()
        self.session_runner = SessionRunner(self.session)
        self.session_runner.log_signal.connect(self.log_text.append)
        self.session_runner.progress_signal.connect(self.on_calculation_progress)
        self.session_runner.result_ready.connect(self.add_result)
        self.session_runner.paused_signal.connect(self.on_session_paused)
        self.session_runner.finished_all.connect(self.on_session_finished)
        total = max(1, len(self.session.aoa_list))
        self.progress.setVisible(True)
        self.progress.setValue(int(self.session.current_index / total * 100))
        self.set_calculation_buttons_enabled(run=False, pause=True,
                                             resume=False, cancel=True)
        self.session_runner.start()

    def on_calculation_progress(self, percent):
        self._set_progress(percent)
        if hasattr(self, "calc_start_time"):
            elapsed = time.time() - self.calc_start_time
            if percent > 2:
                total_est = elapsed / (percent / 100.0)
                remain = max(0.0, total_est - elapsed)
                self._eta_ema = remain if self._eta_ema is None \
                    else 0.3 * remain + 0.7 * self._eta_ema
                mins = int(self._eta_ema // 60)
                secs = int(self._eta_ema % 60)
                self._clock_set_eta(self._eta_ema
                                    if hasattr(self, "_eta_ema") else None)
        if self.session:
            curr_pt = self.session.current_index + 1
            total_pts = len(self.session.aoa_list)
            self.lbl_su2_status.setText(
                f"Расчёт точки {curr_pt}/{total_pts} ({percent}%)")
            self.lbl_su2_status.setStyleSheet("color: #2E5A78; font-style: italic;")

    def pause_calculation(self):
        if self.session_runner and self.session_runner.isRunning():
            self.log_text.append("Запрошена пауза...")
            self.btn_pause.setEnabled(False)
            self.session_runner.request_pause()

    def resume_calculation(self):
        session = CalculationSession(WORK_DIR_BASE)
        if not session.load():
            QMessageBox.warning(self, "Ошибка", "Сохранённая сессия не найдена.")
            return
        if not session.paused:
            QMessageBox.warning(self, "Ошибка", "Сессия не находится на паузе.")
            return
        # Вычислитель всегда CPU: GPU-режим из интерфейса убран, и старый
        # файл сессии не должен его включать.
        self._compute_device_pending = "cpu"
        self._gpu_percent_pending = 0
        self._cpu_cores_pending = int(
            getattr(session, "cpu_cores", self._cpu_cores_pending) or
            self._cpu_cores_pending
        )
        self.session = session
        self.log_text.append(
            f"Возобновление сессии: ядер {self._cpu_cores_pending}"
        )
        self._launch_session_runner()

    def cancel_calculation(self):
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
        if self.session_runner and self.session_runner.isRunning():
            self.log_text.append("Отмена расчёта...")
            self.session_runner.request_cancel()
            self.session_runner.wait(5000)
        elif self.session:
            self.session.mark_cancelled()
            self.log_text.append("Приостановленная сессия отменена.")
        self.cleanup_session_data()
        self.set_calculation_buttons_enabled(run=self.mesh_ready, pause=False,
                                             resume=False, cancel=False)
        self.progress.setVisible(False)
        self._clock_end()
        self.lbl_su2_status.setText("Расчет отменен")
        self.lbl_su2_status.setStyleSheet("color: #9B2C2C; font-weight: bold;")

    def on_session_paused(self):
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
        self.set_calculation_buttons_enabled(run=False, pause=False,
                                             resume=True, cancel=True)
        self.progress.setVisible(False)
        self._clock_end()
        self.lbl_su2_status.setText("На паузе")
        self.lbl_su2_status.setStyleSheet("color: orange; font-weight: bold;")
        self.log_text.append("Сессия на паузе. Можно продолжить или отменить.")
        if not self.project_saved:
            reply = QMessageBox.question(
                self, "Сохранить проект",
                "Сессия поставлена на паузу.\n\n"
                "Хотите сохранить текущий проект сейчас, чтобы не потерять расчеты?",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.save_project()
        if self.session:
            curr_dir = self.session.current_case_dir()
            if curr_dir:
                self._save_pause_history(curr_dir)

    def on_session_finished(self):
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
        self.set_calculation_buttons_enabled(run=self.mesh_ready, pause=False,
                                             resume=False, cancel=False)
        self.progress.setVisible(False)
        self._clock_end()
        self.btn_save_csv.setEnabled(True)
        self.btn_show_flow.setEnabled(True)
        n_ok = sum(1 for r in self.all_results if not r.get("error"))
        n_fail = len(self.all_results) - n_ok
        # Длительность всей сессии — от start_calculation до finished_all.
        # Полезно и при успехе, и при ошибке: видно, сколько реально шёл расчёт.
        elapsed = None
        try:
            start_t = getattr(self, "_session_start_time", None)
            if start_t is not None:
                elapsed = time.time() - float(start_t)
        except Exception:
            elapsed = None
        if n_ok > 0:
            self.lbl_su2_status.setText(
                f"Готово: Расчёт завершён: {n_ok} успешно, {n_fail} с ошибкой")
            self.lbl_su2_status.setStyleSheet("color: #2E6B45; font-weight: bold;")
        else:
            self.lbl_su2_status.setText("Ошибка: Расчёт завершился безуспешно")
            self.lbl_su2_status.setStyleSheet("color: #9B2C2C; font-weight: bold;")
        if elapsed is not None:
            self.log_text.append(
                f"Расчёт шёл {_format_duration(elapsed)} "
                f"(≈ {elapsed:.1f} с)"
            )
        self.log_text.append(
            f"Сессия завершена: {n_ok} успешных, {n_fail} с ошибкой.")

    def _check_pending_session(self):
        tmp_session = CalculationSession(WORK_DIR_BASE)
        if (tmp_session.exists_on_disk() and tmp_session.load()
                and tmp_session.paused and not tmp_session.is_complete):
            QTimer.singleShot(600, lambda: self._offer_resume(tmp_session))

    def _offer_resume(self, tmp_session):
        reply = QMessageBox.question(
            self, "Незавершённый расчёт",
            f"Обнаружена приостановленная сессия ({tmp_session.mode}).\n"
            f"Точка {tmp_session.current_index + 1}/{len(tmp_session.aoa_list)}.\n"
            "\n"
            "Возобновить расчёт?",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            # Вычислитель всегда CPU (см. выше).
            self._compute_device_pending = "cpu"
            self._gpu_percent_pending = 0
            self._cpu_cores_pending = int(
                getattr(tmp_session, "cpu_cores", self._cpu_cores_pending) or
                self._cpu_cores_pending
            )
            self.session = tmp_session
            self._launch_session_runner()
        else:
            tmp_session.clear()

    def _save_pause_history(self, case_dir):
        try:
            src = os.path.join(case_dir, "history.csv")
            if os.path.exists(src):
                dst = os.path.join(
                    case_dir,
                    f"history_paused_{datetime.now().strftime('%H%M%S')}.csv")
                shutil.copy2(src, dst)
                self.log_text.append(f"История расчёта сохранена: {dst}")
        except Exception as e:
            self.log_text.append(f"Внимание: Не удалось сохранить историю паузы: {e}")

    # =============================================================
    # РЕЗУЛЬТАТЫ
    # =============================================================
    def add_result(self, res):
        if res.get("stopped"):
            return
        self.all_results.append(res)
        # Оценка времени до конца серии: среднее на точку × оставшиеся
        try:
            t0 = getattr(self, "calc_start_time", None)
            if t0:
                total = len(self.session_runner.session.aoa_list)
                done = len(self.all_results)
                if 0 < done < total:
                    avg = (time.time() - t0) / done
                    remain = avg * (total - done)
                    self.log_text.append(
                        f"Точка {done}/{total}: среднее {avg:.0f} с/точка, "
                        f"осталось ~{int(remain // 60)}м {int(remain % 60)}с")
        except Exception:
            pass
        row = self.table.rowCount()
        self.table.insertRow(row)
        cl = res.get('cl', 0)
        cd = res.get('cd', 0)
        cm = res.get('cm', 0)
        k = cl / cd if cd > 0.001 else 0
        is_err = res.get('error', True)
        status_text = "OK" if not is_err else res.get("error_msg", "Ошибка расчета")
        values = [str(res.get('aoa', 0)), f"{cl:.4f}", f"{cd:.5f}",
                  f"{cm:.4f}", f"{k:.2f}", status_text]
        for i, v in enumerate(values):
            item = QTableWidgetItem(v)
            if i == 5:
                item.setBackground(QColor(255, 200, 200) if is_err
                                   else QColor(200, 255, 200))
                if is_err:
                    item.setToolTip(res.get("error_msg", ""))
            self.table.setItem(row, i, item)
        self.plot_canvas.update_plots(self.all_results)
        self.project_saved = False
        if is_err:
            msg = (f"Ошибка при расчете точки AoA = {res.get('aoa', 0)}°.\n\n"
                   f"Причина: {status_text}\n\n"
                   "Рекомендации:\n"
                   "1. Проверьте путь к SU2_CFD.exe в Solver Settings.\n"
                   "2. Убедитесь, что сетка корректна и не содержит вырожденных элементов.\n"
                   "3. Проверьте лог файлы в папке расчёта (su2_stdout.log).")
            QMessageBox.warning(self, "Ошибка расчета CFD", msg)

    # =============================================================
    # ОПТИМИЗАЦИЯ
    # =============================================================
    def run_geometric_optimization(self):
        # === T6: проверка лицензии перед оптимизацией ======================
        if self._license is not None:
            allowed, reason = self._license.is_calculation_allowed()
            if not allowed:
                msg_box = QMessageBox(self)
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setWindowTitle("Лицензия")
                msg_box.setTextFormat(Qt.RichText)
                msg_box.setText(
                    f"<b>Оптимизация заблокирована</b><br><br>"
                    f"{reason}<br><br>"
                    'Купить лицензию: '
                    '<a href="https://bergaff.github.io/aeroopt-site/">'
                    'aeroopt.app</a><br><br>'
                    "Введите ключ через меню <b>Лицензия → Активировать</b>."
                )
                msg_box.exec_()
                self.log_text.append(f"{reason}")
                return
        # ================================================================
        if not self.bodies:
            QMessageBox.warning(self, "Ошибка", "Загрузите геометрию.")
            return
        if not self.validate_rules_before_run():
            return
        ref_data = self.calculate_reference_data()
        flight_points = self._get_opt_points()
        self.btn_start_opt.setEnabled(False)
        self._opt_running = True
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.lbl_opt_status.setText("Оптимизация...")
        initial_params = {
            'span': self.w_span.value(),
            'chord_root': self.w_chord_root.value(),
            'chord_tip': self.w_chord_tip.value(),
            'sweep': self.w_sweep.value(),
        }
        if self.flap_enabled.isChecked():
            initial_params['flap_deflection'] = self.flap_deflection.value()
            initial_params['flap_slide'] = self.flap_slide.value()
            initial_params['flap_hinge_depth'] = self.flap_hinge_depth.value()
        if self.slat_enabled.isChecked():
            initial_params['slat_deflection'] = self.slat_deflection.value()
            initial_params['slat_slide'] = self.slat_slide.value()
        active_markers = [b["role"] for b in self.bodies if b.get("visible", True)]
        self.opt_worker = OptimizationWorker(
            target_cl=self.opt_target_cl.value(),
            target_k=self.opt_target_k.value(),
            physics=self.get_physics(),
            solver=self.get_solver(),
            initial_params=initial_params,
            rule_set=self.rule_set,
            flight_points=flight_points,
            ref_data=ref_data,
            body_markers=active_markers,
            cpu_cores=self._resolve_cores_for_level(),
            symmetry_planes=self.get_symmetry_plane_axes(),
        )
        self.opt_worker.mesh_quality = self.combo_mesh_quality.currentText()
        self.opt_worker.log_signal.connect(self.log_text.append)
        self.opt_worker.progress_signal.connect(self._set_progress)
        self.opt_worker.opt_finished.connect(self.optimization_completed)
        self.opt_worker.update_geometry_signal.connect(self._update_geometry_from_opt)
        self.opt_worker.start()

    def _update_geometry_from_opt(self, cand):
        self.w_span.setValue(cand['span'])
        self.w_chord_root.setValue(cand['chord_root'])
        self.w_chord_tip.setValue(cand['chord_tip'])
        self.w_sweep.setValue(cand['sweep'])
        if 'twist' in cand:
            self.w_twist.setValue(cand['twist'])
        if 'flap_deflection' in cand:
            self.flap_deflection.setValue(cand['flap_deflection'])
        if 'flap_slide' in cand:
            self.flap_slide.setValue(cand['flap_slide'])
        if 'flap_hinge_depth' in cand:
            self.flap_hinge_depth.setValue(cand['flap_hinge_depth'])
        if 'slat_deflection' in cand:
            self.slat_deflection.setValue(cand['slat_deflection'])
        if 'slat_slide' in cand:
            self.slat_slide.setValue(cand['slat_slide'])
        self.generate_wing_mesh_parametric(
            cand['span'], cand['chord_root'], cand['chord_tip'], silent=True)
        self._rebuild_mesh_for_optimization()

    def _rebuild_mesh_for_optimization(self):
        visible = [b for b in self.bodies if b.get("visible", True)]
        stl_paths = []
        for b in visible:
            try:
                path = os.path.join(WORK_DIR_BASE, f"_opt_body_{b['id']}_{b['role']}.stl")
                b["mesh"].save(path)
                stl_paths.append(path)
            except Exception:
                pass
        if not stl_paths:
            self.opt_worker.geometry_ready()
            return
        self._meshing = True
        self._mesh_worker = MeshWorker(stl_paths,
                                       self.combo_mesh_quality.currentText(),
                                       parent=self,
                                       symmetry_planes=self.get_symmetry_plane_axes())
        self._mesh_worker.progress_signal.connect(
            lambda p, s: self.statusBar().showMessage(f"Опт. сетка: {s} ({p}%)", 2000))
        self._mesh_worker.finished_signal.connect(
            lambda ok, msg: self._on_opt_mesh_done(ok, msg, stl_paths))
        self._mesh_worker.start()

    def _on_opt_mesh_done(self, ok, msg, stl_paths):
        self._meshing = False
        for p in stl_paths or []:
            try:
                os.remove(p)
            except Exception:
                pass
        if self._mesh_worker:
            self._mesh_worker.deleteLater()
            self._mesh_worker = None
        if not ok:
            self.log_text.append(f"Внимание: Опт. сетка не построена: {msg}")
        if self.opt_worker:
            self.opt_worker.geometry_ready()

    def optimization_completed(self, best):
        self.btn_start_opt.setEnabled(True)
        self._opt_running = False
        self.progress.setVisible(False)
        if best and 'cl_weighted' in best:
            cl = best['cl_weighted']
            k = best['k_weighted']
            self.lbl_opt_status.setText(f"Готово: Cl={cl:.3f}, K={k:.1f}")
            self.log_text.append(f"Оптимизация завершена: Cl={cl:.3f}, K={k:.1f}")
            target_cl = self.opt_target_cl.value()
            target_k = self.opt_target_k.value()
            unmet_recs = []
            if cl < target_cl:
                unmet_recs.append(
                    f"• Подъемная сила (Cl = {cl:.3f} при цели {target_cl:.2f}) не достигнута.\n"
                    "  Рекомендации:\n"
                    "     - Увеличьте хорду крыла или размах для роста площади.\n"
                    "     - Выберите профиль NACA с большей кривизной (напр. 4412).\n"
                    "     - Увеличьте угол атаки или крутку крыла.")
            if k < target_k:
                unmet_recs.append(
                    f"• Качество (K = {k:.1f} при цели {target_k:.1f}) не достигнуто.\n"
                    "  Рекомендации:\n"
                    "     - Увеличьте удлинение крыла (снизится индуктивное сопротивление).\n"
                    "     - Уменьшите толщину профиля NACA.\n"
                    "     - Снизьте стреловидность.")
            if unmet_recs:
                msg = ("Внимание: Оптимизация завершена, но цели не достигнуты полностью "
                       "в заданных границах.\n\n" + "\n\n".join(unmet_recs) +
                       "\n\nСовет: ослабьте ограничения в Rule Set или расширьте бокс.")
                QMessageBox.warning(self, "Цели оптимизации не достигнуты", msg)
            else:
                QMessageBox.information(
                    self, "Успех!",
                    f"Целевые параметры достигнуты!\n\nCl = {cl:.3f}, K = {k:.1f}")
        else:
            self.lbl_opt_status.setText("Ошибка: Решение не найдено")
            QMessageBox.critical(self, "Ошибка",
                                 "Не удалось найти допустимый вариант. "
                                 "Проверьте сетку и граничные условия.")


    # =============================================================
    # DOE: табличный перебор вариантов (ТЗ — параметрическая
    # оптимизация по таблице параметров)
    # =============================================================
    def _doe_param_names(self):
        """Параметры таблицы перебора (ТЗ п.5: расширенный набор)."""
        return ["span", "chord_root", "chord_tip", "sweep", "twist",
                "flap_deflection", "slat_deflection"]

    def _doe_param_labels(self):
        from optimization.doe import SPEC_BY_KEY
        return [SPEC_BY_KEY.get(k, (k,))[0] for k in self._doe_param_names()]

    def _doe_current_values(self):
        return {
            "span": self.w_span.value(),
            "chord_root": self.w_chord_root.value(),
            "chord_tip": self.w_chord_tip.value(),
            "sweep": self.w_sweep.value(),
            "twist": self.w_twist.value(),
            "flap_deflection": self.flap_deflection.value(),
            "slat_deflection": self.slat_deflection.value(),
        }

    def _fill_doe_table(self, rows):
        """Заполняет таблицу перебора списком словарей параметров."""
        names = self._doe_param_names()
        defaults = self._doe_current_values()
        self.doe_table.setRowCount(0)
        for vals in rows:
            row = self.doe_table.rowCount()
            self.doe_table.insertRow(row)
            for col, key in enumerate(names):
                v = vals.get(key, defaults.get(key, 0.0))
                item = QTableWidgetItem(f"{float(v):.3f}")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.doe_table.setItem(row, col, item)
        self.lbl_doe_status.setText(f"Вариантов в таблице: {len(rows)}")

    def show_doe_grid_dialog(self):
        """Диалог генерации сетки вариантов (DOE)."""
        from optimization.doe import (PLAN_FULL, PLAN_LHS, PLAN_OFAT, PLANS,
                                      SPEC_BY_KEY, make_plan, plan_size)
        base = self._doe_current_values()
        names = [k for k in self._doe_param_names() if k in SPEC_BY_KEY]

        dialog = QDialog(self)
        dialog.setWindowTitle("Генерация сетки вариантов (DOE)")
        dialog.setMinimumWidth(520)
        form = QFormLayout(dialog)
        combo_plan = QComboBox()
        combo_plan.addItems(list(PLANS))
        form.addRow("План:", combo_plan)
        spin_levels = QSpinBox()
        spin_levels.setRange(2, 7)
        spin_levels.setValue(3)
        form.addRow("Уровней на параметр:", spin_levels)
        spin_samples = QSpinBox()
        spin_samples.setRange(2, 200)
        spin_samples.setValue(9)
        form.addRow("Вариантов (ЛГК):", spin_samples)

        rows_ui = {}
        for k in names:
            label, lo, hi, _nd = SPEC_BY_KEY[k]
            chk = QCheckBox("варьировать")
            chk.setChecked(k in ("span", "chord_root", "sweep"))
            sp_lo = QDoubleSpinBox()
            sp_lo.setRange(lo, hi)
            sp_lo.setDecimals(3)
            sp_lo.setValue(float(base.get(k, lo)))
            sp_hi = QDoubleSpinBox()
            sp_hi.setRange(lo, hi)
            sp_hi.setDecimals(3)
            sp_hi.setValue(float(base.get(k, hi)))
            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(chk)
            hl.addWidget(sp_lo)
            hl.addWidget(QLabel("…"))
            hl.addWidget(sp_hi)
            rows_ui[k] = (chk, sp_lo, sp_hi)
            form.addRow(label + ":", holder)

        lbl_size = QLabel("—")

        def _ranges():
            out = {}
            for k, (chk, sp_lo, sp_hi) in rows_ui.items():
                if chk.isChecked():
                    a, b = sp_lo.value(), sp_hi.value()
                    out[k] = (min(a, b), max(a, b))
            return out

        def _refresh_size(*_a):
            r = _ranges()
            n = plan_size(combo_plan.currentText(), len(r),
                          spin_levels.value(), spin_samples.value())
            gens = int(self.doe_generations.value())
            lbl_size.setText(f"Расчётов в поколении: {n}"
                             + (f"; всего при {gens} пок.: ≈ {n * gens}"
                                if gens > 1 else ""))

        for k, (chk, sp_lo, sp_hi) in rows_ui.items():
            chk.stateChanged.connect(_refresh_size)
            sp_lo.valueChanged.connect(_refresh_size)
            sp_hi.valueChanged.connect(_refresh_size)
        combo_plan.currentIndexChanged.connect(_refresh_size)
        spin_levels.valueChanged.connect(_refresh_size)
        spin_samples.valueChanged.connect(_refresh_size)
        _refresh_size()
        form.addRow("Размер плана:", lbl_size)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        form.addRow(btns)
        if dialog.exec_() != QDialog.Accepted:
            return
        r = _ranges()
        if not r:
            self.lbl_doe_status.setText("Не выбран ни один параметр для "
                                        "варьирования")
            return
        try:
            rows = make_plan(combo_plan.currentText(), base, r,
                             n_levels=spin_levels.value(),
                             n_samples=spin_samples.value())
        except Exception as e:
            self.lbl_doe_status.setText(f"Не удалось построить план: {e}")
            return
        self._fill_doe_table(rows)
        self._doe_ranges = r
        self._doe_plan = combo_plan.currentText()
        self._doe_levels = int(spin_levels.value())
        self._doe_samples = int(spin_samples.value())
        self.log_text.append(f"Сетка DOE: план «{self._doe_plan}», "
                             f"вариантов {len(rows)}")

    def add_doe_row_from_current(self):
        v = self._doe_current_values()
        row = self.doe_table.rowCount()
        self.doe_table.insertRow(row)
        for col, key in enumerate(self._doe_param_names()):
            item = QTableWidgetItem(f"{v[key]:.3f}")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.doe_table.setItem(row, col, item)

    def add_doe_row_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Вариант перебора (DOE)")
        form = QFormLayout(dialog)
        v = self._doe_current_values()
        spins = {}
        from optimization.doe import SPEC_BY_KEY
        cfg = [(k, SPEC_BY_KEY[k][0] + ":", SPEC_BY_KEY[k][1],
                SPEC_BY_KEY[k][2], v.get(k, 0.0))
               for k in self._doe_param_names() if k in SPEC_BY_KEY]
        for key, label, lo, hi, val in cfg:
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(3)
            sp.setValue(float(val))
            form.addRow(label, sp)
            spins[key] = sp
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        form.addRow(btns)
        if dialog.exec_() == QDialog.Accepted:
            row = self.doe_table.rowCount()
            self.doe_table.insertRow(row)
            for col, key in enumerate(self._doe_param_names()):
                item = QTableWidgetItem(f"{spins[key].value():.3f}")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.doe_table.setItem(row, col, item)

    def remove_doe_rows(self):
        rows = sorted({i.row() for i in self.doe_table.selectedIndexes()},
                      reverse=True)
        for r in rows:
            self.doe_table.removeRow(r)

    def clear_doe_table(self):
        self.doe_table.setRowCount(0)
        self.doe_results.setRowCount(0)
        self.lbl_doe_status.setText("—")

    def _get_doe_candidates(self):
        cands = []
        defaults = self._doe_current_values()
        for row in range(self.doe_table.rowCount()):
            vals = {}
            ok = True
            for col, key in enumerate(self._doe_param_names()):
                it = self.doe_table.item(row, col)
                try:
                    vals[key] = float(it.text()) if it else defaults[key]
                except (ValueError, TypeError):
                    ok = False
                    break
            if ok:
                cands.append(vals)
        return cands

    def run_doe_optimization(self):
        """Старт перебора: инициализирует поколения и запускает первое."""
        self._doe_gen_index = 1
        self._doe_gen_total = max(1, int(self.doe_generations.value()))
        self._doe_best_overall = None
        self._doe_gen_results = []
        self._launch_doe_generation(interactive=True)

    def _launch_doe_generation(self, interactive: bool = False):
        """Одно поколение перебора (ТЗ п.5: несколько поколений)."""
        # === лицензия (та же проверка, что и у обычной оптимизации) ===
        if self._license is not None:
            allowed, reason = self._license.is_calculation_allowed()
            if not allowed:
                QMessageBox.warning(self, "Лицензия", reason)
                self.log_text.append(f"{reason}")
                return
        cands = self._get_doe_candidates()
        if not cands:
            if interactive:
                QMessageBox.warning(self, "Перебор вариантов",
                                    "Таблица вариантов пуста или содержит "
                                    "некорректные значения. Добавьте строки "
                                    "кнопкой «Из текущих параметров» или "
                                    "сгенерируйте сетку «Сетка вариантов».")
            else:
                self.log_text.append("Внимание: Перебор остановлен: нет вариантов")
                self._finish_doe()
            return
        if not self.bodies:
            QMessageBox.warning(self, "Ошибка", "Загрузите геометрию.")
            return
        if not self.validate_rules_before_run():
            return
        ref_data = self.calculate_reference_data()
        flight_points = self._get_opt_points()
        self.btn_start_doe.setEnabled(False)
        self.btn_start_opt.setEnabled(False)
        if getattr(self, "_doe_gen_index", 1) <= 1:
            self.doe_results.setRowCount(0)
        gen = getattr(self, "_doe_gen_index", 1)
        total = getattr(self, "_doe_gen_total", 1)
        prefix = (f"Поколение {gen}/{total}: " if total > 1 else "")
        self.lbl_doe_status.setText(f"{prefix}перебор {len(cands)} "
                                    f"вариантов...")
        self.log_text.append(f"{prefix}запуск {len(cands)} вариантов")
        self.progress.setVisible(True)
        self.progress.setValue(0)
        active_markers = [b["role"] for b in self.bodies if b.get("visible", True)]
        self.opt_worker = OptimizationWorker(
            target_cl=self.opt_target_cl.value(),
            target_k=self.opt_target_k.value(),
            physics=self.get_physics(),
            solver=self.get_solver(),
            initial_params=self._doe_current_values(),
            rule_set=self.rule_set,
            flight_points=flight_points,
            ref_data=ref_data,
            body_markers=active_markers,
            candidates=cands,
            cpu_cores=self._resolve_cores_for_level(),
            symmetry_planes=self.get_symmetry_plane_axes(),
        )
        self.opt_worker.log_signal.connect(self.log_text.append)
        self.opt_worker.progress_signal.connect(self._set_progress)
        self.opt_worker.opt_finished.connect(self.on_doe_finished)
        self.opt_worker.update_geometry_signal.connect(self._update_geometry_from_opt)
        self.opt_worker.variant_ready.connect(self._on_doe_variant_ready)
        self.opt_worker.start()

    def _on_doe_variant_ready(self, info):
        try:
            row = self.doe_results.rowCount()
            self.doe_results.insertRow(row)
            idx = int(info.get("index", row)) + 1
            self.doe_results.setItem(row, 0, QTableWidgetItem(f"#{idx}"))
            self.doe_results.setItem(
                row, 1, QTableWidgetItem(f"{float(info.get('cl_weighted', 0.0)):.4f}"))
            self.doe_results.setItem(
                row, 2, QTableWidgetItem(f"{float(info.get('k_weighted', 0.0)):.2f}"))
            item = QTableWidgetItem("OK" if info.get("ok") else "отклонено")
            if info.get("ok"):
                item.setBackground(QColor(200, 255, 200))
            else:
                item.setBackground(QColor(255, 220, 220))
                item.setToolTip(str(info.get("rejected_reason", "")))
            self.doe_results.setItem(row, 3, item)
        except Exception:
            pass

    def on_doe_finished(self, best):
        self.progress.setVisible(False)
        gen = getattr(self, "_doe_gen_index", 1)
        total = getattr(self, "_doe_gen_total", 1)
        if best and best.get("k_weighted"):
            prev = getattr(self, "_doe_best_overall", None)
            if prev is None or float(best["k_weighted"]) > float(
                    prev.get("k_weighted", 0.0)):
                self._doe_best_overall = best
            self._doe_gen_results.append(
                {"gen": gen, "k": float(best["k_weighted"]),
                 "cl": float(best.get("cl_weighted", 0.0)),
                 "params": {k: best.get(k) for k in self._doe_param_names()}})
        # есть ещё поколения и есть из чего стартовать — сужаем диапазон
        if gen < total and getattr(self, "_doe_best_overall", None):
            try:
                from optimization.doe import make_plan, next_generation
                ranges = next_generation(
                    self._doe_best_overall,
                    getattr(self, "_doe_ranges", None) or {},
                    shrink=float(self.doe_shrink.value()))
                rows = make_plan(getattr(self, "_doe_plan", None)
                                 or "Полный факторный",
                                 self._doe_best_overall, ranges,
                                 n_levels=int(getattr(self, "_doe_levels", 3)),
                                 n_samples=int(getattr(self, "_doe_samples", 9)))
            except Exception as e:
                self.log_text.append(f"Внимание: Не удалось построить следующее "
                                     f"поколение: {e}")
                rows = []
            if rows:
                self._doe_ranges = ranges
                self._fill_doe_table(rows)
                self._doe_gen_index = gen + 1
                self.log_text.append(
                    f"Поколение {gen + 1}: диапазоны сужены до "
                    + ", ".join(f"{k} {v[0]:g}…{v[1]:g}"
                                for k, v in ranges.items()))
                self._launch_doe_generation(interactive=False)
                return
        self._finish_doe()

    def _finish_doe(self):
        """Завершение перебора: лучший вариант по всем поколениям."""
        self.btn_start_doe.setEnabled(True)
        self.btn_start_opt.setEnabled(True)
        self.progress.setVisible(False)
        best = getattr(self, "_doe_best_overall", None)
        gens = getattr(self, "_doe_gen_results", [])
        if len(gens) > 1:
            self.log_text.append(
                "Поколения: " + " → ".join(f"{g['k']:.1f}" for g in gens))
        if best and best.get("k_weighted"):
            text = (f"Готово: Лучший вариант: span={best.get('span', 0):.2f} м, "
                    f"cr={best.get('chord_root', 0):.2f} м, "
                    f"ct={best.get('chord_tip', 0):.2f} м, "
                    f"sweep={best.get('sweep', 0):.1f}° → "
                    f"Cl={best.get('cl_weighted', 0):.3f}, "
                    f"K={best.get('k_weighted', 0):.1f}")
            self.lbl_doe_status.setText(text)
            self.log_text.append(f"{text}")
            try:
                self._update_geometry_from_opt(best)
            except Exception as e:
                self.log_text.append(f"Внимание: Не удалось применить лучший вариант: {e}")
        else:
            self.lbl_doe_status.setText("Ошибка: Допустимых вариантов не найдено")
        if self.opt_worker:
            self.opt_worker.deleteLater()
            self.opt_worker = None

    def _on_cpu_slider_changed(self, value):
        """При изменении слайдера — если анализ идёт, ставим в очередь."""
        cores = max(1, value)
        self.spin_cpu_cores.blockSignals(True)
        self.spin_cpu_cores.setValue(cores)
        self.spin_cpu_cores.blockSignals(False)
        # Расчёт уже запущен — новое число ядер применится на следующем этапе
        # (SessionRunner читает session.cpu_cores при подготовке каждого кейса).
        runner = getattr(self, "session_runner", None)
        running = runner is not None and getattr(runner, "isRunning", lambda: False)()
        if running:
            self.log_text.append(
                f"Изменение числа ядер на {cores} применится при следующем этапе.")
        else:
            self.log_text.append(f"Готово: Применено ядер CPU: {cores}")
        self._refresh_load_status_label()



    # =============================================================
    # ЭКСПОРТ / ВИЗУАЛИЗАЦИЯ
    # =============================================================
    def save_polar_csv(self):
        if not self.all_results:
            return
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(
            RESULTS_DIR, f"polar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=["AoA", "Cl", "Cd", "Cm", "L/D"])
            w.writeheader()
            for r in self.all_results:
                if not r.get('error', True):
                    cl, cd = r.get('cl', 0), r.get('cd', 0)
                    w.writerow({"AoA": r.get('aoa', 0), "Cl": cl, "Cd": cd,
                                "Cm": r.get('cm', 0),
                                "L/D": cl / cd if cd > 0.001 else 0})
        QMessageBox.information(self, "Готово", f"Сохранено: {path}")

    def show_flow_field(self):
        if not os.path.exists(WORK_DIR_BASE):
            self.log_text.append("Внимание: Каталог расчётов не существует. Сначала постройте сетку и запустите расчёт.")
            return
        cases = [d for d in os.listdir(WORK_DIR_BASE)
                 if os.path.isdir(os.path.join(WORK_DIR_BASE, d))]
        if not cases:
            self.log_text.append("Внимание: Нет ни одной папки расчёта. Запустите расчёт хотя бы раз.")
            return
        latest = max(cases, key=lambda d: os.path.getmtime(os.path.join(WORK_DIR_BASE, d)))
        latest_top = os.path.join(WORK_DIR_BASE, latest)

        # Структура бывает двух видов:
        #   1. WORK_DIR_BASE/case_<id>/<files>            (старый код)
        #   2. WORK_DIR_BASE/case_<id>/aoa_<dir>/<files>  (текущий код)
        # Ищем поверхность в обоих вариантах, с приоритетом у вложенного.
        candidate_dirs = []
        subdirs = [os.path.join(latest_top, sd)
                   for sd in os.listdir(latest_top)
                   if os.path.isdir(os.path.join(latest_top, sd))]
        if subdirs:
            # Берём самый свежий aoa_*
            latest_subdir = max(
                subdirs,
                key=lambda d: os.path.getmtime(d),
            )
            candidate_dirs.append(latest_subdir)
        candidate_dirs.append(latest_top)

        surface_file = None
        for d in candidate_dirs:
            for fname in ("surface_flow.vtu", "surface_flow.vtu.vtu",
                          "surface_flow.vtk", "surface.vtu", "surface.vtk"):
                fp = os.path.join(d, fname)
                if os.path.exists(fp):
                    surface_file = fp
                    self.latest_case_dir = d
                    break
            if surface_file:
                break

        if not surface_file:
            # Подробный отчёт: что реально лежит в обеих папках
            self.log_text.append("Внимание: Файл поверхности не найден")
            self.log_text.append(f"   Корневая папка расчёта: {latest_top}")
            for d in candidate_dirs:
                try:
                    on_disk = sorted(os.listdir(d))
                except Exception as e:
                    on_disk = [f"<не удалось прочитать: {e}>"]
                self.log_text.append(f"   {os.path.relpath(d, WORK_DIR_BASE) or '.'}/")
                if on_disk:
                    self.log_text.append(f"      {len(on_disk)} файлов:")
                    for f in on_disk[:20]:
                        self.log_text.append(f"        • {f}")
                    if len(on_disk) > 20:
                        self.log_text.append(f"        … и ещё {len(on_disk) - 20}")
                else:
                    self.log_text.append("      (папка пуста)")
            self.log_text.append(
                "   Возможные причины:\n"
                "     1. В mesh.su2 маркер 'airfoil' пустой (MARKER_ELEMS= 0) —\n"
                "        самолёт не вырезан из сетки.\n"
                "     2. SU2 упал с ошибкой раньше записи .vtu — откройте\n"
                "        su2_stdout.log / console.log в папке расчёта.\n"
                "     3. config.cfg не содержит VOLUME/SURFACE_FILENAME\n"
                "        (должен — мы это добавили в обоих шаблонах)."
            )
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            mesh = pv.read(surface_file)
            self.current_surface_mesh = mesh
            self.current_volume_mesh = None
            self._flow_scene_ready = False
            self.combo_scalar.blockSignals(True)
            self.combo_scalar.clear()
            if mesh.array_names:
                self.combo_scalar.addItems(mesh.array_names)
                self.combo_scalar.setEnabled(True)
                for c in ["Pressure_Coefficient", "Pressure", "Mach", "Velocity"]:
                    if c in mesh.array_names:
                        self.combo_scalar.setCurrentText(c)
                        break
            self.combo_scalar.blockSignals(False)
            self.render_flow_scene()
        except Exception as e:
            self.log_text.append(f"Внимание: Ошибка чтения поверхности: {e}")
        QApplication.restoreOverrideCursor()

    def render_flow_scene(self, *args):
        if not self.current_surface_mesh:
            return
        # Переключение карты поля перерисовывает сцену целиком, а clear()
        # убирает все актёры — вместе с этим уезжает и вид. Пользователь
        # крутит модель один раз и ожидает, что смена отображаемой величины
        # не сдвинет её с места, поэтому камеру запоминаем и возвращаем.
        saved_camera = None
        if self._flow_scene_ready:
            try:
                saved_camera = self.plotter.camera.copy()
            except Exception:
                saved_camera = None
        self.plotter.clear()
        self.plotter.add_axes()
        for b in self.bodies:
            if b.get("visible", True):
                self.plotter.add_mesh(b["mesh"], color=b.get("color", (0.5, 0.5, 0.5)),
                                      opacity=0.7, show_edges=True)
        if self.chk_show_volume.isChecked() and self.latest_case_dir:
            if not self.current_volume_mesh:
                # SU2 7.x: при VOLUME_FILENAME= flow пишет flow.vtu;
                # при VOLUME_FILENAME= flow.vtu пишет flow.vtu.vtu.
                # Поддерживаем оба варианта для совместимости.
                for vf in ("vol_solution.vtu", "flow.vtu", "flow.vtu.vtu"):
                    vp = os.path.join(self.latest_case_dir, vf)
                    if os.path.exists(vp):
                        self.current_volume_mesh = pv.read(vp)
                        break
            if self.current_volume_mesh:
                self.plotter.add_mesh(self.current_volume_mesh, color="lightgray",
                                      opacity=0.05, style="wireframe")
        surface = self.current_surface_mesh.copy()
        field = self.combo_scalar.currentText()
        if field and field in surface.array_names:
            if field in surface.point_data:
                arr = np.asarray(surface.point_data[field])
            elif field in surface.cell_data:
                arr = np.asarray(surface.cell_data[field])
            else:
                arr = np.array([])
            n_points, n_cells = surface.n_points, surface.n_cells
            valid_data = False
            if arr.size > 0:
                if arr.ndim == 2 and arr.shape[1] == 3:
                    mag = np.linalg.norm(arr, axis=1)
                    scalar_name = f"{field}_mag"
                    if field in surface.point_data:
                        surface.point_data[scalar_name] = mag
                    else:
                        surface.cell_data[scalar_name] = mag
                    valid_data = True
                elif arr.ndim == 1 and (len(arr) == n_points or len(arr) == n_cells):
                    scalar_name = field
                    valid_data = True
                else:
                    self.log_text.append(f"Внимание: Поле {field}: неправильный размер {arr.shape}")
            if valid_data:
                self.plotter.add_mesh(surface, scalars=scalar_name, cmap="jet",
                                      show_scalar_bar=True)
            else:
                self.plotter.add_mesh(surface, color="lightblue", show_edges=True)
                self.log_text.append(f"Внимание: Не удалось отобразить поле {field}")
        else:
            self.plotter.add_mesh(surface, color="lightblue", show_edges=True)
        self.update_flow_arrow()
        if saved_camera is not None:
            try:
                self.plotter.camera = saved_camera
            except Exception:
                pass
        self._flow_scene_ready = True
        self.plotter.render()

    def update_flow_arrow(self):
        if self.flow_arrow_actor:
            try:
                self.plotter.remove_actor(self.flow_arrow_actor)
            except Exception:
                pass
            self.flow_arrow_actor = None
        bounds = [b["mesh"].bounds for b in self.bodies if b.get("mesh") is not None]
        if not bounds:
            return
        size = max(max(b[1] for b in bounds) - min(b[0] for b in bounds),
                   max(b[3] for b in bounds) - min(b[2] for b in bounds),
                   max(b[5] for b in bounds) - min(b[4] for b in bounds))
        if size < 1e-6:
            return
        cx = (min(b[0] for b in bounds) + max(b[1] for b in bounds)) / 2
        cy = (min(b[2] for b in bounds) + max(b[3] for b in bounds)) / 2
        cz = (min(b[4] for b in bounds) + max(b[5] for b in bounds)) / 2
        aoa_rad = math.radians(self.input_aoa.value())
        dir_vec = [math.cos(aoa_rad), 0.0, math.sin(aoa_rad)]
        arrow = pv.Arrow(
            start=[cx - dir_vec[0] * size * 2.5, cy, cz - dir_vec[2] * size * 2.5],
            direction=dir_vec, scale=size * 1.5)
        self.flow_arrow_actor = self.plotter.add_mesh(arrow, color="red")
        self.plotter.render()

    # =============================================================
    # ФИЗИКА / СОЛВЕР
    # =============================================================
    def update_isa(self):
        T, P, rho, a = isa_atmosphere(self.input_alt.value())
        self.lbl_isa.setText(
            f"T={T:.1f}K, P={P:.0f}Pa, ρ={rho:.3f}кг/м³, "
            f"M={self.input_speed.value() / a:.3f}")

    def get_physics(self):
        # Источник истины — self.flight; синхронизируем с полями UI,
        # чтобы get_physics() работал даже если пользователь не нажал «Применить».
        # (поведение совпадает с прежним — раньше поля напрямую читались)
        self.flight.speed_m_s = float(self.input_speed.value())
        self.flight.altitude_m = float(self.input_alt.value())
        self.flight.aoa_deg = float(self.input_aoa.value())
        return self.flight.to_physics_dict()

    def get_solver(self) -> str:
        """Возвращает тип решателя для config.cfg.

        T4: добавлен RANS SST (idx=2). Для совместимости со старыми
        вызовами возвращаем "RANS" для обоих вязких режимов — выбор
        SA/SST пробрасывается отдельно через session.turb_model.
        """
        idx = self.combo_solver.currentIndex()
        if idx == 1:
            return "RANS"
        if idx == 2:
            return "RANS"
        return "EULER"

    def get_turb_model(self) -> str:
        """Возвращает 'SA' или 'SST' (T4). По умолчанию 'SA'."""
        try:
            idx = int(self.combo_solver.currentIndex())
        except Exception:
            return "SA"
        if idx == 2:
            return "SST"
        return "SA"

    def update_mode_ui(self):
        single = self.rb_single.isChecked()
        self.input_aoa.setEnabled(single)
        self.input_aoa_start.setEnabled(not single)
        self.input_aoa_end.setEnabled(not single)
        self.input_aoa_step.setEnabled(not single)

    def update_wing_aero_params(self):
        span = self.w_span.value()
        cr = self.w_chord_root.value()
        ct = self.w_chord_tip.value()
        if span > 0 and cr > 0 and ct > 0:
            S = 0.5 * (cr + ct) * span
            self.log_text.append(
                f"Крыло: S={S:.2f}м², λ={span ** 2 / S:.2f}, η={cr / ct:.2f}")

    # =============================================================
    # ПРАВИЛА
    # =============================================================
    def validate_rules_before_run(self):
        if not self.rule_set.rules:
            return True
        conflicts = self.rule_set.check_consistency()
        if conflicts:
            msg = "\n".join(conflicts)
            QMessageBox.critical(self, "Конфликт правил",
                                 f"Набор правил противоречив:\n\n{msg}")
            self.log_text.append("Ошибка: Расчёт остановлен: конфликт правил.")
            return False
        params = self._collect_current_params()
        result = self.rule_set.check_all(params)
        self.log_text.append("\nПроверка правил:")
        for msg in result["messages"]:
            self.log_text.append(msg)
        if not result["passed"]:
            details = "\n".join(f"• {v['rule']} — {v['violation']:.3f}"
                                for v in result["hard_violations"])
            QMessageBox.critical(self, "Нарушены жёсткие правила",
                                 f"Расчёт остановлен.\n\n{details}")
            return False
        if result["soft_violations"]:
            self.log_text.append(
                f"Внимание: Мягких нарушений: {len(result['soft_violations'])}, "
                f"штраф={result['penalty']:.3f}")
        return True

    def load_rule_preset(self):
        name = self.combo_preset.currentText()
        if name in PRESETS:
            self.rule_set = PRESETS[name]()
            self.update_rules_table()
            self.log_text.append(f"Готово: Загружен пресет: {name} "
                                 f"({len(self.rule_set.rules)} правил)")

    def update_rules_table(self):
        self.rules_table.setRowCount(0)
        for rule in self.rule_set.rules:
            row = self.rules_table.rowCount()
            self.rules_table.insertRow(row)
            self.rules_table.setItem(row, 0, QTableWidgetItem(rule.name))
            self.rules_table.setItem(row, 1, QTableWidgetItem(rule.parameter))
            self.rules_table.setItem(row, 2, QTableWidgetItem(rule.operator.value))
            val_str = (f"[{rule.value[0]}, {rule.value[1]}]"
                       if isinstance(rule.value, list) else str(rule.value))
            self.rules_table.setItem(row, 3, QTableWidgetItem(val_str))
            self.rules_table.setItem(row, 4, QTableWidgetItem(rule.severity.value))
            chk = QCheckBox()
            chk.setChecked(rule.enabled)
            chk.stateChanged.connect(
                lambda state, rn=rule.name: self.toggle_rule(rn, state == Qt.Checked))
            self.rules_table.setCellWidget(row, 5, chk)

    def toggle_rule(self, name, enabled):
        for rule in self.rule_set.rules:
            if rule.name == name:
                rule.enabled = enabled
                break

    def add_rule_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Новое правило")
        dialog.setMinimumWidth(400)
        form = QFormLayout(dialog)
        name_edit = QLineEdit("Новое правило")
        param_combo = QComboBox()
        param_combo.addItems([
            "span", "chord_root", "chord_tip", "sweep",
            "aspect_ratio", "taper_ratio", "area",
            "cl", "cd", "cm", "k", "mach", "reynolds",
        ])
        param_combo.setEditable(True)
        op_combo = QComboBox()
        for op in RuleOperator:
            op_combo.addItem(op.value, op)
        val_edit = QLineEdit("10.0")
        val_edit.setToolTip("Для BETWEEN: '5.0, 15.0'")
        sev_combo = QComboBox()
        for s in RuleSeverity:
            sev_combo.addItem(s.value, s)
        weight_spin = QDoubleSpinBox(); weight_spin.setRange(0.1, 10.0); weight_spin.setValue(1.0)
        desc_edit = QLineEdit("")
        form.addRow("Имя:", name_edit)
        form.addRow("Параметр:", param_combo)
        form.addRow("Оператор:", op_combo)
        form.addRow("Значение:", val_edit)
        form.addRow("Жёсткость:", sev_combo)
        form.addRow("Вес:", weight_spin)
        form.addRow("Описание:", desc_edit)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        form.addRow(btns)
        if dialog.exec_() != QDialog.Accepted:
            return
        try:
            op = op_combo.currentData()
            val_str = val_edit.text().strip()
            if op == RuleOperator.BETWEEN:
                parts = [float(x.strip()) for x in val_str.split(",")]
                if len(parts) != 2:
                    raise ValueError("Для BETWEEN нужно два числа")
                value = parts
            else:
                value = float(val_str)
            rule = Rule(name=name_edit.text(), parameter=param_combo.currentText(),
                        operator=op, value=value, severity=sev_combo.currentData(),
                        weight=weight_spin.value(), description=desc_edit.text())
            self.rule_set.add(rule)
            self.update_rules_table()
            self.log_text.append(f"Готово: Добавлено правило: {rule.name}")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось создать правило: {e}")

    def remove_selected_rule(self):
        # Удаляем ВСЕ выделенные строки (Ctrl/Shift поддерживаются в QTableWidget).
        rows = sorted({idx.row() for idx in self.rules_table.selectedIndexes()},
                      reverse=True)
        if not rows:
            return
        names = []
        for r in rows:
            item = self.rules_table.item(r, 0)
            if item is not None:
                names.append(item.text())
        if not names:
            return
        # Без подтверждения — пользователь явно нажал «Удалить» или Delete.
        for r in rows:
            self.rules_table.removeRow(r)
        for rule in self.rule_set.rules:
            if rule.name in names:
                self.rule_set.rules.remove(rule)
        self.log_text.append(
            f"Удалено правил: {len(names)} ({', '.join(names)})"
        )

    def check_rules_consistency(self):
        conflicts = self.rule_set.check_consistency()
        if not conflicts:
            QMessageBox.information(self, "Проверка", "Готово: Конфликтов не найдено.")
            self.log_text.append("Готово: Набор правил непротиворечив.")
        else:
            msg = "\n".join(conflicts)
            QMessageBox.warning(self, "Найдены конфликты!", msg)
            self.log_text.append("Внимание: Конфликты в правилах:")
            for c in conflicts:
                self.log_text.append(f"   {c}")

    def validate_current_design(self):
        params = self._collect_current_params()
        result = self.rule_set.check_all(params)
        self.log_text.append("\n" + "=" * 50)
        self.log_text.append("ПРОВЕРКА ТЕКУЩЕЙ ГЕОМЕТРИИ")
        self.log_text.append("=" * 50)
        for msg in result['messages']:
            self.log_text.append(msg)
        self.log_text.append(f"\nИтог: penalty = {result['penalty']:.3f}")
        if result['passed']:
            QMessageBox.information(
                self, "Проверка",
                f"Готово: Все жёсткие правила соблюдены!\n\n"
                f"Мягких: {len(result['soft_violations'])}\n"
                f"Информ.: {len(result['info_violations'])}\n"
                f"Штраф: {result['penalty']:.3f}")
        else:
            details = "\n".join(f"• {v['rule']} ({v['violation']:.3f})"
                                for v in result['hard_violations'])
            QMessageBox.critical(self, "Нарушения!", f"Ошибка: Жёсткие правила:\n\n{details}")

    def _collect_current_params(self):
        physics = self.get_physics()
        span = self.w_span.value()
        cr = self.w_chord_root.value()
        ct = self.w_chord_tip.value()
        params = {
            'span': span, 'chord_root': cr, 'chord_tip': ct,
            'sweep': self.w_sweep.value(), 'twist': self.w_twist.value(),
            'mach': physics.get('mach', 0), 'reynolds': 0,
            'aspect_ratio': span * 2.0 / max(cr + ct, 1e-6),
            'taper_ratio': ct / max(cr, 1e-6),
            'area': 0.5 * (cr + ct) * span,
        }
        if self.all_results:
            last = self.all_results[-1]
            if not last.get('error', True):
                params['cl'] = last.get('cl', 0)
                params['cd'] = last.get('cd', 0)
                params['cm'] = last.get('cm', 0)
                cd = max(last.get('cd', 0.0), 1e-9)
                params['k'] = last.get('cl', 0.0) / cd
        return params

    def save_rule_set(self):
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить правила",
                                              "rules.json", "JSON (*.json)")
        if path:
            self.rule_set.save(path)
            self.log_text.append(f"Правила сохранены: {path}")

    def load_rule_set(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить правила", "",
                                              "JSON (*.json)")
        if path:
            try:
                self.rule_set = RuleSet.load(path)
                self.update_rules_table()
                self.log_text.append(f"Загружено правил: {len(self.rule_set.rules)}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))

    # =============================================================
    # ТОЧКИ ОПТИМИЗАЦИИ
    # =============================================================
    def add_opt_point(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Новая расчётная точка")
        form = QFormLayout(dialog)
        name_edit = QLineEdit("Взлёт")
        aoa_spin = QDoubleSpinBox(); aoa_spin.setRange(-15, 25); aoa_spin.setValue(8.0)
        weight_spin = QDoubleSpinBox(); weight_spin.setRange(0.1, 10.0); weight_spin.setValue(1.0)
        form.addRow("Имя режима:", name_edit)
        form.addRow("Угол атаки (°):", aoa_spin)
        form.addRow("Вес (важность):", weight_spin)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        form.addRow(btns)
        if dialog.exec_() == QDialog.Accepted:
            row = self.points_table.rowCount()
            self.points_table.insertRow(row)
            self.points_table.setItem(row, 0, QTableWidgetItem(name_edit.text()))
            self.points_table.setItem(row, 1, QTableWidgetItem(f"{aoa_spin.value():.2f}"))
            self.points_table.setItem(row, 2, QTableWidgetItem(f"{weight_spin.value():.2f}"))

    def _get_opt_points(self):
        from optimization.multipoint import FlightPoint
        points = []
        for row in range(self.points_table.rowCount()):
            try:
                name = self.points_table.item(row, 0).text()
                aoa = float(self.points_table.item(row, 1).text())
                wt = self.points_table.item(row, 2).text()
                weight = float(wt) if wt else 1.0
                points.append(FlightPoint(name=name, aoa=aoa, weight=weight))
            except Exception:
                continue
        if not points:
            points.append(FlightPoint(name="Default", aoa=self.input_aoa.value(),
                                      weight=1.0))
        return points

    def load_opt_points_preset(self, preset_name):
        self.points_table.setRowCount(0)
        presets = {
            "cruise": [("Крейсер", 3.0, 1.0)],
            "multi": [("Крейсер", 3.0, 1.0), ("Взлёт", 8.0, 0.7)],
        }
        for name, aoa, weight in presets.get(preset_name, []):
            row = self.points_table.rowCount()
            self.points_table.insertRow(row)
            self.points_table.setItem(row, 0, QTableWidgetItem(name))
            self.points_table.setItem(row, 1, QTableWidgetItem(f"{aoa:.2f}"))
            self.points_table.setItem(row, 2, QTableWidgetItem(f"{weight:.2f}"))

    # =============================================================
    # МЕХАНИЗАЦИЯ / KINK
    # =============================================================
    def _toggle_flap_controls(self, checked):
        self.flap_deflection.setEnabled(checked)
        self.flap_span_ratio.setEnabled(checked)
        self.flap_chord_ratio.setEnabled(checked)
        self.flap_hinge_depth.setEnabled(checked)
        self.flap_slide.setEnabled(checked)

    def _toggle_slat_controls(self, checked):
        self.slat_deflection.setEnabled(checked)
        self.slat_span_ratio.setEnabled(checked)
        self.slat_chord_ratio.setEnabled(checked)
        self.slat_slide.setEnabled(checked)

    def _toggle_kink_controls(self, state):
        checked = (state == Qt.Checked)
        self.w_kink_pos.setEnabled(checked)
        self.w_chord_kink.setEnabled(checked)
        self.w_sweep_outer.setEnabled(checked)

    # =============================================================
    # ОБСЛУЖИВАНИЕ / ИСТОРИЯ / БАЛАНСИРОВКА
    # =============================================================
    def cleanup_old_cases(self, days=7):
        cutoff = time.time() - days * 86400
        if not os.path.exists(WORK_DIR_BASE):
            return
        for d in os.listdir(WORK_DIR_BASE):
            path = os.path.join(WORK_DIR_BASE, d)
            if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                try:
                    shutil.rmtree(path)
                except Exception as e:
                    self.log_text.append(f"Внимание: Не удалось удалить {d}: {e}")

    def update_history_table(self):
        self.history_table.setRowCount(0)
        for rec in self.generation_history:
            row = self.history_table.rowCount()
            self.history_table.insertRow(row)
            self.history_table.setItem(row, 0, QTableWidgetItem(str(rec["id"])))
            self.history_table.setItem(row, 1, QTableWidgetItem(f"{rec['span']:.2f}"))
            self.history_table.setItem(row, 2, QTableWidgetItem(f"{rec['chord_root']:.2f}"))
            self.history_table.setItem(row, 3, QTableWidgetItem(f"{rec['chord_tip']:.2f}"))
            self.history_table.setItem(row, 4, QTableWidgetItem(f"{rec['sweep']:.1f}"))
            self.history_table.setItem(row, 5, QTableWidgetItem(f"{rec['twist']:.1f}"))
            self.history_table.setItem(row, 6, QTableWidgetItem(f"{rec['k']:.2f}"))
            btn_restore = QPushButton("Применить")
            btn_restore.clicked.connect(lambda checked, r=rec: self.restore_history_record(r))
            self.history_table.setCellWidget(row, 7, btn_restore)

    def restore_history_record(self, rec):
        self.w_span.setValue(rec["span"])
        self.w_chord_root.setValue(rec["chord_root"])
        self.w_chord_tip.setValue(rec["chord_tip"])
        self.w_sweep.setValue(rec["sweep"])
        self.w_twist.setValue(rec["twist"])
        self.generate_wing_mesh_parametric(rec["span"], rec["chord_root"],
                                           rec["chord_tip"])
        self.log_text.append(f"Откат к версии #{rec['id']}: размах {rec['span']:.2f}м")

    def clear_generation_history(self):
        self.generation_history.clear()
        self.update_history_table()
        self.log_text.append("История очищена.")

    def calculate_aerodynamic_trim(self):
        if not self.all_results:
            QMessageBox.warning(self, "Нет данных",
                                "Сначала запустите CFD расчёт для получения Cm.")
            return
        last_res = self.all_results[-1]
        if last_res.get("error", True):
            QMessageBox.warning(self, "Ошибка в расчете",
                                "Последний CFD расчёт завершился с ошибкой.")
            return
        cl = last_res.get("cl", 0.0)
        cm = last_res.get("cm", 0.0)
        cm_de = self.trim_eff.value()
        if abs(cm_de) < 1e-6:
            QMessageBox.warning(self, "Ошибка", "Эффективность руля не может быть нулевой.")
            return
        elevator_deflection = -cm / cm_de
        self.lbl_trim_result.setText(
            f"Текущий продольный момент: Cm = {cm:.5f}\n"
            f"Потребный угол руля высоты: δe = {elevator_deflection:.2f}°\n"
            f"(для балансировки Cm_total = 0)")
        self.log_text.append(
            f"Балансировка: для Cm={cm:.5f} требуется δe={elevator_deflection:.2f}°")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            # Удаление выделенных компонентов
            if self.bodies_table.hasFocus():
                self.remove_body()
                event.accept()
                return
            # Удаление выделенных правил (Delete прямо в таблице правил)
            if self.rules_table.hasFocus():
                self.remove_selected_rule()
                event.accept()
                return
        super().keyPressEvent(event)

    def cleanup_session_data(self):
        """Очистка промежуточных расчётов (ТЗ 4.1)."""
        dirs_to_delete = []
        if self.session and self.session.case_dirs:
            for d in self.session.case_dirs:
                if d and os.path.exists(d):
                    dirs_to_delete.append(d)
        if os.path.exists(WORK_DIR_BASE):
            for d in os.listdir(WORK_DIR_BASE):
                path = os.path.join(WORK_DIR_BASE, d)
                if os.path.isdir(path) and d.startswith("OPT_P"):
                    dirs_to_delete.append(path)
        for d in dirs_to_delete:
            try:
                shutil.rmtree(d)
                print(f"Удалена временная расчетная папка: {d}")
            except Exception:
                pass

    def reset_interface(self):
        """Сбрасывает интерфейс к начальному состоянию."""
        reply = QMessageBox.question(
            self, "Сброс интерфейса",
            "Сбросить все настройки к начальным значениям?\n\n"
            "Это НЕ удалит загруженную геометрию и результаты.\n"
            "Сбросятся: полётные условия, правила, параметры крыла,\n"
            "настройки решателя, плоскости симметрии.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # Полётные условия
        self.flight = FlightConditions()
        self.input_speed.setValue(self.flight.speed_m_s)
        self.input_alt.setValue(int(self.flight.altitude_m))
        self.input_aoa.setValue(self.flight.aoa_deg)
        if hasattr(self, 'combo_flight_preset'):
            self.combo_flight_preset.setCurrentIndex(0)
        self.update_isa()

        # Крыло
        self.w_span.setValue(10.0)
        self.w_chord_root.setValue(1.8)
        self.w_chord_tip.setValue(0.9)
        self.w_sweep.setValue(12)
        self.w_twist.setValue(2)
        self.w_naca.setText("2412")
        self.w_pos_x.setValue(3)
        self.w_pos_y.setValue(0)
        self.w_pos_z.setValue(0)
        self.chk_kink.setChecked(False)

        # Фюзеляж
        self.f_length.setValue(8)
        self.f_diameter.setValue(1.2)
        self.f_nose_ratio.setValue(0.25)
        self.f_tail_ratio.setValue(0.3)
        self.f_pos_x.setValue(0)
        self.f_pos_y.setValue(0)
        self.f_pos_z.setValue(0)

        # Механизация
        self.flap_enabled.setChecked(False)
        self.slat_enabled.setChecked(False)

        # Оперение
        self.hs_span.setValue(3.0)
        self.hs_chord.setValue(0.8)
        self.hs_sweep.setValue(15)
        self.hs_pos_x.setValue(6.5)
        self.hs_pos_z.setValue(0.0)
        self.elev_deflection.setValue(0.0)
        self.hs_auto.setChecked(False)
        self.vk_height.setValue(1.5)
        self.vk_chord.setValue(0.7)
        self.vk_sweep.setValue(20)
        self.vk_pos_x.setValue(6.0)
        self.vk_pos_z.setValue(0.0)

        # Решатель
        self.combo_solver.setCurrentIndex(0)
        self.combo_mesh_quality.setCurrentIndex(1)

        # Нагрузка CPU/GPU
        self.slider_cpu_load.setValue(50)
        pass  # слайдера нагрузки GPU в интерфейсе нет
        self.spin_cpu_cores.setValue(0)
        pass  # выбора вычислителя в интерфейсе нет
        self._refresh_load_status_label()

        # RAMP. Галочки симметрии больше нет — состояние симметрии
        # определяется списком плоскостей, который чистится ниже.
        self.chk_use_ramp_aoa.setChecked(False)

        # Плоскости симметрии (3D)
        for p in list(self._symmetry_planes):
            self._remove_symmetry_plane(p["axis"])

        # Правила
        self.rule_set = RuleSet()
        self.update_rules_table()

        # Режим
        self.rb_single.setChecked(True)
        self.input_aoa_start.setValue(-2)
        self.input_aoa_end.setValue(12)
        self.input_aoa_step.setValue(2)

        # Оптимизация
        self.opt_target_cl.setValue(0.45)
        self.opt_target_k.setValue(15)
        if hasattr(self, 'points_table'):
            self.load_opt_points_preset("cruise")

        # Балансировка
        self.trim_arm.setValue(5.0)
        self.trim_eff.setValue(0.015)
        self.lbl_trim_result.setText("Расчет не производился")

        # Дерево → Global Definitions
        self.tree.setCurrentItem(self.item_global_defs)

        self.log_text.append("Интерфейс сброшен к начальным значениям.")
        self.project_saved = False


    def closeEvent(self, event):
        if not self.project_saved:
            reply = QMessageBox.question(
                self, "Несохраненные изменения",
                "У вас есть несохраненные изменения в проекте.\n\n"
                "Сохранить проект перед закрытием?\n"
                "(При выходе без сохранения временные папки расчетов будут удалены)",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                self.save_project()
                if not self.project_saved:
                    event.ignore()
                    return
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
            else:
                self.cleanup_session_data()
        self.log_text.append("Закрытие приложения...")
        if hasattr(self, 'session_runner') and self.session_runner \
                and self.session_runner.isRunning():
            self.session_runner.request_pause()
            self.session_runner.wait(10000)
        if hasattr(self, 'opt_worker') and self.opt_worker \
                and self.opt_worker.isRunning():
            self.opt_worker.stop()
        if hasattr(self, '_mesh_worker') and self._mesh_worker \
                and self._mesh_worker.isRunning():
            self._mesh_worker.cancel()
            self._mesh_worker.wait(5000)
        if hasattr(self, '_adapt_worker') and self._adapt_worker \
                and self._adapt_worker.isRunning():
            self._adapt_worker.wait(5000)
        if hasattr(self, 'plotter'):
            self.plotter.close()
        event.accept()


    # =============================================================
    # АЭРОУПРУГОСТЬ (ТЗ, средний приоритет)
    # =============================================================
    def _aeroelastic_inputs(self) -> dict:
        """Параметры для оценки аэроупругости; при флажке — из модели."""
        w = self.ae_w
        span = w["span"].value()
        chord_root = w["chord_root"].value()
        chord_tip = w["chord_tip"].value()
        if w["fill_from_model"].isChecked():
            try:
                span = float(self.w_span.value())
                chord_root = float(self.w_chord_root.value())
                chord_tip = float(self.w_chord_tip.value())
                w["span"].setValue(span)
                w["chord_root"].setValue(chord_root)
                w["chord_tip"].setValue(chord_tip)
            except Exception:
                pass
        v_dive = w["v_dive"].value()
        return {"span": span, "chord_root": chord_root,
                "chord_tip": chord_tip, "mass_wing": w["mass_wing"].value(),
                "rho": w["rho"].value(), "V_cruise": w["v_cruise"].value(),
                "V_dive": (v_dive if v_dive > 0 else None),
                "t_ratio": w["t_ratio"].value(),
                "x_ea_ratio": w["x_ea_ratio"].value(),
                "x_cg_ratio": w["x_cg_ratio"].value(),
                "safety_factor": w["safety"].value()}

    def _aeroelastic_result(self) -> dict:
        from physics import aeroelastic as AE
        return AE.flutter_assessment(**self._aeroelastic_inputs())

    def run_aeroelastic_check(self):
        """Оценка флатера и дивергенции; результат — в панель и лог."""
        try:
            from physics import aeroelastic as AE
            res = self._aeroelastic_result()
        except Exception as e:
            self.ae_w["out"].setText(f"Внимание: Не удалось выполнить оценку: {e}")
            self.log_text.append(f"Внимание: Аэроупругость: {e}")
            return
        p = res["props"]
        text = AE.format_report(res) + (
            "\n\nИсходные данные:\n"
            f"  размах {p['span']:.2f} м, полу-хорда b = {p['b']:.3f} м, "
            f"e = {p['e']:.3f} м\n"
            f"  погонная масса {p['m']:.2f} кг/м, K_h = {p['K_h']:.4g} Н/м, "
            f"K_alpha = {p['K_alpha']:.4g} Н·м/рад\n"
            f"  x_alpha = {p['x_alpha']:+.3f}, I_alpha = {p['I_alpha']:.4g} "
            f"кг·м²/м\n"
            "\nМетод: типичное сечение (изгиб + кручение), аэродинамика "
            "Теодорсена, p-k метод. Это предварительная оценка для ранней "
            "стадии проектирования, а не замена сертифицированного расчёта "
            "по КЭ-модели.")
        self.ae_w["out"].setText(text)
        self.log_text.append(
            "Аэроупругость: V_F="
            f"{res['V_F'] and round(res['V_F'], 1)} м/с, V_D="
            f"{res['V_D'] and round(res['V_D'], 1)} м/с, запас="
            f"{res['margin'] and round(res['margin'], 2)}")
        self._last_aeroelastic = res

    def plot_vg_diagram(self):
        """V-g диаграмма на вкладке «2D Аэро Графики» (левая ось)."""
        res = getattr(self, "_last_aeroelastic", None)
        if not res:
            try:
                res = self._aeroelastic_result()
            except Exception as e:
                self.ae_w["out"].setText(f"Внимание: {e}")
                return
            self._last_aeroelastic = res
        diag = res.get("vg_diagram") or []
        if not diag:
            self.ae_w["out"].setText("Внимание: Нет данных V-g диаграммы")
            return
        ax = self.plot_canvas.axes1
        ax.clear()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(labelsize=8)
        for i, color in ((0, "#1f77b4"), (1, "#9B2C2C")):
            pts = [(d["modes"][i]["g"], d["V"], d["modes"][i]["freq_hz"])
                   for d in diag if len(d["modes"]) > i]
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                    color=color, lw=1.4, ms=3,
                    label=f"мода {i + 1} (f₀={pts[0][2]:.1f} Гц)")
        ax.axvline(0.0, color="k", lw=0.8, ls="--")
        if res.get("V_F"):
            ax.axvline(float(res["V_F"]), color="r", lw=1.3,
                       label=f"V_F={float(res['V_F']):.1f} м/с")
        ax.set_title("V-g диаграмма (g>0 — нарастание колебаний)", fontsize=9,
                     fontweight="bold", color="#22384A")
        ax.set_xlabel("Структурное демпфирование g", fontsize=8)
        ax.set_ylabel("Скорость, м/с", fontsize=8)
        ax.legend(fontsize=7)
        self.plot_canvas.draw()
        self.bottom_tabs.setCurrentWidget(self.plot_canvas)
        self.log_text.append("V-g диаграмма построена")

    # =============================================================
    # ПРОЧНОСТЬ (ТЗ, низкий приоритет)
    # =============================================================
    def run_structural_check(self):
        """Изгибающий момент и запас прочности корневого сечения."""
        w = self.st_w
        try:
            from physics import structural as ST
            chord = w["chord_root"].value()
            cap_area = (w["cap_frac"].value() * chord
                        * max(w["t_ratio"].value() * chord, 1e-6))
            res = ST.structural_assessment(
                span=w["span"].value(), chord_root=chord,
                mass_aircraft=w["mass_aircraft"].value(),
                mass_wing=w["mass_wing"].value(),
                n_limit=w["n_limit"].value(), dist=w["dist"].currentText(),
                t_ratio=w["t_ratio"].value(), cap_area=cap_area,
                sigma_allow=w["sigma_allow"].value(),
                safety_factor=w["sf"].value())
            text = ST.format_report(res)
        except Exception as e:
            w["out"].setText(f"Внимание: Не удалось выполнить расчёт: {e}")
            return
        w["out"].setText(text)
        self.log_text.append(
            f"Прочность: σ = {res['sigma'] / 1e6:.1f} МПа, "
            f"τ = {res['tau'] / 1e6:.1f} МПа, запас по σ "
            f"{res['MS_sigma']:+.2f}")
        self._last_structural = res

    # =============================================================
    # СПЕЦФУНКЦИИ: ПОЛЯРА И ОТЧЁТЫ (ТЗ)
    # =============================================================
    def _results_rows(self) -> list:
        """Строки таблицы результатов → list[dict] для постобработки."""
        rows = []
        for r in range(self.table.rowCount()):
            def _val(c, row=r):
                it = self.table.item(row, c)
                return it.text().strip() if it else ""
            try:
                rows.append({"aoa": float(_val(0)), "cl": float(_val(1)),
                             "cd": float(_val(2)),
                             "cm": float(_val(3) or 0.0), "converged": True})
            except (TypeError, ValueError):
                continue
        return rows

    def _aspect_ratio_from_ui(self) -> float:
        """Удлинение крыла: размах² / площадь."""
        span = float(self.sp_w["s_ref"].value() and self.ae_w["span"].value())
        s_ref = float(self.sp_w["s_ref"].value())
        if s_ref > 0 and span > 0:
            return span * span / s_ref
        return 10.0

    def _polar_chars(self) -> dict:
        """Интегральные характеристики по текущей таблице результатов."""
        from postprocessing.polar import (build_polar,
                                          integrated_characteristics)
        rows = self._results_rows()
        if len(rows) < 3:
            raise ValueError(
                "в таблице результатов меньше трёх точек — поляру не "
                "построить. Сначала выполните расчёт по нескольким углам "
                "атаки.")
        polar = build_polar(rows)
        chars = integrated_characteristics(
            polar, self._aspect_ratio_from_ui(),
            weight_n=self.sp_w["weight"].value() * 9.80665,
            rho=self.sp_w["rho"].value(), s_ref=self.sp_w["s_ref"].value(),
            mach=self.sp_w["mach"].value())
        self._last_polar_rows = rows
        self._last_polar_chars = chars
        return chars

    @staticmethod
    def _format_polar_chars(ch: dict) -> str:
        def f(key, unit="", nd=4):
            v = ch.get(key)
            return "—" if v is None else f"{float(v):.{nd}f} {unit}".strip()
        return "\n".join([
            "ИНТЕГРАЛЬНЫЕ ХАРАКТЕРИСТИКИ ПО ПОЛЯРЕ",
            "=" * 44,
            f"Точек в поляре        : {ch.get('n_points', 0)}",
            f"Удлинение λ           : {f('aspect_ratio', '', 2)}",
            f"Наклон поляры dCl/dα  : {f('cl_alpha_deg', '1/град')}",
            f"Угол нулевой Cl (α₀)  : {f('alpha0', 'град')}",
            f"Профильное Cd₀        : {f('cd0', '', 5)}",
            f"Коэффициент Освальда e: {f('oswald_e', '', 3)}",
            f"Cl макс                : {f('cl_max')}",
            f"Угол сваливания       : {f('aoa_stall', 'град', 2)}",
            f"K макс                 : {f('k_max', '', 2)}",
            f"Угол при K макс        : {f('aoa_best_k', 'град', 2)}",
            f"Скорость сваливания   : {f('v_stall', 'м/с', 1)}",
            f"Число M               : {f('mach', '', 3)}",
        ])

    def build_polar_from_results(self):
        """Поляра и интегральные характеристики по таблице результатов."""
        try:
            ch = self._polar_chars()
        except Exception as e:
            self.sp_w["out"].setText(f"Внимание: {e}")
            return
        self.sp_w["out"].setText(self._format_polar_chars(ch))
        self.log_text.append(f"Поляра: точек {ch.get('n_points')}, "
                             f"e={ch.get('oswald_e') and round(ch['oswald_e'], 3)}, "
                             f"Cl_max={ch.get('cl_max') and round(ch['cl_max'], 3)}")

    def export_polar_csv(self):
        """Экспорт поляры в CSV (разделитель «;», UTF-8 BOM для Excel)."""
        rows = getattr(self, "_last_polar_rows", None) or self._results_rows()
        if len(rows) < 1:
            self.sp_w["out"].setText("Внимание: Нет данных для экспорта")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить поляру", "polar.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            from postprocessing.report import export_csv
            export_csv(path, rows)
            self.log_text.append(f"Поляра сохранена: {path}")
            self.sp_w["out"].append(f"\nСохранено: {path}")
        except Exception as e:
            self.sp_w["out"].append(f"\nВнимание: Не удалось сохранить: {e}")

    def export_analysis_report(self):
        """Отчёт по шаблону (HTML) + CSV с полярой."""
        rows = self._results_rows()
        if len(rows) < 3:
            self.sp_w["out"].setText(
                "Внимание: В таблице результатов меньше трёх точек — отчёт не "
                "сформировать.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить отчёт", "report.html",
            "HTML (*.html);;Текст (*.txt)")
        if not path:
            return
        try:
            from html import escape as html_escape
            from postprocessing.report import export_csv, render_html
            kw = dict(aspect_ratio=self._aspect_ratio_from_ui(),
                      template=self.sp_w["template"].currentText(),
                      project_info={"name": self.sp_w["project_name"].text()},
                      weight_n=self.sp_w["weight"].value() * 9.80665,
                      rho=self.sp_w["rho"].value(),
                      s_ref=self.sp_w["s_ref"].value(),
                      mach=self.sp_w["mach"].value())
            text = render_html(rows, **kw)
            ae = getattr(self, "_last_aeroelastic", None)
            st = getattr(self, "_last_structural", None)
            extra = ""
            if ae:
                from physics import aeroelastic as AE
                extra += "<pre>" + html_escape(AE.format_report(ae)) + "</pre>"
            if st:
                from physics import structural as ST
                extra += "<pre>" + html_escape(ST.format_report(st)) + "</pre>"
            if extra:
                text = text.replace("</body>", extra + "</body>")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            csv_path = os.path.splitext(path)[0] + "_polar.csv"
            export_csv(csv_path, rows)
            self.log_text.append(f"Отчёт сохранён: {path}")
            self.sp_w["out"].append(f"\nОтчёт: {path}\nПоляра: {csv_path}")
        except Exception as e:
            self.sp_w["out"].append(f"\nВнимание: Не удалось сформировать отчёт: {e}")

    # =============================================================
    # ФОРМАТ КОНФИГУРАЦИИ: ПРЕСЕТЫ (ТЗ)
    # =============================================================
    def _session_params(self) -> dict:
        """Параметры SU2 текущего проекта (для экспорта пресета)."""
        import su2_preset_format as PF
        sess = getattr(self, "session", None)
        if sess is None:
            return {}
        params = {}
        try:
            keys = list(PF.key_catalogue().keys())
        except Exception:
            keys = []
        for k in keys:
            v = getattr(sess, k, None)
            if v is not None:
                params[k] = v
        if not params:
            for attr in ("turb_model", "mesh_quality", "use_ramp_aoa",
                         "cpu_cores", "compute_device"):
                if hasattr(sess, attr):
                    params[attr.upper() if attr.islower() else attr] = \
                        getattr(sess, attr)
        return params

    def export_config_preset(self):
        """Экспорт пресета настроек SU2 в файл .su2preset."""
        import su2_preset_format as PF
        w = self.pr_w
        source = w["source"].currentData()
        try:
            name = w["name"].text().strip() or "Без имени"
            if source == "session":
                params = self._session_params()
                if not params:
                    w["out"].setText(
                        "Внимание: Настройки проекта ещё не созданы — сначала "
                        "подготовьте расчёт либо выберите встроенный шаблон.")
                    return
                description = "Экспорт текущих настроек проекта AeroOpt"
                based_on = None
            else:
                preset = PF.builtin_presets().get(source)
                if not preset:
                    w["out"].setText("Внимание: Встроенный шаблон не найден")
                    return
                params = dict(preset.get("params") or {})
                description = str(preset.get("description") or "")
                based_on = source
            check = PF.validate_preset(PF.make_preset(name, params))
            if not check["ok"]:
                w["out"].setText("Внимание: Пресет не прошёл проверку:\n"
                                 + "\n".join("  • " + e
                                              for e in check["errors"]))
                return
            path, _ = QFileDialog.getSaveFileName(
                self, "Экспорт пресета", f"{name}{PF.EXTENSION}",
                f"Пресет AeroOpt (*{PF.EXTENSION});;JSON (*.json)")
            if not path:
                return
            PF.export_preset(path, name, params, description=description,
                             based_on=based_on)
            w["out"].setText(
                PF.describe_format() + "\n\nСохранено: " + path
                + f"\nПараметров: {len(params)}\n\nСодержимое:\n"
                + "\n".join(f"  {k} = {v}" for k, v in sorted(params.items())))
            self.log_text.append(f"Пресет сохранён: {path}")
        except Exception as e:
            w["out"].setText(f"Внимание: Ошибка экспорта: {e}")

    def import_config_preset(self):
        """Импорт и проверка пресета."""
        import su2_preset_format as PF
        w = self.pr_w
        path, _ = QFileDialog.getOpenFileName(
            self, "Импорт пресета", "",
            f"Пресет AeroOpt (*{PF.EXTENSION});;JSON (*.json);;Все файлы (*)")
        if not path:
            return
        try:
            preset = PF.import_preset(path)
        except Exception as e:
            w["out"].setText(f"Внимание: Не удалось прочитать пресет: {e}")
            return
        self._imported_preset = preset
        w["name"].setText(preset.get("name") or "Импортированный")
        params = preset.get("params") or {}
        lines = [f"Импорт: {os.path.basename(path)}",
                 f"Имя: {preset.get('name')}",
                 f"Версия формата: {preset.get('schema_version')}",
                 f"Параметров: {len(params)}"]
        for wn in preset.get("_warnings", []):
            lines.append(f"Внимание: {wn}")
        lines.append("")
        lines += [f"  {k} = {v}" for k, v in sorted(params.items())]
        w["out"].setText("\n".join(lines))
        self.log_text.append(f"Пресет импортирован: {path}")

    def apply_imported_preset(self):
        """Применяет импортированный пресет к настройкам проекта."""
        import su2_preset_format as PF
        w = self.pr_w
        preset = self._imported_preset
        if not preset:
            w["out"].setText("Внимание: Сначала импортируйте файл пресета")
            return
        params = dict(preset.get("params") or {})
        try:
            catalogue = PF.key_catalogue()
        except Exception:
            catalogue = {}
        applied, skipped = [], []
        for k, v in params.items():
            if k in catalogue:
                applied.append(f"{k} = {v}")
            else:
                skipped.append(f"{k} (нет в каталоге параметров)")
        sess = getattr(self, "session", None)
        if sess is not None:
            for k, v in params.items():
                if k in catalogue:
                    try:
                        setattr(sess, k, v)
                    except Exception:
                        pass
        text = (f"Применено параметров: {len(applied)}\n"
                + "\n".join("  Готово: " + a for a in applied))
        if skipped:
            text += (f"\n\nПропущено: {len(skipped)}\n"
                     + "\n".join("  Внимание: " + s for s in skipped))
        text += ("\n\nЗначения записаны в объект расчёта проекта. Перед "
                 "запуском проверьте их в разделе Solver Settings и в меню "
                 "«SU2»: часть ключей SU2 пишется в config.cfg только при "
                 "подготовке нового расчёта.")
        w["out"].setText(text)
        self.log_text.append(f"Готово: Пресет «{preset.get('name')}» применён: "
                             f"{len(applied)} параметров")


# ---------------------------------------------------------------------------
# Single-instance lock — запрет двойного запуска
# ---------------------------------------------------------------------------
# Используем QLockFile (кросс-платформенный, без сторонних зависимостей).
# Если lock-файл занят, значит приложение уже запущено; вторую копию закрываем.
_INSTANCE_LOCK = None


def _acquire_single_instance_lock(app_name: str = "AeroOpt_v4"):
    """Возвращает QLockFile если это первая копия, иначе None.
    Lock-файл в %TEMP%/AeroOpt_v4.lock (Windows) или /tmp/... (Unix)."""
    try:
        lock_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.TempLocation
        )
        if not lock_dir:
            lock_dir = os.path.expanduser("~")
        os.makedirs(lock_dir, exist_ok=True)
        lock_path = os.path.join(lock_dir, f"{app_name}.lock")
        lock = QLockFile(lock_path)
        lock.setStaleLockTime(0)
        if lock.tryLock(100):
            return lock
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------
def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))

    # Защита от двойного запуска — если AeroOpt уже открыт, вторую копию
    # тихо закрываем (без диалоговых окон, чтобы не раздражать).
    global _INSTANCE_LOCK
    _INSTANCE_LOCK = _acquire_single_instance_lock("AeroOpt_v4")
    if _INSTANCE_LOCK is None:
        # Уже запущено — тихо выходим. Пользователь увидит, что вторая
        # копия просто не открылась, а первая продолжает работать.
        return 0

    os.makedirs(WORK_DIR_BASE, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    window = MainWindow()
    window.show()
    # Меню «SU2»: настройки config.cfg с подсказками, пресеты устойчивости,
    # откат config.cfg.orig.
    try:
        import su2_config_dialog
        su2_config_dialog.install_menu(window)
    except Exception as e:
        try:
            from app_logging import get_logger
            get_logger().warning("Меню SU2 не подключено: %s", e)
        except Exception:
            pass
    # Меню «Справка»: условия использования и политика конфиденциальности.
    try:
        from ui import legal as _legal
        _legal.install_menu(window)
    except Exception as e:
        try:
            from app_logging import get_logger
            get_logger().warning("Меню «Справка» не подключено: %s", e)
        except Exception:
            pass
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
