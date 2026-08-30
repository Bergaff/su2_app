"""Импорт-тест всех модулей проекта со stub-заглушками внешних
GUI/3D-зависимостей (PyQt5, pyvista, pyvistaqt, trimesh), которых нет
в CI-песочнице. Ловит NameError/ImportError на уровне модулей и классов."""
import sys
import types
import unittest.mock as mock

# ---------------------------------------------------------------- stub PyQt5
class _Sig:
    def __init__(self, *a, **k): pass
    def connect(self, *a, **k): pass
    def emit(self, *a, **k): pass
    def disconnect(self, *a, **k): pass

def _pyqtSignal(*a, **k):
    return _Sig()

class _QtStub:
    def __getattr__(self, name):
        # Qt.Checked, Qt.WaitCursor, QEvent.* и т.д.
        return 0

class _MetaBase(type):
    def __getattr__(cls, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        m = mock.MagicMock(name=f"cls.{name}")
        setattr(cls, name, m)
        return m

class _Base(metaclass=_MetaBase):
    def __init__(self, *a, **k): pass
    def __getattr__(self, name):
        m = mock.MagicMock(name=name)
        object.__setattr__(self, name, m)
        return m

pyqt5 = types.ModuleType("PyQt5")
qtwidgets = types.ModuleType("PyQt5.QtWidgets")
qtcore = types.ModuleType("PyQt5.QtCore")
qtgui = types.ModuleType("PyQt5.QtGui")

_WIDGETS = ["QApplication", "QMainWindow", "QWidget", "QVBoxLayout",
           "QHBoxLayout", "QPushButton", "QLabel", "QTextEdit", "QGroupBox",
           "QFileDialog", "QMessageBox", "QProgressBar", "QTableWidget",
           "QTableWidgetItem", "QComboBox", "QCheckBox", "QSpinBox",
           "QDoubleSpinBox", "QFormLayout", "QRadioButton", "QButtonGroup",
           "QTabWidget", "QLineEdit", "QDialog", "QDialogButtonBox", "QMenu",
           "QSplitter", "QScrollArea", "QTreeWidget", "QTreeWidgetItem",
           "QStackedWidget", "QGridLayout", "QSlider", "QToolButton",
           "QWizard", "QWizardPage", "QAction", "QSizePolicy", "QHeaderView",
           "QAbstractItemView", "QStatusBar", "QToolBar", "QStyle",
           "QStyleFactory", "QListWidget", "QListWidgetItem", "QInputDialog",
           "QToolTip", "QFrame", "QSpacerItem", "QMenuBar", "QPlainTextEdit",
           "QTableWidgetSelectionRange", "QProgressDialog", "QSplashScreen"]
for n in _WIDGETS:
    setattr(qtwidgets, n, type(n, (_Base,), {}))

qtcore.Qt = _QtStub()
qtcore.QTimer = type("QTimer", (_Base,), {})
qtcore.QThread = type("QThread", (_Base,), {})
qtcore.QObject = type("QObject", (_Base,), {})
qtcore.QMutex = type("QMutex", (_Base,), {})
qtcore.QWaitCondition = type("QWaitCondition", (_Base,), {})
qtcore.QEvent = _QtStub()
qtcore.QSize = type("QSize", (_Base,), {})
qtcore.QPoint = type("QPoint", (_Base,), {})
qtcore.pyqtSignal = _pyqtSignal
qtcore.pyqtSlot = lambda *a, **k: (lambda f: f)
qtcore.QCoreApplication = type("QCoreApplication", (_Base,), {})
qtcore.PYQT_VERSION_STR = "5.15.9"
qtcore.QT_VERSION_STR = "5.15.2"
qtcore.PYQT_VERSION = 0x50F09
qtcore.QT_VERSION = 0x50F02
qtcore.qVersion = lambda: "5.15.2"
qtcore.qInstallMessageHandler = lambda *a, **k: None
qtcore.QDateTime = type("QDateTime", (_Base,), {})
qtcore.QUrl = type("QUrl", (_Base,), {})
qtcore.QMargins = type("QMargins", (_Base,), {})
qtcore.QRect = type("QRect", (_Base,), {})
qtcore.QRectF = type("QRectF", (_Base,), {})
qtcore.QLineF = type("QLineF", (_Base,), {})
qtcore.QPointF = type("QPointF", (_Base,), {})
qtcore.QTimerEvent = type("QTimerEvent", (_Base,), {})
qtcore.QMetaObject = type("QMetaObject", (_Base,), {})
qtcore.Q_RETURN_ARG = lambda *a, **k: 0
qtcore.Q_ARG = lambda *a, **k: 0
qtcore.Signal = _pyqtSignal
qtcore.Slot = qtcore.pyqtSlot
qtcore.pyqtProperty = lambda *a, **k: property(lambda self: None)
qtcore.Property = qtcore.pyqtProperty
qtcore.pyqtEnum = lambda *a, **k: (lambda cls: cls)
qtcore.QThreadPool = type("QThreadPool", (_Base,), {})
qtcore.QRunnable = type("QRunnable", (_Base,), {})
qtcore.QSettings = type("QSettings", (_Base,), {})
qtcore.QStandardPaths = type("QStandardPaths", (_Base,), {})
qtcore.QLockFile = type("QLockFile", (_Base,), {})
qtcore.QFileInfo = type("QFileInfo", (_Base,), {})
qtcore.QDir = type("QDir", (_Base,), {})
qtcore.QProcess = type("QProcess", (_Base,), {})
qtcore.QSignalBlocker = type("QSignalBlocker", (_Base,), {})
qtcore.QItemSelectionModel = type("QItemSelectionModel", (_Base,), {})
qtcore.QSortFilterProxyModel = type("QSortFilterProxyModel", (_Base,), {})
qtcore.QAbstractTableModel = type("QAbstractTableModel", (_Base,), {})
qtcore.QModelIndex = type("QModelIndex", (_Base,), {})
qtcore.QLocale = type("QLocale", (_Base,), {})
qtcore.QTranslator = type("QTranslator", (_Base,), {})
qtcore.QLibraryInfo = type("QLibraryInfo", (_Base,), {})
qtcore.QtMsgType = _QtStub()
qtcore.qFatal = lambda *a, **k: None
qtcore.qWarning = lambda *a, **k: None
qtcore.qDebug = lambda *a, **k: None

for n in ["QColor", "QFont", "QMouseEvent", "QKeyEvent", "QIcon", "QPixmap",
          "QPalette", "QCursor", "QWheelEvent", "QPainter", "QPen", "QBrush",
          "QFontMetrics", "QGuiApplication", "QSurfaceFormat"]:
    setattr(qtgui, n, type(n, (_Base,), {}))

pyqt5.QtWidgets = qtwidgets
pyqt5.QtCore = qtcore
pyqt5.QtGui = qtgui
sys.modules["PyQt5"] = pyqt5
sys.modules["PyQt5.QtWidgets"] = qtwidgets
sys.modules["PyQt5.QtCore"] = qtcore
sys.modules["PyQt5.QtGui"] = qtgui
sys.modules["PyQt5.QtCore.Qt"] = qtcore.Qt
sip = types.ModuleType("PyQt5.sip")
sip.wrappertype = type
sip.isdeleted = lambda *a, **k: False
sip.unwrapinstance = lambda *a, **k: 0
sip.getapi = lambda *a, **k: 2
sip.setapi = lambda *a, **k: None
sys.modules["PyQt5.sip"] = sip
pyqt5.sip = sip
sys.modules["sip"] = sip  # matplotlib backend_qt5 делает top-level `import sip`
qtsvg = types.ModuleType("PyQt5.QtSvg")
qtsvg.QSvgGenerator = type("QSvgGenerator", (_Base,), {})
sys.modules["PyQt5.QtSvg"] = qtsvg
pyqt5.QtSvg = qtsvg

# ---------------------------------------------------------------- stub pyvista
pv = types.ModuleType("pyvista")

class _PolyData(_Base):
    pass

pv.PolyData = _PolyData
pv.StructuredGrid = type("StructuredGrid", (_Base,), {})
pv.UnstructuredGrid = type("UnstructuredGrid", (_Base,), {})
pv.Plotter = type("Plotter", (_Base,), {})
pv.read = lambda *a, **k: _PolyData()
pv.merge = lambda *a, **k: _PolyData()
pv.lines_from_points = lambda *a, **k: _PolyData()
pv.Sphere = lambda *a, **k: _PolyData()
pv.Box = lambda *a, **k: _PolyData()
pv.Cylinder = lambda *a, **k: _PolyData()
pv.Cone = lambda *a, **k: _PolyData()
pv.Disc = lambda *a, **k: _PolyData()
pv.Plane = lambda *a, **k: _PolyData()
pv.Arrow = lambda *a, **k: _PolyData()
sys.modules["pyvista"] = pv

pvqt = types.ModuleType("pyvistaqt")
pvqt.QtInteractor = type("QtInteractor", (_Base,), {})
pvqt.BackgroundPlotter = type("BackgroundPlotter", (_Base,), {})
sys.modules["pyvistaqt"] = pvqt

# ---------------------------------------------------------------- stub trimesh
trimesh = types.ModuleType("trimesh")
trimesh.load = lambda *a, **k: mock.MagicMock()
trimesh.Trimesh = type("Trimesh", (_Base,), {})
sys.modules["trimesh"] = trimesh
sys.modules["trimesh.repair"] = types.ModuleType("trimesh.repair")
sys.modules["trimesh.smoothing"] = types.ModuleType("trimesh.smoothing")

# ------------------------------------------- stub matplotlib Qt-бэкенда
# (реальный backend_qt5agg проверяет версию Qt через метакласс-стабы)
_qtagg = types.ModuleType("matplotlib.backends.backend_qt5agg")
class FigureCanvasQTAgg(_Base):
    pass
_qtagg.FigureCanvasQTAgg = FigureCanvasQTAgg
sys.modules["matplotlib.backends.backend_qt5agg"] = _qtagg
_qt = types.ModuleType("matplotlib.backends.backend_qt")
_qt.FigureCanvasQT = FigureCanvasQTAgg
sys.modules["matplotlib.backends.backend_qt"] = _qt

# ---------------------------------------------------------------- run imports
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

MODULES = [
    "config.settings",
    "physics.atmosphere",
    "physics.airfoils",
    "optimization.rules",
    "optimization.multipoint",
    "geometry.generators",
    "geometry.stl_healer",
    "mesh.gmsh_generator",
    "mesh.mesh_worker",
    "solver.session",
    "solver.config_builder",
    "solver.workers",
    "config", "physics", "optimization", "geometry", "mesh", "solver",
    "su2_autoconfig",
    "su2_config_dialog",
    "ui.main_window",
    "su2_gui",
    "main",
]

failed = []
for m in MODULES:
    try:
        __import__(m)
        print(f"  ✅ import {m}")
    except Exception as e:
        failed.append((m, e))
        print(f"  ❌ import {m}: {type(e).__name__}: {e}")

if failed:
    print("\nПРОВАЛЕНО:", len(failed))
    sys.exit(1)
print(f"\nВСЕ {len(MODULES)} МОДУЛЕЙ ИМПОРТИРУЮТСЯ БЕЗ ОШИБОК")
