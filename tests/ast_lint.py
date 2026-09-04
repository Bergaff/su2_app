"""AST-линт: каждый self.<attr>, читаемый в классе, должен где-то
присваиваться (self.<attr> = ...) или быть методом/атрибутом базового класса.

Раньше линт проверял один класс MainWindow в ui/main_window.py. Этого
оказалось мало: атрибут _mesh_note_shown был объявлен в SU2Worker, а
читался в SessionRunner.run — и первый же повтор прогона падал с
AttributeError уже после успешно посчитанной сетки. Линт этого не увидел,
потому что SessionRunner в его область не входил. Поэтому теперь проверяется
список классов в нескольких файлах, с учётом наследования внутри файла.
"""
import ast
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# (файл относительно корня, классы для проверки)
TARGETS = [
    (os.path.join("ui", "main_window.py"), ["MainWindow"]),
    (os.path.join("solver", "workers.py"),
     ["SessionRunner", "SweepWorker", "SU2Worker", "OptimizationWorker"]),
    (os.path.join("mesh", "mesh_worker.py"),
     ["MeshWorker", "MeshAdaptWorker"]),
]

# известные атрибуты QMainWindow/QWidget (наследуются)
QT_WIDGET = {
    "statusBar", "menuBar", "addToolBar", "setCentralWidget", "centralWidget",
    "setWindowTitle", "resize", "show", "close", "update", "repaint",
    "setStyleSheet", "setEnabled", "setDisabled", "hasFocus", "keyPressEvent",
    "closeEvent", "setMouseTracking", "cursor", "setCursor", "unsetCursor",
    "move", "pos", "width", "height", "setMinimumSize", "setWindowIcon",
    "findChild", "findChildren", "children", "parent", "deleteLater",
    "setWindowState", "windowState", "isVisible", "raise_", "activateWindow",
    "setTabOrder", "toolTip", "setToolTip", "setWhatsThis", "setAcceptDrops",
    "grab", "saveGeometry", "restoreGeometry", "setFocus", "clearFocus",
    "setGeometry", "geometry", "rect", "frameGeometry", "moveEvent",
    "resizeEvent", "paintEvent", "showEvent", "hideEvent",
}

# известные атрибуты QThread/QObject (наследуются)
QT_THREAD = {
    "start", "run", "quit", "exit", "wait", "isRunning", "isFinished",
    "terminate", "requestInterruption", "isInterruptionRequested",
    "setTerminationEnabled", "currentThread", "msleep", "sleep", "usleep",
    "connect", "disconnect", "emit", "blockSignals", "signalsBlocked",
    "findChild", "findChildren", "children", "parent", "deleteLater",
    "setParent", "objectName", "setObjectName", "metaObject", "sender",
    "thread", "moveToThread", "event", "tr", "priority", "setPriority",
}

KNOWN = QT_WIDGET | QT_THREAD


def collect(cls):
    """(assigned, read, methods, bases) для одного ClassDef.

    В `assigned` идут ТОЛЬКО атрибуты, заданные в __init__ или на уровне
    класса. Присваивание внутри другого метода не считается: атрибут до
    вызова этого метода не существует. Слабое правило («присвоен где
    угодно») баг с _mesh_note_shown пропускало — там спасало условное
    `self._mesh_note_shown = True` внутри if, которое для AST тоже Store.
    """
    assigned, read, methods = set(), set(), set()
    # Атрибуты уровня класса: сигналы pyqtSignal и константы объявляются
    # как `log_signal = pyqtSignal(str)` прямо в теле класса, без self.
    # Без этого линт считает их неприсвоенными.
    for node in cls.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    assigned.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigned.add(node.target.id)
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.add(node.name)
            if node.name == "__init__":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Attribute) \
                            and isinstance(sub.value, ast.Name) \
                            and sub.value.id == "self" \
                            and isinstance(sub.ctx, (ast.Store, ast.Del)):
                        assigned.add(sub.attr)
    # Чтение getattr(self, "x", default) безопасно: дефолт покрывает
    # отсутствие атрибута.
    guarded = set()
    for node in ast.walk(cls):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "getattr" and len(node.args) >= 3 \
                and isinstance(node.args[0], ast.Name) \
                and node.args[0].id == "self" \
                and isinstance(node.args[1], ast.Constant):
            guarded.add(node.args[1].value)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "self":
            if not isinstance(node.ctx, (ast.Store, ast.Del)):
                read.add(node.attr)
    assigned |= guarded
    bases = [b.id for b in cls.bases if isinstance(b, ast.Name)]
    return assigned, read, methods, bases


def resolve(name, by_name, assigned, methods, seen=None):
    """Накопить присваивания и методы по цепочке наследования внутри файла."""
    seen = seen or set()
    if name in seen or name not in by_name:
        return
    seen.add(name)
    a, _r, m, bases = by_name[name]
    assigned |= a
    methods |= m
    for b in bases:
        resolve(b, by_name, assigned, methods, seen)


total_missing = []
totals = []

for rel, class_names in TARGETS:
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        print("Внимание: файл не найден, пропускаю: %s" % rel)
        continue
    tree = ast.parse(open(path, encoding="utf-8").read())
    by_name = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            by_name.setdefault(node.name, collect(node))

    for cname in class_names:
        if cname not in by_name:
            print("ПОТЕНЦИАЛЬНО ОТСУТСТВУЮЩИЙ КЛАСС: %s в %s" % (cname, rel))
            total_missing.append("%s.%s (класс не найден)" % (rel, cname))
            continue
        assigned, read, methods, _bases = by_name[cname]
        # наследование: атрибуты базовых классов этого же файла
        inherited_a, inherited_m = set(), set()
        for b in by_name[cname][3]:
            resolve(b, by_name, inherited_a, inherited_m)
        missing = sorted(read - assigned - inherited_a
                         - methods - inherited_m - KNOWN)
        for a in missing:
            total_missing.append("%s: self.%s" % (cname, a))
        totals.append((cname, len(methods), len(assigned), len(read)))

if total_missing:
    print("ПОТЕНЦИАЛЬНО ОТСУТСТВУЮЩИЕ АТРИБУТЫ:")
    for a in total_missing:
        print("  -", a)
    sys.exit(1)

print("AST-ЛИНТ ЧИСТО: все self-атрибуты проверяемых классов "
      "присваиваются или являются методами")
for cname, m, a, r in totals:
    print("  %-20s методов: %d, присваиваемых: %d, читаемых: %d"
          % (cname, m, a, r))
