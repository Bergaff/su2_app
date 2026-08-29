"""AST-линт: каждый self.<attr>, читаемый в MainWindow, должен где-то
присваиваться (self.<attr> = ...) или быть методом/атрибутом Qt."""
import ast
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "ui", "main_window.py")

tree = ast.parse(open(SRC, encoding="utf-8").read())

cls = next(n for n in ast.walk(tree)
           if isinstance(n, ast.ClassDef) and n.name == "MainWindow")

assigned, read, methods = set(), set(), set()
for node in ast.walk(cls):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        methods.add(node.name)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
            and node.value.id == "self":
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            assigned.add(node.attr)
        else:
            read.add(node.attr)

# атрибуты, читаемые, но нигде не присваиваемые и не методы
missing = sorted(a for a in read - assigned - methods)

# известные атрибуты QMainWindow/Qt (наследуются)
QT_INHERITED = {
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
    "resizeEvent", "paintEvent", "closeEvent", "showEvent", "hideEvent",
}
missing = [a for a in missing if a not in QT_INHERITED]

if missing:
    print("ПОТЕНЦИАЛЬНО ОТСУТСТВУЮЩИЕ АТРИБУТЫ:")
    for a in missing:
        print("  -", a)
    sys.exit(1)
print("AST-ЛИНТ ЧИСТО: все self-атрибуты MainWindow присваиваются или являются методами")
print(f"  методов: {len(methods)}, присваиваемых атрибутов: {len(assigned)}, читаемых: {len(read)}")
