import os
import shutil
import stat
import sys
import subprocess
import time

APP_EXE = "AeroOpt.exe"

def _running_pids(image=APP_EXE):
    """PID процессов с таким именем (пусто, если tasklist недоступен)."""
    if sys.platform != "win32":
        return []
    try:
        r = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq " + image, "/FO", "CSV", "/NH"],
            capture_output=True, text=True, encoding="cp866", errors="replace")
    except Exception:
        return []
    pids = []
    for line in (r.stdout or "").splitlines():
        cols = [c.strip().strip('"') for c in line.split('","')]
        if len(cols) >= 2 and cols[0].lower() == image.lower():
            pids.append(cols[1])
    return pids


def _kill_app():
    """Снимает AeroOpt.exe и ждёт, пока Windows отпустит дескрипторы.

    Раньше taskkill вызывался и сразу шёл rmtree — а Windows освобождает
    файл не мгновенно, отсюда WinError 5 на «вроде бы закрытом» exe.
    """
    if sys.platform != "win32":
        return
    try:
        subprocess.run(["taskkill", "/F", "/T", "/IM", APP_EXE],
                       capture_output=True)
    except Exception:
        pass
    for _ in range(20):
        if not _running_pids():
            break
        time.sleep(0.25)
    time.sleep(0.5)


def _rmtree_force(path, attempts=6, delay=0.7):
    """Удаляет дерево, снимая read-only и пережидая блокировку.

    Возвращает (ok, err). Отсутствие каталога — это УСПЕХ, а не ошибка:
    раньше FileNotFoundError [WinError 3] печатался как сбой, хотя папку
    уже стёрли (вручную или частично предыдущей попыткой).
    """
    def _retry(func, target, _exc):
        try:
            os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
            func(target)
        except OSError:
            pass

    kw = {}
    if sys.version_info >= (3, 12):
        kw["onexc"] = _retry
    else:
        kw["onerror"] = lambda f, t, ei: _retry(f, t, ei)

    err = None
    for attempt in range(1, attempts + 1):
        if not os.path.exists(path):
            return True, None
        try:
            shutil.rmtree(path, **kw)
            if not os.path.exists(path):
                return True, None
            err = "каталог не исчез после rmtree"
        except FileNotFoundError:
            return True, None          # уже удалён — нормально
        except OSError as e:
            err = e
        if attempt < attempts:
            time.sleep(delay * attempt)
    return (not os.path.exists(path)), err


def _prepare_dist(project_root):
    """Освобождает dist\AeroOpt перед сборкой. True — можно собирать.

    Если удалить не выходит, папка ПЕРЕИМЕНОВЫВАЕТСЯ: переименование
    каталога Windows разрешает даже при занятом файле внутри, поэтому
    сборка больше не упирается в блокировку.
    """
    dist_dir = os.path.join(project_root, "dist", "AeroOpt")
    if not os.path.isdir(dist_dir):
        print("Готово: dist/AeroOpt нет — чистая сборка.")
        return True

    # Командная строка, открытая внутри dist, сама держит папку.
    cwd = os.path.abspath(os.getcwd())
    dist_root = os.path.join(project_root, "dist")
    if cwd == dist_root or cwd.startswith(dist_root + os.sep):
        print("Внимание: Текущий каталог внутри dist/ — он и блокирует папку.")
        print("   Перехожу в корень проекта: " + project_root)
        try:
            os.chdir(project_root)
        except OSError:
            pass

    print("Освобождаю dist/AeroOpt ...")
    _kill_app()

    ok, err = _rmtree_force(dist_dir)
    if ok:
        print("Готово: dist/AeroOpt удалён.")
        return True

    print("Внимание: Удалить не удалось: " + str(err))
    pids = _running_pids()
    if pids:
        print("   " + APP_EXE + " всё ещё работает, PID: " + ", ".join(pids))
    left = []
    for root, _dirs, files in os.walk(dist_dir):
        for f in files:
            left.append(os.path.join(root, f))
            if len(left) >= 10:
                break
        if len(left) >= 10:
            break
    if left:
        print("   Остались файлы:")
        for f in left:
            print("     - " + f)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    renamed = os.path.join(os.path.dirname(dist_dir), "AeroOpt_old_" + stamp)
    try:
        os.rename(dist_dir, renamed)
    except OSError as e:
        print("   Переименовать тоже не вышло: " + str(e))
        print()
        print("   Сделайте одно из и запустите сборку заново:")
        print("     1. Диспетчер задач → снимите " + APP_EXE +
              " и зависшие python.exe;")
        print("     2. закройте окно проводника, открытое в dist\\AeroOpt;")
        print("     3. запускайте сборку НЕ изнутри dist\\AeroOpt;")
        print("     4. добавьте C:\\su2_app в исключения антивируса/Defender;")
        print("     5. перезагрузите ПК и сразу запустите python build_exe.py.")
        return False

    print("   Не удалил, а переименовал в " + os.path.basename(renamed) +
          " — сборка продолжается.")
    print("      Сотрите эту папку вручную, когда " + APP_EXE + " точно закрыт.")
    return True


def build():
    print("=" * 60)
    print("AeroOpt v4.1 Standalone Executable Build Script")
    print("=" * 60)

    if sys.platform != "win32":
        print("Внимание:")
        print("   Скрипт запущен в не-Windows окружении. PyInstaller собирает .exe")
        print("   только под ту ОС, где запущен. Скопируйте проект на Windows-ПК и")
        print("   запустите там:  python build_exe.py\n")

    try:
        import PyInstaller  # noqa: F401
        print("Готово: PyInstaller обнаружен.")
    except ImportError:
        print("Установка PyInstaller...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        except Exception as e:
            print(f"Не удалось установить PyInstaller: {e}")
            print("   Выполните вручную: pip install pyinstaller")
            return

    project_root = os.path.dirname(os.path.abspath(__file__))
    lic_dir = os.path.join(project_root, "license_client")

    if not os.path.isdir(lic_dir):
        print(f"Ошибка: Папка license_client не найдена: {lic_dir}")
        print("   Убедитесь, что license_client/ лежит рядом с su2_gui.py")
        return
    if not os.path.isfile(os.path.join(lic_dir, "__init__.py")):
        with open(os.path.join(lic_dir, "__init__.py"), "w", encoding="utf-8") as f:
            f.write('from .license_checker import LicenseChecker, LicenseStatus\n')
            f.write('__all__ = ["LicenseChecker", "LicenseStatus"]\n')
        print("Готово: Создан license_client/__init__.py")
    else:
        print("Готово: license_client/__init__.py найден")

    # Быстрая проверка, что пакет импортируется ДО сборки.
    try:
        subprocess.check_call(
            [sys.executable, "-c",
             "import license_client, license_client.license_checker; print('import OK')"],
            cwd=project_root)
    except Exception:
        print("Ошибка: license_client не импортируется из корня проекта.")
        print("   Ожидается: <project>/license_client/__init__.py и license_checker.py")
        return

    entry = "su2_gui.py"
    if not os.path.isfile(os.path.join(project_root, entry)):
        print(f"Ошибка: Не найден точка входа: {os.path.join(project_root, entry)}")
        return

    # Перед сборкой освобождаем dist/AeroOpt. Именно здесь раньше падало
    # с WinError 5 (exe занят) и WinError 3 (папку уже стёрли вручную,
    # а скрипт считал это ошибкой и трижды требовал Enter).
    if sys.platform == "win32":
        if not _prepare_dist(project_root):
            print("\nОшибка: Сборка прервана: не удалось освободить dist/AeroOpt.")
            print("   dist/AeroOpt остался нетронутым — ничего не сломано.\n")
            return
        # Кэш PyInstaller тоже может быть занят — чистим, но не фатально.
        _rmtree_force(os.path.join(project_root, "build", "AeroOpt"),
                      attempts=2, delay=0.3)

    # Автоматически находим ЛОКАЛЬНЫЕ пакеты проекта (папки с __init__.py
    # прямо в корне: mesh, ui, solver и т.п.) и заставляем PyInstaller
    # собрать ВСЕ их подмодули — иначе динамически импортируемые модули
    # (например mesh.gmsh_generator) в .exe не попадают.
    local_packages = []
    for name in sorted(os.listdir(project_root)):
        p = os.path.join(project_root, name)
        if not os.path.isdir(p) or name in ("build", "dist", "venv", ".venv") \
                or name.startswith("."):
            continue
        # пакет: есть __init__.py ИЛИ внутри лежат .py (namespace-пакет)
        has_py = any(f.endswith(".py") for f in os.listdir(p))
        if has_py:
            local_packages.append(name)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=AeroOpt",
        "--windowed",
        "--clean",
        "--noconfirm",  # перезаписывать dist/AeroOpt без вопроса
        f"--paths={project_root}",
        # license_client — чистый Python (только стандартная библиотека),
        # его подхватывает анализ импортов; collect-all/add-data не нужны.
        "--hidden-import=license_client",
        "--hidden-import=license_client.license_checker",
        "--hidden-import=app_logging",
        # Тяжёлые научные пакеты (не связаны с лицензией) — как было.
        "--hidden-import=pyvista",
        "--hidden-import=vtk",
        "--hidden-import=matplotlib",
        "--hidden-import=scipy.spatial.transform._rotation_groups",
    ]

    # Все подмодули локальных пакетов проекта.
    for pkg in local_packages:
        cmd.append(f"--collect-submodules={pkg}")

    # Официальные кейсы SU2: конфиги (.cfg) — это данные, PyInstaller их сам
    # не подхватит. Сетки в official_cases/meshes/ качаются в рантайме и в
    # сборку не входят (они в .gitignore).
    _oc_cfgs = os.path.join(project_root, "official_cases", "configs")
    if os.path.isdir(_oc_cfgs):
        cmd.append(f"--add-data={_oc_cfgs}{os.pathsep}official_cases/configs")

    # gmsh — сторонняя либа с нативной DLL/данными, собираем целиком.
    try:
        import importlib.util
        if importlib.util.find_spec("gmsh") is not None:
            cmd.append("--collect-all=gmsh")
            print("Готово: gmsh найден — собираю с нативными библиотеками.")
    except Exception:
        pass

    # TetGen — телооблекающая сетка (mesh/bodyfit_tetgen.py). Пакет несёт
    # нативную библиотеку тетраэдрального разбиения, поэтому collect-all.
    # trimesh и manifold3d нужны ему для объединения тел в замкнутую
    # поверхность перед триангуляцией. Если чего-то нет, генератор сам
    # откатится на картезианский фон.
    _mesh_missing = []
    try:
        import importlib.util
        for _pkg in ("tetgen", "trimesh", "manifold3d"):
            try:
                _found = importlib.util.find_spec(_pkg) is not None
            except Exception:
                _found = False
            if _found:
                cmd.append(f"--collect-all={_pkg}")
                print(f"Готово: {_pkg} найден — собираю с нативными "
                      "библиотеками.")
            else:
                _mesh_missing.append(_pkg)
    except Exception:
        _mesh_missing = ["tetgen", "trimesh", "manifold3d"]

    # Без них mesh/bodyfit_tetgen не работает и генератор молча строит
    # картезианский фон: сетка не облегает поверхность, тонкие элементы
    # превращаются в ступеньку, и SU2 на такой сетке расходится. Раньше
    # отсутствие пакетов просто пропускалось, и пользователь узнавал об
    # этом только по расходимости — поэтому здесь это говорится вслух.
    if _mesh_missing:
        print()
        print("ВНИМАНИЕ: не установлены %s." % ", ".join(_mesh_missing))
        print("  Телооблекающая сетка (TetGen) в сборке работать НЕ будет,")
        print("  генератор откатится на картезианский фон. Расчёт на такой")
        print("  сетке может расходиться независимо от настроек решателя.")
        print("  Установка:  python -m pip install %s"
              % " ".join(_mesh_missing))
        print()

    # Если динамически импортируются и другие сторонние пакеты — добавляй
    # их так же здесь, например:  cmd.append("--collect-all=<пакет>")

    cmd.append(entry)

    print("\nЗапуск PyInstaller...")
    print(f"   Корень проекта: {project_root}")
    print(f"   Локальные пакеты (все подмодули): {', '.join(local_packages) or 'нет'}")
    print(f"   Полный лог: build_log.txt\n")

    log_path = os.path.join(project_root, "build_log.txt")
    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        logf.write(" ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=project_root,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace")
        logf.write(proc.stdout or "")
        # И заодно показываем в консоли.
        print(proc.stdout or "")

    exe_path = os.path.join(project_root, "dist", "AeroOpt", APP_EXE)
    if proc.returncode == 0 and not os.path.isfile(exe_path):
        print("\nОшибка: PyInstaller завершился без ошибки, но " + exe_path +
              " не появился.")
        print("   Пришлите build_log.txt.\n")
        return

    if proc.returncode == 0:
        print("\n" + "=" * 60)
        print("СБОРКА УСПЕШНО ЗАВЕРШЕНА!")
        print("   Результат: " + exe_path)
        print("   Папку dist/AeroOpt можно архивировать и переносить.")
        print("=" * 60)
    else:
        print("\nОшибка: Сборка не удалась. Настоящая причина — выше и в build_log.txt.")
        print("   Ищите строки с 'ERROR', 'Traceback', 'Unable to find'.")
        print("   Пришлите последние ~40 строк build_log.txt.\n")

if __name__ == "__main__":
    build()