import os
import sys
import subprocess

def build():
    print("=" * 60)
    print("🚀 AeroOpt v4.1 Standalone Executable Build Script")
    print("=" * 60)

    if sys.platform != "win32":
        print("⚠️ ПРЕДУПРЕЖДЕНИЕ:")
        print("   Скрипт запущен в не-Windows окружении. PyInstaller собирает .exe")
        print("   только под ту ОС, где запущен. Скопируйте проект на Windows-ПК и")
        print("   запустите там:  python build_exe.py\n")

    try:
        import PyInstaller  # noqa: F401
        print("✅ PyInstaller обнаружен.")
    except ImportError:
        print("📦 Установка PyInstaller...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        except Exception as e:
            print(f"❌ Не удалось установить PyInstaller: {e}")
            print("   Выполните вручную: pip install pyinstaller")
            return

    project_root = os.path.dirname(os.path.abspath(__file__))
    lic_dir = os.path.join(project_root, "license_client")

    if not os.path.isdir(lic_dir):
        print(f"❌ Папка license_client не найдена: {lic_dir}")
        print("   Убедитесь, что license_client/ лежит рядом с su2_gui.py")
        return
    if not os.path.isfile(os.path.join(lic_dir, "__init__.py")):
        with open(os.path.join(lic_dir, "__init__.py"), "w", encoding="utf-8") as f:
            f.write('from .license_checker import LicenseChecker, LicenseStatus\n')
            f.write('__all__ = ["LicenseChecker", "LicenseStatus"]\n')
        print("✅ Создан license_client/__init__.py")
    else:
        print("✅ license_client/__init__.py найден")

    # Быстрая проверка, что пакет импортируется ДО сборки.
    try:
        subprocess.check_call(
            [sys.executable, "-c",
             "import license_client, license_client.license_checker; print('import OK')"],
            cwd=project_root)
    except Exception:
        print("❌ license_client не импортируется из корня проекта.")
        print("   Ожидается: <project>/license_client/__init__.py и license_checker.py")
        return

    entry = "su2_gui.py"
    if not os.path.isfile(os.path.join(project_root, entry)):
        print(f"❌ Не найден точка входа: {os.path.join(project_root, entry)}")
        return

    # Перед сборкой закрываем запущенный AeroOpt.exe — иначе Windows держит
    # файл dist\AeroOpt\AeroOpt.exe и PyInstaller падает с WinError 5.
    if sys.platform == "win32":
        try:
            subprocess.run(["taskkill", "/F", "/IM", "AeroOpt.exe"],
                           capture_output=True)
            print("ℹ️  Завершены запущенные процессы AeroOpt.exe (если были).")
        except Exception:
            pass
        dist_dir = os.path.join(project_root, "dist", "AeroOpt")
        if os.path.isdir(dist_dir):
            import shutil
            for attempt in range(3):
                try:
                    shutil.rmtree(dist_dir)
                    break
                except Exception as e:
                    print(f"⚠️  Не удалось удалить {dist_dir} (попытка {attempt+1}/3): {e}")
                    print("   Закройте AeroOpt.exe и окно проводника с этой папкой, затем Enter...")
                    try:
                        input()
                    except Exception:
                        break

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

    # gmsh — сторонняя либа с нативной DLL/данными, собираем целиком.
    try:
        import importlib.util
        if importlib.util.find_spec("gmsh") is not None:
            cmd.append("--collect-all=gmsh")
            print("✅ gmsh найден — собираю с нативными библиотеками.")
    except Exception:
        pass

    # Если динамически импортируются и другие сторонние пакеты — добавляй
    # их так же здесь, например:  cmd.append("--collect-all=<пакет>")

    cmd.append(entry)

    print("\n📦 Запуск PyInstaller...")
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

    if proc.returncode == 0:
        print("\n" + "=" * 60)
        print("🎉 СБОРКА УСПЕШНО ЗАВЕРШЕНА!")
        print("   Результат: dist/AeroOpt/AeroOpt.exe")
        print("   Папку dist/AeroOpt можно архивировать и переносить.")
        print("=" * 60)
    else:
        print("\n❌ Сборка не удалась. Настоящая причина — выше и в build_log.txt.")
        print("   Ищите строки с 'ERROR', 'Traceback', 'Unable to find'.")
        print("   Пришлите последние ~40 строк build_log.txt.\n")

if __name__ == "__main__":
    build()