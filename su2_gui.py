#!/usr/bin/env python3
"""
su2_gui.py — Точка входа AeroOpt v4.0.

Запуск:
    python su2_gui.py

Структура проекта (этот файл — в корне):
    su2_gui.py              ← вы здесь
    config/
    geometry/
    license_client/
    mesh/
    optimization/
    physics/
    solver/
    ui/
        __init__.py
        main_window.py      ← MainWindow + main()
"""
import app_logging
app_logging.setup()
try:
    import su2_config_dialog
except Exception as e:
    su2_config_dialog = None
    print("[AeroOpt] su2_config_dialog не загрузился:", e)
import os
import sys

# ---------------------------------------------------------------------------
# 1. Гарантируем, что корень проекта в sys.path.
#    Без этого `from ui.main_window import ...` падает с ModuleNotFoundError,
#    когда скрипт запускается из подкаталога, ярлыка или через PyInstaller.
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# В собранном .exe sys.executable — это сам AeroOpt.exe. Если что-то
# попытается запустить "python -m pip ...", не стартуем GUI (иначе плодятся
# копии приложения вместо pip).
if getattr(sys, "frozen", False) and len(sys.argv) > 1 and "-m" in sys.argv:
    raise SystemExit(0)
# ---------------------------------------------------------------------------
# 2. Диагностика: если модуль не найден — выводим понятное сообщение
# ---------------------------------------------------------------------------
try:
    from ui.main_window import main
except ModuleNotFoundError as exc:
    # Собираем диагностическую информацию
    missing = str(exc).split("'")[-2] if "'" in str(exc) else str(exc)

    print("=" * 60)
    print("❌ ОШИБКА ИМПОРТА")
    print("=" * 60)
    print(f"  Не найден модуль: {exc}")
    print(f"  Корень проекта:   {_project_root}")
    print(f"  sys.path[0]:      {sys.path[0]}")
    print()

    # Проверяем, существует ли папка ui/
    ui_dir = os.path.join(_project_root, "ui")
    if not os.path.isdir(ui_dir):
        print(f"  ⚠️  Папка '{ui_dir}' НЕ СУЩЕСТВУЕТ!")
        print(f"     Убедитесь, что su2_gui.py лежит в корне проекта")
        print(f"     (рядом с папками config/, solver/, ui/ и т.д.)")
    else:
        init_path = os.path.join(ui_dir, "__init__.py")
        main_path = os.path.join(ui_dir, "main_window.py")
        print(f"  📂 Папка ui/:          {'✅ существует' if os.path.isdir(ui_dir) else '❌ нет'}")
        print(f"  📄 ui/__init__.py:     {'✅ есть' if os.path.isfile(init_path) else '❌ НЕТ — создайте пустой файл!'}")
        print(f"  📄 ui/main_window.py:  {'✅ есть' if os.path.isfile(main_path) else '❌ НЕТ'}")

        if os.path.isdir(ui_dir):
            print(f"\n  Содержимое ui/:")
            for f in os.listdir(ui_dir):
                print(f"    {f}")

    print()
    print("  Возможные причины:")
    print("    1. su2_gui.py лежит НЕ в корне проекта")
    print("    2. В папке ui/ нет файла __init__.py")
    print("    3. Файл main_window.py называется иначе")
    print("=" * 60)

    # На Windows показываем MessageBox, чтобы пользователь увидел ошибку
    # даже если консоль закрывается мгновенно
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0,
                f"Не найден модуль: {missing}\n\n"
                f"Корень проекта: {_project_root}\n\n"
                f"Убедитесь, что su2_gui.py лежит в корне проекта\n"
                f"(рядом с папками config/, solver/, ui/).",
                "AeroOpt — Ошибка запуска",
                0x10,  # MB_ICONERROR
            )
        except Exception:
            pass

    sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Запуск
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import app_logging
    try:
        exit_code = main()
        app_logging.end_session()   # чистое закрытие: метка SESSION END
        sys.exit(exit_code or 0)
    except Exception:
        # Подстраховка: если main() где-то проглотил исключение мимо
        # sys.excepthook (например, в потоке Qt) — пишем полный стек в лог.
        import traceback
        app_logging.get_logger().critical("Падение в main():\n%s", traceback.format_exc())
        app_logging.end_session()
        raise