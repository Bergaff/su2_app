"""
app_logging.py — единое подробное логирование AeroOpt.

Подключение: в САМОМ НАЧАЛЕ su2_gui.py (до любых других импортов проекта):

    import app_logging
    app_logging.setup()

Что даёт:
- Папка с логами: %APPDATA%\\AeroOpt\\logs (Windows),
  ~/.local/share/AeroOpt/logs (Linux), ~/Library/Logs/AeroOpt (macOS).
- На каждую сессию — отдельный файл session_ГГГГммдд_ЧЧММСС_<pid>.log
  (строки «SESSION START» / «SESSION END»), плюс общий aeroopt.log.
- Ловятся ВСЕ необработанные исключения (в т.ч. в слотах PyQt и в
  фоновых потоках) — пишется полный traceback; в windowed-сборке (.exe),
  где консоли нет, это единственный способ увидеть причину вылета.
- ВАЖНО: никакой рекурсии — перехват stdout/stderr защищён от
  повторного входа, консольный обработчик создаётся только при наличии
  реальной консоли (в --windowed .exe его нет).

Посмотреть путь к логам: app_logging.log_dir()
Открыть папку логов:   app_logging.open_log_folder()
"""

import os
import sys
import glob
import logging
import datetime
import traceback
import threading

LOGGER_NAME = "aeroopt"
_KEEP_DAYS = 50
_ROTATE_BYTES = 2_000_000
_ROTATE_BACKUPS = 10

_session_file_handler = None
_log_dir = None
_session_started_at = None
# Защита от рекурсии «log → stderr → log» на каждый поток.
_tls = threading.local()


def log_dir() -> str:
    global _log_dir
    if _log_dir:
        return _log_dir
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        d = os.path.join(base, "AeroOpt", "logs")
    elif sys.platform == "darwin":
        d = os.path.join(os.path.expanduser("~/Library/Logs"), "AeroOpt")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(
            os.path.expanduser("~"), ".local", "share")
        d = os.path.join(base, "AeroOpt", "logs")
    os.makedirs(d, exist_ok=True)
    _log_dir = d
    return d


def _cleanup_old_sessions(d: str) -> None:
    try:
        cutoff = datetime.datetime.now() - datetime.timedelta(days=_KEEP_DAYS)
        for f in glob.glob(os.path.join(d, "session_*.log")):
            try:
                if datetime.datetime.fromtimestamp(os.path.getmtime(f)) < cutoff:
                    os.remove(f)
            except Exception:
                pass
    except Exception:
        pass


def _fmt_ts() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class _StreamTee:
    """Дублирует stdout/stderr в логгер. РЕЕНТРАНТНО-БЕЗОПАСЕН:
    если запись в лог сама что-то выводит в поток — не зацикливаемся."""

    def __init__(self, original, level):
        self._original = original          # НАСТОЯЩИЙ поток (захвачен до подмены)
        self._level = level
        self._buf = ""

    def _in_tee(self):
        return getattr(_tls, "in_tee", False)

    def write(self, data):
        # 1) всегда сначала пишем в реальный поток (если он есть)
        if self._original is not None:
            try:
                self._original.write(data)
            except Exception:
                pass
        # 2) дублируем в лог, но только если не внутри логирования уже
        if self._in_tee():
            return
        try:
            _tls.in_tee = True
            self._buf += data
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.rstrip("\r")
                if line.strip():
                    logging.getLogger(LOGGER_NAME).log(self._level, line)
        except Exception:
            pass
        finally:
            _tls.in_tee = False

    def flush(self):
        if self._original is not None:
            try:
                self._original.flush()
            except Exception:
                pass

    def __getattr__(self, name):
        # Делегируем реальному потоку (encoding, isatty, fileno и т.п.).
        return getattr(self._original, name)


def _install_exception_hooks(log: logging.Logger) -> None:
    def _flush_all():
        for h in logging.getLogger(LOGGER_NAME).handlers:
            try:
                h.flush()
            except Exception:
                pass

    def _show_crash_box():
        try:
            if sys.platform != "win32":
                return
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0,
                f"AeroOpt столкнулось с ошибкой и будет закрыто.\n\n"
                f"Подробности в логе:\n{log_dir()}\n\n"
                f"Пришлите самый свежий файл session_*.log.",
                "AeroOpt — ошибка", 0x10)
        except Exception:
            pass

    def _excepthook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            try:
                sys.__excepthook__(exc_type, exc_value, exc_tb)
            except Exception:
                pass
            return
        try:
            log.critical("Необработанное исключение (главный поток):\n%s",
                         "".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
        except Exception:
            pass
        _flush_all()
        _show_crash_box()

    def _thread_excepthook(args):
        try:
            log.critical(
                "Необработанное исключение в потоке %r:\n%s",
                getattr(args.thread, "name", "?"),
                "".join(traceback.format_exception(
                    args.exc_type, args.exc_value, args.exc_traceback)),
            )
        except Exception:
            pass
        _flush_all()

    sys.excepthook = _excepthook
    try:
        threading.excepthook = _thread_excepthook
    except Exception:
        pass


def setup(level=logging.DEBUG, console=None) -> logging.Logger:
    """Инициализировать логирование. Безопасно вызывать повторно и
    безопасно падать (любой сбой не должен ронять запуск приложения)."""
    global _session_file_handler, _session_started_at

    log = logging.getLogger(LOGGER_NAME)
    if getattr(log, "_aero_configured", False):
        return log

    # Захватываем НАСТОЯЩИЕ потоки ДО любой подмены.
    real_stdout = sys.stdout
    real_stderr = sys.stderr

    # Консольный лог имеет смысл только когда реально есть консоль
    # (в PyInstaller --windowed stderr/stdout == None).
    if console is None:
        console = bool(real_stderr is not None)

    try:
        log.setLevel(level)
        log.propagate = False
        fmt = logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s | "
            "%(threadName)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        d = log_dir()
        _cleanup_old_sessions(d)
        _session_started_at = datetime.datetime.now()
        sess_path = os.path.join(d, f"session_{_fmt_ts()}_{os.getpid()}.log")

        # Файловые обработчики добавляем ПЕРВЫМИ и отдельно — даже если
        # остальное упадёт, файл уже будет создан и будет писаться.
        _session_file_handler = logging.FileHandler(sess_path, encoding="utf-8")
        _session_file_handler.setLevel(logging.DEBUG)
        _session_file_handler.setFormatter(fmt)
        log.addHandler(_session_file_handler)

        try:
            from logging.handlers import RotatingFileHandler
            rot = RotatingFileHandler(os.path.join(d, "aeroopt.log"),
                                      maxBytes=_ROTATE_BYTES,
                                      backupCount=_ROTATE_BACKUPS,
                                      encoding="utf-8")
            rot.setLevel(logging.DEBUG)
            rot.setFormatter(fmt)
            log.addHandler(rot)
        except Exception:
            pass

        # Консольный обработчик — строго на ЗАХВАЧЕННЫЙ реальный поток
        # (не на sys.stderr, который мы вот-вот подменим).
        if console and real_stderr is not None:
            try:
                ch = logging.StreamHandler(real_stderr)
                ch.setLevel(logging.INFO)
                ch.setFormatter(fmt)
                log.addHandler(ch)
            except Exception:
                pass

        _install_exception_hooks(log)

        # Подмена stdout/stderr — ПОСЛЕ настройки обработчиков и с
        # реентрант-защитой внутри Tee.
        try:
            sys.stdout = _StreamTee(real_stdout, logging.INFO)
            sys.stderr = _StreamTee(real_stderr, logging.WARNING)
        except Exception:
            pass

        log._aero_configured = True

        log.info("=" * 70)
        log.info("SESSION START  pid=%s python=%s exe=%s", os.getpid(),
                 sys.version.split()[0], sys.executable)
        log.info("frozen=%s platform=%s argv=%r",
                 getattr(sys, "frozen", False), sys.platform, sys.argv)
        log.info("log_dir=%s", d)
        log.info("session_log=%s", sess_path)
        log.info("=" * 70)
    except Exception:
        # Совсем аварийный путь: логирование не должно мешать запуску.
        try:
            with open(os.path.join(log_dir(), "logging_setup_error.log"),
                      "a", encoding="utf-8") as f:
                f.write(traceback.format_exc() + "\n")
        except Exception:
            pass

    return log


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    if name in (None, LOGGER_NAME, ""):
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def end_session() -> None:
    log = logging.getLogger(LOGGER_NAME)
    if not getattr(log, "_aero_configured", False):
        return
    try:
        log.info("=" * 70)
        log.info("SESSION END  длительность=%s",
                 datetime.datetime.now() - _session_started_at
                 if _session_started_at else "?")
        log.info("=" * 70)
    except Exception:
        pass
    for h in log.handlers:
        try:
            h.flush()
        except Exception:
            pass


def open_log_folder() -> str:
    d = log_dir()
    try:
        if sys.platform == "win32":
            os.startfile(d)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", d])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", d])
    except Exception:
        pass
    return d