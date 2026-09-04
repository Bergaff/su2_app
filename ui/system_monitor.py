# -*- coding: utf-8 -*-
"""Живой мониторинг ЦПУ, ГПУ и памяти для панели состояния.

Без сторонних зависимостей: на Windows используется ctypes, на Linux —
/proc. Чистая математика (пересчёт приращений в проценты, подписи)
вынесена в отдельные функции, чтобы её можно было проверить тестами без
Windows.

Принцип: если источник данных недоступен, функция возвращает None, а
подпись показывает «н/д». Выдуманное число хуже честного пробела.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Чистая математика (проверяется тестами)
# ---------------------------------------------------------------------------

def cpu_percent_from_times(prev, cur):
    """Процент загрузки ЦПУ по двум замерам времён.

    Принимает кортежи ``(idle, total)`` в любых одинаковых единицах.
    Возвращает число 0..100 либо None, если посчитать нельзя.
    """
    if not prev or not cur or len(prev) != 2 or len(cur) != 2:
        return None
    d_total = cur[1] - prev[1]
    d_idle = cur[0] - prev[0]
    if d_total <= 0:
        return None
    pct = 100.0 * (d_total - d_idle) / d_total
    return max(0.0, min(100.0, pct))


def format_cpu_label(percent, cores_used=None, cores_total=None):
    """Подпись вида «ЦПУ 63% · 6 из 8 ядер»."""
    if percent is None:
        return "ЦПУ н/д"
    text = f"ЦПУ {percent:.0f}%"
    if cores_used and cores_total:
        # Пользователь мог выставить ядер больше, чем есть physically —
        # показываем честно, без «6 из 2».
        if cores_used > cores_total:
            text += f" · {cores_used} ядер"
        else:
            text += f" · {cores_used} из {cores_total} ядер"
    elif cores_total:
        text += f" · {cores_total} ядер"
    return text


def format_gpu_label(percent):
    """Подпись вида «ГПУ 41%» либо «ГПУ н/д»."""
    if percent is None:
        return "ГПУ н/д"
    return f"ГПУ {percent:.0f}%"


def format_memory_label(process_bytes, total_bytes=None, available_bytes=None):
    """Подпись памяти процесса и, если известно, всей системы."""
    if not process_bytes or process_bytes <= 0:
        return "Память н/д"
    text = f"Память {_fmt_bytes(process_bytes)}"
    if total_bytes and total_bytes > 0:
        text += f" из {_fmt_bytes(total_bytes)}"
    if available_bytes is not None and available_bytes >= 0:
        text += f" · своб. {_fmt_bytes(available_bytes)}"
    return text


def _fmt_bytes(num_bytes):
    mb = num_bytes / (1024.0 * 1024.0)
    if mb >= 1024.0:
        return f"{mb / 1024.0:.2f} ГБ"
    if mb >= 1.0:
        return f"{mb:.0f} МБ"
    return f"{num_bytes / 1024.0:.0f} КБ"


def clamp_percent(value):
    """None остаётся None, остальное приводится к 0..100."""
    if value is None:
        return None
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# ЦПУ
# ---------------------------------------------------------------------------

def read_cpu_times():
    """Возвращает ``(idle, total)`` в условных единицах либо None."""
    try:
        if sys.platform == "win32":
            idle = ctypes.c_ulonglong()
            kernel = ctypes.c_ulonglong()
            user = ctypes.c_ulonglong()
            ok = ctypes.windll.kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
            if not ok:
                return None
            # kernel уже включает idle
            total = kernel.value + user.value
            return (idle.value, total)
        path = "/proc/stat"
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="ascii", errors="replace") as f:
            line = f.readline()
        parts = line.split()
        if not parts or parts[0] != "cpu" or len(parts) < 5:
            return None
        vals = [int(x) for x in parts[1:]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        return (idle, sum(vals))
    except Exception:
        return None


def cpu_core_counts():
    """Возвращает ``(логические, физические)``; физическое может быть None."""
    logical = os.cpu_count()
    try:
        import psutil  # type: ignore
        physical = psutil.cpu_count(logical=False) or logical
    except Exception:
        physical = logical
    return logical, physical


# ---------------------------------------------------------------------------
# ГПУ
# ---------------------------------------------------------------------------

class GpuUtilization:
    """Загрузка ГПУ как в диспетчере задач.

    Порядок источников:
      1. Windows: счётчик производительности «GPU Engine / Utilization
         Percentage» (то же, что показывает диспетчер задач);
      2. nvidia-smi, если он есть в PATH;
      3. иначе None — подпись покажет «н/д».
    """

    PDH_FMT_DOUBLE = 0x00000200
    PDH_FMT_NOCAP100 = 0x40000000
    PDH_CSTATUS_VALID_DATA = 0x0
    COUNTER = "\\GPU Engine(*)\\Utilization Percentage"

    # Запускать nvidia-smi чаще нет смысла, а каждый запуск — это новый
    # процесс. На Windows из GUI-приложения это показывало мигающее окно
    # консоли раз в 2 секунды, поэтому опрос прорежен и окно подавлено.
    NVIDIA_POLL_S = 5.0

    def __init__(self):
        self._query = None
        self._counter = None
        self._pdh = None
        self._mode = None
        self._last = None
        self._nv_at = 0.0
        self._nv_val = None
        self._nv_broken = False

    # -- PDH --------------------------------------------------------------
    def _pdh_open(self):
        if sys.platform != "win32":
            return False
        try:
            pdh = ctypes.windll.pdh
            query = ctypes.c_void_p()
            counter = ctypes.c_void_p()
            if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) != 0:
                return False
            add = getattr(pdh, "PdhAddEnglishCounterW", None) or pdh.PdhAddCounterW
            if add(query, self.COUNTER, 0, ctypes.byref(counter)) != 0:
                pdh.PdhCloseQuery(query)
                return False
            self._pdh = pdh
            self._query = query
            self._counter = counter
            pdh.PdhCollectQueryData(query)   # первый замер — базовый
            return True
        except Exception:
            self._pdh_close()
            return False

    def _pdh_close(self):
        try:
            if self._pdh is not None and self._query is not None:
                self._pdh.PdhCloseQuery(self._query)
        except Exception:
            pass
        self._pdh = None
        self._query = None
        self._counter = None

    def _pdh_read(self):
        try:
            pdh = self._pdh
            pdh.PdhCollectQueryData(self._query)
            count = ctypes.c_ulong(0)
            size = ctypes.c_ulong(0)
            fmt = self.PDH_FMT_DOUBLE | self.PDH_FMT_NOCAP100
            # первый вызов — узнаём размер буфера
            pdh.PdhGetFormattedCounterArrayW(
                self._counter, fmt, ctypes.byref(size), None)
            if size.value <= 0:
                return None
            buf = ctypes.create_string_buffer(size.value)
            if pdh.PdhGetFormattedCounterArrayW(
                    self._counter, fmt, ctypes.byref(count), buf) != 0:
                return None
            # PDH_FORMATTED_DATA_COUNTER_ITEM_W: DWORD dwName + PWSTR + DWORD
            # dwCStatus + union{...} — значение DOUBLE лежит после выравнивания
            ptr_size = ctypes.sizeof(ctypes.c_void_p)
            stride = 8 + ptr_size + 4 + 4 + 8
            total = 0.0
            used = 0
            for i in range(count.value):
                off = i * stride + 8 + ptr_size + 4 + 4
                if off + 8 > size.value:
                    break
                total += ctypes.c_double.from_buffer_copy(
                    buf.raw[off:off + 8]).value
                used += 1
            if used == 0:
                return None
            # сумма по всем движкам всех процессов; берём максимум как
            # «загрузку ГПУ», иначе число уезжает за 100
            return max(0.0, min(100.0, total))
        except Exception:
            return None

    # -- nvidia-smi -------------------------------------------------------
    @staticmethod
    def _nvidia_read():
        try:
            # Без CREATE_NO_WINDOW дочерний консольный процесс, запущенный
            # из GUI-приложения, на мгновение показывает своё окно.
            flags = 0
            if sys.platform == "win32":
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
                creationflags=flags)
            if out.returncode != 0:
                return None
            vals = []
            for line in (out.stdout or "").splitlines():
                line = line.strip()
                if line.isdigit():
                    vals.append(float(line))
            return max(vals) if vals else None
        except Exception:
            return None

    # -- публичный интерфейс ---------------------------------------------
    def read(self):
        if self._mode is None:
            self._mode = "pdh" if self._pdh_open() else "nvidia"
        if self._mode == "pdh":
            value = self._pdh_read()
            if value is not None:
                return clamp_percent(value)
            self._pdh_close()
            self._mode = "nvidia"
        if self._nv_broken:
            return None
        now = time.monotonic()
        if now - self._nv_at >= self.NVIDIA_POLL_S:
            self._nv_at = now
            self._nv_val = self._nvidia_read()
            # Если утилиты просто нет — не стучимся в неё каждые 2 секунды.
            if self._nv_val is None and shutil.which("nvidia-smi") is None:
                self._nv_broken = True
        return clamp_percent(self._nv_val)

    def close(self):
        self._pdh_close()
        self._mode = None


# ---------------------------------------------------------------------------
# Память
# ---------------------------------------------------------------------------

class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
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


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def read_process_rss():
    """RSS текущего процесса в байтах либо None."""
    try:
        if sys.platform == "win32":
            return _read_process_rss_windows()
        return _read_process_rss_unix()
    except Exception:
        return None


def _read_process_rss_windows():
    """GetProcessMemoryInfo с явными argtypes.

    Раньше вызов шёл без объявления типов, из-за чего на 64-битной
    Windows дескриптор и размер структуры передавались урезанными и
    функция возвращала ноль — индикатор памяти всегда был пустым.
    """
    counters = _PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
    handle = ctypes.windll.kernel32.GetCurrentProcess()

    for lib, fname in (("kernel32", "K32GetProcessMemoryInfo"),
                       ("psapi", "GetProcessMemoryInfo")):
        try:
            fn = getattr(ctypes.windll, lib).__getattr__(fname) \
                if hasattr(ctypes.windll, lib) else None
        except Exception:
            fn = None
        try:
            if fn is None:
                fn = getattr(getattr(ctypes, "WinDLL")(lib + ".dll"), fname)
            fn.argtypes = [ctypes.c_void_p,
                           ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
                           ctypes.c_ulong]
            fn.restype = ctypes.c_int
            if fn(ctypes.c_void_p(handle), ctypes.byref(counters),
                  counters.cb):
                return int(counters.WorkingSetSize)
        except Exception:
            continue
    return None


def _read_process_rss_unix():
    path = "/proc/self/statm"
    if os.path.exists(path):
        with open(path, "r", encoding="ascii", errors="replace") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    import resource  # noqa: F401
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return int(usage.ru_maxrss) * 1024


def read_system_memory():
    """Возвращает ``(всего, свободно)`` в байтах либо ``(None, None)``."""
    try:
        if sys.platform == "win32":
            st = _MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullTotalPhys), int(st.ullAvailPhys)
            return None, None
        with open("/proc/meminfo", "r", encoding="ascii") as f:
            info = {}
            for line in f:
                key, _, rest = line.partition(":")
                info[key.strip()] = int(rest.strip().split()[0]) * 1024
        return info.get("MemTotal"), info.get("MemAvailable")
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Сборка снимка
# ---------------------------------------------------------------------------

class SystemMonitor:
    """Хранит предыдущий замер и отдаёт готовый снимок для подписей."""

    def __init__(self):
        self._prev_times = None
        self._gpu = GpuUtilization()
        self.logical_cores, self.physical_cores = cpu_core_counts()

    def snapshot(self, cores_used=None):
        """Один замер. Первый вызов даёт cpu=None: нужен второй для дельты."""
        times = read_cpu_times()
        cpu = cpu_percent_from_times(self._prev_times, times)
        self._prev_times = times
        total, avail = read_system_memory()
        return {
            "cpu": cpu,
            "cpu_cores_used": cores_used,
            "cpu_cores_total": self.physical_cores or self.logical_cores,
            "gpu": self._gpu.read(),
            "rss": read_process_rss(),
            "mem_total": total,
            "mem_avail": avail,
        }

    @staticmethod
    def labels(snap):
        """Текст подписей по снимку."""
        return {
            "cpu": format_cpu_label(snap["cpu"], snap["cpu_cores_used"],
                                    snap["cpu_cores_total"]),
            "gpu": format_gpu_label(snap["gpu"]),
            "mem": format_memory_label(snap["rss"], snap["mem_total"],
                                       snap["mem_avail"]),
        }

    def close(self):
        self._gpu.close()
