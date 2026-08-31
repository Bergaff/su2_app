"""
solver/workers.py — НЕблокирующее исполнение SU2 и прогон сессий.

Патч: добавлен гибридный CPU+GPU запуск (без потери существующего
CPU-функционала). Использует helper-функции из solver.gpu_launcher.

Стратегия авто-фоллбэка (если compute_device == "cpu_gpu"):
  1. mpiexec -n N -gpu 0,1,...  (Microsoft MPI ≥10.0)
  2. mpiexec -n N + OMP_TARGET_OFFLOAD=MANDATORY  (OpenMP offload)
  3. mpiexec -n N  (чистый CPU — как раньше)

Для compute_device == "cpu" всё работает ровно как раньше.
"""

import os
import re
import sys
import time
import shutil
import subprocess
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition

from config.settings import config, MESH_FILE, WORK_DIR_BASE
from solver.config_builder import write_case_config
# === ПАТЧ: импорт помощников гибридного GPU-режима =====================
from solver.gpu_launcher import (
    build_hybrid_command,
    build_cpu_fallback_command,
    is_unknown_gpu_option,
    is_openmp_offload_unavailable,
    detect_mpi_implementation,
    gpu_config_args,
)
# ======================================================================

# === Автоконфиг SU2: устойчивые пресеты при расхождении ==============
try:
    import su2_autoconfig
except Exception as _e:
    su2_autoconfig = None
    print("[AeroOpt] su2_autoconfig не загружен:", _e)
try:
    import su2_config_dialog
except Exception as _e:
    su2_config_dialog = None
    print("[AeroOpt] su2_config_dialog не загружен:", _e)
# ======================================================================
# ======================================================================

MPIEXEC_EXE = "mpiexec"


# ---------------------------------------------------------------------------
# T2: SU2_PARTITION — mesh-декомпозиция для многоядерных расчётов
# ---------------------------------------------------------------------------
def find_su2_partition_exe() -> str:
    """Ищет исполняемый файл SU2_PARTITION (или SU2_PARTITION.exe).

    Источники поиска (по убыванию приоритета):
      1. config.su2_partition_exe (если задан вручную в Solver Settings)
      2. Каталог с config.su2_exe (часто SU2 идёт в комплекте с партиционером)
      3. %SU2_HOME% / $SU2_HOME
      4. shutil.which("SU2_PARTITION")
    """
    candidates: list = []
    explicit = getattr(config, "su2_partition_exe", None)
    if explicit:
        candidates.append(explicit)
    su2_dir = os.path.dirname(config.su2_exe) if getattr(config, "su2_exe", None) else ""
    if su2_dir:
        for name in ("SU2_PARTITION.exe", "SU2_PARTITION",
                     "su2_partition.exe", "su2_partition"):
            candidates.append(os.path.join(su2_dir, name))
    su2_home = os.environ.get("SU2_HOME") or os.environ.get("SU2_RUN")
    if su2_home:
        for sub in ("bin", ""):
            for name in ("SU2_PARTITION.exe", "SU2_PARTITION",
                         "su2_partition.exe", "su2_partition"):
                candidates.append(os.path.join(su2_home, sub, name))
    for path in candidates:
        try:
            if path and os.path.isfile(path):
                return os.path.abspath(path)
        except Exception:
            continue
    which = shutil.which("SU2_PARTITION") or shutil.which("SU2_PARTITION.exe")
    if which:
        return which
    return ""


def find_su2_adapt_exe() -> str:
    """Ищет исполняемый файл SU2_ADAPT (адаптация сетки по решению).

    SU2_ADAPT обычно лежит рядом с SU2_CFD.exe в каталоге bin.
    """
    candidates: list = []
    su2_dir = os.path.dirname(config.su2_exe) if getattr(config, "su2_exe", None) else ""
    if su2_dir:
        for name in ("SU2_ADAPT.exe", "SU2_ADAPT", "su2_adapt.exe", "su2_adapt"):
            candidates.append(os.path.join(su2_dir, name))
    su2_home = os.environ.get("SU2_HOME") or os.environ.get("SU2_RUN")
    if su2_home:
        for sub in ("bin", ""):
            for name in ("SU2_ADAPT.exe", "SU2_ADAPT", "su2_adapt.exe", "su2_adapt"):
                candidates.append(os.path.join(su2_home, sub, name))
    for path in candidates:
        try:
            if path and os.path.isfile(path):
                return os.path.abspath(path)
        except Exception:
            continue
    which = shutil.which("SU2_ADAPT") or shutil.which("SU2_ADAPT.exe")
    if which:
        return which
    return ""


def run_su2_adapt(case_dir: str, mesh_path: str, restart_path: str,
                  adapt_markers=("airfoil",), abs_error: float = 1e-6,
                  log_cb=None) -> str:
    """Запускает SU2_ADAPT: адаптивная перестройка сетки по решению.

    Требования:
      * SU2_ADAPT найден рядом с SU2_CFD.exe (иначе RuntimeError);
      * restart.dat — решение из завершённого расчёта той же сетки.

    Работает в каталоге case_dir (туда кладутся mesh.su2, restart.dat,
    адаптационный config.cfg). Результат — mesh_adapt.su2 в case_dir;
    возвращается путь к нему.
    """
    adapt_exe = find_su2_adapt_exe()
    if not adapt_exe:
        raise RuntimeError(
            "SU2_ADAPT не найден рядом с SU2_CFD.exe. "
            "Установите полный дистрибутив SU2 (с адаптивным модулем).")

    def _log(m):
        if log_cb:
            try:
                log_cb(m)
            except Exception:
                pass

    os.makedirs(case_dir, exist_ok=True)
    shutil.copy2(mesh_path, os.path.join(case_dir, "mesh.su2"))
    shutil.copy2(restart_path, os.path.join(case_dir, "restart.dat"))

    markers = [str(m).strip() for m in (adapt_markers or []) if str(m).strip()]
    if not markers:
        markers = ["airfoil"]

    cfg_path = os.path.join(case_dir, "adapt.cfg")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(
            "SOLVER= EULER\n"
            "MATH_PROBLEM= DIRECT\n"
            "MESH_FORMAT= SU2\n"
            "MESH_FILENAME= mesh.su2\n"
            "RESTART_FILENAME= restart.dat\n"
            "MESH_OUT_FILENAME= mesh_adapt.su2\n"
            f"MARKER_ADAPT= ( {' '.join(markers)} )\n"
            f"ADAPT_ABS_ERROR= {float(abs_error):g}\n"
            "ADAPT_BOUNDARY= YES\n"
            "ADAPT_NUM_ADAPT= 1\n"
            "ADAPT_STATISTICS= YES\n"
        )

    _log(f"SU2_ADAPT: {adapt_exe}")
    _log(f"   Сетка: mesh.su2, решение: restart.dat, маркеры: {markers}")
    try:
        proc = subprocess.run(
            [adapt_exe, "adapt.cfg"],
            cwd=case_dir,
            capture_output=True,
            text=True,
            encoding="utf-8", errors="replace",
            timeout=3600,
            **hidden_subprocess_kwargs(),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("SU2_ADAPT превысил таймаут 60 мин.") from None
    except Exception as e:
        raise RuntimeError(f"Не удалось запустить SU2_ADAPT: {e}") from e

    out_mesh = os.path.join(case_dir, "mesh_adapt.su2")
    if proc.returncode != 0 or not os.path.isfile(out_mesh):
        tail = (proc.stdout or "")[-1500:] + (proc.stderr or "")[-1500:]
        raise RuntimeError("SU2_ADAPT завершился ошибкой.\n" + tail)
    _log("Готово: Адаптация завершена: mesh_adapt.su2")
    return out_mesh


def _mesh_npoin(mesh_path: str):
    """Читает число точек из mesh.su2 (NMARK-формат). Возвращает int|None."""
    try:
        with open(mesh_path, "r", encoding="ascii", errors="ignore") as f:
            for ln in f:
                s = ln.strip()
                if s.startswith("NPOIN="):
                    try:
                        return int(s.split("=", 1)[1].strip())
                    except (ValueError, IndexError):
                        return None
                if s.startswith(("NELEM=", "NMARK=")):
                    continue
    except OSError:
        pass
    return None


def partition_mesh(case_dir: str, n_proc: int, log_cb=None) -> bool:
    """Запускает SU2_PARTITION для декомпозиции mesh.su2 на n_proc частей.

    Возвращает True, если декомпозиция прошла (или не нужна).
    Возвращает False, если упало — в этом случае SU2 попробует
    auto-partition на старте (медленнее, но работает).
    """
    log_cb = log_cb or (lambda m: None)
    if n_proc <= 1:
        return True
    mesh_path = os.path.join(case_dir, "mesh.su2")
    if not os.path.exists(mesh_path):
        log_cb(f"  Внимание: partition: нет mesh.su2 в {case_dir}")
        return False
    part_exe = find_su2_partition_exe()
    if not part_exe:
        log_cb(
            "  Внимание: SU2_PARTITION не найден — SU2 сделает auto-partition "
            "(медленнее на старте, но работает)."
        )
        return False
    mpiexec_exe = (config.mpiexec
                   if hasattr(config, "mpiexec") and config.mpiexec
                   else "mpiexec")
    if not shutil.which(mpiexec_exe):
        # На Windows mpiexec может не быть в PATH
        log_cb(
            "  Внимание: mpiexec не найден в PATH — пропускаем partition."
        )
        return False
    cmd = [mpiexec_exe, "-n", str(int(n_proc)), part_exe, mesh_path]
    log_cb(f"  Partition: {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd, cwd=case_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=120,
            **hidden_subprocess_kwargs(),
        )
        if proc.returncode != 0:
            log_cb(
                f"  Внимание: SU2_PARTITION завершился с кодом {proc.returncode}: "
                f"{(proc.stdout or '')[-300:]}"
            )
            return False
        # SU2_PARTITION создаёт mesh_*.su2 файлы. Проверим, что они есть.
        n_parts = sum(
            1 for f in os.listdir(case_dir)
            if f.startswith("mesh_") and f.endswith(".su2")
        )
        if n_parts < n_proc:
            log_cb(
                f"  Внимание: SU2_PARTITION создал только {n_parts}/{n_proc} частей — "
                f"SU2 попробует auto-partition."
            )
            return False
        log_cb(f"  Готово: Partition: {n_parts} частей")
        return True
    except subprocess.TimeoutExpired:
        log_cb("  Внимание: SU2_PARTITION превысил таймаут 120 с")
        return False
    except Exception as e:
        log_cb(f"  Внимание: SU2_PARTITION: {e}")
        return False

    def request_cores_change(self, cores: int):
        """Запрос на смену числа ядер. Применится при следующем mpiexec."""
        self._pending_cores = cores
        
    def _apply_pending_cores(self):
        """Применяет отложенную смену ядер."""
        if self._pending_cores is not None:
            self._cores = self._pending_cores
            self._pending_cores = None
            self.log(f"CPU cores changed to {self._cores} (applied for next phase)")
# ---------------------------------------------------------------------------
# ТЗ п.8 — скрытие «чёрного окна» консоли SU2 на Windows
# ---------------------------------------------------------------------------
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
STARTF_USESHOWWINDOW = 0x00000001
SW_HIDE = 0


def hidden_subprocess_kwargs() -> dict:
    if sys.platform != "win32":
        return {}
    si = subprocess.STARTUPINFO()
    si.dwFlags |= STARTF_USESHOWWINDOW
    si.wShowWindow = SW_HIDE
    return {"creationflags": CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
            "startupinfo": si}


# ---------------------------------------------------------------------------
# parse helpers
# ---------------------------------------------------------------------------
_SU2_ERROR_KEYWORDS = ("error", "exception", "cannot", "fail", "not found",
                       "invalid option", "appears twice")


def su2_log_gate(line, budget=0):
    """Решает, показывать ли строку вывода SU2 в логе приложения.

    Возвращает ``(показывать, новый_бюджет)``.

    SU2 печатает ошибку в три приёма: ``Error in "func":``, разделитель
    ``---- Error Exit ----`` и только ПОТОМ настоящую причину, например
    ``Line 52 TIME_DISCRETE_FLOW: invalid option name``. Раньше в лог
    попадали лишь строки со словом "error", поэтому причина терялась и
    приходилось гадать. Теперь после строки-признака следующие ``budget``
    строк печатаются как есть.
    """
    if budget > 0:
        return True, budget - 1
    low = (line or "").lower()
    if any(kw in low for kw in _SU2_ERROR_KEYWORDS):
        return True, 30
    return False, 0


_HISTORY_LINE = re.compile(r"^\s*(\d+)\s*\|\s*([-\deE.+]+)")


def parse_iteration_line(line: str):
    """Строка таблицы итераций SU2: |  123|  -3.45| ..."""
    cells = [c.strip() for c in line.split("|") if c.strip()]
    if len(cells) >= 2:
        try:
            return int(float(cells[0])), float(cells[1])
        except ValueError:
            return None
    return None


def parse_history(case_dir: str):
    """Читает итоговые CL/CD/CMz из history*.csv. Возвращает dict или None."""
    try:
        hist = None
        for name in os.listdir(case_dir):
            low = name.lower()
            if low.startswith("history") and low.endswith((".csv", ".dat")):
                hist = os.path.join(case_dir, name)
                break
        if not hist:
            return None
        with open(hist, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        header, data = None, []
        for i, ln in enumerate(lines):
            if "," in ln and re.search(r"[A-Za-z]", ln):
                header = [h.strip().strip('"').lower() for h in ln.split(",")]
                data = lines[i + 1:]
                break
        if header is None or not data:
            return None

        def col(*names):
            for nm in names:
                if nm in header:
                    return header.index(nm)
            return None

        icl = col("cl")
        icd = col("cd")
        icm = col("cmz", "cm", "cmy")
        if icl is None or icd is None:
            return None
        out = {"cl": None, "cd": None, "cm": 0.0, "iters": len(data)}
        for ln in reversed(data):
            parts = [p.strip().strip('"') for p in ln.split(",")]
            if len(parts) <= max(icl, icd):
                continue
            try:
                out["cl"] = float(parts[icl])
                out["cd"] = float(parts[icd])
                if icm is not None and len(parts) > icm:
                    out["cm"] = float(parts[icm])
                break
            except ValueError:
                continue
        if out["cl"] is None or out["cd"] is None:
            return None
        return out
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SU2Worker — исполнение одной расчётной точки в указанном каталоге
# ---------------------------------------------------------------------------
class SU2Worker:
    """Неблокирующий «исполнитель»; вызывается из потока раннера.

    Колбэки: log_cb(str), progress_cb(percent 0..100 точки).

    ПАТЧ: добавлены атрибуты compute_device ('cpu' | 'cpu_gpu') и
    gpu_percent (0..100). При compute_device='cpu' поведение
    полностью совпадает с прежним — никаких регрессий.
    """

    def __init__(self, case_dir: str, cpu_cores: int = 1,
                 log_cb=None, progress_cb=None,
                 compute_device: str = "cpu", gpu_percent: int = 0):
        self.case_dir = case_dir
        self.cpu_cores = max(1, int(cpu_cores))
        self.log_cb = log_cb or (lambda m: None)
        self.progress_cb = progress_cb or (lambda p: None)
        self._stop = False
        self._tail = []
        # === ПАТЧ: гибридный GPU-режим ===================================
        self.compute_device = str(compute_device or "cpu")
        self.gpu_percent = int(gpu_percent or 0)
        # Счётчик попыток для авто-фоллбэка: 0 = ещё не пробовали,
        # 1 = mpiexec -gpu не сработал, 2 = OpenMP offload не сработал.
        self._gpu_attempts = 0
        # =================================================================

    def stop(self):
        self._stop = True

    # ------------------------------------------------------------------
    def _result(self, aoa, ok, error_msg="", stopped=False):
        return {"aoa": aoa, "cl": 0.0, "cd": 0.0, "cm": 0.0,
                "error": (not ok) and (not stopped), "error_msg": error_msg,
                "stopped": stopped}

    def _tail_text(self, n=12):
        return "\n".join(self._tail[-n:])

    # ------------------------------------------------------------------
    def _build_cmd(self, exe: str) -> tuple:
        """Формирует (cmd, env_overlay) с учётом compute_device/gpu_percent.

        Для compute_device=='cpu' возвращает ту же команду, что и раньше:
            [mpiexec, '-n', N, exe, 'config.cfg']  (если cores>1 и есть mpiexec)
            [exe, 'config.cfg']                    (иначе — однопроцессный запуск)

        Для compute_device=='cpu_gpu':
            mpiexec -n N -gpu 0,1,... exe config.cfg + OMP_TARGET_OFFLOAD=MANDATORY
        """
        cfg_path_rel = "config.cfg"  # cwd = case_dir

        if self.compute_device == "cpu_gpu" and self.gpu_percent > 0:
            # Гибридный путь: ищем mpiexec. Если его нет — fallback на однопроцессный
            # запуск с OMP_TARGET_OFFLOAD (SU2 может использовать OpenMP offload
            # и без mpiexec, если собрана с -fopenmp).
            mpiexec_exe = shutil.which(config.mpiexec) if hasattr(config, "mpiexec") else None
            if mpiexec_exe and self.cpu_cores > 1:
                if self._gpu_attempts == 0:
                    cmd, env_overlay = build_hybrid_command(
                        mpiexec=mpiexec_exe,
                        n_proc=self.cpu_cores,
                        su2_exe=exe,
                        cfg_path=cfg_path_rel,
                        compute_device="cpu_gpu",
                        gpu_percent=self.gpu_percent,
                    )
                    return cmd, env_overlay, "mpiexec-gpu"
                else:
                    # Уже пробовали mpiexec-gpu и не вышло — OpenMP offload
                    cmd, env_overlay = build_cpu_fallback_command(
                        mpiexec=mpiexec_exe,
                        n_proc=self.cpu_cores,
                        su2_exe=exe,
                        cfg_path=cfg_path_rel,
                        use_openmp_offload=True,
                        gpu_percent=self.gpu_percent,
                    )
                    return cmd, env_overlay, "omp-offload"
            else:
                # Нет mpiexec (или cores==1) — однопроцессный запуск с OpenMP offload
                # SU2 в одном процессе сама выгрузит ядра на GPU при OMP_TARGET_OFFLOAD.
                env_overlay = {
                    "OMP_TARGET_OFFLOAD": "MANDATORY",
                }
                if shutil.which("nvidia-smi"):
                    env_overlay.setdefault("ACC_DEVICE_TYPE", "NVIDIA")
                if shutil.which("rocm-smi"):
                    env_overlay.setdefault("ACC_DEVICE_TYPE", "AMD")
                return [exe, cfg_path_rel], env_overlay, "omp-offload-single"

        # === СТАРОЕ ПОВЕДЕНИЕ (compute_device='cpu' / fallback) ==========
        use_mpi = self.cpu_cores > 1 and shutil.which(config.mpiexec) is not None \
            if hasattr(config, "mpiexec") else self.cpu_cores > 1
        if use_mpi:
            mpiexec_exe = config.mpiexec if hasattr(config, "mpiexec") else "mpiexec"
            return [mpiexec_exe, "-n", str(self.cpu_cores), exe, cfg_path_rel], {}, "cpu"
        return [exe, cfg_path_rel], {}, "cpu"

    # ------------------------------------------------------------------
    def run(self, aoa: float):
        exe = config.su2_exe
        cfg_path = os.path.join(self.case_dir, "config.cfg")
        mesh_path = os.path.join(self.case_dir, "mesh.su2")
        if not exe or not os.path.exists(exe):
            return self._result(aoa, False,
                f"Не найден SU2_CFD:\n{exe}\n\n"
                "Укажите путь в Solver Settings или установите SU2 автоматически.")
        if not os.path.exists(cfg_path):
            return self._result(aoa, False, "Отсутствует config.cfg в каталоге расчёта.")
        if not os.path.exists(mesh_path):
            return self._result(aoa, False,
                "Отсутствует mesh.su2. Постройте расчётную сетку заново.")

        # Проверяем config.cfg ДО запуска SU2. Иначе SU2 падает с
        # «Error in TokenizeString(): ... no "=" sign», а в логе это
        # выглядит как загадочный обрыв без указания строки.
        try:
            import su2_autoconfig as _ac
            _ok, _problems = _ac.validate_config(cfg_path)
        except Exception:
            _ok, _problems = True, []
        if not _ok:
            _lines = "\n".join(f"   стр. {n}: {txt!r}  <- {why}"
                                for n, txt, why in _problems[:5])
            return self._result(
                aoa, False,
                "config.cfg нечитаем для SU2 — запуск отменён.\n\n"
                f"{_lines}\n\n"
                "SU2 понимает как комментарий только '%', а не '#': любая "
                "строка без '=' роняет решатель. Исправьте файл или "
                "восстановите config.cfg.orig.")

        cmd, env_overlay, launch_mode = self._build_cmd(exe)

        # === ПАТЧ: лог GPU-режима =========================================
        if launch_mode != "cpu":
            self.log_cb(
                f"Гибридный режим ({launch_mode}): GPU {self.gpu_percent}%, "
                f"ядер CPU {self.cpu_cores}"
            )
            # Пишем в лог подсказку, какие строки можно добавить в config.cfg
            # при ручной настройке (но фактически cfg не меняем).
            for line in gpu_config_args(self.compute_device, self.gpu_percent):
                self.log_cb(f"  {line}")
            # Детектим MPI один раз — для информационного лога
            mpi_impl = detect_mpi_implementation()
            if mpi_impl != "unknown":
                self.log_cb(f"  MPI: {mpi_impl}")
        # =================================================================

        self.log_cb(f"[{datetime.now().strftime('%H:%M:%S')}] Запуск: {' '.join(cmd)} "
                    f"(cwd={self.case_dir})")

        try:
            proc = subprocess.Popen(
                cmd, cwd=self.case_dir,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, errors="replace",
                env={**os.environ, **env_overlay},
                **hidden_subprocess_kwargs())
        except Exception as e:
            return self._result(aoa, False, f"Не удалось запустить SU2: {e}")

        max_iter = 300
        try:
            with open(cfg_path, "r", encoding="utf-8", errors="ignore") as f:
                m = re.search(
                    r"^\s*(?:INNER_ITER|ITER)\s*=\s*(\d+)",
                    f.read(),
                    re.M,
                )
                if m:
                    max_iter = int(m.group(1)) or 300
        except Exception:
            pass

        nan_count = 0
        last_rms = None
        # === ПАТЧ: для авто-фоллбэка собираем весь вывод ==================
        all_output_lines: list = []
        err_mode = 0
        # =================================================================

        while True:
            if self._stop:
                proc.terminate()
                return self._result(aoa, False, stopped=True)
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if not line:
                time.sleep(0.05)
                continue
            line = line.rstrip()
            if line:
                self._tail.append(line)
                if len(self._tail) > 200:
                    self._tail = self._tail[-200:]
                # === ПАТЧ: копим вывод для детекта ошибок MPI/SU2 =========
                all_output_lines.append(line)
                if len(all_output_lines) > 1000:
                    all_output_lines = all_output_lines[-1000:]
                # =========================================================

            show, err_mode = su2_log_gate(line, err_mode)
            if show:
                self.log_cb(f"  SU2: {line}")

            parsed = parse_iteration_line(line)
            if parsed:
                it, rms = parsed
                last_rms = rms
                if rms != rms:
                    nan_count += 1
                    if nan_count >= 3:
                        proc.terminate()
                        return self._result(
                            aoa, False,
                            "Решение разошлось (NaN в невязках).\n"
                            "Попробуйте снизить CFL или сделать сетку грубее.\n\n"
                            f"Хвост лога:\n{self._tail_text()}")
                self.progress_cb(max(0, min(99, int(round(100.0 * it / max_iter)))))
                if it % max(max_iter // 4, 1) == 0:
                    self.log_cb(f"  итерация {it}/{max_iter}, log10(res)={rms:.3f}")

        rc = proc.wait()
        self.progress_cb(100)
        if self._stop:
            return self._result(aoa, False, stopped=True)

        # === ПАТЧ: авто-фоллбэк для GPU-режима ============================
        if rc != 0 and launch_mode == "mpiexec-gpu":
            full_output = "\n".join(all_output_lines)
            if is_unknown_gpu_option(full_output):
                self.log_cb(
                    "Внимание: mpiexec не поддерживает опцию -gpu → фоллбэк "
                    "на OpenMP target offload."
                )
                self._gpu_attempts = 1
                return self.run(aoa)  # перезапуск с omp-offload
        if rc != 0 and launch_mode == "omp-offload" and self.compute_device == "cpu_gpu":
            full_output = "\n".join(all_output_lines)
            if is_openmp_offload_unavailable(full_output):
                self.log_cb(
                    "Внимание: SU2 собрана без GPU-поддержки (CUDA/HIP) → фоллбэк "
                    "на чистый CPU. Гибридный режим не активен."
                )
                self._gpu_attempts = 2
                self.compute_device = "cpu"  # дальше идём как обычный CPU
                return self.run(aoa)
        # =================================================================

        if rc != 0:
            return self._result(aoa, False,
                f"SU2 завершился с кодом ошибки {rc}.\n\n"
                f"Хвост лога:\n{self._tail_text()}")

        hist = parse_history(self.case_dir)
        if hist is None:
            return self._result(aoa, False,
                "SU2 отработал, но history не содержит CL/CD.\n"
                "Возможные причины: в mesh.su2 нет маркера 'airfoil', "
                "решение разошлось, итераций слишком мало.\n\n"
                f"Хвост лога:\n{self._tail_text()}")
        if abs(hist["cl"]) < 1e-12 and abs(hist["cd"]) < 1e-12:
            return self._result(aoa, False,
                "SU2 вернул нулевые CL/CD: силы не посчитаны.\n"
                "Проверьте, что тело самолёта нанесено на границу 'airfoil' "
                "и геометрия замкнута.")

        res = self._result(aoa, True)
        res.update({"cl": hist["cl"], "cd": hist["cd"], "cm": hist["cm"]})
        return res


# ---------------------------------------------------------------------------
# SessionRunner — последовательный прогон точек сессии (single/sweep)
# ---------------------------------------------------------------------------
class SessionRunner(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)          # общий прогресс 0..100
    result_ready = pyqtSignal(object)          # dict результата
    paused_signal = pyqtSignal()
    finished_all = pyqtSignal()
    # Автоконфиг: поток расчёта шлёт запрос, диалог показывается в GUI-потоке
    recovery_signal = pyqtSignal(object)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._pause_requested = False
        self._cancel_requested = False
        self._worker = None
        # Автоконфиг
        self._recovery_declined = False   # пользователь отказался — больше не спрашивать
        self._auto_preset = None          # 'safe'|'ultra' — применять ко всем точкам серии
        self._recovery_mutex = QMutex()
        self._recovery_cond = QWaitCondition()
        self._recovery_answer = None
        # Слот-приёмник живёт в главном потоке (объект QThread создан там),
        # поэтому модальный диалог внутри него безопасен.
        self.recovery_signal.connect(self._handle_recovery_in_gui)

    # ------------------------------------------------------------------
    def request_pause(self):
        self._pause_requested = True
        if self._worker is not None:
            self._worker.stop()

    def request_cancel(self):
        self._cancel_requested = True
        if self._worker is not None:
            self._worker.stop()

    # ------------------------------------------------------------------
    # Автоконфиг: предложение устойчивого пресета после расхождения
    # ------------------------------------------------------------------
    def _recover(self, case_dir, res, log_cb):
        """Вызывается из потока расчёта. Определяет причину провала по
        history.csv и ждёт ответ из GUI-диалога.
        Возвращает 'rerun_safe'|'rerun_ultra'|'settings'|'abort'."""
        text = res.get("error_msg", "") or ""
        if su2_autoconfig is None:
            return "abort"
        det = su2_autoconfig.detect_result(case_dir, screen_text=text)
        if det["status"] == "converged":
            return "abort"
        log_cb(f"Внимание: {det.get('detail', 'Расчёт не сошёлся.')}")

        # Если диалоговый модуль недоступен — безопасный авто-пресет без вопросов
        if su2_config_dialog is None:
            try:
                su2_autoconfig.apply_preset(
                    os.path.join(case_dir, "config.cfg"), "safe")
                log_cb("Автоматически применён устойчивый пресет 'safe'.")
                return "rerun_safe"
            except Exception as e:
                log_cb(f"Внимание: Автоконфиг недоступен: {e}")
                return "abort"

        self._recovery_answer = None
        self.recovery_signal.emit(
            {"case_dir": case_dir, "text": text})
        self._recovery_mutex.lock()
        if self._recovery_answer is None:
            self._recovery_cond.wait(self._recovery_mutex, 600000)  # 10 мин на ответ
        ans = self._recovery_answer
        self._recovery_mutex.unlock()
        return ans or "abort"

    def provide_recovery_answer(self, answer):
        """Вызывается из GUI-потока после выбора пользователя."""
        self._recovery_mutex.lock()
        self._recovery_answer = answer
        self._recovery_cond.wakeAll()
        self._recovery_mutex.unlock()

    def _handle_recovery_in_gui(self, payload):
        """Слот в ГЛАВНОМ потоке: здесь можно показывать модальный диалог."""
        try:
            if su2_config_dialog is not None:
                verdict = su2_config_dialog.offer_recovery_after_failure(
                    None, payload["case_dir"], payload.get("text", ""))
            else:
                verdict = "abort"
        except Exception:
            verdict = "abort"
        self.provide_recovery_answer(verdict)

    # ------------------------------------------------------------------
    def run(self):
        sess = self.session
        total = max(1, len(sess.aoa_list))
        sess.paused = False
        # === ПАТЧ: вычислитель/GPU берём из session =====================
        compute_device = getattr(sess, "compute_device", "cpu")
        gpu_percent = int(getattr(sess, "gpu_percent", 0) or 0)
        if compute_device == "cpu_gpu":
            self.log_signal.emit(
                f"Сессия в гибридном режиме: GPU {gpu_percent}%, "
                f"ядер CPU {sess.cpu_cores}"
            )
        # =================================================================
        for idx in range(sess.next_index, len(sess.aoa_list)):
            if self._pause_requested or self._cancel_requested:
                sess.save()
                if self._pause_requested:
                    sess.mark_paused()
                    self.paused_signal.emit()
                return
            aoa = sess.aoa_list[idx]
            sess.current_index = idx
            sess.save()
            case_dir = sess.case_dir_for(idx)
            local_ref = {"p": 0}

            def log_cb(m):
                self.log_signal.emit(m)

            def progress_cb(p):
                local_ref["p"] = p
                overall = int(100.0 * (idx + p / 100.0) / total)
                self.progress_signal.emit(max(0, min(100, overall)))

            ok = self._prepare_case(case_dir, aoa, sess, log_cb)
            if ok:
                # Если пользователь ранее выбрал устойчивый пресет —
                # применяем его и ко всем последующим точкам серии.
                if self._auto_preset and su2_autoconfig is not None:
                    try:
                        su2_autoconfig.apply_preset(
                            os.path.join(case_dir, "config.cfg"),
                            self._auto_preset)
                        log_cb(f"К точке применён устойчивый пресет "
                               f"'{self._auto_preset}'.")
                    except Exception as e:
                        log_cb(f"Внимание: Не удалось применить пресет: {e}")

                def _make_worker():
                    return SU2Worker(
                        case_dir, sess.cpu_cores, log_cb, progress_cb,
                        compute_device=compute_device,
                        gpu_percent=gpu_percent,
                    )

                self._worker = _make_worker()
                res = self._worker.run(aoa)

                # Автоконфиг: до двух попыток (safe, затем ultra)
                for _attempt in range(2):
                    if not (res.get("error") and not res.get("stopped")):
                        break
                    if self._recovery_declined or su2_autoconfig is None:
                        break
                    verdict = self._recover(case_dir, res, log_cb)
                    if verdict in ("rerun_safe", "rerun_ultra", "settings"):
                        if verdict == "rerun_ultra":
                            self._auto_preset = "ultra"
                        elif verdict == "rerun_safe":
                            self._auto_preset = "safe"
                        log_cb("Повторный расчёт точки с новыми настройками...")
                        self._worker = _make_worker()
                        res = self._worker.run(aoa)
                    else:
                        self._recovery_declined = True
                        log_cb("Автоконфиг отклонён — продолжаю серию "
                               "без повторных вопросов.")
                        break
            else:
                res = {"aoa": aoa, "cl": 0.0, "cd": 0.0, "cm": 0.0,
                       "error": True,
                       "error_msg": "Не удалось подготовить каталог расчёта.",
                       "stopped": False}

            self._worker = None
            if self._pause_requested or self._cancel_requested or res.get("stopped"):
                sess.save()
                if self._pause_requested and not self._cancel_requested:
                    sess.mark_paused()
                    self.paused_signal.emit()
                return

            sess.mark_point_processed(idx)
            sess.results.append(res)
            sess.save()
            if res.get("error") and "Не найден SU2_CFD" in res.get("error_msg", ""):
                self.result_ready.emit(res)
                self.log_signal.emit("Серия прервана: нет исполняемого SU2.")
                self.finished_all.emit()
                return
            self.result_ready.emit(res)

        sess.mark_finished()
        self.progress_signal.emit(100)
        self.finished_all.emit()

    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_case(case_dir, aoa, sess, log_cb) -> bool:
        try:
            os.makedirs(case_dir, exist_ok=True)
            if os.path.exists(MESH_FILE):
                shutil.copy2(MESH_FILE, os.path.join(case_dir, "mesh.su2"))
            write_case_config(case_dir, aoa, sess)
            # === T2: mesh-декомпозиция для многоядерных расчётов ==========
            # Проверяем флаг use_partition — пользователь мог отключить его
            # в Solver Settings (для маленьких сеток partition не нужен).
            use_partition = bool(getattr(sess, "use_partition", True))
            n_proc = int(getattr(sess, "cpu_cores", 1) or 1)
            if n_proc > 1 and use_partition:
                partition_mesh(case_dir, n_proc, log_cb)
            elif n_proc > 1 and not use_partition:
                log_cb(
                    "  Mesh partition отключён пользователем — "
                    "SU2 сделает auto-partition на старте."
                )
            # ==============================================================
            return True
        except Exception as e:
            log_cb(f"Ошибка: Подготовка каталога {case_dir}: {e}")
            return False


class SweepWorker(SessionRunner):
    """Совместимость: прогон поляры (sweep) — тот же SessionRunner."""
    pass


# ---------------------------------------------------------------------------
# OptimizationWorker — многоточечная геометрическая оптимизация
# ---------------------------------------------------------------------------
class OptimizationWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    opt_finished = pyqtSignal(object)             # dict лучшего кандидата
    update_geometry_signal = pyqtSignal(object)   # dict параметров → GUI перестраивает крыло
    variant_ready = pyqtSignal(object)            # dict результата одного варианта (DOE)

    def __init__(self, target_cl, target_k, physics, solver, initial_params,
                 rule_set, flight_points, ref_data, body_markers, parent=None,
                 candidates=None, cpu_cores=1, use_symmetry=False,
                 symmetry_planes=None):
        super().__init__(parent)
        self.target_cl = target_cl
        self.target_k = target_k
        self.physics = physics
        self.solver = solver
        self.initial_params = dict(initial_params)
        self.rule_set = rule_set
        self.flight_points = flight_points
        self.ref_data = ref_data
        self.body_markers = body_markers
        # ТЗ №1 (многоядерность) и №2 (плоскость симметрии): оптимизация
        # раньше всегда считала в одно ядро и без симметрии — теперь
        # настройки обычного расчёта пробрасываются и сюда.
        self.cpu_cores = max(1, int(cpu_cores or 1))
        self.use_symmetry = bool(use_symmetry)
        self.symmetry_planes = (list(symmetry_planes) if symmetry_planes
                                else (["xz"] if self.use_symmetry else None))
        self._stop = False
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        self._geom_ready = False
        self.n_iterations = 6
        # Табличная оптимизация (DOE): если передан список кандидатов,
        # перебираем его вместо случайного поиска.
        self.candidates = list(candidates) if candidates else None

    # ------------------------------------------------------------------
    def stop(self):
        self._stop = True
        self._geom_ready = True
        self._cond.wakeAll()

    def geometry_ready(self):
        self._mutex.lock()
        self._geom_ready = True
        self._cond.wakeAll()
        self._mutex.unlock()

    def _wait_geometry(self, timeout_ms=120000) -> bool:
        self._mutex.lock()
        ok = True
        if not self._geom_ready:
            ok = self._cond.wait(self._mutex, timeout_ms)
        self._geom_ready = False
        self._mutex.unlock()
        return ok and not self._stop

    # ------------------------------------------------------------------
    def _candidate(self, it: int) -> dict:
        base = self.initial_params
        rng = __import__("random").Random(it * 7919 + 13)
        cand = dict(base)
        if it == 0:
            return cand
        scale = 1.0 + rng.uniform(-0.06, 0.06)
        shrink = 1.0 - 0.03 * min(it, 4)
        cand["span"] = round(max(base.get("span", 8.0) * scale, 0.5), 3)
        cand["chord_root"] = round(max(base.get("chord_root", 1.5), 0.05), 3)
        cand["chord_tip"] = round(max(base.get("chord_tip", 0.8) * shrink, 0.05), 3)
        if cand["chord_tip"] > cand["chord_root"]:
            cand["chord_tip"] = cand["chord_root"]
        cand["sweep"] = round(base.get("sweep", 10.0) + rng.uniform(-5.0, 5.0), 2)
        for key in ("flap_deflection", "flap_slide", "flap_hinge_depth",
                    "slat_deflection", "slat_slide"):
            if key in base:
                cand[key] = round(base[key] * (1.0 + rng.uniform(-0.1, 0.1)), 4)
        if "flap_deflection" in cand:
            cand["flap_deflection"] = min(max(cand["flap_deflection"], 0.0), 45.0)
        if "slat_deflection" in cand:
            cand["slat_deflection"] = min(max(cand["slat_deflection"], 0.0), 30.0)
        return cand

    # ------------------------------------------------------------------
    def _enabled_symmetry_planes(self, mesh_path: str) -> list:
        """Плоскости симметрии, которые действительно есть в сетке.

        MARKER_SYM с несуществующим маркером роняет SU2, поэтому каждая
        заявленная плоскость проверяется по ``mesh.su2`` (как это делает
        :func:`solver.config_builder.write_case_config` для обычного
        расчёта).
        """
        if not self.symmetry_planes:
            return []
        try:
            from solver.config_builder import _mesh_has_marker
        except Exception:
            return []
        tags = {"xz": ("symmetry_plane", "symmetry_xz"),
                "xy": ("symmetry_xy",), "yz": ("symmetry_yz",)}
        out = []
        for p in self.symmetry_planes:
            p = str(p).lower()
            if any(_mesh_has_marker(mesh_path, t) for t in tags.get(p, ())):
                out.append(p)
        return out

    # ------------------------------------------------------------------
    def _evaluate(self, cand: dict) -> dict:
        cl_w = cd_w = w_sum = 0.0
        results = []
        for p_i, fp in enumerate(self.flight_points):
            if self._stop:
                break
            case_dir = os.path.join(
                WORK_DIR_BASE, f"OPT_P{id(cand) % 1000}_{p_i}")
            os.makedirs(case_dir, exist_ok=True)
            try:
                mesh_dst = os.path.join(case_dir, "mesh.su2")
                if os.path.exists(MESH_FILE):
                    shutil.copy2(MESH_FILE, mesh_dst)
                from solver.config_builder import build_su2_config
                planes = self._enabled_symmetry_planes(mesh_dst)
                text = build_su2_config(
                    fp.aoa, self.physics, self.solver, self.ref_data,
                    mesh_quality=getattr(self, "mesh_quality", None),
                    use_symmetry=bool(planes),
                    symmetry_planes=planes,
                )
                with open(os.path.join(case_dir, "config.cfg"), "w",
                          encoding="utf-8") as f:
                    f.write(text)
            except Exception as e:
                self.log_signal.emit(f"Внимание: Кейс оптимизации: {e}")
                continue
            # === ПАТЧ: пробрасываем compute_device/gpu_percent в оценку ===
            compute_device = getattr(self, "compute_device", "cpu")
            gpu_percent = int(getattr(self, "gpu_percent", 0) or 0)
            worker = SU2Worker(case_dir, self.cpu_cores, None, None,
                               compute_device=compute_device,
                               gpu_percent=gpu_percent)
            # =============================================================
            res = worker.run(fp.aoa)
            results.append(res)
            if not res.get("error"):
                w = getattr(fp, "weight", 1.0)
                cl_w += w * res["cl"]
                cd_w += w * max(res["cd"], 1e-9)
                w_sum += w
        if self._stop or w_sum <= 0:
            return {"ok": False, "results": results}
        cl_avg = cl_w / w_sum
        cd_avg = max(cd_w / w_sum, 1e-9)
        return {"ok": True, "cl_weighted": cl_avg,
                "k_weighted": cl_avg / cd_avg, "results": results}

    # ------------------------------------------------------------------
    def run(self):
        import time as _time
        best = {"cl_weighted": 0.0, "k_weighted": 0.0}
        best_score = -1e18
        t0 = _time.time()
        if self.candidates is not None:
            cands = self.candidates
            mode = "табличный перебор"
        else:
            cands = [self._candidate(it) for it in range(self.n_iterations)]
            mode = "случайный поиск"
        n_total = max(1, len(cands))
        self.log_signal.emit(f"Старт оптимизации ({mode}): {n_total} кандидатов, "
                             f"точек на кандидата: {len(self.flight_points)}")
        for it, cand in enumerate(cands):
            if self._stop:
                break
            self.log_signal.emit(f"→ Кандидат #{it + 1}: span={cand.get('span')}, "
                                 f"cr={cand.get('chord_root')}, ct={cand.get('chord_tip')}, "
                                 f"sweep={cand.get('sweep')}")
            self.update_geometry_signal.emit(dict(cand))
            if not self._wait_geometry():
                self.log_signal.emit("Внимание: Перестройка геометрии прервана.")
                break
            ev = self._evaluate(cand)
            params_for_rules = dict(cand)
            try:
                span = max(cand.get("span", 1.0), 1e-6)
                cr = max(cand.get("chord_root", 1.0), 1e-6)
                ct = max(cand.get("chord_tip", 0.5), 1e-6)
                params_for_rules["aspect_ratio"] = span * 2 / (cr + ct)
                params_for_rules["taper_ratio"] = ct / cr
                params_for_rules["area"] = 0.5 * (cr + ct) * span
                params_for_rules["mach"] = self.physics.get("mach", 0.0)
                params_for_rules["cl"] = ev.get("cl_weighted", 0.0)
                params_for_rules["k"] = ev.get("k_weighted", 0.0)
            except Exception:
                pass

            penalty = 0.0
            hard_block = False
            if self.rule_set is not None:
                rep = self.rule_set.check_all(params_for_rules)
                penalty = rep["penalty"]
                hard_block = not rep["passed"]

            if ev.get("ok") and not hard_block:
                score = (ev["k_weighted"] - penalty
                         - max(0.0, self.target_cl - ev["cl_weighted"]) * 10.0)
                self.log_signal.emit(
                    f"  кандидат #{it + 1}: Cl={ev['cl_weighted']:.4f}, "
                    f"K={ev['k_weighted']:.2f}, штраф={penalty:.3f}, score={score:.3f}")
                if score > best_score:
                    best_score = score
                    best = dict(cand)
                    best["cl_weighted"] = ev["cl_weighted"]
                    best["k_weighted"] = ev["k_weighted"]
            else:
                reason = "жёсткое правило" if hard_block else "нет расчёта"
                self.log_signal.emit(f"  кандидат #{it + 1} отклонён: {reason}.")

            # Результат варианта — для таблицы DOE в GUI
            try:
                self.variant_ready.emit({
                    "index": it,
                    "params": dict(cand),
                    "ok": bool(ev.get("ok")) and not hard_block,
                    "cl_weighted": ev.get("cl_weighted", 0.0),
                    "k_weighted": ev.get("k_weighted", 0.0),
                    "rejected_reason": reason if (not ev.get("ok") or hard_block) else "",
                })
            except Exception:
                pass

            pct = int(100.0 * (it + 1) / n_total)
            self.progress_signal.emit(pct)

        if self._stop:
            self.log_signal.emit("Оптимизация остановлена пользователем.")
        else:
            dt = _time.time() - t0
            self.log_signal.emit(f"Оптимизация завершена за {dt:.0f} сек. "
                                 f"Лучший K={best.get('k_weighted', 0):.2f}")
        self.opt_finished.emit(best if best_score > -1e17 else None)
