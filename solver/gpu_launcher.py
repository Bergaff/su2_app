# solver/gpu_launcher.py
"""Помощники для гибридного (CPU + GPU) запуска SU2.

Файл не запускается отдельно — это набор функций, которые импортируются
из ``solver.workers.SU2Worker`` (или ``SessionRunner._run_case``).

Использование из workers.py (точечная правка, ничего не удаляем):

    from solver.gpu_launcher import (
        build_hybrid_command, hybrid_env, gpu_config_args,
    )

    # в SU2Worker.run() / _run_case() — вместо:
    #   cmd = [mpiexec, "-n", str(n), su2_exe, "config.cfg"]
    # пишем:
    cmd, env_overlay = build_hybrid_command(
        mpiexec=mpiexec,
        n_proc=n,
        su2_exe=su2_exe,
        cfg_path=cfg_path,
        compute_device=session.compute_device,   # "cpu" | "cpu_gpu"
        gpu_percent=session.gpu_percent,         # 0..100
    )
    proc = subprocess.Popen(cmd, env={**os.environ, **env_overlay}, ...)

Возможности:
    1. ``build_hybrid_command`` — собирает итоговую команду. Стратегия:
       * "cpu"      → классический mpiexec -n N (как раньше).
       * "cpu_gpu"  → пытаемся ``mpiexec -gpu``. Если процесс сразу
         упал с «unknown option -gpu» — авто-фоллбэк на
         ``OMP_TARGET_OFFLOAD=MANDATORY`` (OpenMP offload). Если и
         это не сработало — на чистый CPU (как раньше).
    2. ``hybrid_env`` — словарь env-переменных (OMP_TARGET_OFFLOAD,
       OMP_NUM_DEVICES, ACC_DEVICE_TYPE) для OpenMP-фоллбэка.
    3. ``gpu_config_args`` — список строк, которые НЕ пишутся в
       config.cfg, а просто логируются (если пользователь захочет
       потом добавить их в config.cfg вручную).
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
from typing import List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Детекция MPI-реализации
# ---------------------------------------------------------------------------
def detect_mpi_implementation() -> str:
    """Возвращает 'msmpi' | 'openmpi' | 'mpich' | 'unknown'.

    Используется только для информационного лога; команда запуска
    формируется одинаково — через mpiexec.
    """
    mpiexec = shutil.which("mpiexec") or shutil.which("mpiexec.exe")
    if not mpiexec:
        return "unknown"
    try:
        r = subprocess.run(
            [mpiexec, "--version"],
            capture_output=True, text=True, timeout=5,
        )
        out = (r.stdout or "") + (r.stderr or "")
        out_l = out.lower()
        if "microsoft mpi" in out_l or "ms-mpi" in out_l:
            return "msmpi"
        if "open mpi" in out_l or "open-mpi" in out_l:
            return "openmpi"
        if "mpich" in out_l:
            return "mpich"
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Словарь env для OpenMP target offload
# ---------------------------------------------------------------------------
def hybrid_env(gpu_percent: int = 50,
               num_devices: Optional[int] = None) -> dict:
    """Возвращает переменные окружения для OpenMP target offload.

    Параметры:
        gpu_percent  — 0..100, чисто информационно; для OpenMP offload
                       точная доля GPU регулируется через
                       OMP_TARGET_OFFLOAD=MANDATORY + расписание
                       внутри SU2.
        num_devices  — сколько GPU задействовать (None = все доступные).
    """
    env = {
        # Заставляет компилятор OpenMP реально выгружать ядра на GPU
        "OMP_TARGET_OFFLOAD": "MANDATORY",
    }
    if num_devices is not None and num_devices > 0:
        env["OMP_NUM_DEVICES"] = str(int(num_devices))
    # Если у пользователя стоит AMD ROCm — подсказываем OpenACC устройство.
    # Сама SU2 при сборке с HIP/OpenMP target это использует.
    if shutil.which("rocm-smi") and "ACC_DEVICE_TYPE" not in os.environ:
        env["ACC_DEVICE_TYPE"] = "AMD"
    if shutil.which("nvidia-smi") and "ACC_DEVICE_TYPE" not in os.environ:
        # Не перебиваем AMD, если оба доступны — пользователь решит сам.
        env.setdefault("ACC_DEVICE_TYPE", "NVIDIA")
    return env


# ---------------------------------------------------------------------------
# Опции для config.cfg (НЕ пишем в файл, только логируем)
# ---------------------------------------------------------------------------
def gpu_config_args(compute_device: str,
                    gpu_percent: int = 50) -> List[str]:
    """Список строк, которые можно добавить в config.cfg, если SU2
    собран с поддержкой GPU (--enable-cuda / --enable-hip).

    По умолчанию НЕ используем — config.cfg остаётся без изменений,
    гибридный режим активируется через mpiexec/env.
    Возвращаемый список — для логирования.
    """
    if compute_device != "cpu_gpu":
        return []
    return [
        "% GPU_PLATFORM= CUDA        # раскомментируйте, если SU2 собран с CUDA",
        "% GPU_DEVICE= 0            # индекс GPU (0..N-1)",
        f"% GPU_COMPUTE_PERCENT= {int(gpu_percent)}",
    ]


# ---------------------------------------------------------------------------
# Главная функция: формирование команды запуска
# ---------------------------------------------------------------------------
def build_hybrid_command(
    mpiexec: str,
    n_proc: int,
    su2_exe: str,
    cfg_path: str,
    compute_device: str = "cpu",
    gpu_percent: int = 0,
    gpu_device_ids: Optional[Sequence[int]] = None,
) -> Tuple[List[str], dict]:
    """Собирает (cmd, env_overlay) для запуска SU2.

    Возвращает:
        cmd          — список аргументов (готов для subprocess.Popen).
        env_overlay  — словарь env, который мерджится с os.environ.

    Стратегия (compute_device == "cpu_gpu"):
        1. mpiexec -n N -gpu 0,1,... su2_exe config.cfg
           (поддерживается Microsoft MPI >= 10.0; OpenMPI игнорирует -gpu)
        2. fallback: mpiexec -n N su2_exe config.cfg + OMP_TARGET_OFFLOAD=MANDATORY
        3. fallback: mpiexec -n N su2_exe config.cfg (как раньше)
    """
    cmd: List[str] = []
    env_overlay: dict = {}

    # --- Базовый случай: чистый CPU (как раньше, ничего не сломано) ------
    if compute_device == "cpu" or gpu_percent <= 0:
        cmd = [mpiexec, "-n", str(int(n_proc)), su2_exe, cfg_path]
        return cmd, env_overlay

    # --- GPU-режим --------------------------------------------------------
    # 1. Пытаемся -gpu (msmpi). Неизвестно, поддерживает ли MPI — поэтому
    #    не тестируем заранее: если опция не поддерживается, mpiexec
    #    напишет "unrecognized option -gpu" в stderr и вернёт код 7.
    #    Это мы обработаем в SU2Worker.run() и сделаем авто-фоллбэк.
    if gpu_device_ids is None:
        # По умолчанию: берём все 0..gpu_count-1, если их <=8, иначе только 0
        # (больше 8 GPU — экзотика, и mpiexec может не любить длинный список).
        gpu_device_ids = list(range(0, min(8, max(1, n_proc))))
    gpu_csv = ",".join(str(int(i)) for i in gpu_device_ids)

    cmd_gpu = [
        mpiexec, "-n", str(int(n_proc)),
        "-gpu", gpu_csv,
        su2_exe, cfg_path,
    ]
    env_overlay = hybrid_env(gpu_percent=gpu_percent,
                             num_devices=len(gpu_device_ids))
    # cmd_gpu — основной; если mpiexec не поддерживает -gpu, SU2Worker
    # обнаружит это в stderr и перезапустит с cmd_cpu_offload ниже.
    # Чтобы не усложнять публичный API, возвращаем «лучшую попытку»
    # и тегируем через env_overlay. Прямо cmd_cpu_offload не возвращаем
    # — иначе было бы два вызова subprocess, а нам нужно один.
    cmd = cmd_gpu
    return cmd, env_overlay


def build_cpu_fallback_command(
    mpiexec: str,
    n_proc: int,
    su2_exe: str,
    cfg_path: str,
    use_openmp_offload: bool = True,
    gpu_percent: int = 50,
) -> Tuple[List[str], dict]:
    """Команда фоллбэка: OpenMP offload или чистый CPU.

    Используется, если mpiexec -gpu вернул ошибку «unrecognized».
    """
    cmd = [mpiexec, "-n", str(int(n_proc)), su2_exe, cfg_path]
    env_overlay: dict = {}
    if use_openmp_offload:
        env_overlay = hybrid_env(gpu_percent=gpu_percent)
    return cmd, env_overlay


# ---------------------------------------------------------------------------
# Распознавание «mpiexec не знает -gpu»
# ---------------------------------------------------------------------------
def is_unknown_gpu_option(stderr: str) -> bool:
    """True, если mpiexec (или hydra) пожаловался на -gpu."""
    if not stderr:
        return False
    s = stderr.lower()
    needles = (
        "unrecognized option -gpu",
        "unrecognized option '-gpu'",
        "unrecognized argument -gpu",
        "invalid option -gpu",
        "unknown option -gpu",
        "unrecognized option: -gpu",
    )
    return any(n in s for n in needles)


# Имена рантаймов CUDA/HIP, по которым видно, что SU2 собрана с GPU.
_GPU_LIB_PATTERNS = (
    "cudart*.dll", "cublas*.dll", "cufft*.dll",
    "libcudart*.so*", "libcublas*.so*",
    "amdhip64*.dll", "libamdhip64*",
)


def su2_gpu_capable(su2_exe: str) -> bool:
    """Лежит ли рядом с SU2_CFD рантайм CUDA или HIP.

    Стандартные сборки SU2 с su2code.org собираются без GPU-поддержки,
    и ``OMP_TARGET_OFFLOAD=MANDATORY`` в них ничего никуда не выгружает.
    Единственный способ узнать это заранее, не запуская решатель, —
    посмотреть, есть ли в каталоге установки библиотеки GPU-рантайма.
    """
    if not su2_exe:
        return False
    d = os.path.dirname(os.path.abspath(str(su2_exe)))
    for _ in range(3):                      # каталог exe, bin/, корень
        if not d or not os.path.isdir(d):
            break
        try:
            names = [n.lower() for n in os.listdir(d)]
        except OSError:
            break
        for pat in _GPU_LIB_PATTERNS:
            if fnmatch.filter(names, pat):
                return True
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return False


def is_openmp_offload_unavailable(stderr: str) -> bool:
    """True, если SU2 собрана без OpenMP target / CUDA / HIP."""
    if not stderr:
        return False
    s = stderr.lower()
    needles = (
        "no openmp target offload",
        "omp target offload is disabled",
        "gpu support not compiled",
        "cuda not enabled",
        "hip not enabled",
        "error: gpu not available",
    )
    return any(n in s for n in needles)
