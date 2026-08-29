# Интеграция гибридного GPU-режима в `solver/workers.py`

## Что уже сделано в этой итерации (без правки workers.py)

1. **`ui/main_window.py`** — расширено:
   - Добавлены атрибуты `self._compute_device_pending` (`"cpu"` / `"cpu_gpu"`)
     и `self._gpu_percent_pending` (0..100).
   - Кнопка «✅ Применить» (`apply_load_level`) фиксирует эти поля.
   - `start_calculation` / `resume_calculation` / `_offer_resume` /
     блок «resume существующей» пробрасывают `compute_device` и
     `gpu_percent` в `session` и затем в `session_runner`.
   - Лог старта сессии теперь пишет, какой вычислитель и доля GPU
     будут использованы.
   - **Обратная совместимость**: если `session.compute_device` не задан
     (старые сессии) — везде фоллбэк на `"cpu"`, `gpu_percent=0`.
2. **`solver/gpu_launcher.py`** (новый файл) — helper-функции:
   - `build_hybrid_command(mpiexec, n_proc, su2_exe, cfg_path, compute_device, gpu_percent)`
     → возвращает `(cmd, env_overlay)`. Для `"cpu"` — обычный
     `mpiexec -n N su2_exe config.cfg` (никаких изменений в поведении).
   - `build_cpu_fallback_command(...)` — для фоллбэка на OpenMP offload
     или чистый CPU.
   - `is_unknown_gpu_option(stderr)` / `is_openmp_offload_unavailable(stderr)`
     — распознавание ошибок MPI / SU2.
   - `detect_mpi_implementation()` — для информационного лога.

## Что нужно сделать в `C:\su2_app\solver\workers.py` (точечно)

Файл `workers.py` в workspace не лежит (он только у вас), поэтому
ниже — минимальная инструкция «где что поменять» на 2 правки по 5–10 строк.
**Никакие существующие ветки кода не удаляются** — добавляется
только обёртка вокруг `subprocess.Popen`.

### Правка 1: импорт

В начале файла добавьте:

```python
from solver.gpu_launcher import (
    build_hybrid_command, build_cpu_fallback_command,
    is_unknown_gpu_option, is_openmp_offload_unavailable,
    detect_mpi_implementation, gpu_config_args,
)
```

### Правка 2: в `SU2Worker.run()` (или там, где формируется команда)

Найдите место, где сейчас примерно такое:

```python
cmd = ["mpiexec", "-n", str(self.n_proc), su2_exe, "config.cfg"]
proc = subprocess.Popen(cmd, ...)
```

и замените на:

```python
# Гибридный GPU-режим (если включён в UI)
compute_device = getattr(self, "compute_device", "cpu")  # пробрасывается из session
gpu_percent = int(getattr(self, "gpu_percent", 0) or 0)
cmd, env_overlay = build_hybrid_command(
    mpiexec=mpiexec_exe,                    # ваша переменная с путём к mpiexec
    n_proc=self.n_proc,
    su2_exe=su2_exe,
    cfg_path=cfg_path,
    compute_device=compute_device,
    gpu_percent=gpu_percent,
)
proc = subprocess.Popen(
    cmd,
    env={**os.environ, **env_overlay},
    cwd=case_dir,
    stdout=log_fh, stderr=subprocess.STDOUT,
    **hidden_subprocess_kwargs(),
)
self._gpu_launched_with = "mpiexec-gpu" if compute_device == "cpu_gpu" else "cpu"
```

### Правка 3: авто-фоллбэк (если mpiexec не знает `-gpu`)

После `proc.wait()` / чтения `stderr`/`stdout` (там, где вы
обрабатываете падение SU2), добавьте:

```python
# Если первый запуск с -gpu упал из-за того, что MPI не знает такой опции —
# перезапускаем с OpenMP offload (или с чистым CPU).
stderr_text = ""  # подставьте свой код чтения stderr, если он есть
if (compute_device == "cpu_gpu"
        and self._gpu_launched_with == "mpiexec-gpu"
        and is_unknown_gpu_option(stderr_text)):
    self.log_signal.emit(
        "⚠️ mpiexec не поддерживает -gpu → фоллбэк на OpenMP offload."
    )
    cmd2, env2 = build_cpu_fallback_command(
        mpiexec=mpiexec_exe,
        n_proc=self.n_proc,
        su2_exe=su2_exe,
        cfg_path=cfg_path,
        use_openmp_offload=True,
        gpu_percent=gpu_percent,
    )
    self._gpu_launched_with = "omp-offload"
    proc2 = subprocess.Popen(
        cmd2, env={**os.environ, **env2},
        cwd=case_dir, stdout=log_fh, stderr=subprocess.STDOUT,
        **hidden_subprocess_kwargs(),
    )
    # дальше работаем с proc2 как раньше
```

### Правка 4: проброс `compute_device` / `gpu_percent` из `session`

В `SessionRunner._run_case` (или там, где создаётся `SU2Worker`),
перед запуском:

```python
self.worker.compute_device = getattr(self.session, "compute_device", "cpu")
self.worker.gpu_percent = int(getattr(self.session, "gpu_percent", 0) or 0)
self.worker.n_proc = self.session.cpu_cores  # или откуда берётся сейчас
```

## Стратегия «CPU + GPU» (авто-фоллбэк, без правки config.cfg)

1. **`mpiexec -gpu 0,1,2,...`** — Microsoft MPI >= 10.0 + SU2 с
   `-DENABLE_CUDA=ON` / `-DENABLE_HIP=ON`. Если MPI не знает `-gpu`
   или SU2 собрана без GPU → падение → ловим и пробуем следующий шаг.
2. **`OMP_TARGET_OFFLOAD=MANDATORY`** — OpenMP target offload. Подходит
   для SU2, собранной с Clang/OpenMP GPU. Плюс `OMP_NUM_DEVICES=...`.
3. **Чистый CPU** — `mpiexec -n N su2_exe config.cfg` (без изменений).

`config.cfg` остаётся **как был** (VOLUME/SURFACE_FILENAME без .vtu и т. д.) —
по вашему выбору «только env, без правки config.cfg».

## Что увидите в логе после правки workers.py

```
🚀 Запуск сессии (sweep) в 16:42:13, точек: 5, ядер: 6, решатель: EULER,
   сетка: Средняя, вычислитель: cpu_gpu (GPU 50%)
🎮 Гибридный режим: SU2_CFD должен быть собран с -DENABLE_CUDA=ON...
🔧 MPI: msmpi. Пробуем `mpiexec -n 6 -gpu 0,1,2,3,4,5 SU2_CFD.exe config.cfg`...
```

Или (если MPI не знает `-gpu`):

```
⚠️ mpiexec не поддерживает -gpu → фоллбэк на OpenMP offload.
🔧 MPI: openmpi. OMP_TARGET_OFFLOAD=MANDATORY + OMP_NUM_DEVICES=4
```

## Команды для пользователя

Замените файлы:
- `C:\su2_app\ui\main_window.py` ← `fixes\ui\main_window.py`
- `C:\su2_app\solver\gpu_launcher.py` ← `fixes\solver\gpu_launcher.py`
  (новый файл)
- `C:\su2_app\solver\workers.py` ← точечные правки по инструкции выше

Пересоберите:
```bat
cd /d C:\su2_app
rmdir /s /q build
rmdir /s /q dist
pyinstaller --clean --noconfirm AeroOpt.spec
```

## Что НЕ меняется

- `solver/config_builder.py` — НЕ правим (config.cfg остаётся как был).
- `solver/session.py` — НЕ обязательно править: `compute_device` и
  `gpu_percent` пишутся как обычные атрибуты. Опционально можно
  добавить их в `__init__`/`to_dict`/`load` — но и без этого работает.
- `mesh/*` — не трогаем.
- Все старые вызовы и проекты работают (compute_device по умолчанию = "cpu").
