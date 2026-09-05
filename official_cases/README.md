# official_cases — библиотека официальных конфигов и 3D-моделей SU2

Этот пакет добавляет в AeroOpt эталонные тест-кейсы **от разработчиков SU2**
(репозитории [`su2code/SU2`](https://github.com/su2code/SU2) и
[`su2code/Tutorials`](https://github.com/su2code/Tutorials), ветка `master`).

Зачем он нужен, если «расчёт даёт неправдоподобно большие значения»:

> На типичном режиме AeroOpt (V≈60 м/с у земли, M≈0.176) сжимаемый
> `EULER`/`RANS` становится жёстким по акустике (звук в 1/M раз быстрее
> потока). Схема Роу на низких махах вносит лишнюю вязкость, поэтому
> невязкий `Cd` вместо ~0 получается большим, а `RANS` — завышенным.
> Официальное решение SU2 — **несжимаемый решатель** `INC_EULER` /
> `INC_RANS` (кейсы `inc_naca0012`, `inc_turb_naca0012`) на малых скоростях.

## Что внутри

* `configs/` — официальные `config.cfg` (встроенные, маленькие текстовые файлы).
* `catalog.py` — реестр кейсов с эталонными `Cl/Cd/Cm` из
  `TestCases/serial_regression.py` SU2 и ссылками на официальную сетку.
* `downloader.py` — скачивание официальных 3D-сеток (ONERA M6 и др.) в
  `official_cases/meshes/` (в Git не попадает).
* `compare.py` — сравнение вашего `config.cfg` с официальным кейсом и
  объяснение, **почему** значения могут быть неправдоподобными.
* `loader.py` — разбор `config.cfg` (stdlib-only).
* `__main__.py` — CLI, работает без Qt/numpy.

## CLI

```bash
python -m official_cases list                     # список кейсов
python -m official_cases show inv_oneram6         # описание кейса
python -m official_cases show --full inv_oneram6  # + полный config.cfg
python -m official_cases download inv_oneram6     # скачать официальную 3D-сетку
python -m official_cases prepare inv_oneram6 ./case_oneram6
python -m official_cases compare path/to/config.cfg [--case inv_oneram6]
python -m official_cases meshes                   # таблица 3D-сеток
```

## Python API

```python
from official_cases import (
    list_cases, get_case,              # реестр
    download_mesh, is_downloaded,      # официальные 3D-модели
    prepare_case_dir,
    compare_file, render_diagnosis,    # диагностика «больших значений»
)
```

`compare_file(".../config.cfg")` вернёт структурированный ответ: режим
(compressible / compressible-low-mach / incompressible), рекомендуемый
решатель, список замечаний (severity) и дифф ключей с официальным кейсом.

## Ограничения

* Сетки SU2 хранятся в `su2code/Tutorials` и не коммитятся в `su2code/SU2`;
  качаются **по требованию** (нужен доступ к api.github.com). В Git их не
  кладём (`official_cases/meshes/` в `.gitignore`).
* Эталонные `ref_*` — числа из `TestCases/serial_regression.py` SU2, снятые
  на конкретной итерации (`ref_iter`). У ряда RANS/невязких кейсов это
  **далеко от полной сходимости**, поэтому их нужно читать вместе с
  `ref_source`. Для итоговой калибровки лучше смотреть на сходящиеся
  значения (закомментированы в `notes`).
