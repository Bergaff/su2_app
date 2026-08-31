# -*- coding: utf-8 -*-
"""
su2_autoconfig.py - устойчивые пресеты config.cfg для SU2 и детектор
расхождения расчёта.

Зачем: на сложных/«рваных» сетках штатные настройки (CFL-адаптация до 5.0,
2-й порядок MUSCL) могут приводить к расхождению:
    SU2 has diverged (Residual > 10^20 detected)
    Error in "void CSolver::SetResidual_RMS(...)"
Модуль умеет:
  1. apply_preset() - переписать config.cfg на устойчивый пресет
     (оригинал один раз бэкапится в config.cfg.orig);
  2. detect_result() - по history.csv (и тексту экрана SU2) понять,
     сошёлся расчёт, разошёлся или упал;
  3. suggest() - вернуть рекомендованный следующий шаг.

Зависимостей нет (только stdlib). Можно использовать:
  * из GUI (импорт функций);
  * вручную из командной строки:
      python su2_autoconfig.py "C:\...\aoa_+3.00\config.cfg"
      python su2_autoconfig.py "C:\...\aoa_+3.00\config.cfg" --preset ultra
      python su2_autoconfig.py --check "C:\...\aoa_+3.00"

Пресеты:
  safe  - CFL 2.0 без адаптации, 1-й порядок (MUSCL=NO), entropy fix 0.1;
          в ~95% случаев расходящаяся задача на этом идёт устойчиво.
  ultra - если не помог и safe: CFL 0.5, entropy fix 0.2, более строгий
          линейный решатель. Самый медленный, но самый «вездеходный».
"""

from __future__ import annotations

import csv
import math
import os
import re
import shutil

# ---------------------------------------------------------------------------
# Пресеты. Трогаем ТОЛЬКО численную схему; граничные условия, маркеры,
# ссылочные величины, поток и т.п. не меняются.
# ---------------------------------------------------------------------------

PRESETS = {
    "safe": {
        "TIME_DISCRE_FLOW": "EULER_IMPLICIT",
        "CFL_NUMBER": "2.0",
        "CFL_ADAPT": "NO",
        "MUSCL_FLOW": "NO",
        "ENTROPY_FIX_COEFF": "0.1",
        "NUM_METHOD_GRAD": "WEIGHTED_LEAST_SQUARES",
    },
    "ultra": {
        "TIME_DISCRE_FLOW": "EULER_IMPLICIT",
        "CFL_NUMBER": "0.5",
        "CFL_ADAPT": "NO",
        "MUSCL_FLOW": "NO",
        "ENTROPY_FIX_COEFF": "0.2",
        "NUM_METHOD_GRAD": "WEIGHTED_LEAST_SQUARES",
        "LINEAR_SOLVER_ITER": "20",
    },
}

PRESET_ORDER = ["safe", "ultra"]

# ВАЖНО: единственный символ комментария в config.cfg у SU2 - это '%'.
# Знак '#' SU2 комментарием НЕ считает: в CConfig::TokenizeString() строка
# ищется только по '%' (pos = str.find_first_of('%')), а дальше требуется
# '='. Строка "# ..." без '=' роняет SU2 с
#   Error in TokenizeString(): line in the configuration file with no "=" sign
# (проверено по SU2 v8.5.0, Common/src/CConfig.cpp).
BLOCK_HEADER = "% ===== AEROOPT-AUTOCONFIG: устойчивый пресет ====="
BLOCK_FOOTER = "% ===== /AEROOPT-AUTOCONFIG ======================="

# Старые версии писали блок с '#' - их тоже надо уметь вычищать.
_LEGACY_BLOCK_STARTS = ("# ===== AEROOPT-AUTOCONFIG",
                        "% ===== AEROOPT-AUTOCONFIG")
_LEGACY_BLOCK_ENDS = ("# ===== /AEROOPT-AUTOCONFIG",
                      "% ===== /AEROOPT-AUTOCONFIG")

# Разделители SU2 (CConfig::TokenizeString) - нужны для линтера конфига.
_SU2_DELIMS = " (){}:,\t\n\v\f\r"

# Допустимое имя опции SU2: латиница, цифры, подчёркивание.
_VALID_OPTION_NAME = re.compile(r"^[A-Za-z0-9_]+$")


# ---------------------------------------------------------------------------
# Работа с config.cfg
# ---------------------------------------------------------------------------

def _read_lines(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().splitlines(keepends=True)


def _write_lines(path, lines):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.writelines(lines)
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, AttributeError):
            pass
    os.replace(tmp, path)


def _strip_managed_block(lines):
    """Удаляет ранее вставленный блок автоконфига (чтобы пресеты наслаивались
    корректно и без дублей ключей)."""
    out = []
    skip = False
    for ln in lines:
        s = ln.strip()
        if s.startswith(_LEGACY_BLOCK_STARTS):
            skip = True
            continue
        if s.startswith(_LEGACY_BLOCK_ENDS):
            skip = False
            continue
        if not skip:
            out.append(ln)
    return out


def _set_key(lines, key, value):
    """Меняет существующую активную строку KEY= ... (закомментированные
    строки, начинающиеся с '%', не трогает). Возвращает (новые_строки,
    bool_ключ_найден)."""
    pat = re.compile(r'^([ \t]*)' + re.escape(key) + r'[ \t]*=.*$')
    out = []
    found = False
    for ln in lines:
        if not found and pat.match(ln.rstrip("\r\n")):
            indent = re.match(r'^[ \t]*', ln).group(0)
            out.append(f"{indent}{key}= {value}\n")
            found = True
        else:
            out.append(ln if ln.endswith("\n") else ln + "\n")
    return out, found



def su2_lint_lines(text):
    """Проверяет config.cfg ровно так, как это делает SU2.

    Повторяет логику CConfig::TokenizeString() (SU2 v8.5.0):
      * комментарий - только '%', причём всё от первого '%' отбрасывается;
      * пустая строка и строка из одних разделителей пропускаются;
      * в оставшейся части обязан быть '=', до '=' - ровно одно имя.

    Возвращает список (номер_строки, строка, причина). Пустой список -
    SU2 такой файл прочитает.
    """
    problems = []
    seen = {}             # имя опции -> номер строки (дубликат у SU2 фатален)
    pending = ""          # склейка продолжений через обратный слэш
    start_no = 0
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip("\r")
        if not pending:
            start_no = i
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        line = pending + line
        pending = ""

        pos = line.find("%")
        if not line or pos == 0:
            continue                      # пусто или комментарий
        if pos != -1:
            line = line[:pos]             # SU2 отбрасывает хвост от '%'
        if not line.strip(_SU2_DELIMS):
            continue                      # строка из одних разделителей
        if "=" not in line:
            problems.append((start_no, raw.rstrip("\r"),
                             'нет знака "=" — SU2 не считает "#" '
                             "комментарием, только %"))
            continue
        name_part = line.split("=", 1)[0]
        stripped = name_part.strip(_SU2_DELIMS)
        if not stripped:
            problems.append((start_no, raw.rstrip("\r"),
                             'перед "=" нет имени параметра'))
            continue
        if len(stripped.split()) > 1 or any(
                c in _SU2_DELIMS for c in stripped):
            problems.append((start_no, raw.rstrip("\r"),
                             'перед "=" два и более слова'))
            continue
        # Имя параметра SU2 приводит к верхнему регистру и ищет в таблице
        # опций. Строка вида "# =====" даёт имя "#" - SU2 ответит
        # "invalid option name", поэтому ловим это здесь же.
        if not _VALID_OPTION_NAME.match(stripped):
            problems.append((start_no, raw.rstrip("\r"),
                             f"недопустимое имя параметра {stripped!r} "
                             "(SU2 понимает как комментарий только '%')"))
            continue
        # SU2 v8 считает повтор опции ошибкой разбора:
        #   "Line N KEY: option appears twice" -> SetConfig_Parsing падает.
        name = stripped.upper()
        if name in seen:
            problems.append((start_no, raw.rstrip("\r"),
                             f"параметр {name} встречается дважды "
                             f"(первый раз — стр. {seen[name]}); SU2 v8 "
                             "считает это ошибкой разбора"))
        else:
            seen[name] = start_no
    return problems


def validate_config(path):
    """Возвращает (ok, problems) для config.cfg на диске."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as e:
        return False, [(0, "", f"не удалось прочитать файл: {e}")]
    problems = su2_lint_lines(text)
    return (not problems), problems


def apply_preset(config_path, preset="safe"):
    """Переписывает config.cfg под устойчивый пресет.

    Первый вызов делает резервную копию config.cfg.orig (если её ещё нет).
    Возвращает (out_path, list[str]) - путь и список изменений.
    """
    if preset not in PRESETS:
        raise ValueError(f"Неизвестный пресет: {preset!r}. "
                         f"Доступны: {', '.join(PRESETS)}")
    config_path = os.path.abspath(config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(config_path)

    backup = config_path + ".orig"
    if not os.path.exists(backup):
        shutil.copy2(config_path, backup)

    lines = _read_lines(config_path)
    lines = _strip_managed_block(lines)

    changes = []
    appended = []
    for key, value in PRESETS[preset].items():
        lines, existed = _set_key(lines, key, value)
        if existed:
            changes.append(f"  {key} = {value}")
        else:
            appended.append(f"{key}= {value}\n")
            changes.append(f"  + {key} = {value} (ключа не было)")

    if appended:
        if not lines[-1].endswith("\n"):
            lines[-1] = lines[-1] + "\n"
        lines.append("\n")
        lines.append(BLOCK_HEADER + "\n")
        lines.append(f"% Пресет: {preset}. Откат: вернуть config.cfg.orig\n")
        lines.extend(appended)
        lines.append(BLOCK_FOOTER + "\n")

    _write_lines(config_path, lines)

    # Самопроверка: если после правки конфиг стал нечитаемым для SU2 -
    # откатываемся на бэкап и честно сообщаем, а не запускаем SU2 в падение.
    ok, problems = validate_config(config_path)
    if not ok:
        bad = "; ".join(f"стр.{n}: {txt!r}" for n, txt, _r in problems[:3])
        shutil.copy2(backup, config_path)
        raise RuntimeError(
            "После применения пресета config.cfg стал нечитаемым для SU2 "
            f"({bad}). Откатил config.cfg.orig на место.")
    return config_path, changes


def restore_original(config_path):
    """Возвращает config.cfg.orig на место (если бэкап есть)."""
    config_path = os.path.abspath(config_path)
    backup = config_path + ".orig"
    if not os.path.exists(backup):
        return False
    shutil.copy2(backup, config_path)
    return True


# ---------------------------------------------------------------------------
# Детектор результата расчёта
# ---------------------------------------------------------------------------

def _case_dir(path):
    """Принимает и папку кейса, и путь к config.cfg - возвращает папку."""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return os.path.dirname(path)
    return path


def _parse_history(case_dir):
    """Читает history.csv. Возвращает (last_iter, last_log10_rho,
    first_log10_rho, n_rows) или None."""
    hist = os.path.join(case_dir, "history.csv")
    if not os.path.isfile(hist):
        return None
    try:
        with open(hist, "r", encoding="utf-8", errors="replace", newline="") as f:
            rows = list(csv.reader(f))
    except Exception:
        return None
    if len(rows) < 2:
        return None

    header = [h.strip().strip('"').strip() for h in rows[0]]
    try:
        i_iter = header.index("Inner_Iter")
        i_rho = next(i for i, h in enumerate(header)
                     if h.lower().replace(" ", "") in ("rms[rho]",))
    except (ValueError, StopIteration):
        return None

    def _val(row):
        try:
            v = float(row[i_rho])
            return v if math.isfinite(v) else None
        except (ValueError, IndexError):
            return None

    first_v = _val(rows[1])
    last_v = None
    last_iter = None
    n = 0
    for r in rows[1:]:
        if len(r) <= max(i_iter, i_rho):
            continue
        v = _val(r)
        if v is None:
            continue
        last_v = v
        n += 1
        try:
            last_iter = int(float(r[i_iter]))
        except ValueError:
            pass
    return last_iter, last_v, first_v, n


#: Признаки того, что SU2 умер на РАЗБОРЕ config.cfg, а не на счёте.
#: В этом случае history.csv остаётся от прошлой попытки, и вердикт
#: «расхождение на итерации N» был бы ложным - SU2 ни одной итерации
#: не сделал.
CONFIG_PARSE_MARKERS = (
    'error in tokenizestring()',
    'no "=" sign',
    "no '=' sign",
    "two or more options before",
    "invalid option name",
    "no value assigned",
)


def is_config_parse_error(text):
    """True, если SU2 упал на разборе config.cfg (не на расчёте)."""
    t = (text or "").lower()
    return any(m in t for m in CONFIG_PARSE_MARKERS)


def detect_from_text(text):
    """Быстрая проверка по экрану/логам SU2 (если history.csv нет)."""
    t = (text or "").lower()
    if is_config_parse_error(t):
        return "config_error"
    if "has diverged" in t or "residual > 10^20" in t or "residual > 1e20" in t:
        return "diverged"
    if "error exit" in t or "error in " in t:
        return "error"
    if "convergence" in t and "reached" in t:
        return "converged"
    return "unknown"


_CONFIG_ERROR_DETAIL = (
    "SU2 не смог прочитать config.cfg (ошибка разбора, ни одной итерации "
    "не выполнено). Показатели из history.csv относятся к ПРЕДЫДУЩЕЙ "
    "попытке и не отражают этот запуск.\n"
    "Причина почти всегда одна: в config.cfg появилась строка без знака "
    "'='. SU2 считает комментарием только '%', а не '#'.")


def detect_result(path, screen_text=None):
    """Вердикт по итогам прогона.

    Возвращает dict:
      status: 'converged' | 'diverged' | 'error' | 'unknown'
      detail: человекочитаемая строка (рус.)
      last_iter, last_log10_rho: если удалось прочитать history.csv
    """
    case_dir = _case_dir(path)
    info = _parse_history(case_dir)

    text_status = detect_from_text(screen_text) if screen_text else "unknown"

    if info is None:
        if text_status != "unknown":
            msg = {
                "diverged": "Расчёт разошёлся (по логу SU2: Residual > 10^20).",
                "error": "SU2 завершился с ошибкой (см. лог).",
                "converged": "Сходимость достигнута (по логу SU2).",
                "config_error": _CONFIG_ERROR_DETAIL,
            }[text_status]
            return {"status": text_status, "detail": msg}
        return {"status": "unknown",
                "detail": "history.csv не найден или пуст - результат неясен."}

    if text_status == "config_error":
        return {"status": "config_error", "detail": _CONFIG_ERROR_DETAIL}

    last_iter, last_v, first_v, n = info
    base = {
        "last_iter": last_iter, "last_log10_rho": last_v,
        "first_log10_rho": first_v, "n_rows": n,
    }

    if last_v is None:
        return {**base, "status": "error",
                "detail": "В history.csv нет конечного значения невязки."}

    # rms[Rho] в history.csv - это log10 RMS-невязки плотности.
    # SU2 аварийно выходит при Residual > 10^20 (т.е. log10 > 20);
    # log10 >= 8 - это уже уверенный взрыв (начальная невязка ~0.0,
    # при норме она падает до отрицательных значений).
    if last_v >= 8.0 or text_status == "diverged":
        return {**base, "status": "diverged",
                "detail": (f"Расчёт разошёлся: log10(rms[Rho])={last_v:.2f} "
                           f"на итерации {last_iter} (норма: падение до -4…-6).")}
    if last_v <= -4.0:
        return {**base, "status": "converged",
                "detail": (f"Сходимость хорошая: log10(rms[Rho])={last_v:.2f} "
                           f"на итерации {last_iter}.")}
    if text_status == "error":
        return {**base, "status": "error",
                "detail": (f"SU2 завершился ошибкой; log10(rms[Rho])="
                           f"{last_v:.2f} на итерации {last_iter}.")}
    return {**base, "status": "unknown",
            "detail": (f"Прогон остановлен при log10(rms[Rho])={last_v:.2f} "
                       f"(итерация {last_iter}); явного расхождения нет.")}


def suggest(path, screen_text=None, current_preset=None):
    """Рекомендация следующего шага: (action, preset|None, текст)."""
    res = detect_result(path, screen_text)
    st = res["status"]
    if st == "converged":
        return "none", None, "Расчёт сошёлся - настройки менять не нужно."
    if st in ("diverged", "error", "unknown"):
        if current_preset is None:
            return "apply_preset", "safe", \
                "Расчёт не сошёлся. Применить устойчивый пресет (CFL 2.0, " \
                "1-й порядок) и перезапустить?"
        idx = PRESET_ORDER.index(current_preset) if current_preset in PRESET_ORDER else -1
        if idx < len(PRESET_ORDER) - 1:
            nxt = PRESET_ORDER[idx + 1]
            return "apply_preset", nxt, \
                f"Пресет '{current_preset}' не помог. Попробовать более " \
                f"мягкий пресет '{nxt}' (CFL 0.5) и перезапустить?"
        return "abort", None, \
            "Даже самый мягкий пресет не помог - дело, вероятно, в сетке " \
            "(качество ячеек у тела). Нужно перегенерировать сетку " \
            "качеством 'Точная' или проверить геометрию."
    return "none", None, res["detail"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Автоконфиг SU2 (AeroOpt)")
    ap.add_argument("path", nargs="?",
                    help="путь к config.cfg (или папке кейса для --check)")
    ap.add_argument("--preset", default="safe", choices=list(PRESETS),
                    help="пресет устойчивости (по умолчанию safe)")
    ap.add_argument("--check", action="store_true",
                    help="только показать вердикт по history.csv, не менять конфиг")
    ap.add_argument("--restore", action="store_true",
                    help="вернуть config.cfg.orig на место")
    args = ap.parse_args(argv)

    if not args.path:
        ap.error("укажи путь к config.cfg (или папку кейса с --check)")

    if args.check:
        res = detect_result(args.path)
        print(f"Статус: {res['status']}")
        print(res["detail"])
        action, preset, text = suggest(args.path)
        print(f"Рекомендация: {text}")
        return

    cfg = args.path
    if os.path.isdir(cfg):
        cfg = os.path.join(cfg, "config.cfg")

    if args.restore:
        ok = restore_original(cfg)
        print("config.cfg.orig восстановлен." if ok
              else "Бэкап config.cfg.orig не найден.")
        return

    out, changes = apply_preset(cfg, args.preset)
    print(f"Пресет '{args.preset}' применён: {out}")
    print("Изменения:")
    for c in changes:
        print(c)
    print("Оригинал сохранён рядом: config.cfg.orig")
    print("Теперь запусти расчёт как обычно (GUI возьмёт этот же config.cfg).")


if __name__ == "__main__":
    _main()
