# -*- coding: utf-8 -*-
"""
official_cases/compare.py — сравнение конфига AeroOpt с официальными SU2.

Основная цель — помочь понять, **почему расчёт даёт неправдоподобно большие
Cl/Cd**. Почти всегда это не геометрия: это численная схема или нормировка.
Здесь мы сравниваем ключи ``config.cfg`` с ближайшим официальным кейсом и
возвращаем список замечаний с приоритетами.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from .catalog import (
    OFFICIAL_CASES, OfficialCase, nearest_for,
)
from .loader import (
    parse_keys, get_float, bundled_config_text, bundled_config_path,
)

# Порог, ниже которого сжимаемый решатель становится «жёстким» по акустике.
LOW_MACH = 0.30
# Очень низкий Mach — режим по умолчанию AeroOpt (V≈60 м/с у земли).
VERY_LOW_MACH = 0.20


def _mach_from_keys(cfg: Dict[str, str]) -> Optional[float]:
    mach = get_float(cfg, "MACH_NUMBER")
    if mach is not None:
        return mach
    # Incompressible: режим задаётся скоростью, а не Mach.
    if any(cfg.get(k, "").upper().startswith("INC_")
           for k in ("SOLVER",) if k in cfg):
        return None
    return None


def _solver_of(cfg: Dict[str, str]) -> str:
    return str(cfg.get("SOLVER", "")).strip().upper()


def _key_diff(own: Dict[str, str], official: Dict[str, str],
              keys: List[str]) -> List[dict]:
    """Сравнивает набор ключей own vs official."""
    out = []
    for k in keys:
        ko = own.get(k)
        of = official.get(k)
        if ko is None and of is None:
            continue
        if (ko or "").strip() == (of or "").strip():
            continue
        out.append({
            "key": k,
            "own": ko,
            "official": of,
            "note": "",
        })
    return out


def pick_case(text: str, case_id: Optional[str] = None) -> Optional[OfficialCase]:
    """Выбирает официальный кейс для сравнения.

    Если ``case_id`` задан — берём его; иначе — ближайший по режиму (Mach).
    """
    if case_id:
        return OFFICIAL_CASES.get(case_id)
    cfg = parse_keys(text)
    mach = _mach_from_keys(cfg)
    solver = _solver_of(cfg)
    if mach is None:
        # Несжимаемый (или не удалось определить) — берём низкомаховые кейсы.
        cands = [c for c in OFFICIAL_CASES.values() if c.solver.startswith("INC_")]
        if cands:
            return cands[0]
    cands = nearest_for(mach or 0.18, solver)
    if not cands:
        return None
    # Берём самый близкий по Mach.
    if mach is not None:
        cands.sort(key=lambda c: abs((c.mach or 0.0) - mach))
    return cands[0]


def diagnose(text: str, case_id: Optional[str] = None) -> dict:
    """Анализирует конфиг и возвращает структурированный диагноз.

    ``text`` — содержимое config.cfg (строка).
    Возвращает dict с полями:
        regime, mach, solver, case_id, suggested_solver,
        findings: [{severity, title, detail}],
        diff: [{key, own, official, note}]
    """
    cfg = parse_keys(text)
    solver = _solver_of(cfg)
    mach = _mach_from_keys(cfg)
    case = pick_case(text, case_id)

    findings: List[dict] = []
    diff = []

    regime = "compressible"
    if solver.startswith("INC_"):
        regime = "incompressible"
    elif mach is not None and mach < LOW_MACH:
        regime = "compressible-low-mach"

    suggested_solver = None

    # --- 1. Сжимаемый решатель на малом Mach — главный источник «больших Cd»
    if solver in ("EULER", "NAVIER_STOKES", "RANS") and mach is not None and mach < LOW_MACH:
        incomp = "INC_EULER" if solver == "EULER" else "INC_RANS"
        suggested_solver = incomp
        findings.append({
            "severity": "high",
            "title": f"Сжимаемый {solver} на M={mach:.3f} — источник большого Cd",
            "detail": (
                "На M<0.3 сжимаемый решатель становится жёстким по акустике "
                "(звук в 1/M раз быстрее потока). Схема Роу на низких махах "
                "вносит лишнюю вязкость: невязкий Cd вместо ~0 получается "
                "большим, а RANS — завышенным. Официальный способ SU2 для "
                "малых скоростей — несжимаемый решатель " + incomp +
                " (см. кейсы inc_naca0012 / inc_turb_naca0012)."
            ),
        })
    elif solver in ("EULER", "NAVIER_STOKES", "RANS") and mach is None and regime != "incompressible":
        findings.append({
            "severity": "low",
            "title": "Не удаётся определить Mach (нет MACH_NUMBER)",
            "detail": (
                "В конфиге не задан MACH_NUMBER, но решатель сжимаемый. "
                "Либо это incompressible SOLVER= INC_*, либо Mach задаётся "
                "иначе — проверьте FREESTREAM_OPTION / INC_* поля."
            ),
        })

    # --- 2. Низкомаховый прекондиционер выключен --------------------------
    if regime == "compressible-low-mach":
        prec = cfg.get("LOW_MACH_PREC", "NO").upper()
        corr = cfg.get("LOW_MACH_CORR", "NO").upper()
        if prec == "NO" and corr == "NO":
            findings.append({
                "severity": "medium",
                "title": "LOW_MACH_PREC / LOW_MACH_CORR выключены",
                "detail": (
                    "На M<0.3 желательно включить прекондиционер Roe–Turkel "
                    "(LOW_MACH_PREC= YES) и поправку MUSCL (LOW_MACH_CORR= "
                    "YES), либо перейти на несжимаемый решатель. Проверено "
                    "кейсом SU2 turbofan_MFR_coupling на M≈0.167."
                ),
            })

    # --- 3. EOF EULER: невязкий Cd должен быть близок к нулю --------------
    if solver == "EULER" and regime != "incompressible":
        findings.append({
            "severity": "medium",
            "title": "Невязкий EULER: Cd физически должен быть ~0",
            "detail": (
                "В невязкой постановке сопротивление замкнутого тела должно "
                "стремиться к нулю. Если ваш EULER-расчёт даёт большой Cd — "
                "это численная (схемная) вязкость, а не физика сопротивления. "
                "Для правдоподобного Cd считайте RANS."
            ),
        })

    # --- 4. REF_AREA: 0 = автоподсчёт, иначе важно совпадение с сеткой ----
    ref_area = cfg.get("REF_AREA")
    if ref_area is not None:
        fa = str(ref_area).strip()
        if fa == "0":
            findings.append({
                "severity": "low",
                "title": "REF_AREA= 0 (автоподсчёт по сетке)",
                "detail": (
                    "SU2 сам посчитает площадь по поверхности. Для сравнения "
                    "с кейсом, где REF_AREA= 1.0 (2D), помните, что это "
                    "разные нормировки."
                ),
            })

    # --- 5. CFL адаптация с низким потолком ------------------------------
    if cfg.get("CFL_ADAPT", "").upper() == "YES":
        cfl = cfg.get("CFL_ADAPT_PARAM", "")
        if cfl and ("5.0" in cfl or " 5 " in cfl or "," in cfl and "5.0" in str(cfl)):
            findings.append({
                "severity": "low",
                "title": "Потолок CFL_ADAPT_PARAM низкий",
                "detail": (
                    "В официальных кейсах SU2 потолок CFL в разы выше "
                    "(100–1000 и более), а CFL_ADAPT чаще NO. Низкий потолок "
                    "не даёт невязке упасть — это плоскость (застой), а не "
                    "правдоподобие значений."
                ),
            })

    # --- diff с официальным кейсом ----------------------------------------
    if case is not None:
        off_cfg = parse_keys(bundled_config_text(case.config_file))
        diff = _key_diff(cfg, off_cfg, [
            "SOLVER", "MACH_NUMBER", "AOA", "REYNOLDS_NUMBER", "REYNOLDS_LENGTH",
            "FREESTREAM_PRESSURE", "FREESTREAM_TEMPERATURE",
            "REF_LENGTH", "REF_AREA",
            "CONV_NUM_METHOD_FLOW", "MUSCL_FLOW", "SLOPE_LIMITER_FLOW",
            "CFL_NUMBER", "CFL_ADAPT", "CFL_ADAPT_PARAM",
            "TIME_DISCRE_FLOW", "LINEAR_SOLVER", "LINEAR_SOLVER_PREC",
            "LINEAR_SOLVER_ITER", "KIND_TURB_MODEL",
        ])

    return {
        "regime": regime,
        "mach": mach,
        "solver": solver,
        "case_id": case.id if case else None,
        "case_name": case.name if case else None,
        "suggested_solver": suggested_solver,
        "findings": findings,
        "diff": diff,
    }


def compare_file(config_path: str, case_id: Optional[str] = None) -> dict:
    """То же, что :func:`diagnose`, но читает файл по пути."""
    with open(config_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return diagnose(text, case_id=case_id)


def render_diagnosis(res: dict) -> str:
    """Человекочитаемый текст диагноза для лога/CLI."""
    lines = []
    if res.get("case_id"):
        lines.append(f"Официальный кейс для сравнения: {res['case_id']} "
                     f"({res['case_name']})")
    lines.append(f"Режим: {res['regime']}, решатель: {res['solver']}, "
                 f"Mach: {res['mach'] if res['mach'] is not None else '—'}")
    if res.get("suggested_solver"):
        lines.append(f"Рекомендуемый решатель: {res['suggested_solver']}")
    lines.append("")
    if res["findings"]:
        lines.append("Замечания:")
        for f in res["findings"]:
            lines.append(f"  [{f['severity']}] {f['title']}")
            lines.append(f"      {f['detail']}")
    else:
        lines.append("Замечаний нет.")
    lines.append("")
    if res["diff"]:
        lines.append("Отличия от официального кейса:")
        for d in res["diff"]:
            lines.append(f"  {d['key']}: ваш={d['own'] or '—'} "
                         f"официальный={d['official'] or '—'}")
    return "\n".join(lines)
