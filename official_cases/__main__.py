# -*- coding: utf-8 -*-
"""
official_cases.__main__ — CLI-обёртка над библиотекой официальных кейсов SU2.

Примеры:
    python -m official_cases list
    python -m official_cases show inv_oneram6
    python -m official_cases show --full inv_oneram6
    python -m official_cases download inv_oneram6
    python -m official_cases prepare inv_oneram6 ./case_oneram6
    python -m official_cases compare path/to/config.cfg [--case inv_oneram6]
    python -m official_cases meshes
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .catalog import OFFICIAL_CASES, get_case, list_cases
from .compare import compare_file, diagnose, render_diagnosis
from .downloader import (
    download_mesh, is_downloaded, meshes_dir, meshes_report,
    prepare_case_dir,
)
from .loader import bundled_config_text


def _print_case(case, full=False):
    print(f"=== {case.id}: {case.name} ===")
    print(f"   {case.description}")
    print(f"   Решатель: {case.solver} | {case.dimension}D | "
          f"SU2 {case.su2_version or '?'}")
    if case.mach is not None:
        print(f"   Mach={case.mach}, AoA={case.aoa}°, "
              f"Re={case.reynolds if case.reynolds is not None else '—'}")
    if case.ref_cl is not None:
        print(f"   Эталон (SU2 регрессия, {case.ref_iter or '?'} итер.): "
              f"CL={case.ref_cl}, CD={case.ref_cd}, "
              f"CM={case.ref_cm if case.ref_cm is not None else '—'}")
    if case.mesh_filename:
        print(f"   Сетка: {case.mesh_filename} "
              f"({case.mesh_repo}:{case.mesh_path})")
    print(f"   Источник: {case.repo}:{case.source_path}")
    if case.notes:
        print(f"   Примечание: {case.notes}")
    if full:
        print("--- config.cfg (официальный) ---")
        print(bundled_config_text(case.config_file))
    print()


def cmd_list(args):
    for cid in list_cases():
        c = get_case(cid)
        print(f"{cid:<20} {c.solver:<10} {c.dimension}D  {c.name}")
    print(f"\nВсего кейсов: {len(list_cases())}")


def cmd_show(args):
    _print_case(get_case(args.case), full=args.full)


def cmd_download(args):
    cid = args.case
    c = get_case(cid)
    if not c.mesh_path:
        print(f"Кейс {cid!r} не имеет официальной сетки в репозитории.")
        sys.exit(1)
    if is_downloaded(cid) and not args.force:
        print(f"Уже скачано: {__import__('official_cases').mesh_local_path(cid)}")
        return
    print(f"Скачивание сетки {c.mesh_filename} "
          f"из {c.mesh_repo} …")
    try:
        path = download_mesh(cid, overwrite=args.force)
    except Exception as e:
        print(f"Ошибка: {type(e).__name__}: {e}")
        sys.exit(1)
    print(f"OK: {path} ({os.path.getsize(path)} байт)")


def cmd_prepare(args):
    try:
        res = prepare_case_dir(args.case, args.out_dir, download=not args.no_download)
    except Exception as e:
        print(f"Ошибка: {type(e).__name__}: {e}")
        sys.exit(1)
    print(f"Каталог: {res['case_dir']}")
    print(f"Конфиг:  {res['config']}")
    if res.get('mesh_in_dir'):
        print(f"Сетка:   {res['mesh_in_dir']}  (рядом с config.cfg)")
    elif res.get('mesh'):
        print(f"Сетка:   {res['mesh']}")
    elif res.get('mesh_error'):
        print(f"Сетка:   НЕ скачана — {res['mesh_error']}")
    else:
        print("Сетка:   не запрашивалась (--no-download)")


def cmd_compare(args):
    cfg = args.config
    if not os.path.exists(cfg):
        print(f"config.cfg не найден: {cfg}")
        sys.exit(1)
    res = compare_file(cfg, case_id=args.case)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    else:
        print(render_diagnosis(res))


def cmd_meshes(args):
    for row in meshes_report():
        if row["available"]:
            print(f"{row['id']:<20} {'уже скачана' if row['downloaded'] else 'доступна':<14}" 
                  f"{row['size']:>8}  {row['mesh']}  [{row['repo']}]")
        else:
            print(f"{row['id']:<20} {'нет сетки':<14}  —")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m official_cases",
        description="Библиотека официальных конфигов и 3D-сеток SU2.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="список кейсов").set_defaults(fn=cmd_list)

    sp = sub.add_parser("show", help="показать кейс")
    sp.add_argument("case")
    sp.add_argument("--full", action="store_true",
                    help="показать ещё и полный config.cfg")
    sp.set_defaults(fn=cmd_show)

    dp = sub.add_parser("download", help="скачать официальную сетку (3D-модель)")
    dp.add_argument("case")
    dp.add_argument("--force", action="store_true", help="перекачать заново")
    dp.set_defaults(fn=cmd_download)

    pp = sub.add_parser("prepare",
                        help="подготовить каталог: конфиг + (опц.) сетка")
    pp.add_argument("case")
    pp.add_argument("out_dir")
    pp.add_argument("--no-download", action="store_true",
                    help="не скачивать сетку, только конфиг")
    pp.set_defaults(fn=cmd_prepare)

    cp = sub.add_parser("compare",
                        help="сравнить свой config.cfg с официальным кейсом")
    cp.add_argument("config")
    cp.add_argument("--case", help="id официального кейса (иначе — по Mach)")
    cp.add_argument("--json", action="store_true", help="вывести JSON")
    cp.set_defaults(fn=cmd_compare)

    mp = sub.add_parser("meshes", help="таблица официальных 3D-сеток")
    mp.set_defaults(fn=cmd_meshes)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
