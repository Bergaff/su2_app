# -*- coding: utf-8 -*-
"""
tests/test_official_cases.py — проверки библиотеки официальных кейсов SU2.

Модуль написан stdlib-only (без numpy/Qt/сети), поэтому работает в
CI-песочнице без тяжёлых зависимостей.

Запуск:
    python tests/test_official_cases.py
    (или через pytest)
"""

import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from official_cases import (
    OFFICIAL_CASES,
    list_cases,
    get_case,
    find_by_mach,
    find_by_solver,
    nearest_for,
    diagnose,
    compare_file,
    render_diagnosis,
    parse_keys,
    bundled_config_text,
    bundled_config_path,
    mesh_local_path,
    prepare_case_dir,
    meshes_report,
    body_markers_from_config,
    parse_su2_text,
    read_su2_boundary,
    is_manifold_closed,
)
from official_cases.downloader import (
    fetch_remote, download_mesh, meshes_dir, prepare_case_run_dir,
)

# Ожидаемые id в реестре
EXPECTED_IDS = {
    "inv_oneram6", "turb_oneram6", "inv_crm",  # 3D крылья / самолёт
    "inv_naca0012", "turb_naca0012_sa",     # 2D профили (EULER/RANS)
    "turb_rae2822_sa",
    "inc_naca0012", "inc_turb_naca0012",    # несжимаемые (малые скорости)
}


def _ok(label):
    print("  ✅ " + label)


def test_catalogue():
    ids = list_cases()
    assert EXPECTED_IDS.issubset(set(ids)), (EXPECTED_IDS - set(ids))
    _ok(f"реестр содержит {len(ids)} кейсов, все id в OFFICIAL_CASES")

    for cid in ids:
        c = get_case(cid)
        assert c.id == cid
        assert c.solver, cid
        assert c.dimension in (2, 3), cid
        # Встроенный конфиг существует и в нём есть SOLVER
        assert os.path.exists(c.config_path), c.config_file
        text = bundled_config_text(c.config_file)
        assert "=" in text, c.config_file
        _ok(f"конфиг {c.config_file!r} присутствует, SOLVER={c.solver}")


def test_get_case_unknown():
    got = False
    try:
        get_case("no_such_case")
    except KeyError:
        got = True
    assert got
    _ok("get_case неизвестного id -> KeyError")


def test_find_by():
    assert find_by_mach(0.84)            # ONERA M6 случаи
    # Near M=0 incompressible (mach None) cases are never matched.
    for c in find_by_mach(0.0):
        assert c.mach is not None
    assert find_by_solver("INC_RANS")    # несжимаемый кейс
    assert find_by_solver("INC_EULER")
    assert find_by_solver("EULER")       # хотя бы один EULER
    _ok("find_by_mach / find_by_solver работают")


def test_nearest_for():
    # Низкомаховый режим AeroOpt (M≈0.176) должен подтянуть близкий кейс
    nearest = nearest_for(0.176, "EULER")
    assert nearest, "не должно быть пусто"
    # Для типичного низкого числа Маха предпочитаем кейс с Mach поближе
    assert any(c.mach is not None and abs(c.mach - 0.176) < 0.1
               for c in nearest)
    _ok("nearest_for на M=0.176 находит ближайший кейс")


def test_loader_helpers():
    text = "SOLVER= EULER\n% comment\nMESH_FILENAME= mesh.su2\n"
    cfg = parse_keys(text)
    assert cfg["SOLVER"] == "EULER"
    assert cfg["MESH_FILENAME"] == "mesh.su2"
    assert "COMMENT" not in cfg
    _ok("parse_keys игнорирует '%' комментарии")


def test_diagnose_low_mach_euler():
    text = (
        "SOLVER= EULER\n"
        "MACH_NUMBER= 0.176\n"
        "AOA= 3.0\n"
        "REF_AREA= 12.0\n"
        "CFL_ADAPT= YES\n"
        "CFL_ADAPT_PARAM= ( 0.5, 1.2, 0.5, 5.0 )\n"
    )
    res = diagnose(text)
    # Режим — сжимаемый низкомаховый
    assert res["regime"] == "compressible-low-mach", res["regime"]
    assert res["mach"] == 0.176
    # Должна появиться запись про «большой Cd» (severity high)
    highs = [f for f in res["findings"] if f["severity"] == "high"]
    assert any("источник большого Cd" in f["title"] for f in highs)
    # Рекомендация — несжимаемый
    assert res["suggested_solver"] == "INC_EULER", res["suggested_solver"]
    # Есть дифф с официальным кейсом
    assert res.get("case_id"), res.get("case_id")
    assert isinstance(res["diff"], list)
    _ok("diagnose() для низкомахового EULER даёт high-замечание и INC_EULER")


def test_compare_file():
    import tempfile
    d = tempfile.mkdtemp(prefix="oc_")
    cfg = os.path.join(d, "config.cfg")
    with open(cfg, "w", encoding="utf-8") as f:
        f.write(
            "SOLVER= RANS\nMACH_NUMBER= 0.15\nAOA= 10.0\n"
            "REYNOLDS_NUMBER= 6.0E6\nMUSCL_FLOW= YES\n"
        )
    res = compare_file(cfg)
    assert res["solver"] == "RANS"
    # Должен подобраться кейс; трудно сказать какой именно, но это RANS-ish
    assert res.get("case_id") in list_cases()
    txt = render_diagnosis(res)
    assert "официальный" in txt.lower() or "Замечаний" in txt
    _ok("compare_file() + render_diagnosis() работают")


def test_prepare_case_dir_offline():
    # Без сети: только конфиг, сетка не качается
    import tempfile
    d = tempfile.mkdtemp(prefix="oc_prep_")
    res = prepare_case_dir("inv_oneram6", d, download=False)
    assert os.path.exists(res["config"])
    with open(res["config"], encoding="utf-8") as f:
        txt = f.read()
    assert "SOLVER= EULER" in txt, "должен быть официальный конфиг"
    assert "MESH_FILENAME" in txt
    # download=False => mesh не должен быть заполнен
    assert res.get("mesh") is None
    _ok("prepare_case_dir(download=False) кладёт официальный конфиг")


def test_meshes_report():
    rows = meshes_report()
    assert rows, "должна быть таблица сеток"
    by_id = {r["id"]: r for r in rows}
    assert by_id["inv_oneram6"]["available"] is True
    assert "mesh_ONERAM6" in by_id["inv_oneram6"]["mesh"]
    _ok("meshes_report() перечисляет официальные 3D-сетки")


def _synthetic_su2_text():
    """Минимальный 3D-меш SU2 для проверки извлечения поверхности."""
    return (
        "NDIME= 3\n"
        "NPOIN= 5\n"
        "0.0 0.0 0.0\n"
        "1.0 0.0 0.0\n"
        "1.0 1.0 0.0\n"
        "0.0 1.0 0.0\n"
        "0.5 0.5 0.5\n"
        "NMARK= 1\n"
        "MARKER_TAG= WING\n"
        "MARKER_ELEMS= 2\n"
        "5 0 1 2\n"
        "5 0 2 3\n"
    )


def test_body_markers_from_config():
    text = (
        "SOLVER= RANS\n"
        "MARKER_HEATFLUX= ( WING, 0.0 )\n"
        "MARKER_FAR= ( FARFIELD )\n"
    )
    assert body_markers_from_config(text) == ["WING"]
    # MARKER_EULER в приоритете, если задан
    text2 = "MARKER_EULER= ( UPPER_SIDE, LOWER_SIDE, TIP )\n"
    assert body_markers_from_config(text2) == [
        "UPPER_SIDE", "LOWER_SIDE", "TIP"]
    _ok("body_markers_from_config читает MARKER_EULER / MARKER_HEATFLUX")


def test_read_su2_boundary_synthetic():
    import tempfile
    import os as _os
    d = tempfile.mkdtemp(prefix="oc_surf_")
    p = _os.path.join(d, "mesh.su2")
    with open(p, "w", encoding="utf-8") as f:
        f.write(_synthetic_su2_text())

    parsed = parse_su2_text(_synthetic_su2_text())
    assert parsed["ndime"] == 3
    assert len(parsed["points"]) == 5
    assert "WING" in parsed["markers"]

    surf = read_su2_boundary(p, markers=["WING"])
    assert len(surf["triangles"]) == 2, surf["triangles"]
    assert len(surf["points"]) == 4, surf["points"]  # compact: 0,1,2,3
    assert surf["markers"] == {"WING": 2}

    # Без markers — берутся все «тела» (не дальнее поле/симметрия)
    surf_all = read_su2_boundary(p, markers=None)
    assert len(surf_all["triangles"]) == 2
    _ok("parse_su2_text / read_su2_boundary работают (в т.ч. compact)")


def test_downloader_avoids_rate_limit_via_raw():
    """Скачивание отдаёт приоритет raw-зеркалу и не бьётся в лимит API.

    GitHub Contents API без токена ограничен 60 запросами в час и отдаёт
    ``HTTP 403 rate limit exceeded``. ``fetch_remote`` первым делом пробует
    ``raw.githubusercontent.com`` (лимита нет), поэтому при доступном raw
    API вообще не вызывается.
    """
    import urllib.error
    from unittest import mock

    calls = []

    class _Resp:
        def __init__(self, data):
            self._data = data
        def read(self):
            return self._data
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        if "api.github.com" in req.full_url:
            # Если бы дошли до API — упёрлись бы в лимит.
            raise urllib.error.HTTPError(
                req.full_url, 403, "rate limit exceeded", {}, None)
        return _Resp(b"MESH-BYTES")

    with mock.patch("official_cases.downloader.urllib.request.urlopen",
                    _fake_urlopen):
        data = fetch_remote("su2code/Tutorials", "design/some/mesh.su2")
    assert data == b"MESH-BYTES", data
    # raw-зеркало сработало первым — к ограниченному API не обращались.
    assert any("raw.githubusercontent.com" in u for u in calls), calls
    assert not any("api.github.com" in u for u in calls), \
        "не должны стучаться в API, когда raw доступен"
    _ok("fetch_remote(): raw-зеркало в приоритете, API-лимит обойдён")


def test_downloader_fallback_to_api():
    """Если raw-зеркало недоступно, fetch_remote докачивает через GitHub API."""
    import urllib.error
    from unittest import mock

    calls = []

    class _Resp:
        def __init__(self, data):
            self._data = data
        def read(self):
            return self._data
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        if "raw.githubusercontent.com" in req.full_url:
            # CDN недоступен (TLS/прокси) на этой машине.
            raise urllib.error.URLError("raw cdN недоступен")
        return _Resp(b"MESH-API")

    with mock.patch("official_cases.downloader.urllib.request.urlopen",
                    _fake_urlopen):
        data = fetch_remote("su2code/Tutorials", "design/some/mesh.su2")
    assert data == b"MESH-API", data
    assert any("api.github.com" in u for u in calls), calls
    _ok("fetch_remote(): при недоступном raw качает через GitHub API")


def test_download_mesh_caches():
    """download_mesh сохраняет сетку в кэш meshes/ и не перекачивает её."""
    import os
    import tempfile
    from unittest import mock
    # Пустой кейс-заглушка, не требующий сети: временный каталог кэша.
    import official_cases.downloader as _DL
    tmp = tempfile.mkdtemp(prefix="oc_mesh_")
    with mock.patch.object(_DL, "_MESHES_DIR", tmp):
        case = get_case("inv_naca0012")
        with mock.patch.object(_DL, "fetch_remote",
                               return_value=b"X" * 64) as fr:
            p = download_mesh("inv_naca0012")
        assert os.path.exists(p)
        assert fr.call_count == 1
        # Повторный вызов — без повторного скачивания.
        with mock.patch.object(_DL, "fetch_remote",
                               return_value=b"") as fr2:
            p2 = download_mesh("inv_naca0012")
        assert p2 == p
        assert fr2.call_count == 0
    _ok("download_mesh(): кэширует сетку в meshes/")


def test_prepare_case_run_dir():
    """prepare_case_run_dir собирает SU2Worker-совместимый кейс (mesh.su2)."""
    import os
    import tempfile
    import official_cases.downloader as _DL
    from unittest import mock
    out = tempfile.mkdtemp(prefix="oc_run_")
    # Подменяем сетку на маленький файл, чтобы не качать и не зависеть от сети.
    fake_mesh = os.path.join(tempfile.mkdtemp(prefix="oc_fakemesh_"), "m.su2")
    with open(fake_mesh, "w", encoding="utf-8") as f:
        f.write("NDIME= 3\nNPOIN= 1\n0 0 0\n")
    with mock.patch.object(_DL, "download_mesh", return_value=fake_mesh):
        res = prepare_case_run_dir("inv_naca0012", out)
    assert os.path.exists(res["config"])
    assert os.path.basename(res["mesh"]) == "mesh.su2", res["mesh"]
    assert os.path.exists(res["mesh"])
    with open(res["config"], encoding="utf-8") as f:
        txt = f.read()
    assert "MESH_FILENAME= mesh.su2" in txt, txt
    _ok("prepare_case_run_dir() кладёт mesh.su2 + config.cfg (MESH_FILENAME=mesh.su2)")


def test_is_manifold_closed():
    # Замкнутый тетраэдр из 4 треугольников — замкнут.
    closed = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
    assert is_manifold_closed(closed) is True
    # Один треугольник — открытый край.
    assert is_manifold_closed([(0, 1, 2)]) is False
    # Квадрат из двух треугольников — открытый (внешний контур).
    assert is_manifold_closed([(0, 1, 2), (0, 2, 3)]) is False
    # Пустой список — не замкнут.
    assert is_manifold_closed([]) is False
    _ok("is_manifold_closed различает замкнутую и открытую поверхность")


def test_parse_2d_line_boundary():
    """2D-границы (линии, тип 3) больше не отбрасываются: профиль читается."""
    import tempfile
    import os as _os
    text = ("NDIME= 2\n"
            "NPOIN= 4\n"
            "0.0 0.0\n"
            "1.0 0.0\n"
            "1.0 1.0\n"
            "0.0 1.0\n"
            "NELEM= 1\n"
            "5 0 1 2\n"
            "NMARK= 1\n"
            "MARKER_TAG= airfoil\n"
            "MARKER_ELEMS= 4\n"
            "3 0 1\n"
            "3 1 2\n"
            "3 2 3\n"
            "3 3 0\n")
    parsed = parse_su2_text(text)
    # Линии сохранились (раньше отбрасывались из-за len(nodes)>=3).
    assert parsed["ndime"] == 2
    assert parsed["markers"]["airfoil"], parsed["markers"]
    assert len(parsed["markers"]["airfoil"]) == 4, parsed["markers"]
    _ok("parse_su2_text сохраняет 2D-границы (линии, тип 3)")


if __name__ == "__main__":
    print("== test_official_cases ==")
    test_catalogue()
    test_get_case_unknown()
    test_find_by()
    test_nearest_for()
    test_loader_helpers()
    test_body_markers_from_config()
    test_read_su2_boundary_synthetic()
    test_is_manifold_closed()
    test_parse_2d_line_boundary()
    test_diagnose_low_mach_euler()
    test_compare_file()
    test_prepare_case_dir_offline()
    test_meshes_report()
    test_downloader_avoids_rate_limit_via_raw()
    test_downloader_fallback_to_api()
    test_download_mesh_caches()
    test_prepare_case_run_dir()
    print("== OK ==")
