# -*- coding: utf-8 -*-
"""
ui/analysis_pages.py — страницы панели настроек для новых разделов ТЗ.

Вынесено из ``ui/main_window.py``, чтобы тот не разрастался: каждая
функция собирает свой ``QWidget`` и возвращает словарь созданных виджетов
— ``MainWindow`` хранит его как атрибут и читает в обработчиках.

Страницы:
  * :func:`build_aeroelastic_page` — аэроупругость (флатер/дивергенция),
    ТЗ «Средний приоритет»;
  * :func:`build_structural_page` — прочность корневого сечения,
    ТЗ «Низкий приоритет»;
  * :func:`build_specials_page` — спецфункции постобработки (поляра,
    отчёты по шаблону);
  * :func:`build_presets_page` — именованные пресеты ``config.cfg``.
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget,
)

from postprocessing.report import TEMPLATES


# ---------------------------------------------------------------------------
# Аэроупругость
# ---------------------------------------------------------------------------

def build_aeroelastic_page(on_check=None, on_plot=None) -> tuple:
    """Страница «Аэроупругость»: параметры крыла и кнопка проверки."""
    page = QWidget()
    lay = QVBoxLayout(page)
    lay.setContentsMargins(10, 10, 10, 10)

    gb = QGroupBox("Параметры крыла для оценки (метод типичного сечения)")
    f = QFormLayout(gb)
    w = {}

    w["span"] = QDoubleSpinBox(); w["span"].setRange(0.5, 200.0)
    w["span"].setValue(15.0); w["span"].setSuffix(" м")
    w["chord_root"] = QDoubleSpinBox(); w["chord_root"].setRange(0.05, 30.0)
    w["chord_root"].setValue(1.5); w["chord_root"].setSuffix(" м")
    w["chord_tip"] = QDoubleSpinBox(); w["chord_tip"].setRange(0.02, 30.0)
    w["chord_tip"].setValue(0.7); w["chord_tip"].setSuffix(" м")
    w["mass_wing"] = QDoubleSpinBox(); w["mass_wing"].setRange(0.1, 200000.0)
    w["mass_wing"].setValue(600.0); w["mass_wing"].setSuffix(" кг")
    w["x_ea_ratio"] = QDoubleSpinBox(); w["x_ea_ratio"].setRange(0.05, 0.95)
    w["x_ea_ratio"].setValue(0.40); w["x_ea_ratio"].setSingleStep(0.01)
    w["x_ea_ratio"].setToolTip("Положение оси упругости в долях хорды от носка")
    w["x_cg_ratio"] = QDoubleSpinBox(); w["x_cg_ratio"].setRange(0.05, 0.95)
    w["x_cg_ratio"].setValue(0.38); w["x_cg_ratio"].setSingleStep(0.01)
    w["x_cg_ratio"].setToolTip("Положение центра масс сечения в долях хорды")
    w["t_ratio"] = QDoubleSpinBox(); w["t_ratio"].setRange(0.01, 0.5)
    w["t_ratio"].setValue(0.12); w["t_ratio"].setSingleStep(0.01)
    w["t_ratio"].setToolTip("Относительная толщина (высота лонжерона-коробки)")

    f.addRow("Размах:", w["span"])
    f.addRow("Хорда в корне:", w["chord_root"])
    f.addRow("Хорда на конце:", w["chord_tip"])
    f.addRow("Масса крыла:", w["mass_wing"])
    f.addRow("Ось упругости x/c:", w["x_ea_ratio"])
    f.addRow("Центр масс x/c:", w["x_cg_ratio"])
    f.addRow("Толщина t/c:", w["t_ratio"])
    lay.addWidget(gb)

    gb2 = QGroupBox("Условия полёта")
    f2 = QFormLayout(gb2)
    w["rho"] = QDoubleSpinBox(); w["rho"].setRange(0.01, 2.0)
    w["rho"].setValue(1.225); w["rho"].setDecimals(4); w["rho"].setSuffix(" кг/м³")
    w["v_cruise"] = QDoubleSpinBox(); w["v_cruise"].setRange(1.0, 2000.0)
    w["v_cruise"].setValue(120.0); w["v_cruise"].setSuffix(" м/с")
    w["v_dive"] = QDoubleSpinBox(); w["v_dive"].setRange(0.0, 3000.0)
    w["v_dive"].setValue(180.0); w["v_dive"].setSuffix(" м/с")
    w["v_dive"].setSpecialValueText("= крейсерская")
    w["safety"] = QDoubleSpinBox(); w["safety"].setRange(1.0, 3.0)
    w["safety"].setValue(1.15); w["safety"].setSingleStep(0.05)
    w["safety"].setToolTip("Требуемый запас по критической скорости")
    w["fill_from_model"] = QCheckBox("Подставить размеры из текущей модели")
    w["fill_from_model"].setChecked(True)

    f2.addRow("Плотность ρ:", w["rho"])
    f2.addRow("Крейсерская V:", w["v_cruise"])
    f2.addRow("Скорость пикирования:", w["v_dive"])
    f2.addRow("Требуемый запас:", w["safety"])
    f2.addRow(w["fill_from_model"])
    lay.addWidget(gb2)

    btns = QHBoxLayout()
    w["btn_check"] = QPushButton("Проверить аэроупругость")
    w["btn_plot"] = QPushButton("V-g диаграмма")
    if on_check:
        w["btn_check"].clicked.connect(on_check)
    if on_plot:
        w["btn_plot"].clicked.connect(on_plot)
    btns.addWidget(w["btn_check"])
    btns.addWidget(w["btn_plot"])
    lay.addLayout(btns)

    w["out"] = QTextEdit()
    w["out"].setReadOnly(True)
    w["out"].setPlaceholderText(
        "Здесь появится оценка: частоты изгиба и кручения, скорость "
        "флатера V_F, скорость дивергенции V_D, запас по скорости и "
        "вердикт.\n\nМетод — типичное сечение (2 степени свободы) с "
        "полной аэродинамикой Теодорсена; это предварительная оценка, "
        "а не замена сертифицированного расчёта.")
    lay.addWidget(w["out"], stretch=1)
    return page, w


# ---------------------------------------------------------------------------
# Прочность
# ---------------------------------------------------------------------------

def build_structural_page(on_calc=None) -> tuple:
    """Страница «Прочность»: изгибающий момент и запас в корневом сечении."""
    page = QWidget()
    lay = QVBoxLayout(page)
    lay.setContentsMargins(10, 10, 10, 10)

    gb = QGroupBox("Нагружение")
    f = QFormLayout(gb)
    w = {}
    w["mass_aircraft"] = QDoubleSpinBox(); w["mass_aircraft"].setRange(1.0, 5e5)
    w["mass_aircraft"].setValue(1200.0); w["mass_aircraft"].setSuffix(" кг")
    w["mass_wing"] = QDoubleSpinBox(); w["mass_wing"].setRange(0.0, 1e5)
    w["mass_wing"].setValue(600.0); w["mass_wing"].setSuffix(" кг")
    w["n_limit"] = QDoubleSpinBox(); w["n_limit"].setRange(1.0, 15.0)
    w["n_limit"].setValue(3.8); w["n_limit"].setSingleStep(0.1)
    w["n_limit"].setToolTip("Эксплуатационная перегрузка (для ЛА общего "
                            "назначения обычно 3.8, для пилотажных — 6…9)")
    w["span"] = QDoubleSpinBox(); w["span"].setRange(0.5, 200.0)
    w["span"].setValue(15.0); w["span"].setSuffix(" м")
    w["chord_root"] = QDoubleSpinBox(); w["chord_root"].setRange(0.05, 30.0)
    w["chord_root"].setValue(1.5); w["chord_root"].setSuffix(" м")
    w["dist"] = QComboBox()
    w["dist"].addItems(["elliptic", "uniform", "triangular"])
    w["dist"].setToolTip("Закон распределения подъёмной силы по размаху")
    f.addRow("Масса ЛА:", w["mass_aircraft"])
    f.addRow("Масса крыла:", w["mass_wing"])
    f.addRow("Перегрузка n_y:", w["n_limit"])
    f.addRow("Размах:", w["span"])
    f.addRow("Хорда в корне:", w["chord_root"])
    f.addRow("Распределение:", w["dist"])
    lay.addWidget(gb)

    gb2 = QGroupBox("Сечение лонжерона")
    f2 = QFormLayout(gb2)
    w["t_ratio"] = QDoubleSpinBox(); w["t_ratio"].setRange(0.01, 0.5)
    w["t_ratio"].setValue(0.12); w["t_ratio"].setSingleStep(0.01)
    w["cap_frac"] = QDoubleSpinBox(); w["cap_frac"].setRange(0.001, 0.5)
    w["cap_frac"].setValue(0.02); w["cap_frac"].setDecimals(4)
    w["cap_frac"].setToolTip("Доля площади полок в площади сечения")
    w["sigma_allow"] = QDoubleSpinBox(); w["sigma_allow"].setRange(1e6, 2e9)
    w["sigma_allow"].setValue(2.8e8); w["sigma_allow"].setDecimals(0)
    w["sigma_allow"].setSuffix(" Па")
    w["sf"] = QDoubleSpinBox(); w["sf"].setRange(1.0, 5.0)
    w["sf"].setValue(1.5); w["sf"].setSingleStep(0.1)
    f2.addRow("Толщина t/c:", w["t_ratio"])
    f2.addRow("Доля полок:", w["cap_frac"])
    f2.addRow("σ допускаемое:", w["sigma_allow"])
    f2.addRow("Коэф. запаса:", w["sf"])
    lay.addWidget(gb2)

    w["btn_calc"] = QPushButton("Рассчитать прочность")
    if on_calc:
        w["btn_calc"].clicked.connect(on_calc)
    lay.addWidget(w["btn_calc"])

    w["out"] = QTextEdit(); w["out"].setReadOnly(True)
    w["out"].setPlaceholderText(
        "Результат: перерезывающая сила и изгибающий момент в корневом "
        "сечении, нормальные и касательные напряжения, коэффициент запаса "
        "(margin of safety) и вердикт.")
    lay.addWidget(w["out"], stretch=1)
    return page, w


# ---------------------------------------------------------------------------
# Спецфункции: поляра и отчёты
# ---------------------------------------------------------------------------

def build_specials_page(on_polar=None, on_report=None, on_csv=None) -> tuple:
    """Страница «Спецфункции»: поляра по точкам и отчёты по шаблону."""
    page = QWidget()
    lay = QVBoxLayout(page)
    lay.setContentsMargins(10, 10, 10, 10)

    gb = QGroupBox("Поляра по результатам расчётов")
    f = QFormLayout(gb)
    w = {}
    w["weight"] = QDoubleSpinBox(); w["weight"].setRange(0.1, 1e6)
    w["weight"].setValue(1200.0); w["weight"].setSuffix(" кгс")
    w["weight"].setToolTip("Полётный вес — нужен для скорости сваливания")
    w["rho"] = QDoubleSpinBox(); w["rho"].setRange(0.01, 2.0)
    w["rho"].setValue(1.225); w["rho"].setDecimals(4); w["rho"].setSuffix(" кг/м³")
    w["s_ref"] = QDoubleSpinBox(); w["s_ref"].setRange(0.01, 1000.0)
    w["s_ref"].setValue(12.0); w["s_ref"].setSuffix(" м²")
    w["mach"] = QDoubleSpinBox(); w["mach"].setRange(0.0, 0.95)
    w["mach"].setValue(0.15); w["mach"].setDecimals(3)
    w["btn_polar"] = QPushButton("Построить поляру по таблице результатов")
    if on_polar:
        w["btn_polar"].clicked.connect(on_polar)
    f.addRow("Полётный вес:", w["weight"])
    f.addRow("Плотность ρ:", w["rho"])
    f.addRow("Площадь S:", w["s_ref"])
    f.addRow("Число M:", w["mach"])
    f.addRow(w["btn_polar"])
    lay.addWidget(gb)

    gb2 = QGroupBox("Отчёт")
    f2 = QFormLayout(gb2)
    w["template"] = QComboBox(); w["template"].addItems(list(TEMPLATES))
    w["project_name"] = QLineEdit("Без имени")
    w["btn_report"] = QPushButton("Сформировать отчёт (HTML)")
    w["btn_csv"] = QPushButton("Выгрузить поляру в CSV")
    if on_report:
        w["btn_report"].clicked.connect(on_report)
    if on_csv:
        w["btn_csv"].clicked.connect(on_csv)
    f2.addRow("Шаблон:", w["template"])
    f2.addRow("Название проекта:", w["project_name"])
    f2.addRow(w["btn_report"])
    f2.addRow(w["btn_csv"])
    lay.addWidget(gb2)

    w["out"] = QTextEdit(); w["out"].setReadOnly(True)
    w["out"].setPlaceholderText(
        "Спецфункции: наклон поляры C_Lα, угол нулевой подъёмной силы, "
        "C_D0, коэффициент Освальда e, C_L max и угол сваливания, "
        "максимальное качество K и скорость сваливания.")
    lay.addWidget(w["out"], stretch=1)
    return page, w


# ---------------------------------------------------------------------------
# Пресеты конфигурации
# ---------------------------------------------------------------------------

def _collect_all_presets() -> dict:
    """Все конфиги пресетов: встроенные + официальные + пользовательские.

    Возвращает ``{подпись: {kind, name, params, desc}}``. Подпись
    («Встроенный: …», «Официальный: …», «Мой: …») — это текст пункта в
    выпадающем списке. ``kind``: builtin / official / user (по нему
    разрешаются «Сохранить» / «Удалить»).
    """
    import su2_autoconfig as _AC
    items = {}
    for name in _AC.PRESET_ORDER:
        label, desc = _AC.PRESET_INFO.get(name, (name, ""))
        items["Встроенный: %s (%s)" % (label, name)] = {
            "kind": "builtin", "name": name,
            "params": dict(_AC.PRESETS[name]), "desc": desc,
        }
    try:
        import su2_config_dialog as _D
        for label, meta in _D.OFFICIAL_PRESETS.items():
            items[label] = {                       # label = «Официальный: …»
                "kind": "official", "name": label,
                "params": dict(meta["params"]),
                "desc": meta.get("description", ""),
            }
        for name, params in _D.load_user_presets().items():
            items["Мой: %s" % name] = {
                "kind": "user", "name": name,
                "params": dict(params), "desc": "",
            }
    except Exception:                              # pragma: no cover
        pass
    return items


def _reload_preset_combo(combo) -> None:
    combo.clear()
    for label in _collect_all_presets():
        combo.addItem(label)


def preset_table_params(table) -> dict:
    """Собирает ``{ключ: значение}`` из редактируемой таблицы пресета."""
    params = {}
    if table is None:
        return params
    for i in range(table.rowCount()):
        k_item = table.item(i, 0)
        if k_item is None:
            continue
        k = str(k_item.text()).strip()
        if not k:
            continue
        v_item = table.item(i, 1)
        params[k] = str(v_item.text()) if v_item is not None else ""
    return params


def _load_selected_preset(w) -> None:
    """Показывает параметры выбранного конфига в редактируемой таблице."""
    label = w["combo"].currentText()
    info = _collect_all_presets().get(label)
    table = w["table"]
    table.setRowCount(0)
    if not info:
        w["out"].setText("Выберите конфиг — здесь появятся его параметры.")
        return
    params = info["params"]
    table.setRowCount(len(params))
    for i, (k, v) in enumerate(sorted(params.items())):
        table.setItem(i, 0, QTableWidgetItem(str(k)))
        table.setItem(i, 1, QTableWidgetItem(str(v)))
    head = "%s — параметров: %d." % (label, len(params))
    if info.get("desc"):
        head += "\n" + info["desc"]
    w["out"].setText(head)


def _save_user_preset(w) -> None:
    """Сохраняет текущие значения таблицы как пользовательский пресет."""
    import su2_config_dialog as _D
    params = preset_table_params(w["table"])
    if not params:
        w["out"].setText("Внимание: нет ни одного параметра для сохранения.")
        return
    name, ok = QInputDialog.getText(None, "Мой пресет", "Название пресета:")
    if not ok or not name.strip():
        return
    name = name.strip()
    presets = _D.load_user_presets()
    presets[name] = params
    _D.save_user_presets(presets)
    _reload_preset_combo(w["combo"])
    idx = w["combo"].findText("Мой: %s" % name)
    if idx >= 0:
        w["combo"].setCurrentIndex(idx)
    w["out"].setText(f"Пресет «{name}» сохранён ({len(params)} параметров).")


def _delete_user_preset(w) -> None:
    """Удаляет выбранный пользовательский пресет (с подтверждением)."""
    import su2_config_dialog as _D
    label = w["combo"].currentText()
    info = _collect_all_presets().get(label)
    if not info or info["kind"] != "user":
        w["out"].setText(
            "Удалить можно только свой пресет. Встроенные и официальные "
            "конфиги защищены от изменения, их можно скопировать под своим "
            "именем через «Сохранить как мой пресет».")
        return
    name = info["name"]
    ret = QMessageBox.question(
        None, "Удалить пресет",
        f"Удалить пресет «{name}»?",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    if ret != QMessageBox.Yes:
        return
    presets = _D.load_user_presets()
    if name in presets:
        del presets[name]
        _D.save_user_presets(presets)
    _reload_preset_combo(w["combo"])
    w["out"].setText(f"Пресет «{name}» удалён.")


def build_presets_page(on_export=None, on_import=None, on_apply=None) -> tuple:
    """Страница «Формат конфигурации»: выбор и правка пресетов config.cfg.

    Выпадающий список содержит ВСЕ конфиги (встроенные ultra/safe,
    официальные кейсы SU2 и пользовательские). Ниже — редактируемая
    таблица параметров выбранного конфига; значения можно менять. Под ней —
    «Применить к проекту», «Сохранить как мой пресет» и «Удалить мой пресет»,
    а также импорт/экспорт файла .su2preset.
    """
    page = QWidget()
    lay = QVBoxLayout(page)
    lay.setContentsMargins(10, 10, 10, 10)

    lbl = QLabel(
        "Конфиг — именованный набор параметров SU2. Выберите его в списке, "
        "при необходимости правьте значения в таблице, затем примените к "
        "проекту, сохраните как свой или удалите. Импорт/экспорт позволяют "
        "переносить настройки файлом .su2preset.")
    lbl.setWordWrap(True)
    lay.addWidget(lbl)

    gb = QGroupBox("Конфиг")
    f = QFormLayout(gb)
    w = {}
    w["combo"] = QComboBox()
    _reload_preset_combo(w["combo"])
    w["combo"].currentTextChanged.connect(
        lambda _t: _load_selected_preset(w))
    f.addRow("Конфиг:", w["combo"])

    w["table"] = QTableWidget(0, 2)
    w["table"].setHorizontalHeaderLabels(["Ключ", "Значение"])
    w["table"].horizontalHeader().setStretchLastSection(True)
    w["table"].horizontalHeader().setSectionResizeMode(
        0, QHeaderView.ResizeToContents)
    w["table"].horizontalHeader().setSectionResizeMode(
        1, QHeaderView.Stretch)
    w["table"].setSelectionBehavior(QAbstractItemView.SelectRows)
    w["table"].setMinimumHeight(220)
    w["table"].verticalHeader().setVisible(False)
    w["table"].setToolTip(
        "Значения можно менять. Ключи с ошибкой распознаются при "
        "применении к проекту как предупреждение (SU2 развивается).")
    f.addRow(w["table"])
    lay.addWidget(gb)

    row = QHBoxLayout()
    w["btn_apply"] = QPushButton("Готово: Применить к проекту")
    w["btn_save"] = QPushButton("Сохранить как мой пресет")
    w["btn_delete"] = QPushButton("Удалить мой пресет")
    w["btn_export"] = QPushButton("Экспорт…")
    w["btn_import"] = QPushButton("Импорт…")
    for b in (w["btn_apply"], w["btn_save"], w["btn_delete"],
              w["btn_export"], w["btn_import"]):
        row.addWidget(b)
    lay.addLayout(row)

    if on_apply:
        w["btn_apply"].clicked.connect(on_apply)
    if on_export:
        w["btn_export"].clicked.connect(on_export)
    if on_import:
        w["btn_import"].clicked.connect(on_import)
    w["btn_save"].clicked.connect(lambda: _save_user_preset(w))
    w["btn_delete"].clicked.connect(lambda: _delete_user_preset(w))

    w["out"] = QTextEdit(); w["out"].setReadOnly(True)
    w["out"].setPlaceholderText(
        "Описание конфига и результат проверки. Неизвестные ключи "
        "считаются предупреждением (SU2 развивается, и новый ключ не "
        "должен ломать старый пресет).")
    lay.addWidget(w["out"], stretch=1)

    _load_selected_preset(w)
    return page, w


# ---------------------------------------------------------------------------
# Пояснительные страницы корневых узлов дерева
# ---------------------------------------------------------------------------

INFO_PAGES = {
    "global_defs": {
        "title": "Global Definitions",
        "lead": (
            "Общие параметры, которые действуют на весь проект целиком, "
            "а не на отдельную деталь."),
        "body": [
            ("Условия полёта",
             "Скорость, высота и угол атаки, при которых считается модель. "
             "Из них приложение получает плотность, температуру и скоростной "
             "напор, а дальше подставляет их в config.cfg и в инженерные "
             "оценки. Менять их можно в любой момент — перестраивать сетку "
             "для этого не нужно."),
            ("Правила проектирования",
             "Ограничения и целевые значения, по которым приложение судит, "
             "удачен ли вариант: размах, стреловидность, удлинение, "
             "коэффициент подъёмной силы и прочие. Используются при "
             "переборе вариантов и при проверке результата."),
            ("Формат конфигурации",
             "Именованные пресеты настроек SU2. Нужны, чтобы переносить "
             "удачную конфигурацию решателя между проектами и хранить её "
             "в системе контроля версий."),
        ],
        "hint": ("Порядок работы: задайте условия полёта, затем при "
                 "необходимости уточните правила проектирования. Формат "
                 "конфигурации понадобится позже, при настройке расчёта."),
    },
    "component": {
        "title": "Component 1",
        "lead": "Геометрия модели и расчётная сетка, построенная по ней.",
        "body": [
            ("Состав модели",
             "Фюзеляж, крыло, механизация и оперение собираются либо "
             "параметрическими генераторами, либо импортом готовой "
             "геометрии. Каждой детали назначается роль — от неё зависит, "
             "какой маркер получит поверхность в сетке и как она будет "
             "учтена в коэффициентах."),
            ("Расчётная сетка",
             "Объёмная тетраэдральная сетка с внешней границей. Именно она, "
             "а не геометрия, передаётся решателю, поэтому любое изменение "
             "геометрии или плоскости симметрии требует перестроения сетки — "
             "приложение при этом помечает её устаревшей."),
        ],
        "hint": ("Порядок работы: сначала состав модели, затем плоскости "
                 "симметрии, и только после этого построение сетки. Если "
                 "поменять их местами, симметрия не применится."),
    },
    "study": {
        "title": "Study 1",
        "lead": "Как именно считать: решатель, режим и дополнительные оценки.",
        "body": [
            ("Настройки решателя",
             "Тип уравнений (EULER или RANS), число итераций, число ядер "
             "и вычислительное устройство. Здесь же выбирается пресет "
             "устойчивости — он подбирает CFL и схему, при которых расчёт "
             "не расходится на сложных сетках."),
            ("Многоточечная оптимизация",
             "Перебор вариантов по диапазонам параметров: полный факторный "
             "план, варьирование по одному параметру или латинский "
             "гиперкуб. Каждый вариант считается отдельным прогоном."),
            ("Аэроупругость и прочность",
             "Инженерные оценки, которые считаются внутри приложения, а не "
             "в SU2: скорости флаттера и дивергенции по методу p-k, "
             "напряжения в корневом сечении и запасы прочности. Нагрузки "
             "берутся из аналитической эпюры по размаху, поэтому эти "
             "оценки не зависят от сходимости аэродинамического расчёта."),
        ],
        "hint": ("Порядок работы: сначала настройки решателя и пробный "
                 "прогон, и только после сошедшегося решения — перебор "
                 "вариантов."),
    },
    "results": {
        "title": "Results",
        "lead": "Обработка того, что уже посчитано.",
        "body": [
            ("Балансировка",
             "Подбор отклонения руля высоты, при котором сумма моментов "
             "относительно центра масс равна нулю."),
            ("Поле обтекания",
             "Отображение распределения давлений или скоростей по "
             "поверхности по файлу решения SU2."),
            ("Поляра и отчёты",
             "Построение поляры по серии прогонов, поиск наилучшего "
             "качества и точки срыва, выгрузка отчёта и CSV."),
            ("История генерации",
             "Журнал построенных вариантов геометрии — позволяет вернуться "
             "к предыдущему состоянию модели."),
        ],
        "hint": ("Раздел доступен после завершённого расчёта: все страницы "
                 "читают файлы решения из каталога прогона."),
    },
}


def build_info_page(key: str, actions=None) -> QWidget:
    """Пояснительная страница корневого узла дерева.

    Вместо полей ввода показывает текст: что это за раздел и зачем он
    нужен. Кнопки перехода собраны в самом низу, под пояснениями.

    :param key: ключ в :data:`INFO_PAGES`.
    :param actions: список кортежей ``(подпись, подсказка, обработчик)``.
    """
    spec = INFO_PAGES[key]
    page = QWidget()
    lay = QVBoxLayout(page)
    lay.setContentsMargins(12, 12, 12, 12)
    lay.setSpacing(8)

    title = QLabel(spec["title"])
    title.setStyleSheet(
        "font-size: 13px; font-weight: bold; color: #22384A; border: none;")
    lay.addWidget(title)

    lead = QLabel(spec["lead"])
    lead.setWordWrap(True)
    lead.setStyleSheet("font-size: 11px; color: #2c4257; border: none;")
    lay.addWidget(lead)

    for name, text in spec["body"]:
        head = QLabel(name)
        head.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #2c4257;"
            " border: none; margin-top: 4px;")
        lay.addWidget(head)
        para = QLabel(text)
        para.setWordWrap(True)
        para.setStyleSheet(
            "font-size: 11px; color: #3d4a57; border: none;"
            " margin-left: 10px;")
        lay.addWidget(para)

    hint = QLabel(spec["hint"])
    hint.setWordWrap(True)
    hint.setStyleSheet(
        "font-size: 10px; color: #4A4A4A; font-style: italic;"
        " border: none; margin-top: 6px;")
    lay.addWidget(hint)

    # Пояснения занимают верх, кнопки перехода — самый низ страницы.
    lay.addStretch(1)

    if actions:
        sep = QLabel("Перейти к настройке")
        sep.setStyleSheet(
            "font-size: 10px; color: #6B7280; border: none;")
        lay.addWidget(sep)
        row = QHBoxLayout()
        row.setSpacing(6)
        for label, tip, handler in actions:
            btn = QPushButton(label)
            if tip:
                btn.setToolTip(tip)
            if handler:
                btn.clicked.connect(handler)
            row.addWidget(btn)
        row.addStretch(1)
        lay.addLayout(row)

    return page
