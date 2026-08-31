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
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
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

def build_presets_page(on_export=None, on_import=None, on_apply=None) -> tuple:
    """Страница «Формат конфигурации»: именованные пресеты config.cfg."""
    page = QWidget()
    lay = QVBoxLayout(page)
    lay.setContentsMargins(10, 10, 10, 10)

    lbl = QLabel(
        "Пресет — именованный набор параметров SU2 (JSON, расширение "
        ".su2preset). Импорт/экспорт позволяют переносить удачные "
        "настройки между проектами и хранить их в системе контроля версий.")
    lbl.setWordWrap(True)
    lay.addWidget(lbl)

    gb = QGroupBox("Пресет")
    f = QFormLayout(gb)
    w = {}
    w["name"] = QLineEdit("Мой пресет")
    w["source"] = QComboBox()
    w["source"].addItem("Текущие настройки проекта", "session")
    w["source"].addItem("Встроенный: Стандартный", "std")
    w["source"].addItem("Встроенный: Устойчивый (safe)", "safe")
    w["source"].addItem("Встроенный: Ультра-устойчивый (ultra)", "ultra")
    w["btn_export"] = QPushButton("Экспортировать пресет…")
    w["btn_import"] = QPushButton("Импортировать пресет…")
    w["btn_apply"] = QPushButton("Готово: Применить импортированный к проекту")
    if on_export:
        w["btn_export"].clicked.connect(on_export)
    if on_import:
        w["btn_import"].clicked.connect(on_import)
    if on_apply:
        w["btn_apply"].clicked.connect(on_apply)
    f.addRow("Имя:", w["name"])
    f.addRow("Источник:", w["source"])
    f.addRow(w["btn_export"])
    f.addRow(w["btn_import"])
    f.addRow(w["btn_apply"])
    lay.addWidget(gb)

    w["out"] = QTextEdit(); w["out"].setReadOnly(True)
    w["out"].setPlaceholderText(
        "Содержимое пресета и результат проверки: неизвестные ключи "
        "считаются предупреждением (SU2 развивается, и новый ключ не "
        "должен ломать старый пресет), отсутствующие значения — тоже.")
    lay.addWidget(w["out"], stretch=1)
    return page, w
