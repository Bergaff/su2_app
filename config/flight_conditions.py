"""
config/flight_conditions.py

Полётные условия: скорость, высота, AoA, ISA-атмосфера, пресеты.

Содержит:
- dataclass FlightConditions с полями speed_m_s, altitude_m, aoa_deg
- FLIGHT_PRESETS — словарь готовых наборов (Cessna 172, Piper PA-28, Beechcraft Bonanza, ...)
- compute_isa() — пересчёт T/P/rho/a для текущих условий
- to_physics_dict() — формирование словаря 'physics', который передаётся в
  solver/config_builder.build_su2_config / write_case_config.

Логика вынесена из MainWindow, чтобы UI только отображал и редактировал значения,
а всякие магические числа и формулы жили в одном месте и были покрыты типами.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

from physics.atmosphere import (
    isa_atmosphere,
    sutherland_viscosity,
)


# ---------------------------------------------------------------------------
# Модель данных
# ---------------------------------------------------------------------------

@dataclass
class FlightConditions:
    """Полётные условия одной расчётной точки / режима."""

    name: str = "Крейсерский"
    speed_m_s: float = 60.0
    altitude_m: float = 0.0
    aoa_deg: float = 3.0
    # привязка к пресету (если выбран из списка — отображаем имя)
    preset_name: str = ""

    # ------------------------------------------------------------------
    # ISA
    # ------------------------------------------------------------------
    def isa(self):
        """Возвращает (T [K], P [Pa], rho [кг/м^3], a [м/с])."""
        return isa_atmosphere(self.altitude_m)

    # ------------------------------------------------------------------
    # Физика для SU2
    # ------------------------------------------------------------------
    def to_physics_dict(self) -> dict:
        """
        Словарь physics, который передаётся в solver.config_builder.

        Все ключи соответствуют тому, что ожидает build_su2_config / workers.
        """
        T, P, rho, a = self.isa()
        mach = self.speed_m_s / max(a, 1e-9)
        return {
            "speed": self.speed_m_s,
            "aoa": self.aoa_deg,
            "altitude": self.altitude_m,
            "temperature": T,
            "pressure": P,
            "rho": rho,
            "a": a,
            "mach": mach,
            "mu": sutherland_viscosity(T),
            "name": self.name,
            "preset": self.preset_name,
        }

    # ------------------------------------------------------------------
    # Сериализация
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "FlightConditions":
        if d is None:
            return FlightConditions()
        return FlightConditions(
            name=str(d.get("name", "Крейсерский")),
            speed_m_s=float(d.get("speed_m_s", d.get("speed", 60.0))),
            altitude_m=float(d.get("altitude_m", d.get("altitude", 0.0))),
            aoa_deg=float(d.get("aoa_deg", d.get("aoa", 3.0))),
            preset_name=str(d.get("preset_name", "")),
        )

    @staticmethod
    def from_preset(name: str) -> "FlightConditions":
        if name not in FLIGHT_PRESETS:
            return FlightConditions(name=name)
        fc = FLIGHT_PRESETS[name]
        fc.preset_name = name
        return fc


# ---------------------------------------------------------------------------
# Пресеты — общая авиация (не БПЛА)
# ---------------------------------------------------------------------------
#
# Скорости в м/с, высоты в м, AoA в градусах. Это типичные крейсерские
# режимы для учебных/прогулочных самолётов, чтобы было от чего отталкиваться.
# Числа носят ориентировочный характер и подходят для предварительной
# прикидки аэродинамики.
#
FLIGHT_PRESETS: dict = {
    # --- Общая авиация (GA) ---
    "Cessna 172 (крейсер)": FlightConditions(
        name="Cessna 172 (крейсер)",
        speed_m_s=58.0,    # ~115 узлов
        altitude_m=1500.0, # 5000 ft
        aoa_deg=2.0,
    ),
    "Piper PA-28 Cherokee (крейсер)": FlightConditions(
        name="Piper PA-28 Cherokee (крейсер)",
        speed_m_s=55.0,    # ~110 узлов
        altitude_m=1200.0,
        aoa_deg=2.0,
    ),
    "Beechcraft Bonanza (крейсер)": FlightConditions(
        name="Beechcraft Bonanza (крейсер)",
        speed_m_s=75.0,    # ~150 узлов
        altitude_m=2400.0, # 8000 ft
        aoa_deg=2.0,
    ),

    # --- Чуть «соседних» режимов на будущее (не удалять, можно расширять) ---
    "Произвольный (ручной ввод)": FlightConditions(
        name="Произвольный (ручной ввод)",
        speed_m_s=60.0,
        altitude_m=0.0,
        aoa_deg=3.0,
    ),
}


def list_presets() -> list:
    """Возвращает список имён пресетов в порядке объявления."""
    return list(FLIGHT_PRESETS.keys())


# Алиас — чтобы и `from config.flight_conditions import list_flight_presets`,
# и `from config import list_flight_presets` работали одинаково.
list_flight_presets = list_presets
