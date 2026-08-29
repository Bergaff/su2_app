import numpy as np


def isa_atmosphere(altitude_m: float):
    """МСА до 11 км. Возвращает (T[K], P[Pa], rho[kg/m3], a[m/s])."""
    T0, p0 = 288.15, 101325.0
    L = 0.0065
    R = 287.058
    g = 9.80665
    h = max(0.0, min(float(altitude_m), 11000.0))
    T = T0 - L * h
    P = p0 * (T / T0) ** (g / (L * R))
    rho = P / (R * T)
    a = float(np.sqrt(1.4 * R * T))
    return float(T), float(P), float(rho), a


def sutherland_viscosity(T: float) -> float:
    """Динамическая вязкость воздуха по формуле Сазерленда, Па·с."""
    mu0, T0, S = 1.716e-5, 273.15, 110.4
    return float(mu0 * ((T / T0) ** 1.5) * (T0 + S) / (T + S))


# --- совместимые алиасы прежнего API ---
def get_isa_atmosphere(altitude_m: float):
    return isa_atmosphere(altitude_m)


def calculate_reynolds(rho: float, velocity: float, length: float, T: float):
    length = max(float(length), 1e-6)
    mu = sutherland_viscosity(T)
    return float(rho * velocity * length / mu)


def compute_aero_forces(CL, CD, rho, velocity, area):
    q = 0.5 * rho * velocity ** 2
    return CL * q * area, CD * q * area


def compute_non_dim(lift, drag, rho, velocity, area):
    q = max(0.5 * rho * velocity ** 2 * area, 1e-12)
    return lift / q, drag / q
