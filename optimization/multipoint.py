from dataclasses import dataclass


@dataclass
class FlightPoint:
    """Расчётная (полётная) точка многоточечной оптимизации."""
    name: str = "Крейсер"
    aoa: float = 3.0
    weight: float = 1.0
    mach: float = 0.0
    altitude: float = 0.0

    def to_dict(self):
        return dict(name=self.name, aoa=self.aoa, weight=self.weight,
                    mach=self.mach, altitude=self.altitude)

    @staticmethod
    def from_dict(d):
        return FlightPoint(**d)


# --- совместимые имена прежнего API ---
OptimizationPoint = FlightPoint


def standard_cruise_points():
    return [FlightPoint("Крейсер", 3.0, 1.0)]


def takeoff_landing_points():
    return [FlightPoint("Взлёт", 8.0, 0.7), FlightPoint("Посадка", 10.0, 0.3)]


def high_speed_points():
    return [FlightPoint("Скоростной", 0.0, 1.0)]


PRESETS = {
    "Крейсерский режим": standard_cruise_points,
    "Взлёт/посадка": takeoff_landing_points,
    "Скоростной режим": high_speed_points,
}


def optimize_multipoint(points, evaluate_fn, param_bounds: dict, n_iter=10,
                        rule_set=None, log_fn=None):
    """Простой случайный поиск по средневзвешенной целевой функции точек."""
    import numpy as np
    names = list(param_bounds.keys())
    bounds = np.array([param_bounds[n] for n in names], dtype=float)
    best, best_cost = None, np.inf
    for it in range(n_iter):
        vec = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * np.random.rand(len(names))
        params = dict(zip(names, vec))
        penalty = 0.0
        if rule_set is not None:
            penalty = rule_set.check_all(params)["penalty"]
        total = wsum = 0.0
        for p in points:
            CL, CD = evaluate_fn(params, p)
            w = getattr(p, "weight", 1.0)
            total += w * (-CL / max(CD, 1e-9))
            wsum += w
        cost = total / max(wsum, 1e-9) + penalty
        if log_fn:
            log_fn(f"Итерация {it + 1}/{n_iter}: стоимость {cost:.4f}")
        if cost < best_cost:
            best_cost, best = cost, dict(params)
    return best, best_cost
