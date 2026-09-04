import ast
import json
import operator
from dataclasses import dataclass, field
from enum import Enum


class RuleOperator(Enum):
    LE = "<="
    GE = ">="
    LT = "<"
    GT = ">"
    EQ = "=="
    NE = "!="
    BETWEEN = "BETWEEN"


class RuleSeverity(Enum):
    HARD = "Жёсткое"
    SOFT = "Мягкое"
    INFO = "Информ."


class RuleType(Enum):
    GEOMETRY = "Геометрия"
    AERO = "Аэродинамика"
    CUSTOM = "Прочее"


_OPS = {
    RuleOperator.LE: operator.le, RuleOperator.GE: operator.ge,
    RuleOperator.LT: operator.lt, RuleOperator.GT: operator.gt,
    RuleOperator.EQ: operator.eq, RuleOperator.NE: operator.ne,
}


@dataclass
class Rule:
    name: str
    parameter: str
    operator: RuleOperator
    value: object            # число или [min, max] для BETWEEN
    severity: RuleSeverity = RuleSeverity.SOFT
    rule_type: RuleType = RuleType.GEOMETRY
    weight: float = 1.0
    enabled: bool = True
    description: str = ""

    # ------------------------------------------------------------------
    def check(self, params: dict):
        """Возвращает (ok, величина_нарушения) или None, если параметра нет."""
        if self.parameter not in params:
            return None
        lhs = params[self.parameter]
        if self.operator == RuleOperator.BETWEEN:
            lo, hi = self.value
            if lo <= lhs <= hi:
                return True, 0.0
            viol = min(abs(lhs - lo), abs(lhs - hi))
            return False, float(viol)
        ok = bool(_OPS[self.operator](lhs, self.value))
        viol = 0.0 if ok else abs(float(lhs) - float(self.value))
        return ok, float(viol)

    def to_dict(self):
        return dict(name=self.name, parameter=self.parameter,
                    operator=self.operator.value, value=self.value,
                    severity=self.severity.value, rule_type=self.rule_type.value,
                    weight=self.weight, enabled=self.enabled,
                    description=self.description)

    @staticmethod
    def from_dict(d):
        return Rule(
            name=d["name"], parameter=d["parameter"],
            operator=RuleOperator(d["operator"]), value=d["value"],
            severity=RuleSeverity(d.get("severity", "Мягкое")),
            rule_type=RuleType(d.get("rule_type", "Геометрия")),
            weight=d.get("weight", 1.0), enabled=d.get("enabled", True),
            description=d.get("description", ""))


class RuleSet:
    def __init__(self, name: str = "default"):
        self.name = name
        self.rules = []

    # ------------------------------------------------------------------
    def add(self, rule: Rule):
        self.rules.append(rule)

    add_rule = add  # совместимость

    def remove(self, name: str) -> bool:
        for i, r in enumerate(self.rules):
            if r.name == name:
                del self.rules[i]
                return True
        return False

    remove_rule = remove

    def get(self, name: str):
        return next((r for r in self.rules if r.name == name), None)

    # ------------------------------------------------------------------
    def check_all(self, params: dict) -> dict:
        """Полная проверка. Возвращает dict:
        passed, messages, penalty, hard_violations, soft_violations, info_violations"""
        hard, soft, info = [], [], []
        messages = []
        penalty = 0.0
        for rule in self.rules:
            if not rule.enabled:
                messages.append(f"⊘ {rule.name}: отключено")
                continue
            res = rule.check(params)
            if res is None:
                messages.append(f"? {rule.name}: параметр '{rule.parameter}' не задан")
                continue
            ok, viol = res
            if ok:
                messages.append(f"Готово: {rule.name}: {rule.parameter} {rule.operator.value} "
                                f"{rule.value} — OK")
                continue
            entry = {"rule": rule.name, "violation": viol,
                     "severity": rule.severity.value}
            messages.append(f"Ошибка: {rule.name}: {rule.parameter} нарушает "
                            f"{rule.operator.value} {rule.value} (дельта {viol:.4f})")
            if rule.severity == RuleSeverity.HARD:
                hard.append(entry)
            elif rule.severity == RuleSeverity.SOFT:
                soft.append(entry)
                penalty += rule.weight * viol
            else:
                info.append(entry)
        return {"passed": not hard, "messages": messages, "penalty": penalty,
                "hard_violations": hard, "soft_violations": soft,
                "info_violations": info}

    # ------------------------------------------------------------------
    def check_consistency(self):
        """Поиск противоречий вида 'span >= 15 и span <= 10'."""
        conflicts = []
        bounds = {}
        for r in self.rules:
            if not r.enabled:
                continue
            lo, hi = bounds.setdefault(r.parameter, [None, None])
            if r.operator in (RuleOperator.GE, RuleOperator.GT):
                bounds[r.parameter][0] = r.value if lo is None else max(lo, r.value)
            elif r.operator in (RuleOperator.LE, RuleOperator.LT):
                bounds[r.parameter][1] = r.value if hi is None else min(hi, r.value)
            elif r.operator == RuleOperator.EQ:
                bounds[r.parameter][0] = r.value
                bounds[r.parameter][1] = r.value
            elif r.operator == RuleOperator.BETWEEN:
                bounds[r.parameter][0] = r.value[0] if lo is None else max(lo, r.value[0])
                bounds[r.parameter][1] = r.value[1] if hi is None else min(hi, r.value[1])
        for param, (lo, hi) in bounds.items():
            if lo is not None and hi is not None and lo > hi:
                conflicts.append(
                    f"Противоречие по '{param}': нижняя граница {lo} > верхней {hi}")
        return conflicts

    # ------------------------------------------------------------------
    def to_dict(self):
        return {"name": self.name, "rules": [r.to_dict() for r in self.rules]}

    @staticmethod
    def from_dict(d):
        rs = RuleSet(d.get("name", "loaded"))
        for rd in d.get("rules", []):
            rs.add(Rule.from_dict(rd))
        return rs

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "RuleSet":
        with open(path, "r", encoding="utf-8") as f:
            return RuleSet.from_dict(json.load(f))

    def serialize(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @staticmethod
    def deserialize(text: str) -> "RuleSet":
        return RuleSet.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Пресеты наборов правил
# ---------------------------------------------------------------------------

def _preset_basic() -> RuleSet:
    rs = RuleSet("Базовые ограничения")
    rs.add(Rule("Сужение крыла снизу", "taper_ratio", RuleOperator.GE, 0.25,
                RuleSeverity.HARD, RuleType.GEOMETRY))
    rs.add(Rule("Сужение крыла сверху", "taper_ratio", RuleOperator.LE, 1.0,
                RuleSeverity.HARD, RuleType.GEOMETRY))
    rs.add(Rule("Стреловидность", "sweep", RuleOperator.LE, 45.0,
                RuleSeverity.SOFT, RuleType.GEOMETRY))
    rs.add(Rule("Удлинение снизу", "aspect_ratio", RuleOperator.GE, 4.0,
                RuleSeverity.SOFT, RuleType.GEOMETRY))
    return rs


def _preset_uav() -> RuleSet:
    rs = RuleSet("БПЛА")
    rs.add(Rule("Размах БПЛА", "span", RuleOperator.LE, 6.0,
                RuleSeverity.HARD, RuleType.GEOMETRY))
    rs.add(Rule("Мах БПЛА", "mach", RuleOperator.LE, 0.5,
                RuleSeverity.HARD, RuleType.AERO))
    rs.add(Rule("Качество", "k", RuleOperator.GE, 10.0,
                RuleSeverity.SOFT, RuleType.AERO, weight=0.5))
    return rs


def _preset_transport() -> RuleSet:
    rs = RuleSet("Транспортный самолёт")
    rs.add(Rule("Размах", "span", RuleOperator.BETWEEN, [20.0, 80.0],
                RuleSeverity.HARD, RuleType.GEOMETRY))
    rs.add(Rule("Стреловидность", "sweep", RuleOperator.BETWEEN, [15.0, 40.0],
                RuleSeverity.SOFT, RuleType.GEOMETRY))
    rs.add(Rule("Мах крейсера", "mach", RuleOperator.BETWEEN, [0.6, 0.9],
                RuleSeverity.INFO, RuleType.AERO))
    return rs


PRESETS = {
    "Базовые ограничения": _preset_basic,
    "БПЛА": _preset_uav,
    "Транспортный самолёт": _preset_transport,
}


# --- совместимые алиасы прежнего API ---
OptimizationRule = Rule


def create_default_rules() -> RuleSet:
    return _preset_basic()
