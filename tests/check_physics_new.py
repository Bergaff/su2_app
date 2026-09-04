# -*- coding: utf-8 -*-
"""Быстрая проверка физики новых модулей (прогоняется разработчиком).

Это не часть tests/test_backend.py — отдельный сценарий для отладки
аэроупругости и прочности с подробным выводом.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from physics import aeroelastic as AE
from physics import structural as ST

fails = []


def chk(name, cond, extra=""):
    print(("  OK  " if cond else "  FAIL") + f" {name} {extra}")
    if not cond:
        fails.append(name)


print("== aeroelastic ==")
chk("C(0)=1", AE.theodorsen(0.0) == complex(1.0, 0.0))
_c = AE.theodorsen(0.2)
chk("|C(k)| < 1 при k>0", abs(_c) < 1.0, f"|C(0.2)|={abs(_c):.4f}")
chk("G(k) < 0 при k>0 (запаздывание следа)", _c.imag < 0.0,
    f"G(0.2)={_c.imag:+.4f}")
chk("C(∞) = 0.5", abs(AE.theodorsen(1.0e6) - 0.5) < 1e-4,
    f"C(1e6)={AE.theodorsen(1.0e6).real:.5f}")
chk("F(k) монотонно стремится к 0.5",
    all(AE.theodorsen(k1).real > AE.theodorsen(k2).real
        for k1, k2 in ((0.1, 0.5), (0.5, 2.0)))
    and AE.theodorsen(50.0).real > 0.5)
chk("рациональная аппроксимация C(k) близка к точной",
    max(abs(AE.theodorsen_rational(k) - AE.theodorsen(k))
        for k in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)) < 0.05,
    f"макс. |dC|={max(abs(AE.theodorsen_rational(k) - AE.theodorsen(k)) for k in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)):.4f}")

# --- дивергенция: аналитика ---
K_a, c, e, rho = 2.0e5, 2.0, 0.3, 1.225
v_d = AE.divergence_speed(K_a, c, e, rho)
v_ref = math.sqrt(2.0 * (K_a / (c * 2 * math.pi * e)) / rho)
chk("V_D совпадает с аналитикой", abs(v_d - v_ref) < 1e-6, f"{v_d:.4f}")
chk("e<=0 -> дивергенции нет", AE.divergence_speed(K_a, c, -0.3, rho) is None)

# --- нулевая скорость: частоты с учётом присоединённых масс ---
kw = dict(m=60.0, x_alpha=0.2, b=1.0, I_alpha=12.0, K_h=8.0e4,
          K_alpha=2.0e5, chord=2.0, e=0.3, rho=1.225)
a_par = AE.a_from_e(0.3, 2.0)
m_app = math.pi * rho * 1.0 ** 2
i_app = math.pi * rho * 1.0 ** 4 * (0.125 + a_par ** 2)
M = np.array([[60.0 + m_app, 60.0 * 0.2 * 1.0],
              [60.0 * 0.2 * 1.0, 12.0 + i_app]])
K = np.array([[8.0e4, 0.0], [0.0, 2.0e5]])
w_exact = np.sqrt(np.abs(np.linalg.eigvals(np.linalg.inv(M) @ K)))
f_exact = sorted(float(x) / (2 * math.pi) for x in w_exact)
f_pk = sorted(x["freq_hz"] for x in AE.pk_point(1e-9, **kw))
chk("U->0: частоты = собственные с присоединёнными массами",
    all(abs(p - q) / q < 0.02 for p, q in zip(f_pk, f_exact)),
    f"pk={[round(x,3) for x in f_pk]} exact={[round(x,3) for x in f_exact]}")
chk("U->0: демпфирование нулевое",
    all(abs(x["g"]) < 1e-6 for x in AE.pk_point(1e-9, **kw)))

# --- изолированный изгиб устойчив на всех скоростях ---
kw_iso = dict(kw)
kw_iso.update(x_alpha=0.0, e=0.0)
gs = [AE.pk_point(V, **kw_iso)[0]["g"] for V in (10, 50, 100, 200, 400)]
chk("изолированный изгиб: g<0 на всех V", all(g < 0 for g in gs),
    str([round(g, 4) for g in gs]))

# --- флатер: существует и растёт с жёсткостью кручения ---
v_f, diag = AE.flutter_speed(V_max=400, n_steps=120, **kw)
kw5 = dict(kw)
kw5["K_alpha"] = 6.0e5
v_f5, _ = AE.flutter_speed(V_max=800, n_steps=160, **kw5)
chk("V_F найден", v_f is not None, f"V_F={v_f and round(v_f,1)}")
chk("V_F растёт с K_alpha", v_f is not None and v_f5 is not None and v_f5 > v_f,
    f"{v_f and round(v_f,1)} -> {v_f5 and round(v_f5,1)}")
chk("g меняет знак около V_F", v_f is not None and
    any(d["g"] < 0 for d in diag if d["V"] < v_f) and
    any(d["g"] > 0 for d in diag if d["V"] > v_f))

# --- e<=0 (ОУ перед фокусом): флатер позже, чем при e>0 ---
kw_fwd = dict(kw)
kw_fwd["e"] = -0.3
v_f_fwd, _ = AE.flutter_speed(V_max=800, n_steps=160, **kw_fwd)
chk("при ОУ впереди фокуса V_F выше",
    (v_f_fwd is None) or (v_f is not None and v_f_fwd > v_f),
    f"e>0: {v_f and round(v_f,1)}, e<0: {v_f_fwd and round(v_f_fwd,1)}")

res = AE.flutter_assessment(span=30.0, chord_root=4.0, chord_tip=1.5,
                            mass_wing=9000.0, rho=1.225,
                            V_cruise=230.0, V_dive=350.0)
chk("flutter_assessment возвращает вердикт", bool(res.get("verdict")))

# --- сверка с тонкой аэродинамической теорией (Глауэрт) в пределе k->0 ---
print("== сверка с теорией тонкого профиля (k->0) ==")
rho_, U_, b_, a_ = 1.225, 120.0, 0.75, -0.15
B0, B1, B2 = AE.aero_matrices(rho_, U_, b_, a_, AE.theodorsen(0.0))
q_st = np.array([0.0, 0.3])           # угол атаки 0.3 рад
qd_st = np.array([0.4, 0.7])          # ḣ = 0.4 м/с (вниз), α̇ = 0.7 рад/с
aero = B0 @ q_st + B1 @ qd_st         # = [-L, +M]
L_model, M_model = -aero[0].real, aero[1].real
# Эталон (Глауэрт): A0 = ḣ/U − a·q̄, A1 = q̄, q̄ = α̇·b/U
qbar = qd_st[1] * b_ / U_
A0 = q_st[1] + qd_st[0] / U_ - a_ * qbar
A1 = qbar
CL = 2 * math.pi * (A0 + A1 / 2.0)
Cm_ac = (math.pi / 4.0) * (0.0 - A1)
e_ = b_ * (0.5 + a_)
L_ref = 0.5 * rho_ * U_ * U_ * (2 * b_) * CL
M_ref = 0.5 * rho_ * U_ * U_ * (2 * b_) ** 2 * Cm_ac + e_ * L_ref
chk("стационарная подъёмная сила = теории тонкого профиля",
    abs(L_model - L_ref) < 1e-6 * max(1.0, abs(L_ref)),
    f"модель={L_model:.4f} Н/м, теория={L_ref:.4f} Н/м")
chk("стационарный момент = теории тонкого профиля",
    abs(M_model - M_ref) < 1e-6 * max(1.0, abs(M_ref)),
    f"модель={M_model:.4f} Н·м/м, теория={M_ref:.4f} Н·м/м")
# квазистационарное демпфирование кручения не может быть отрицательным
dmp = [-(AE.aero_matrices(rho_, V, b_, aa, AE.theodorsen(0.0))[1][1, 1].real)
       for V in (30.0, 100.0, 250.0) for aa in (-0.4, -0.1, 0.2)]
chk("квазистационарное демпфирование кручения ≥ 0 при любом a",
    all(x >= -1e-9 for x in dmp))

print("== structural ==")
fr = ST.root_forces(L_total=1.0e6, span=30.0, mass_wing=0.0, dist="elliptic")
chk("эллипс: Q_root = L", abs(fr["Q"] - 1.0e6) < 1.0, f"Q={fr['Q']:.1f}")
chk("эллипс: M_root = 4/(3pi)·L·s",
    abs(fr["M"] - fr["M_analytic"]) / fr["M_analytic"] < 1e-6,
    f"M={fr['M']:.1f} analytic={fr['M_analytic']:.1f}")
fr_tri = ST.root_forces(L_total=1.0e6, span=30.0, dist="triangular")
chk("треугольник: M_root = L·s/3",
    abs(fr_tri["M"] - 1.0e6 * 15.0 / 3.0) / (1.0e6 * 15.0 / 3.0) < 1e-6,
    f"M={fr_tri['M']:.1f}")
fr_uni = ST.root_forces(L_total=1.0e6, span=30.0, dist="uniform")
chk("равномерно: M_root = L·s/2",
    abs(fr_uni["M"] - 1.0e6 * 7.5) / (1.0e6 * 7.5) < 1e-6)
chk("разгрузка массой уменьшает момент",
    ST.root_forces(1e6, 30.0, mass_wing=9000.0)["M"] < fr["M"])

try:
    ST.root_forces(1e6, 30.0, dist="косинус")
    chk("ValueError на плохом распределении", False)
except ValueError:
    chk("ValueError на плохом распределении", True)

sa = ST.structural_assessment(span=30.0, chord_root=4.0, mass_aircraft=4.0e4,
                              mass_wing=9000.0)
chk("structural_assessment: σ>0 и масса>0",
    sa["sigma"] > 0 and sa["mass_estimate"] > 0,
    f"σ={sa['sigma']/1e6:.1f} МПа, m={sa['mass_estimate']:.0f} кг")
chk("structural_assessment: вердикт есть", bool(sa["verdict"]))
sa2 = ST.structural_assessment(span=30.0, chord_root=4.0, mass_aircraft=4.0e4,
                               mass_wing=9000.0, cap_area=0.05)
chk("большая полка -> меньше напряжения", sa2["sigma"] < sa["sigma"])

print()
if fails:
    print("ПРОВАЛЕНО:", len(fails), fails)
    sys.exit(1)
print("ВСЕ ПРОВЕРКИ ФИЗИКИ ПРОЙДЕНЫ")
