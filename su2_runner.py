import os
import re
import subprocess
import csv
import shutil
from datetime import datetime

# Автоконфиг SU2 (устойчивые пресеты + детектор расхождения).
# Файл su2_autoconfig.py лежит рядом с этим скриптом.
try:
    import su2_autoconfig
except Exception as _e:
    su2_autoconfig = None
    print("⚠️  su2_autoconfig не загружен (", _e, ") — авто-пресеты отключены")


def _config_is_safe(cfg_path):
    """True, если в config.cfg уже стоят устойчивые настройки."""
    try:
        t = re.sub(r"\s+", "",
                   open(cfg_path, encoding="utf-8", errors="replace").read().upper())
        return "CFL_ADAPT=NO" in t and "MUSCL_FLOW=NO" in t
    except OSError:
        return False

# ================= НАСТРОЙКИ =================
SU2_EXE = r"C:\SU2\bin\SU2_CFD.exe"
TEMPLATE_CFG = "template.cfg"
MESH_FILE = "mesh.su2"
WORK_DIR_BASE = os.path.join(os.getcwd(), "cases")
RESULTS_DIR = os.path.join(os.getcwd(), "results")
# =============================================

def get_input():
    print("\n" + "="*50)
    print("   SU2 Console Runner v1.1 (Sweep Mode)")
    print("="*50)
    print("Режимы работы:")
    print("  1. Одиночный расчёт")
    print("  2. Серия расчётов (Поляра)")
    
    mode = input("Выберите режим (1 или 2): ").strip()
    
    try:
        mach = float(input("Число Маха: "))
        
        if mode == '2':
            aoa_start = float(input("Начальный AoA (град): "))
            aoa_end = float(input("Конечный AoA (град): "))
            aoa_step = float(input("Шаг AoA (град): "))
            
            # Генерируем список углов атаки
            aoa_list = []
            current_aoa = aoa_start
            while current_aoa <= aoa_end + (aoa_step / 10): # защита от ошибок float
                aoa_list.append(round(current_aoa, 2))
                current_aoa += aoa_step
            return mach, aoa_list
        else:
            aoa = float(input("Угол атаки (AoA, град): "))
            return mach, [aoa]
            
    except ValueError:
        print("❌ Ошибка: введите числа!")
        return None, None

def create_case_dir(aoa, mach):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    case_name = f"M{mach}_AoA{aoa}_{ts}"
    case_dir = os.path.join(WORK_DIR_BASE, case_name)
    os.makedirs(case_dir, exist_ok=True)
    return case_dir

def generate_config(case_dir, aoa, mach):
    with open(TEMPLATE_CFG, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        stripped = line.strip()
        # Динамическая подстановка параметров
        if stripped.startswith("AOA="):
            new_lines.append(f"AOA= {aoa}\n")
        elif stripped.startswith("MACH_NUMBER="):
            new_lines.append(f"MACH_NUMBER= {mach}\n")
        elif stripped.startswith("MESH_FILENAME="):
            new_lines.append(f"MESH_FILENAME= {MESH_FILE}\n") # Гарантируем правильное имя
        else:
            new_lines.append(line)

    cfg_path = os.path.join(case_dir, "config.cfg")
    with open(cfg_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    shutil.copy2(MESH_FILE, case_dir)
    return cfg_path

def run_su2(cfg_path):
    """Запуск SU2 с автоконфигом: при расхождении предлагает устойчивый
    пресет (safe, затем ultra) и перезапускает расчёт в той же папке."""
    case_dir = os.path.dirname(cfg_path)
    print(f"\n🚀 Запуск SU2 для {os.path.basename(case_dir)}...")

    def _once():
        try:
            return subprocess.run(
                [SU2_EXE, os.path.basename(cfg_path)],
                cwd=case_dir,
                capture_output=True,
                text=True,
                timeout=600
            )
        except subprocess.TimeoutExpired:
            print("⏱️ Превышено время ожидания (10 мин).")
            return None
        except FileNotFoundError:
            print(f"❌ Не найден SU2: {SU2_EXE}")
            return "NOEXE"

    def _state(res):
        out = ((res.stdout or "") + (res.stderr or "")) if res is not None else ""
        crashed = res is not None and res.returncode != 0
        converged = ("Converged: Yes" in out) or ("Converged : Yes" in out)
        return out, crashed, converged

    result = _once()
    if result == "NOEXE":
        return False, False
    if result is None:
        return False, False

    out, crashed, converged = _state(result)

    # ---- Автоконфиг: расхождение/ошибка SU2 ----
    if su2_autoconfig is not None and (crashed or not converged):
        verdict = su2_autoconfig.detect_result(case_dir, screen_text=out)
        if verdict["status"] in ("diverged", "error", "unknown"):
            print(f"\n⚠️  {verdict['detail']}")
            preset = "ultra" if _config_is_safe(cfg_path) else "safe"
            ans = input(
                f"Применить устойчивый пресет '{preset}' "
                "(CFL↓, 1-й порядок) и пересчитать? [д/н]: "
            ).strip().lower()
            if ans in ("д", "y", "yes", ""):
                _, changes = su2_autoconfig.apply_preset(cfg_path, preset)
                print(f"✅ Пресет '{preset}' применён:")
                for c in changes:
                    print("  ", c)
                r2 = _once()
                if r2 is None:
                    return False, False
                out, crashed, converged = _state(r2)
                if not crashed and converged:
                    print("✅ После пресета расчёт сошёлся.")
                elif preset == "safe":
                    print("⚠️  На safe не сошёлся.")
                    ans2 = input("Попробовать ultra (CFL 0.5)? [д/н]: ").strip().lower()
                    if ans2 in ("д", "y", "yes", ""):
                        su2_autoconfig.apply_preset(cfg_path, "ultra")
                        r3 = _once()
                        if r3 is not None:
                            out, crashed, converged = _state(r3)
                if crashed or not converged:
                    print("❌ Даже ultra не помог — вероятно, дело в сетке: "
                          "перегенерируй её качеством «Точная (медленно)».")

    return (not crashed), converged

def parse_results(case_dir):
    history_path = os.path.join(case_dir, "history.csv")
    if not os.path.exists(history_path):
        return None

    with open(history_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if len(lines) < 2: return None

    # Парсим заголовки (убираем мусор)
    headers = [h.strip().strip('"').strip() for h in lines[0].split(',')]
    # Парсим последнюю строку
    values = []
    for v in lines[-1].split(','):
        try: values.append(float(v.strip()))
        except ValueError: values.append(0.0)

    data = dict(zip(headers, values))
    
    cl = data.get("CL", 0.0)
    cd = data.get("CD", 0.0)
    cm = data.get("CMz", data.get("CM", 0.0))
    iters = int(data.get("Inner_Iter", 0))
    
    return cl, cd, cm, iters

def evaluate_results(cl, cd, cm, converged):
    status = "OK"
    warnings = []
    ld = "N/A"

    if not converged:
        status = "WARNING"
        warnings.append("Не сошёлся")
        
    if cd <= 0:
        status = "WARNING"
        warnings.append("Cd <= 0 (ошибка физики/сетки)")
    elif cd < 0.001:
        status = "WARNING"
        warnings.append("Cd слишком мал")
    else:
        ld = round(cl / cd, 2)

    return ld, status, ", ".join(warnings) if warnings else "Нет"

def save_polar_csv(all_results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"polar_{ts}.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Mach", "AoA", "CL", "CD", "CM", "L/D", "Status", "Warnings"])
        writer.writeheader()
        writer.writerows(all_results)
        
    return csv_path

def main():
    mach, aoa_list = get_input()
    if not aoa_list: return

    print(f"\n📋 Запланировано расчётов: {len(aoa_list)}")
    all_results = []

    for aoa in aoa_list:
        print("\n" + "-"*50)
        case_dir = create_case_dir(aoa, mach)
        cfg_path = generate_config(case_dir, aoa, mach)
        
        success, converged = run_su2(cfg_path)
        
        if success:
            res = parse_results(case_dir)
            if res:
                cl, cd, cm, iters = res
                ld, status, warns = evaluate_results(cl, cd, cm, converged)
                
                # Сохраняем в общую кучу
                all_results.append({
                    "Mach": mach, "AoA": aoa, "CL": cl, "CD": cd, 
                    "CM": cm, "L/D": ld, "Status": status, "Warnings": warns
                })
                
                # Красивый вывод
                print(f"✅ AoA={aoa:5.1f}° | CL={cl:7.4f} | CD={cd:7.4f} | L/D={str(ld):>5} | [{status}]")
                if warns != "Нет": print(f"   ⚠️ Предупреждения: {warns}")
            else:
                print(f"❌ AoA={aoa}: Не удалось прочитать history.csv")
        else:
            print(f"❌ AoA={aoa}: Расчёт прерван")

    # Итоговая таблица и сохранение
    if all_results:
        csv_path = save_polar_csv(all_results)
        print("\n" + "="*50)
        print("📈 ИТОГОВАЯ ПОЛЯРА")
        print("="*50)
        print(f"{'AoA':>5} | {'CL':>7} | {'CD':>7} | {'L/D':>6} | {'Status'}")
        print("-" * 40)
        for r in all_results:
            print(f"{r['AoA']:5.1f} | {r['CL']:7.4f} | {r['CD']:7.4f} | {str(r['L/D']):>6} | {r['Status']}")
        print("="*50)
        print(f"💾 Результаты сохранены в:\n   {csv_path}")

if __name__ == "__main__":
    main()