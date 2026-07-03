"""
run_extrapolation.py
====================
Fuehrt zwei Extrapolations-Simulationen ausserhalb des Trainingsbereichs durch:
    - sturm_25ms_45deg:       U_ref = 25.0 m/s, angle = 45 Grad  (ueber Trainingsmax ~19.3 m/s)
    - schwachwind_1_5ms_45deg: U_ref =  1.5 m/s, angle = 45 Grad  (unter Trainingsmin ~3.5 m/s)

Misst separat:
    - Vernetzungszeit  (blockMesh bis reconstructParMesh, Schritte 2-6)
    - Solver-Zeit      (simpleFoam, Schritt 10)
    - Gesamtzeit

Ergebnis wird in results/extrapolation_timing.yaml gespeichert.
"""

import os
import sys
import time
import yaml
from datetime import datetime

import createGeometry
import runSimulation


SIMULATIONS = [
    {"id": "sturm_25ms_45deg",        "U_ref": 25.0, "angle": 45.0},
    {"id": "schwachwind_1_5ms_45deg", "U_ref":  1.5, "angle": 45.0},
]

# Schritte, die zur Vernetzung gehoeren
MESH_STEPS = {2, 3, 4, 5, 6}
# Schritt des Solvers
SOLVER_STEPS = {10}


def load_config(path="config.yaml"):
    if not os.path.exists(path):
        print(f"FEHLER: '{path}' nicht gefunden.")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def run_with_timing(task, stl_path, bounds, cfg):
    """Fuehrt eine Simulation aus und misst Schritt-Timings."""

    step_times = {}   # step -> start_time
    step_names = {}   # step -> description

    def status_cb(case_name, step, total, desc):
        now = time.time()
        step_times[step] = now
        step_names[step] = desc
        label = f"[{case_name}] Schritt {step}/{total}: {desc}"
        print(f"  {datetime.now().strftime('%H:%M:%S')}  {label}")

    total_start = time.time()
    result = runSimulation.run_case(
        case_name=task["id"],
        stl_source=stl_path,
        params=task,
        bounds=bounds,
        cfg=cfg,
        status_callback=status_cb,
    )
    total_end = time.time()

    # Schritt-Dauern berechnen:
    # Dauer eines Schritts = Start des naechsten Schritts - Start dieses Schritts
    # Letzter Schritt: total_end - Start dieses Schritts
    sorted_steps = sorted(step_times.keys())
    step_durations = {}
    for i, step in enumerate(sorted_steps):
        if i + 1 < len(sorted_steps):
            duration = step_times[sorted_steps[i + 1]] - step_times[step]
        else:
            duration = total_end - step_times[step]
        step_durations[step] = {
            "name": step_names[step],
            "duration_s": round(duration, 1),
        }

    mesh_s   = sum(step_durations[s]["duration_s"] for s in MESH_STEPS   if s in step_durations)
    solver_s = sum(step_durations[s]["duration_s"] for s in SOLVER_STEPS if s in step_durations)
    total_s  = round(total_end - total_start, 1)

    return result, step_durations, mesh_s, solver_s, total_s


def fmt_time(seconds):
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}min {s}s"


def main():
    cfg = load_config("config.yaml")
    geo = cfg["geometry"]
    gen = cfg["general"]

    # S3-Upload deaktivieren (Extrapolations-Sims gehoeren nicht ins Training-Dataset)
    if "s3" in cfg:
        cfg["s3"]["enabled"] = False

    base_path = os.getcwd()
    for d in [gen["results_dir"], gen["geometry_dir"], gen["simulation_dir"]]:
        os.makedirs(os.path.join(base_path, d), exist_ok=True)

    timing_report = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "simulationen": [],
    }

    for task in SIMULATIONS:
        print(f"\n{'=' * 64}")
        print(f"  STARTE: {task['id']}")
        print(f"  U_ref = {task['U_ref']} m/s  |  angle = {task['angle']} Grad")
        print(f"{'=' * 64}\n")

        # Geometrie erzeugen
        stl_path = os.path.join(base_path, gen["geometry_dir"], f"{task['id']}.stl")
        bounds = createGeometry.create_building(
            geo["width"],
            geo["depth"],
            geo["wall_height"],
            geo["roof_height"],
            task["angle"],
            stl_path,
        )

        result, step_durations, mesh_s, solver_s, total_s = run_with_timing(
            task, stl_path, bounds, cfg
        )

        print(f"\n  --- Timing {task['id']} ---")
        print(f"  Vernetzung:  {fmt_time(mesh_s)}  ({mesh_s:.0f}s)")
        print(f"  Solver:      {fmt_time(solver_s)}  ({solver_s:.0f}s)")
        print(f"  Gesamt:      {fmt_time(total_s)}  ({total_s:.0f}s)")
        print(f"  Status:      {result['status']}")

        sim_entry = {
            "id":         task["id"],
            "U_ref":      task["U_ref"],
            "angle":      task["angle"],
            "status":     result["status"],
            "timing": {
                "vernetzung_s":  round(mesh_s, 1),
                "vernetzung":    fmt_time(mesh_s),
                "solver_s":      round(solver_s, 1),
                "solver":        fmt_time(solver_s),
                "gesamt_s":      total_s,
                "gesamt":        fmt_time(total_s),
            },
            "schritte": {
                str(step): {
                    "name":       info["name"],
                    "dauer_s":    info["duration_s"],
                    "dauer":      fmt_time(info["duration_s"]),
                }
                for step, info in step_durations.items()
            },
            "residuen":  result.get("residuals", {}),
        }
        timing_report["simulationen"].append(sim_entry)

    # Bericht speichern
    report_path = os.path.join(base_path, gen["results_dir"], "extrapolation_timing.yaml")
    with open(report_path, "w") as f:
        yaml.dump(timing_report, f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)

    print(f"\n{'=' * 64}")
    print(f"  FERTIG — Timing-Report: {report_path}")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
