"""Einzeltest: angle=0, U_ref=7.0 mit Zwei-Phasen-Solver."""
import os
import sys
import yaml

import test_createGeometry as createGeometry
import test_runSimulation as runSimulation


def status_cb(name, step, total, desc):
    sys.stdout.write(f"\r   [{name}] Schritt {step}/{total}: {desc:<50}")
    sys.stdout.flush()


def main():
    cfg = yaml.safe_load(open("test_config.yaml"))

    # Einzelsimulation mit den Problemparametern
    task = {"id": "test_angle0", "U_ref": 7.0, "angle": 0.0}
    geo = cfg["geometry"]

    stl_path = os.path.join("geometry", f"{task['id']}.stl")
    os.makedirs("geometry", exist_ok=True)

    bounds = createGeometry.create_building(
        geo["width"], geo["depth"],
        geo["wall_height"], geo["roof_height"],
        task["angle"], stl_path,
    )

    print(f"\n   Starte Test: angle={task['angle']}, U_ref={task['U_ref']}")
    print(f"   Bounds: {bounds}")
    print(f"   first_order_iterations: {cfg['solver'].get('first_order_iterations', 0)}")
    print()

    result = runSimulation.run_case(
        case_name=task["id"],
        stl_source=stl_path,
        params=task,
        bounds=bounds,
        cfg=cfg,
        status_callback=status_cb,
    )

    print(f"\n\n   === ERGEBNIS ===")
    print(f"   Status:    {result['status']}")
    if result.get("failed_step"):
        print(f"   Fehler:    {result['failed_step']}")
        print(f"   Meldung:   {result.get('error_message', '')}")
    print(f"   Residuen:  {result.get('residuals', {})}")
    print(f"   Drag:      {result.get('drag')}")
    print(f"   Lift:      {result.get('lift')}")
    print()


if __name__ == "__main__":
    main()
